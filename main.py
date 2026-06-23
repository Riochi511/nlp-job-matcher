import io
import os
import joblib
import numpy as np
import pandas as pd
import pdfplumber
import docx
import gdown

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

# ── Download artifacts from Google Drive on cold start ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS = {
    "tfidf_vectorizer.pkl": "1Hxjw3hZOGdym32IfMKrnrUM4PZ3MlEOl",
    "tfidf_matrix.pkl":     "1ad1dQC9UXwipP5jhEr4jP1S_WAmqSLNm",
    "jobs_clean.csv":       "19DW0X-8bUV6S2nWayf5p1DsWp0m4NqqY",
}

for filename, file_id in ARTIFACTS.items():
    dest = os.path.join(BASE_DIR, filename)
    if not os.path.exists(dest):
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", dest, quiet=False)
    else:
        print(f"{filename} already exists, skipping download.")

# ── Load artifacts ───────────────────────────────────────────────────────────
vectorizer   = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
tfidf_matrix = joblib.load(os.path.join(BASE_DIR, "tfidf_matrix.pkl"))
jobs_df      = pd.read_csv(os.path.join(BASE_DIR, "jobs_clean.csv"))

app = FastAPI(title="NLP Job Matcher")

# ── Text extraction helpers ──────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_resume_text(file):
    file_bytes = file.file.read()
    if file.filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif file.filename.lower().endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Upload a PDF or DOCX file.")

# ── Matching logic ───────────────────────────────────────────────────────────
def match_jobs(resume_text, top_n=10):
    resume_vec   = vectorizer.transform([resume_text])
    similarities = cosine_similarity(resume_vec, tfidf_matrix).flatten()
    top_indices  = np.argsort(similarities)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = jobs_df.iloc[idx]
        results.append({
            "rank":       rank,
            "title":      str(row.get("title", "N/A")),
            "company":    str(row.get("company_name", "N/A")),
            "preview":    str(row.get("description", ""))[:150] + "...",
            "similarity": round(float(similarities[idx]), 4),
        })
    return results

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "NLP Job Matcher is running. POST to /match"}

@app.post("/match")
async def match(
    resume_text: Optional[str]        = Form(None),
    resume_file: Optional[UploadFile] = File(None),
    top_n:       int                  = Form(10),
):
    if resume_file and resume_file.filename:
        text = extract_resume_text(resume_file)
    elif resume_text and resume_text.strip():
        text = resume_text.strip()
    else:
        raise HTTPException(status_code=422, detail="Provide resume_text or upload a PDF/DOCX.")
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from file.")
    matches = match_jobs(text, top_n=top_n)
    return JSONResponse(content={"matches": matches, "total": len(matches)})