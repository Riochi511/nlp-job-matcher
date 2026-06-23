import io
import joblib
import numpy as np
import pandas as pd
import pdfplumber
import docx

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

# Load artifacts directly from repo
vectorizer   = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
jobs_df      = pd.read_csv("jobs_clean.csv")

app = FastAPI(title="NLP Job Matcher")

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

def match_jobs(resume_text, top_n=10):
    resume_vec   = vectorizer.transform([resume_text])
    similarities = cosine_similarity(resume_vec, tfidf_matrix).flatten()
    top_indices  = np.argsort(similarities)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = jobs_df.iloc[idx]
        results.append({
    "rank":        rank,
    "title":       str(row.get("title", "N/A")),
    "company":     str(row.get("company_name", "N/A")),
    "preview":     str(row.get("description", ""))[:150] + "...",
    "similarity":  round(float(similarities[idx]), 4),
})
    return results

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