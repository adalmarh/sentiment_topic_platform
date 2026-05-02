import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline import ReviewPipeline

app = FastAPI(title="Análisis de Sentimiento API")

# Inicializar pipeline (una sola vez al iniciar)
pipeline = ReviewPipeline(lang='es')

class ReviewRequest(BaseModel):
    text: str

class ReviewResponse(BaseModel):
    original_text: str
    clean_text: str
    sentiment: str
    rating: int
    confidence: float

@app.post("/analyze", response_model=ReviewResponse)
async def analyze_review(request: ReviewRequest):
    try:
        result = pipeline.process_single(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "message": "API funcionando correctamente"}

@app.get("/")
async def root():
    return {"message": "API de Análisis de Sentimiento - Usa POST /analyze"}