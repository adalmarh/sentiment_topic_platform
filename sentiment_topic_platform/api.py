from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Modelo de entrada
class Review(BaseModel):
    text: str

# Modelo de sentimiento
sentiment_model = pipeline("sentiment-analysis")

@app.post("/analyze")
def analyze(review: Review):
    result = sentiment_model(review.text)
    return {
        "text": review.text,
        "sentiment": result
    }