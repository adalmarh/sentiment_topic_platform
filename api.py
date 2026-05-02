from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API funcionando 🚀"}

# ⚡ versión ligera SIN transformers (para probar que Railway funciona)
@app.post("/analyze")
def analyze(request: TextRequest):
    text = request.text.lower()

    if "bueno" in text or "excelente" in text:
        sentiment = "positivo"
    elif "malo" in text or "terrible" in text:
        sentiment = "negativo"
    else:
        sentiment = "neutral"

    return {
        "text": request.text,
        "sentiment": sentiment
    }

@app.get("/health")
def health():
    return {"status": "ok"}