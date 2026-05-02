from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, model_name='nlptown/bert-base-multilingual-uncased-sentiment'):
        # Modelo real entrenado en reseñas (1-5 estrellas)
        self.classifier = pipeline('sentiment-analysis', model=model_name)
    
    def analyze(self, text):
        result = self.classifier(text[:512])[0]  # BERT limita a 512 tokens
        label = result['label']
        score = result['score']
        
        # Convertir rating a sentimiento comprensible
        rating = int(label.split()[0])  # '1 star' -> 1
        if rating <= 2:
            sentiment = 'negativo'
        elif rating == 3:
            sentiment = 'neutro'
        else:
            sentiment = 'positivo'
        
        return {
            'sentiment': sentiment,
            'rating': rating,
            'confidence': score
        }