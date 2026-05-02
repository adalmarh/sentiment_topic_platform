from datasets import load_dataset
import pandas as pd
from src.pipeline import ReviewPipeline
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='es')

def translate_text(text):
    try:
        return translator.translate(text)
    except:
        return text  # si falla, deja el original

print("🌐 Traduciendo reseñas... (puede tardar)")

df['review_es'] = df['review'].apply(translate_text)

# Usar texto traducido
df['review'] = df['review_es']
# -----------------------------

df = df.sample(500)

#-----------------------------
# 🎯 Función para mapear sentimiento
# -----------------------------
def map_sentiment(stars):
    if stars <= 2:
        return "negativo"
    elif stars == 3:
        return "neutral"
    else:
        return "positivo"

# -----------------------------
# 🚀 MAIN
# -----------------------------
def main():
    print("=== INICIANDO PIPELINE DE ANÁLISIS ===")

    # -----------------------------
    # 1. Cargar dataset real
    # -----------------------------
    print("\n1. Cargando dataset de reseñas en español...")
    
    dataset = load_dataset("SetFit/amazon_reviews_multi_es")
    df = dataset['train'].to_pandas()

    print(f"   ✅ Reseñas cargadas: {len(df)}")

    # -----------------------------
    # 2. Preprocesar columnas
    # -----------------------------
    print("\n2. Preparando datos...")

    df = df[['review_body', 'stars']]  # columnas necesarias
    df = df.dropna()

    df['sentiment'] = df['stars'].apply(map_sentiment)
    df.rename(columns={'review_body': 'review'}, inplace=True)

    # (Opcional) reducir tamaño para pruebas
    df = df.sample(2000, random_state=42)

    # -----------------------------
    # 3. Inicializar pipeline
    # -----------------------------
    print("\n3. Inicializando pipeline...")
    pipeline = ReviewPipeline(lang='es', n_topics=5)

    # -----------------------------
    # 4. Ejecutar pipeline
    # -----------------------------
    print("\n4. Procesando reseñas...")
    results = pipeline.run(df['review'].tolist())

    # Convertir resultados a DataFrame
    results_df = pd.DataFrame(results)

    # Unir con datos originales
    df_final = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    # -----------------------------
    # 5. Guardar resultados
    # -----------------------------
    output_path = "data/processed/test.csv"
    df_final.to_csv(output_path, index=False)

    print(f"\n✅ Resultados guardados en: {output_path}")

# -----------------------------
if __name__ == "__main__":
    main()