import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import warnings
import requests
import streamlit as st
from bertopic import BERTopic

topic_model = BERTopic()

docs = ["El producto es bueno", "El envío fue lento"]
topics, probs = topic_model.fit_transform(docs)
text = st.text_area("Escribe una reseña")

if st.button("Analizar"):
    response = requests.post(
        "http://127.0.0.1:8000/analyze",
        json={"text": text}
    )
    
    st.write(response.json())

warnings.filterwarnings("ignore")

# -----------------------------
# ⚙️ CONFIG
# -----------------------------
st.set_page_config(page_title="Dashboard de Reseñas", layout="wide")
st.title("📊 Dashboard de Análisis de Reseñas")

# -----------------------------
# 📂 CARGAR ARCHIVO (FLEXIBLE)
# -----------------------------
file_path_1 = "data/processed/resultados.csv"
file_path_2 = "data/processed/test.csv"

if os.path.exists(file_path_1):
    file_path = file_path_1
elif os.path.exists(file_path_2):
    file_path = file_path_2
else:
    st.error("❌ No se encontró ningún archivo CSV en data/processed/")
    st.stop()

df = pd.read_csv(file_path)

# -----------------------------
# 🧹 LIMPIAR COLUMNAS
# -----------------------------
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# 🧠 CREAR SENTIMIENTO SI NO EXISTE
# -----------------------------
if 'sentiment' not in df.columns:
    if 'stars' in df.columns:
        def map_sentiment(stars):
            try:
                stars = float(stars)
            except:
                return "neutral"

            if stars <= 2:
                return "negativo"
            elif stars == 3:
                return "neutral"
            else:
                return "positivo"

        df['sentiment'] = df['stars'].apply(map_sentiment)
    else:
        st.error("❌ No existe columna 'stars' para generar sentimiento")
        st.stop()

# -----------------------------
# 🧾 DETECTAR TEXTO
# -----------------------------
if 'review' not in df.columns:
    if 'review_body' in df.columns:
        df.rename(columns={'review_body': 'review'}, inplace=True)
    elif 'text' in df.columns:
        df.rename(columns={'text': 'review'}, inplace=True)
    else:
        st.error(f"❌ No se encontró columna de texto. Columnas: {df.columns.tolist()}")
        st.stop()

# -----------------------------
# 📊 VISTA PREVIA
# -----------------------------
st.subheader("📄 Vista previa")
st.dataframe(df.head())

# -----------------------------
# 📊 SENTIMIENTO
# -----------------------------
st.subheader("📊 Distribución de Sentimientos")

sent_counts = df['sentiment'].value_counts()

if sent_counts.empty:
    st.warning("⚠️ No hay datos de sentimiento")
else:
    fig1, ax1 = plt.subplots()
    ax1.bar(sent_counts.index, sent_counts.values)
    ax1.set_xlabel("Sentimiento")
    ax1.set_ylabel("Cantidad")
    st.pyplot(fig1)

# -----------------------------
# 📊 TÓPICOS
# -----------------------------
if 'topic' in df.columns:
    st.subheader("📊 Distribución de Tópicos")

    topic_counts = df['topic'].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.bar(topic_counts.index.astype(str), topic_counts.values)
    ax2.set_xlabel("Tópico")
    ax2.set_ylabel("Cantidad")
    st.pyplot(fig2)
else:
    st.warning("⚠️ No existe la columna 'topic'")

# -----------------------------
# ☁️ WORDCLOUD
# -----------------------------
st.subheader("☁️ Nube de palabras")

if 'clean_text' in df.columns:
    text_data = " ".join(df['clean_text'].dropna().astype(str))
else:
    text_data = " ".join(df['review'].dropna().astype(str))

if text_data.strip() == "":
    st.warning("⚠️ No hay texto suficiente")
else:
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    fig3, ax3 = plt.subplots()
    ax3.imshow(wc, interpolation='bilinear')
    ax3.axis("off")
    st.pyplot(fig3)

# -----------------------------
# 🎛️ FILTRO
# -----------------------------
st.subheader("🎛️ Filtrar por sentimiento")

selected = st.selectbox("Selecciona", df['sentiment'].unique())

filtered = df[df['sentiment'] == selected]

st.write(f"Resultados: {len(filtered)}")
st.dataframe(filtered.head())