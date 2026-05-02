FROM python:3.10

WORKDIR /app

# Primero copiamos solo los requerimientos e instalamos
# Esto hace que Railway no reinstale todo si solo cambias una línea de código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Luego copiamos el resto del código
COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]