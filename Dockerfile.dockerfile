# Usamos uma imagem base do Python
FROM python:3.9-slim

# Instala o Tesseract e as dependências do OpenCV
RUN apt-get update && apt-get install -y tesseract-ocr libgl1-mesa-glx

# Prepara o ambiente para a nossa aplicação
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do nosso código
COPY . .

# Comando para iniciar o servidor
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app", "--bind", "0.0.0.0:10000"]