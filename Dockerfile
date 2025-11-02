# Базовый образ со всем нужным
FROM python:3.11-slim

# Чтобы apt не задавал вопросы
ENV DEBIAN_FRONTEND=noninteractive

# Попплер (для pdf2image), tesseract, рус/англ языки
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Директория приложения
WORKDIR /app

# Зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY app.py .

# Важное: Render даёт PORT, слушаем его
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV PYTHONUNBUFFERED=1

# Старт
CMD ["/bin/sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
