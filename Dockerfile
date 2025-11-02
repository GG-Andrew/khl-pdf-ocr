FROM python:3.11-slim

# Системные зависимости и Tesseract с языками
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Tesseract на Debian/Ubuntu кладёт модели сюда:
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Render передаст порт через $PORT, по умолчанию 8000
ENV PORT=8000
EXPOSE 8000

CMD ["python", "app.py"]
