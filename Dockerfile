FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Системные пакеты: poppler для pdf2image, tesseract (включая рус. язык), либы для PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ставим Python-зависимости
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Render сам пробрасывает $PORT. Откроем дефолтный порт для локала.
EXPOSE 10000

# Старт
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
