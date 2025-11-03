# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Tesseract + рус/англ языки
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    fonts-dejavu \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

# Render стартует так:
CMD ["gunicorn","app:app","-b","0.0.0.0:10000","-w","2","--timeout","180"]
