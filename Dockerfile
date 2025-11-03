# ---- base image
FROM python:3.11-slim

# ---- system deps (для PyMuPDF и Tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# ---- app files
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---- runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Render задаёт порт в $PORT — используем его
CMD ["sh","-c","gunicorn app:app -b 0.0.0.0:$PORT -w 2 --timeout 120"]
