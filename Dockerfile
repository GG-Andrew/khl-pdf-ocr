FROM python:3.11-slim

# Нужные системные либы для PyMuPDF/Pillow (никакого poppler и tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Render сам прокинет $PORT
CMD gunicorn app:app -b 0.0.0.0:${PORT:-10000} -w 2 --timeout 120
