FROM python:3.11-slim

# Библиотеки для PyMuPDF и Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render сам подставит $PORT. НИКАКИХ Start Command в настройках Render.
CMD gunicorn app:app -b 0.0.0.0:${PORT:-10000} -w 2 --timeout 120
