FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# ВАЖНО: shell-форма, чтобы $PORT подставился Render'ом
CMD sh -c 'gunicorn -w 2 -b 0.0.0.0:$PORT app:app'
