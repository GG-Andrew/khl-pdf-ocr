FROM python:3.11-slim

# Системные зависимости (poppler для pdf2image, tesseract с рус/англ)
COPY apt.txt /tmp/apt.txt
RUN apt-get update && xargs -a /tmp/apt.txt apt-get install -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

# Переменные окружения для tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV PYTHONUNBUFFERED=1

# Питон-зависимости
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY app.py /app/app.py

# gunicorn
EXPOSE 10000
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:10000", "app:app"]
