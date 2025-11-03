# Базовый образ
FROM python:3.11-slim

# Устанавливаем системные зависимости + Tesseract с рус/англ языковыми пакетами
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Рабочая папка
WORKDIR /app

# Устанавливаем зависимости Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY app.py ./

# Tesseract будет искать языки здесь
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Стартуем через gunicorn; Render сам подставит $PORT
CMD bash -lc 'gunicorn app:app -b 0.0.0.0:${PORT:-8000} -w 2 --timeout 120'
