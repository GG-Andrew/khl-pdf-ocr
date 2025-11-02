# Python образ
FROM python:3.11-slim

# Системные пакеты для Tesseract и рендеринга PDF в изображения
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Каталог приложения
WORKDIR /app

# Сначала зависимости
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY app.py ./
# (если у тебя есть players_master.csv / referees_master.csv — тоже скопируй)
# COPY players_master.csv referees_master.csv ./

# Важно: Render прокидывает порт в $PORT
ENV PORT=10000
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Gunicorn — продуктивный сервер
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:${PORT}", "app:app"]
