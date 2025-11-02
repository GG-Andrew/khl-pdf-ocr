# ---- Base image
FROM python:3.11-slim

# Системные зависимости для PyMuPDF, Tesseract (если понадобится в дальнейшем)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    libglib2.0-0 libgl1 libpoppler-cpp0 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Чтобы логи сразу шли в stdout
ENV PYTHONUNBUFFERED=1

# Запуск (PORT даёт Render)
CMD sh -c "gunicorn app:app -b 0.0.0.0:${PORT:-10000} -w 2 --timeout 120"
