# ---- base image ----
FROM python:3.11-slim

# Опционально: ускоряем и делаем тоньше
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app

# 1) зависимости
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2) код и словари
COPY . .

# 3) порт и команда запуска
EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
