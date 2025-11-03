FROM python:3.12-slim

# системные либы для PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Render сам подставит $PORT
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:$PORT", "-w", "2", "--timeout", "120"]
