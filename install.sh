#!/usr/bin/env bash
apt-get update
apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus poppler-utils
pip install --no-cache-dir -r requirements.txt
