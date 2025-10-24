from fastapi import FastAPI, Query
import pytesseract
from pdf2image import convert_from_path
import tempfile
import requests
import os
import json

app = FastAPI(title="KHL PDF OCR Server")

@app.get("/")
def root():
    return {"ok": True, "service": "KHL PDF OCR", "status": "ready"}

@app.get("/ocr")
def ocr_parse(
    match_id: int = Query(...),
    pdf_url: str = Query(...)
):
    """
    Пример запроса:
    /ocr?match_id=897678&pdf_url=https://www.khl.ru/pdf/1369/897678/game-897678-start-ru.pdf
    """
    try:
        # скачиваем PDF
        r = requests.get(pdf_url, timeout=20)
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code} при скачивании PDF"}

        # сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

        # конвертируем в изображения
        pages = convert_from_path(tmp_path, dpi=200)
        text_all = ""
        for page in pages:
            text = pytesseract.image_to_string(page, lang="rus+eng")
            text_all += text + "\n"

        os.unlink(tmp_path)

        # возвращаем результат
        return {
            "ok": True,
            "match_id": match_id,
            "text_len": len(text_all),
            "snippet": text_all[:400]
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
