from fastapi import FastAPI, Query
from pdf2image import convert_from_path
import pytesseract
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
    Пример:
    /ocr?match_id=897678&pdf_url=https://www.khl.ru/pdf/1369/897678/game-897678-start-ru.pdf
    """
    try:
        # скачиваем PDF с khl.ru (нужен referer, иначе 403)
        headers = {"Referer": "https://www.khl.ru"}
        r = requests.get(pdf_url, headers=headers, timeout=20)
        if r.status_code != 200:
            return {"ok": False, "step": "GET", "status": r.status_code}

        # временный файл PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

        # конвертация страниц в изображения
        pages = convert_from_path(tmp_path, dpi=200)
        text_full = ""
        for page in pages:
            text_full += pytesseract.image_to_string(page, lang="rus+eng") + "\n"

        os.remove(tmp_path)

        # чистим и упрощаем результат
        lines = [l.strip() for l in text_full.splitlines() if l.strip()]
        sample = "\n".join(lines[:30])

        return {
            "ok": True,
            "match_id": match_id,
            "pdf_len": len(r.content),
            "text_len": len(text_full),
            "snippet": sample
        }

    except Exception as e:
        return {"ok": False, "step": "PARSE", "error": str(e)}
