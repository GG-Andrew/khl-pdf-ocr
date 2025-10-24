from fastapi import FastAPI, Query
from pdf2image import convert_from_path
import pytesseract
import tempfile
import requests
import os
import re

app = FastAPI(title="KHL PDF OCR Server")

@app.get("/")
def root():
    return {"ok": True, "service": "KHL PDF OCR", "status": "ready"}

@app.get("/ocr")
def ocr_parse(match_id: int = Query(...), pdf_url: str = Query(...)):
    """
    Пример:
    /ocr?match_id=897678&pdf_url=https://www.khl.ru/pdf/1369/897678/game-897678-start-ru.pdf
    """
    try:
        # --- Формируем правильный Referer для khl.ru ---
        m = re.search(r"/pdf/(\d{3,5})/(\d{6})/", pdf_url)
        if m:
            season, mid = m.group(1), m.group(2)
            referer = f"https://www.khl.ru/game/{season}/{mid}/preview/"
        else:
            referer = "https://www.khl.ru/"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Referer": referer,
            "Accept": "application/pdf",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        }

        # --- Скачиваем PDF с нужными заголовками ---
        r = requests.get(pdf_url, headers=headers, timeout=30, allow_redirects=True)
        if r.status_code != 200 or "pdf" not in r.headers.get("content-type", "").lower():
            return {"ok": False, "step": "GET", "status": r.status_code}

        # --- OCR обработка PDF ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

        pages = convert_from_path(tmp_path, dpi=200)
        text_full = ""
        for page in pages:
            text_full += pytesseract.image_to_string(page, lang="rus+eng") + "\n"

        os.remove(tmp_path)

        # --- Формируем текстовый ответ ---
        lines = [l.strip() for l in text_full.splitlines() if l.strip()]
        return {
            "ok": True,
            "match_id": match_id,
            "pdf_len": len(r.content),
            "text_len": len(text_full),
            "snippet": "\n".join(lines[:30])
        }

    except Exception as e:
        return {"ok": False, "step": "PARSE", "error": str(e)}
