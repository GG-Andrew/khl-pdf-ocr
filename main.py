from fastapi import FastAPI, Query
from pdf2image import convert_from_path
import pytesseract
import tempfile, requests, os, re, time
from typing import List
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

app = FastAPI(title="KHL PDF OCR Server")

def preprocess(img: Image.Image, scale: float = 1.5, bin_thresh: int = 190) -> Image.Image:
    """
    Простая и быстрая предобработка:
      - апскейл (1.5x)
      - в градации серого
      - автоконтраст
      - лёгкая резкость
      - лёгкая бинаризация
    """
    w, h = img.size
    if scale != 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    img = img.convert("L")                                    # grayscale
    img = ImageOps.autocontrast(img, cutoff=1)               # автоконтраст
    img = ImageEnhance.Sharpness(img).enhance(1.2)           # чуть резкости
    # мягкая бинаризация (оставляет полутона, но «гасит» шум)
    img = img.point(lambda p: 255 if p > bin_thresh else 0)
    return img

@app.get("/")
def root():
    return {"ok": True, "service": "KHL PDF OCR", "status": "ready"}

@app.get("/ocr")
def ocr_parse(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    dpi: int = Query(300, ge=120, le=360, description="DPI растрирования PDF"),
    max_pages: int = Query(2, ge=1, le=5, description="Сколько первых страниц OCR’ить"),
    scale: float = Query(1.5, ge=1.0, le=2.0, description="Апскейл картинки перед OCR"),
    bin_thresh: int = Query(190, ge=120, le=230, description="Порог бинаризации 0-255")
):
    """
    Пример:
    /ocr?match_id=897678&pdf_url=https://www.khl.ru/pdf/1369/897678/game-897678-start-ru.pdf&dpi=300&max_pages=2
    """
    t0 = time.time()
    # корректный Referer для khl.ru
    m = re.search(r"/pdf/(\d{3,5})/(\d{6})/", pdf_url)
    referer = f"https://www.khl.ru/game/{m.group(1)}/{m.group(2)}/preview/" if m else "https://www.khl.ru/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Referer": referer,
        "Accept": "application/pdf",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }

    try:
        r = requests.get(pdf_url, headers=headers, timeout=30, allow_redirects=True)
        if r.status_code != 200 or "pdf" not in r.headers.get("content-type","").lower():
            return {"ok": False, "step": "GET", "status": r.status_code}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

        # Берём только первые max_pages для скорости
        pages = convert_from_path(tmp_path, dpi=dpi, first_page=1, last_page=max_pages)
        os.remove(tmp_path)

        # Тессеракт: LSTM + табличный текст
        # preserve_interword_spaces=1 — меньше склеенных фамилий
        tconfig = "--oem 1 --psm 6 -c preserve_interword_spaces=1"

        parts: List[str] = []
        ocr_start = time.time()
        for p in pages:
            pp = preprocess(p, scale=scale, bin_thresh=bin_thresh)
            parts.append(pytesseract.image_to_string(pp, lang="rus+eng", config=tconfig))

        text = "\n".join(parts)
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        return {
            "ok": True,
            "match_id": match_id,
            "pdf_len": len(r.content),
            "dpi": dpi,
            "pages_ocr": len(pages),
            "dur_total_s": round(time.time() - t0, 3),
            "dur_ocr_s": round(time.time() - ocr_start, 3),
            "text_len": len(text),
            "snippet": "\n".join(lines[:40])
        }
    except Exception as e:
        return {"ok": False, "step": "PARSE", "error": str(e)}
