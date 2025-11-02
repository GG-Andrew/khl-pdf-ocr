import os
import io
import re
import time
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# --- окружение для tesseract ---
# Render задаёт PORT, мы его используем; TESSDATA_PREFIX выставим на стандартный путь в образе
os.environ.setdefault("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata")

app = FastAPI(title="KHL PDF OCR", version="1.0.0")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.khl.ru/",
    "Accept": "application/pdf,*/*",
}

def fetch_pdf(url: str, timeout: int = 20) -> bytes:
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.content

def ocr_referees_from_image(img: Image.Image) -> Dict[str, List[str]]:
    """
    Очень простой извлекатель блока судей:
    - OCR всей страницы
    - берём строки рядом с ключевыми словами
    - чистим роли и хвосты
    """
    txt = pytesseract.image_to_string(img, lang="rus+eng")
    # нормализация пробелов
    lines = [re.sub(r"\s+", " ", x).strip() for x in txt.splitlines()]
    lines = [x for x in lines if x]

    main, liney = [], []

    # ключевые маркеры — как в твоём скрине PDF
    for i, s in enumerate(lines):
        s_low = s.lower()
        if "главный судья" in s_low:
            # берём текущую строку и следующую, выбрасывая слово "главный судья"
            blob = " ".join(lines[i:i+2])
            blob = re.sub(r"главн\w*\s*суд\w*", " ", blob, flags=re.I).strip(" -:.,")
            # несколько имён через запятую/пробелы
            for name in re.split(r"[;,]| {2,}", blob):
                name = name.strip(" -:.,")
                if name and len(name.split()) >= 2:
                    main.append(name)
        if "линейный судья" in s_low:
            blob = " ".join(lines[i:i+2])
            blob = re.sub(r"линейн\w*\s*суд\w*", " ", blob, flags=re.I).strip(" -:.,")
            for name in re.split(r"[;,]| {2,}", blob):
                name = name.strip(" -:.,")
                if name and len(name.split()) >= 2:
                    liney.append(name)

    # удаляем дубликаты, резервы и мусорные хвосты
    def clean(names: List[str]) -> List[str]:
        out, seen = [], set()
        for n in names:
            n = re.sub(r"\b(резервн\w*|судья|судьи)\b", "", n, flags=re.I).strip(" -:.,")
            n = re.sub(r"\s{2,}", " ", n)
            if n and n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out

    return {"main": clean(main), "linesmen": clean(liney)}

def ocr_goalies_from_image(img: Image.Image) -> Dict[str, Any]:
    """
    Базовый MVP: вытягиваем 3 номера/ФИО из левой и правой половины,
    где встречаются маркеры вратарей (буквы 'В' или 'G' в столбце позиций).
    Это очень упрощённо — задача сейчас запустить конвейер на Render.
    """
    W, H = img.size
    left_crop  = img.crop((0, int(H*0.05), int(W*0.5), int(H*0.95)))
    right_crop = img.crop((int(W*0.5), int(H*0.05), W, int(H*0.95)))

    def pull_side(side_img: Image.Image) -> List[Dict[str, str]]:
        text = pytesseract.image_to_string(side_img, lang="rus+eng")
        text = re.sub(r"[|]+", " ", text)
        lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
        lines = [x for x in lines if x]
        out = []
        for s in lines:
            # ищем строки с номером + фамилией, где рядом есть маркер позиций 'В' (вратарь)
            m = re.search(r"\b(\d{1,2})\b.*\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\.]+)*)", s)
            if m and (" В " in f" {s} " or " в " in f" {s} " or " G " in f" {s} "):
                out.append({"number": m.group(1), "name": m.group(2), "gk_status": ""})
            if len(out) >= 3:
                break
        return out

    return {
        "home": pull_side(left_crop),
        "away": pull_side(right_crop),
    }

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "healthy"

@app.get("/ocr")
def ocr_endpoint(
    pdf_url: str = Query(..., description="Прямая ссылка на PDF KHL протокола"),
    dpi: int = Query(300, ge=150, le=400),
):
    t0 = time.time()
    try:
        pdf_bytes = fetch_pdf(pdf_url)
        # конвертим только 1 страницу — этого хватает для блока судей
        images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
        img = images[0]

        refs = ocr_referees_from_image(img)
        goalies = ocr_goalies_from_image(img)

        return JSONResponse({
            "ok": True,
            "engine": "ocr",
            "referees": refs,
            "goalies": goalies,
            "duration_s": round(time.time() - t0, 3),
            "source_url": pdf_url,
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
