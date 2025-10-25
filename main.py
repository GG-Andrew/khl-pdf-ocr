# main.py
# KHL-OCR_Core_v1.0.0 — FastAPI-сервис OCR/EXTRACT для протоколов КХЛ (Render)
# Включает:
#  • Робастную загрузку PDF (браузерные заголовки, варианты URL, tried[])
#  • PyMuPDF текстовый парсинг → fallback Tesseract (rus+eng)
#  • /ocr — сырой текст + метаданные
#  • /extract — структурный разбор: refs / goalies / lineups (базовые правила)
#  • Единый формат JSON с ok/step/status/tried

import io
import os
import re
import time
from typing import List, Tuple, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse

# ==== OCR/IMAGE STACK ====
# PyMuPDF (fitz) — быстрый извлекатель текста, и рендер в растровое изображение при необходимости
import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

# -----------------------------------------------------------------------------
# Конфиг по умолчанию (можно переопределить через ENV)
# -----------------------------------------------------------------------------
DEFAULT_DPI = int(os.getenv("OCR_DPI", "300"))
DEFAULT_SCALE = float(os.getenv("OCR_SCALE", "1.6"))
DEFAULT_BIN_THRESH = int(os.getenv("OCR_BIN_THRESH", "185"))
DEFAULT_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "2"))

# Внешние словари (опционально): пути к CSV с реестрами игроков/судей
PLAYERS_CSV = os.getenv("PLAYERS_CSV")  # например: "/app/players.csv"
REFEREES_CSV = os.getenv("REFEREES_CSV")  # например: "/app/referees.csv"

# -----------------------------------------------------------------------------
# Утилиты: загрузка PDF робастно (браузерные заголовки, варианты URL)
# -----------------------------------------------------------------------------
def fetch_pdf_bytes(url: str) -> Tuple[bytes, List[str]]:
    """
    Качаем PDF с «браузерными» заголовками и fallback'ами.
    Возвращаем (bytes, tried_urls) — для прозрачного дебага.
    """
    tried: List[str] = []

    base = (url or "").strip()
    variants: List[str] = []
    if base:
        variants.append(base)
        if "www.khl.ru" in base:
            variants.append(base.replace("www.khl.ru", "khl.ru"))

        # /pdf/YYYY/XXXXXX/game-<id>-start-ru.pdf  →  /documents/YYYY/<id>.pdf
        m = re.search(r"/pdf/(\d{4})/(\d{6})/game-(\d+)-start-ru\.pdf$", base)
        if m:
            season, mid6, mid = m.groups()
            variants.append(f"https://www.khl.ru/documents/{season}/{mid}.pdf")
            # иногда лежит под похожим именем
            variants.append(f"https://www.khl.ru/documents/{season}/game-{mid}-start-ru.pdf")

        # без -ru
        if base.endswith("-start-ru.pdf"):
            variants.append(base.replace("-start-ru.pdf", "-start.pdf"))

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.1 Safari/537.36"
        ),
        "Referer": "https://www.khl.ru/",
        "Accept": "application/pdf,application/*;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru,en;q=0.9",
        "Connection": "keep-alive",
    }

    # 1) httpx (HTTP/2 + follow_redirects)
    try:
        import httpx
        with httpx.Client(http2=True, timeout=30.0, headers=headers, follow_redirects=True) as client:
            for v in variants:
                try:
                    r = client.get(v)
                    tried.append(f"{v} [{r.status_code}]")
                    ctype = (r.headers.get("content-type") or "").lower()
                    if r.status_code == 200 and "pdf" in ctype:
                        return r.content, tried
                except Exception as e:
                    tried.append(f"{v} [httpx err: {e}]")
    except Exception as e:
        tried.append(f"httpx unavailable: {e}")

    # 2) urllib fallback (на первый вариант)
    if variants:
        try:
            import urllib.request
            req = urllib.request.Request(variants[0], headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                tried.append(f"{variants[0]} [urllib {getattr(resp, 'status', 200)}]")
                return data, tried
        except Exception as e:
            tried.append(f"{variants[0]} [urllib err: {e}]")

    return b"", tried

# -----------------------------------------------------------------------------
# Преобразования изображения (для OCR)
# -----------------------------------------------------------------------------
def binarize(img: Image.Image, threshold: int) -> Image.Image:
    return img.convert("L").point(lambda p: 255 if p > threshold else 0, mode="1").convert("L")

def preprocess_for_ocr(pix: "fitz.Pixmap", scale: float, bin_thresh: int) -> Image.Image:
    # fitz.Pixmap → PIL.Image
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    # upscale
    if scale and scale != 1.0:
        w = int(img.width * scale)
        h = int(img.height * scale)
        img = img.resize((w, h), Image.LANCZOS)
    # автоконтраст
    img = ImageOps.autocontrast(img)
    # лёгкая резкость
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    # бинаризация
    img = binarize(img, bin_thresh)
    return img

# -----------------------------------------------------------------------------
# Текст из PDF: PyMuPDF текст → если пусто, OCR Tesseract
# -----------------------------------------------------------------------------
def pdf_to_text(pdf_bytes: bytes, dpi: int, scale: float, bin_thresh: int, max_pages: int) -> Tuple[str, int]:
    """
    Возвращает (text, pages_used).
    1) Пытается достать «живой» текст через PyMuPDF
    2) Если мало/пусто — рендерит страницы и прогоняет Tesseract (rus+eng)
    """
    text_parts: List[str] = []
    pages_used = 0

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages_to_process = min(len(doc), max_pages)

        # Шаг 1: живой текст
        raw_parts = []
        for i in range(pages_to_process):
            page = doc.load_page(i)
            words = page.get_text("words")
            if words:
                # собрать по координатам → в строки
                words_sorted = sorted(words, key=lambda w: (round(w[3] / 20), w[0]))  # грубая группировка по y
                line = []
                cur_y = None
                lines = []
                for (x0, y0, x1, y1, wtext, block_no, line_no, word_no) in words_sorted:
                    if cur_y is None:
                        cur_y = y1
                    if abs(y0 - cur_y) > 8:
                        if line:
                            lines.append(" ".join(line))
                        line = [wtext]
                        cur_y = y1
                    else:
                        line.append(wtext)
                        cur_y = max(cur_y, y1)
                if line:
                    lines.append(" ".join(line))
                raw_parts.append("\n".join(lines))

        raw_text = "\n".join(raw_parts).strip()
        if len(raw_text) >= 500:  # если текста достаточно — берём PyMuPDF
            return raw_text, pages_to_process

        # Шаг 2: OCR Tesseract (по страницам)
        ocr_parts = []
        for i in range(pages_to_process):
            page = doc.load_page(i)
            # dpi → матрица зума
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = preprocess_for_ocr(pix, scale=scale, bin_thresh=bin_thresh)
            txt = pytesseract.image_to_string(img, lang="rus+eng", config="--oem 1 --psm 6 -c preserve_interword_spaces=1")
            ocr_parts.append(txt)
        ocr_text = "\n".join(ocr_parts).strip()
        pages_used = pages_to_process
        return (ocr_text or raw_text), pages_used

# -----------------------------------------------------------------------------
# Простые экстракторы сущностей из текста (эвристики)
# -----------------------------------------------------------------------------
NAME_RE = r"[А-ЯЁA-Z][а-яёa-z'\-]+(?:\s+[А-ЯЁA-Z][а-яёa-z'\-\.]+){0,2}"

def extract_refs(text: str) -> List[Dict[str, str]]:
    """
    Ищем блок 'Судьи'/'Referees' и вытаскиваем фамилии.
    Возвращает список {"name": "...", "role": "Referee|Linesman|Unknown"}
    """
    refs: List[Dict[str, str]] = []
    # упрощённая эвристика — берём строки рядом со словом "Судьи"
    m = re.search(r"Судьи[:\s]*\n?(.{0,200})", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if m:
        chunk = m.group(1).split("\n")[0:3]  # 2-3 строки далее
        chunk = " ".join(chunk)
        # делим по запятым/тире/точкам с запятой
        cand = re.split(r"[;,•\-\u2013]\s*", chunk)
        for c in cand:
            c = c.strip()
            if len(c) < 3:
                continue
            nm = re.findall(NAME_RE, c)
            for n in nm:
                role = "Unknown"
                if re.search(r"лайнсмен|linesman", c, re.I):
                    role = "Linesman"
                if re.search(r"судья|referee", c, re.I):
                    role = "Referee"
                refs.append({"name": n, "role": role})
    # dedup
    uniq = []
    seen = set()
    for r in refs:
        k = (r["name"].lower(), r["role"])
        if k not in seen:
            uniq.append(r)
            seen.add(k)
    return uniq

def extract_goalies(text: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Ищем секции 'Вратари' для home/away. Эвристика: две колонки часто идут подряд.
    Статусы: 'С'→starter, 'Р'→reserve, 'scratch' если явно отмечен.
    """
    res = {"home": [], "away": []}

    # Найдём два подряд блока 'Вратари' (левый/правый)
    blocks = list(re.finditer(r"Вратари[^\n]*\n(.{0,400})", text, flags=re.IGNORECASE | re.DOTALL))
    # Если нашли хотя бы один — попытаемся распарсить строки имён/меток
    def parse_block(bl: str) -> List[Dict[str, str]]:
        out = []
        lines = [ln.strip() for ln in bl.splitlines() if ln.strip()]
        for ln in lines[:6]:  # первые строки
            # Попробуем выковырять имя и метку С/Р/в скобках
            # Примеры: "60 | В | Бочаров Иван С 18.05.1995"  /  "Фукале Зак (С)"
            status = None
            if re.search(r"\bС\b|\(С\)", ln):
                status = "starter"
            elif re.search(r"\bР\b|\(Р\)", ln):
                status = "reserve"
            elif re.search(r"scratch", ln, re.I):
                status = "scratch"

            nm = re.findall(NAME_RE, ln)
            if nm:
                out.append({"name": nm[0], "status": status or "unknown"})
        return out

    if blocks:
        # первый блок считаем "home", второй — "away"
        home_blk = blocks[0].group(1)
        res["home"] = parse_block(home_blk)
        if len(blocks) > 1:
            away_blk = blocks[1].group(1)
            res["away"] = parse_block(away_blk)

    return res

def extract_lineups(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Базовая заготовка для ростеров (по строкам с '№', позициями F/D/G и именем).
    Возвращает {"home":[{...}], "away":[{...}]}
    """
    lineup = {"home": [], "away": []}

    # Эвристика: искать строки с "№" + позицию + имя
    # Пример строки: "60 | В | Бочаров Иван  С 18.05.1995 30"
    LINE_RE = re.compile(r"(?P<num>\d{1,2})\s*\|\s*(?P<pos>[ВДНFG])\s*\|\s*(?P<name>"+NAME_RE+")", re.U)

    # Разделим документ на «две колонки» грубо: по «Составы команд»/названиям клубов
    chunks = re.split(r"Составы команд|ЛАДА|ДИНАМО|СОСТАВЫ", text, flags=re.I)
    # Пройдем по всем кускам — первые два осмысленных считаем home/away
    buckets: List[List[Dict[str, Any]]] = []
    for ch in chunks:
        lst: List[Dict[str, Any]] = []
        for m in LINE_RE.finditer(ch):
            d = m.groupdict()
            pos_map = {"В": "G", "Д": "D", "Н": "F", "F": "F", "G": "G", "D": "D"}
            lst.append({
                "num": int(d["num"]),
                "pos": pos_map.get(d["pos"], d["pos"]),
                "name": d["name"]
            })
        if lst:
            buckets.append(lst)

    if buckets:
        lineup["home"] = buckets[0]
        if len(buckets) > 1:
            lineup["away"] = buckets[1]

    return lineup

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="KHL PDF OCR", version="1.0.0")

@app.get("/")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "khl-pdf-ocr", "version": "1.0.0"}

# === /ocr =====================================================================
@app.get("/ocr")
@app.post("/ocr")
def ocr_endpoint(
    match_id: int = Query(...),
    pdf_url: str = Query(None),
    season: int = Query(1369),
    dpi: int = Query(DEFAULT_DPI),
    scale: float = Query(DEFAULT_SCALE),
    bin_thresh: int = Query(DEFAULT_BIN_THRESH),
    max_pages: int = Query(DEFAULT_MAX_PAGES),
    body: Optional[Dict[str, Any]] = Body(None)
):
    """
    Возвращает сырой текст OCR + метаданные. Поддерживает GET (query) и POST (JSON).
    """
    t0 = time.perf_counter()

    # JSON-параметры имеют приоритет над query
    if body:
        match_id = body.get("match_id", match_id)
        pdf_url = body.get("pdf_url", pdf_url)
        season = body.get("season", season)
        dpi = body.get("dpi", dpi)
        scale = body.get("scale", scale)
        bin_thresh = body.get("bin_thresh", bin_thresh)
        max_pages = body.get("max_pages", max_pages)

    b, tried = fetch_pdf_bytes(pdf_url or "")
    if not b:
        return JSONResponse({
            "ok": False,
            "match_id": match_id,
            "season": season,
            "step": "GET",
            "status": 404,
            "tried": tried
        }, status_code=404)

    text, pages = pdf_to_text(b, dpi=dpi, scale=scale, bin_thresh=bin_thresh, max_pages=max_pages)
    dur = round(time.perf_counter() - t0, 3)

    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": pdf_url,
        "pdf_len": len(b),
        "dpi": dpi,
        "pages_ocr": pages,
        "dur_total_s": dur,
        "text_len": len(text),
        "snippet": text[:600],
        "tried": tried
    }

# === /extract =================================================================
@app.get("/extract")
@app.post("/extract")
def extract_endpoint(
    match_id: int = Query(...),
    pdf_url: str = Query(None),
    season: int = Query(1369),
    target: str = Query("all"),  # refs|goalies|lineups|all
    dpi: int = Query(DEFAULT_DPI),
    scale: float = Query(DEFAULT_SCALE),
    bin_thresh: int = Query(DEFAULT_BIN_THRESH),
    max_pages: int = Query(DEFAULT_MAX_PAGES),
    body: Optional[Dict[str, Any]] = Body(None)
):
    """
    Возвращает структурированные сущности: refs/goalies/lineups (по target).
    """
    t0 = time.perf_counter()

    if body:
        match_id = body.get("match_id", match_id)
        pdf_url = body.get("pdf_url", pdf_url)
        season = body.get("season", season)
        target = body.get("target", target)
        dpi = body.get("dpi", dpi)
        scale = body.get("scale", scale)
        bin_thresh = body.get("bin_thresh", bin_thresh)
        max_pages = body.get("max_pages", max_pages)

    b, tried = fetch_pdf_bytes(pdf_url or "")
    if not b:
        return JSONResponse({
            "ok": False,
            "match_id": match_id,
            "season": season,
            "step": "GET",
            "status": 404,
            "tried": tried
        }, status_code=404)

    # Для извлечения сначала получим текст (PyMuPDF→Tesseract)
    text, pages = pdf_to_text(b, dpi=dpi, scale=scale, bin_thresh=bin_thresh, max_pages=max_pages)

    data: Dict[str, Any] = {}

    if target in ("refs", "all"):
        data["refs"] = extract_refs(text)
    if target in ("goalies", "all"):
        data["goalies"] = extract_goalies(text)
    if target in ("lineups", "all"):
        data["lineups"] = extract_lineups(text)

    dur = round(time.perf_counter() - t0, 3)
    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": pdf_url,
        "pdf_len": len(b),
        "dpi": dpi,
        "pages_ocr": pages,
        "dur_total_s": dur,
        "data": data,
        "tried": tried
    }

# -----------------------------------------------------------------------------
# Запуск локально (Render запускает через команду из Procfile/Start Command)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
