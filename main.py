# main.py
# KHL PDF OCR Server — v1.2.1 (robust errors)
# - /          : health
# - /ocr       : OCR PDF с авто-фолбэком URL, прогревом cookies, «браузерными» заголовками
# - /extract   : черновая структура (refs[], goalies{home,away}, lineups_raw)
# При любой ошибке возвращаем JSON {"ok": false, "step": "...","error":"..."} вместо 500.

import re
import time
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps
import pytesseract

APP_VERSION = "1.2.1"
DEFAULT_SEASON = 1369

app = FastAPI(title="KHL PDF OCR Server", version=APP_VERSION)

# ---------------------------- Заголовки / URL-кандидаты ----------------------------

HEADERS = {
    "Referer": "https://www.khl.ru/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.9",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
}

PDF_TEMPLATES = [
    "{pdf_url}",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-en.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-protocol-ru.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-official-ru.pdf",
    "https://khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf",
    "https://www.khl.ru/documents/{season}/{match_id}.pdf",
]

# ---------------------------- Нормализация текста ----------------------------

LAT_TO_CYR = str.maketrans({
    "A":"А","B":"В","C":"С","E":"Е","H":"Н","K":"К","M":"М","O":"О","P":"Р","T":"Т","X":"Х","Y":"У",
    "a":"а","c":"с","e":"е","o":"о","p":"р","x":"х","y":"у",
})

COMMON_FIXES = [
    (r"(?i)\bIван\b", "Иван"),
    (r"(?i)\bДанийл\b", "Даниил"),
    (r"(?i)\bАндрёй\b", "Андрей"),
    (r"(?i)\bОбйдин\b", "Обидин"),
    (r"(?i)\bМаксйм\b", "Максим"),
    (r"(?i)\bЕгбр\b", "Егор"),
    (r"(?i)\bКсавьё\b", "Ксавье"),
    (r"(?i)Ё", "Е"),
]

def normalize_cyrillic(text: str) -> str:
    if not text:
        return text
    s = text.translate(LAT_TO_CYR)
    s = re.sub(r"([А-ЯЁ][а-яё]{2,})([А-ЯЁ][а-яё]{2,})", r"\1 \2", s)  # "БелоусовГеоргий" -> "Белоусов Георгий"
    s = re.sub(r"([А-ЯЁ][а-яё]+)([А-ЯЁ]\.)", r"\1 \2", s)              # "ФамилияИ." -> "Фамилия И."
    for pat, repl in COMMON_FIXES:
        s = re.sub(pat, repl, s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---------------------------- OCR-препроцесс / OCR ----------------------------

def preprocess(img, scale: float = 1.25, bin_thresh: int = 185):
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    img = ImageOps.autocontrast(img)
    img = img.convert("L")
    img = img.point(lambda x: 255 if x > bin_thresh else 0, mode="1")
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_images(images: List, lang: str = "rus+eng") -> str:
    cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    blocks = []
    for im in images:
        blocks.append(pytesseract.image_to_string(im, lang=lang, config=cfg))
    return "\n".join(blocks)

# ---------------------------- Скачивание PDF (с прогревом) ----------------------------

async def fetch_pdf_with_fallback(pdf_url: str, match_id: int, season: int) -> Tuple[Optional[bytes], Optional[str], List[str], Optional[str]]:
    tried: List[str] = []
    params = {"pdf_url": (pdf_url or "").strip(), "match_id": match_id, "season": season}
    if "khl.ru/documents/" in params["pdf_url"] and "/pdf/" not in params["pdf_url"]:
        params["pdf_url"] = f"https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf"

    last_error: Optional[str] = None
    timeout = httpx.Timeout(25.0)
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        headers=HEADERS
    ) as client:
        # прогрев куков
        try:
            await client.get("https://www.khl.ru/", headers=HEADERS)
            await client.get(f"https://www.khl.ru/game/{match_id}/", headers=HEADERS)
        except Exception as e:
            last_error = f"warmup:{e}"

        for tpl in PDF_TEMPLATES:
            url = tpl.format(**params).strip()
            if not url or url in tried:
                continue
            tried.append(url)
            local_headers = dict(HEADERS)
            local_headers["Referer"] = f"https://www.khl.ru/game/{match_id}/"
            try:
                r = await client.get(url, headers=local_headers)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
                    return r.content, url, tried, None
                else:
                    last_error = f"status:{r.status_code} ct:{r.headers.get('content-type','')}"
            except Exception as e:
                last_error = f"get:{type(e).__name__}:{e}"

        # облегчённые заголовки
        for url in list(tried):
            try:
                r = await client.get(url, headers={
                    "User-Agent": HEADERS["User-Agent"],
                    "Referer": f"https://www.khl.ru/game/{match_id}/"
                })
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
                    return r.content, url, tried, None
                else:
                    last_error = f"fallback_status:{r.status_code}"
            except Exception as e:
                last_error = f"fallback_get:{type(e).__name__}:{e}"

    return None, None, tried, last_error

# ---------------------------- Вспомогательные парсеры ----------------------------

SECTION_PATTERNS = {
    "goalies": r"(?:\bВратари\b[\s\S]{0,900})",
    "refs": r"(?:\bСудьи\b|Главный судья|Линейный судья|Резервный(?: главный)? судья)[\s\S]{0,900}",
    "lineups": r"(?:\bСоставы команд\b[\s\S]{0,5000})",
}

def extract_block(text: str, key: str) -> str:
    m = re.search(SECTION_PATTERNS[key], text, flags=re.IGNORECASE)
    return normalize_cyrillic(m.group(0) if m else "")

def parse_referees(block: str) -> List[Dict[str, str]]:
    out = []
    if not block:
        return out
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    role_map = []
    for ln in lines:
        if re.search(r"(Главный судья|Линейный судья|Резервный(?: главный)? судья)", ln, flags=re.I):
            role_map.append({"role": re.sub(r".*?(Главный судья|Линейный судья|Резервный(?: главный)? судья).*", r"\1", ln, flags=re.I), "name": ""})
        else:
            if role_map and not role_map[-1]["name"]:
                role_map[-1]["name"] = ln
            elif re.search(r"[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+", ln):
                out.append({"role": "Судья", "name": ln})
    for r in role_map:
        if r["name"]:
            out.append(r)
    seen, uniq = set(), []
    for j in out:
        k = (j["role"], j["name"])
        if k in seen: 
            continue
        seen.add(k)
        uniq.append(j)
    return uniq

def parse_goalies(block: str) -> Dict[str, List[Dict[str, str]]]:
    res = {"home": [], "away": []}
    if not block:
        return res
    parts = re.split(r"\bВратари\b", block, flags=re.I)
    cols = [p.strip() for p in parts[1:3] if p.strip()]
    fio_re = re.compile(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.)?\b")
    for idx, col in enumerate(cols):
        side = "home" if idx == 0 else "away"
        buf = " ".join(col.splitlines())
        found = []
        for m in fio_re.finditer(buf):
            nm = m.group(0).strip()
            if len(nm) >= 5:
                found.append({"name": nm})
        seen, uniq = set(), []
        for f in found:
            if f["name"] in seen:
                continue
            seen.add(f["name"])
            uniq.append(f)
        res[side] = uniq[:3]
    return res

# ---------------------------- Модели ответов ----------------------------

class OCRResponse(BaseModel):
    ok: bool
    match_id: int
    season: int
    source_pdf: Optional[str] = None
    pdf_len: Optional[int] = None
    dpi: Optional[int] = None
    pages_ocr: Optional[int] = None
    dur_total_s: Optional[float] = None
    dur_download_s: Optional[float] = None
    dur_preproc_s: Optional[float] = None
    dur_ocr_s: Optional[float] = None
    text_len: Optional[int] = None
    snippet: Optional[str] = None
    step: Optional[str] = None
    status: Optional[int] = None
    tried: Optional[List[str]] = None
    error: Optional[str] = None

# ---------------------------- Эндпойнты ----------------------------

@app.get("/")
def root():
    return {"ok": True, "service": "khl-pdf-ocr", "version": APP_VERSION, "ready": True}

@app.get("/ocr", response_model=OCRResponse)
async def ocr_parse(
    match_id: int = Query(..., description="UID матча КХЛ"),
    pdf_url: str = Query(..., description="URL на PDF протокола"),
    season: int = Query(DEFAULT_SEASON),
    dpi: int = Query(200, ge=120, le=360, description="DPI растрирования PDF"),
    max_pages: int = Query(1, ge=1, le=3, description="Сколько первых страниц OCR’ить"),
    scale: float = Query(1.25, ge=1.0, le=2.0, description="Апскейл картинки перед OCR"),
    bin_thresh: int = Query(185, ge=120, le=230, description="Порог бинаризации 0-255"),
):
    t0 = time.time()
    try:
        pdf_bytes, final_url, tried, last_err = await fetch_pdf_with_fallback(pdf_url, match_id, season)
        if not pdf_bytes:
            return OCRResponse(ok=False, step="GET", status=404, match_id=match_id, season=season, tried=tried, error=last_err)

        t_pdf = time.time()
        pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
        proc = [preprocess(p, scale=scale, bin_thresh=bin_thresh) for p in pages]
        t_pre = time.time()

        raw_text = ocr_images(proc, lang="rus+eng")
        text = normalize_cyrillic(raw_text)
        t_ocr = time.time()

        snippet = re.sub(r"\s+", " ", text.strip())[:480]
        return OCRResponse(
            ok=True,
            match_id=match_id,
            season=season,
            source_pdf=final_url,
            pdf_len=len(pdf_bytes),
            dpi=dpi,
            pages_ocr=len(proc),
            dur_total_s=round(t_ocr - t0, 3),
            dur_download_s=round(t_pdf - t0, 3),
            dur_preproc_s=round(t_pre - t_pdf, 3),
            dur_ocr_s=round(t_ocr - t_pre, 3),
            text_len=len(text),
            snippet=snippet,
        )
    except Exception as e:
        return OCRResponse(ok=False, step="OCR", status=500, match_id=match_id, season=season, error=f"{type(e).__name__}: {e}")

@app.get("/extract")
async def extract_structured(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(DEFAULT_SEASON),
    dpi: int = Query(200, ge=120, le=360),
    max_pages: int = Query(1, ge=1, le=3),
    scale: float = Query(1.25, ge=1.0, le=2.0),
    bin_thresh: int = Query(185, ge=120, le=230),
    target: str = Query("all", description="refs|goalies|lineups|all"),
):
    t0 = time.time()
    try:
        pdf_bytes, final_url, tried, last_err = await fetch_pdf_with_fallback(pdf_url, match_id, season)
        if not pdf_bytes:
            return {"ok": False, "step": "GET", "status": 404, "match_id": match_id, "season": season, "tried": tried, "error": last_err}

        pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
        proc = [preprocess(p, scale=scale, bin_thresh=bin_thresh) for p in pages]
        raw_text = ocr_images(proc, lang="rus+eng")
        text = normalize_cyrillic(raw_text)

        keys = ["refs", "goalies", "lineups"] if target == "all" else [target]
        data: Dict[str, object] = {}
        for k in keys:
            blk = extract_block(text, k)
            if k == "refs":
                data["refs"] = parse_referees(blk)
            elif k == "goalies":
                data["goalies"] = parse_goalies(blk)
            elif k == "lineups":
                data["lineups_raw"] = blk

        return {
            "ok": True,
            "match_id": match_id,
            "season": season,
            "source_pdf": final_url,
            "dpi": dpi,
            "pages_ocr": len(proc),
            "dur_total_s": round(time.time() - t0, 3),
            "text_len": len(text),
            "data": data,
        }
    except Exception as e:
        return {"ok": False, "step": "EXTRACT", "status": 500, "match_id": match_id, "season": season, "error": f"{type(e).__name__}: {e}"}
