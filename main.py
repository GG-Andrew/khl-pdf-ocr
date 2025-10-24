import re
import time
from functools import lru_cache
from typing import Tuple, List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps
import pytesseract

app = FastAPI(title="khl-pdf-ocr-1", version="1.0.1")

# ---------- Константы ----------
HEADERS = {
    "Referer": "https://www.khl.ru/online/",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/pdf,*/*;q=0.9",
}

PDF_TEMPLATES = [
    "{pdf_url}",  # как пришло
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf",
    "https://www.khl.ru/documents/{season}/{match_id}.pdf",
]

SECTION_PATTERNS = {
    "goalies": r"(?:\bВратари\b[\s\S]{0,800})",
    "refs":    r"(?:\bСудьи\b|Главный судья|Линейный судья)[\s\S]{0,800}",
    "lineups": r"(?:\bСоставы команд\b|Звено\s*1|Нападающие|Защитники)[\s\S]{0,3500}",
}

# ---------- Вспомогалки ----------

@lru_cache(maxsize=256)
def _cache_key(s: str) -> str:
    return (s or "").strip()

async def fetch_pdf_with_fallback(pdf_url: str, match_id: int, season: int) -> Tuple[Optional[bytes], Optional[str], List[str]]:
    """Пробуем скачать PDF: исходный url -> pdf-шаблон -> documents-шаблон"""
    tried: List[str] = []
    params = {"pdf_url": pdf_url or "", "match_id": match_id, "season": season}

    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(20.0)) as client:
        for tpl in PDF_TEMPLATES:
            url = tpl.format(**params).strip()
            if not url or url in tried:
                continue
            tried.append(url)
            r = await client.get(url, headers=HEADERS)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
                return r.content, url, tried

    return None, None, tried

def preprocess(img: Image.Image, scale: float = 1.3, bin_thresh: int = 185) -> Image.Image:
    """Лёгкая предобработка: апскейл, автоконтраст, бинаризация, шарп."""
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    img = ImageOps.autocontrast(img)
    img = img.convert("L")
    img = img.point(lambda x: 255 if x > bin_thresh else 0, mode="1")
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_images(images: List[Image.Image], lang: str = "rus+eng") -> str:
    """OCR по списку готовых изображений."""
    cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    blocks = []
    for im in images:
        blocks.append(pytesseract.image_to_string(im, lang=lang, config=cfg))
    return "\n".join(blocks)

def normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", s)).strip()

def grab_section(text: str, key: str) -> str:
    m = re.search(SECTION_PATTERNS[key], text, flags=re.IGNORECASE)
    return (m.group(0) if m else "").strip()

# ---------- Модели ----------

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

# ---------- Маршруты ----------

@app.get("/")
async def health():
    return {"ok": True, "service": "khl-pdf-ocr-1", "version": "1.0.1"}

@app.get("/ocr", response_model=OCRResponse)
@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    match_id: int = Query(...),
    pdf_url: str = Query(""),
    season: int = Query(1369),
    dpi: int = Query(200, ge=100, le=400),
    max_pages: int = Query(2, ge=1, le=4),
    scale: float = Query(1.3, ge=1.0, le=2.0),
    bin_thresh: int = Query(185, ge=100, le=220),
):
    t0 = time.time()

    pdf_bytes, final_url, tried = await fetch_pdf_with_fallback(pdf_url, match_id, season)
    if not pdf_bytes:
        return OCRResponse(ok=False, step="GET", status=404, match_id=match_id, season=season, tried=tried)

    t_pdf = time.time()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
    proc = [preprocess(p, scale=scale, bin_thresh=bin_thresh) for p in pages]
    t_pre = time.time()

    text = ocr_images(proc, lang="rus+eng")
    t_ocr = time.time()

    snippet = re.sub(r"\s+", " ", text.strip())[:420]
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

@app.get("/extract")
@app.post("/extract")
async def extract_endpoint(
    match_id: int = Query(...),
    pdf_url: str = Query(""),
    season: int = Query(1369),
    dpi: int = Query(200, ge=100, le=400),
    max_pages: int = Query(2, ge=1, le=4),
    scale: float = Query(1.3, ge=1.0, le=2.0),
    bin_thresh: int = Query(185, ge=100, le=220),
    target: str = Query("all"),  # refs|goalies|lineups|all
):
    """
    Черновой структурный эндпойнт.
    Возвращает сырые блоки: refs / goalies / lineups.
    """
    # 1) Скачиваем и OCR (как в /ocr)
    t0 = time.time()
    pdf_bytes, final_url, tried = await fetch_pdf_with_fallback(pdf_url, match_id, season)
    if not pdf_bytes:
        return {"ok": False, "step": "GET", "status": 404, "match_id": match_id, "season": season, "tried": tried}

    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
    proc = [preprocess(p, scale=scale, bin_thresh=bin_thresh) for p in pages]
    text = ocr_images(proc, lang="rus+eng")

    # 2) Достаём блоки
    keys = ["refs", "goalies", "lineups"] if target == "all" else [target]
    data: Dict[str, str] = {}
    for k in keys:
        raw = grab_section(text, k)
        data[k] = normalize_whitespace(raw)

    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": final_url,
        "dpi": dpi,
        "pages_ocr": len(proc),
        "dur_total_s": round(time.time() - t0, 3),
        "text_len": len(text),
        "data": data
    }
