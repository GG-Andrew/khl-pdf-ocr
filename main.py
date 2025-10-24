# main.py
# KHL PDF OCR Server — v1.1.0
# - /ocr: OCR PDF с авто-фолбэком URL (documents -> pdf)
# - /extract: структурный JSON (refs, goalies, lineups_raw) + нормализация кириллицы

import re
import time
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps
import pytesseract

APP_VERSION = "1.1.0"

app = FastAPI(title="KHL PDF OCR Server", version=APP_VERSION)

# ---------------------------- Константы / Настройки ----------------------------

DEFAULT_SEASON = 1369  # можно поменять при деплое
HEADERS = {
    "Referer": "https://www.khl.ru/online/",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/pdf,*/*;q=0.9",
}
# порядок важен: сначала используем пришедший url, затем шаблон pdf, затем documents
PDF_TEMPLATES = [
    "{pdf_url}",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf",
    "https://www.khl.ru/documents/{season}/{match_id}.pdf",
]

# Грубые паттерны для извлечения блоков
SECTION_PATTERNS = {
    "goalies": r"(?:\bВратари\b[\s\S]{0,900})",
    "refs": r"(?:\bСудьи\b|Главный судья|Линейный судья|Резервный судья)[\s\S]{0,900}",
    "lineups": r"(?:\bСоставы команд\b[\s\S]{0,5000})",
}

# Карта латиница→кириллица (для нормализации ФИО и заголовков)
LAT_TO_CYR = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
})
# Исправления частых OCR-ошибок (кейсы «м/п», «й/и», «ё/е», «б/6», цифры/буквы)
COMMON_FIXES = [
    (r"(?i)\bIван\b", "Иван"),
    (r"(?i)\bГебргий\b", "Георгий"),
    (r"(?i)\bДанийл\b", "Даниил"),
    (r"(?i)\bАндрёй\b", "Андрей"),
    (r"(?i)\bМаксйм\b", "Максим"),
    (r"(?i)\bЕгбр\b", "Егор"),
    (r"(?i)\bКсавьё\b", "Ксавье"),
    (r"(?i)\bОбйдин\b", "Обидин"),
    (r"(?i)\bБашкйров\b", "Башкиров"),
    (r"(?i)\bЛимож\b", "Лимож"),  # оставить как есть (иностр.)
    (r"(?i)\bЧивилёв\b", "Чивилёв"),
    (r"(?i)\bТрушк[её]в\b", "Трушков"),
    (r"(?i)\bД[её]мченко\b", "Дёмченко"),
    # Символьные подмены
    (r"(?i)Ё", "Е"),
    (r"(?i)\bЙ\b", "И"),
    (r"(?i)0", "О"),
    (r"(?i)1", "І"),  # визуально, но в ФИО редко критично
    # Заголовки
    (r"(?i)Вратары", "Вратари"),
    (r"(?i)Судьи", "Судьи"),
    (r"(?i)Составы команд", "Составы команд"),
]

# ---------------------------- Вспомогательные функции ----------------------------

def normalize_cyrillic(text: str) -> str:
    """Нормализует OCR-текст: латиницу → кириллицу, фиксы частых ошибок, сжатие пробелов."""
    if not text:
        return text
    s = text.translate(LAT_TO_CYR)
    for pat, repl in COMMON_FIXES:
        s = re.sub(pat, repl, s)
    # сжать множественные пробелы и пустые строки
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def preprocess(img: Image.Image, scale: float = 1.25, bin_thresh: int = 185) -> Image.Image:
    """Лёгкая предобработка: апскейл, автоконтраст, бинаризация, шарп."""
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    img = ImageOps.autocontrast(img)
    img = img.convert("L")
    img = img.point(lambda x: 255 if x > bin_thresh else 0, mode="1")
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_images(images: List[Image.Image], lang: str = "rus+eng") -> str:
    """OCR по списку изображений."""
    cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    blocks = []
    for im in images:
        blocks.append(pytesseract.image_to_string(im, lang=lang, config=cfg))
    return "\n".join(blocks)

async def fetch_pdf_with_fallback(pdf_url: str, match_id: int, season: int) -> Tuple[Optional[bytes], Optional[str], List[str]]:
    """Скачивает PDF, пробуя несколько шаблонов URL. Возвращает (bytes, final_url, tried_urls)."""
    tried: List[str] = []
    params = {"pdf_url": (pdf_url or "").strip(), "match_id": match_id, "season": season}
    # быстрая подмена, если пришёл documents-ссылкой
    if "khl.ru/documents/" in params["pdf_url"] and "/pdf/" not in params["pdf_url"]:
        params["pdf_url"] = f"https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf"

    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        for tpl in PDF_TEMPLATES:
            url = tpl.format(**params).strip()
            if not url or url in tried:
                continue
            tried.append(url)
            r = await client.get(url, headers=HEADERS)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
                return r.content, url, tried

    return None, None, tried

def extract_block(text: str, key: str) -> str:
    pat = SECTION_PATTERNS[key]
    m = re.search(pat, text, flags=re.IGNORECASE)
    return normalize_cyrillic(m.group(0) if m else "")

def parse_referees(block: str) -> List[Dict[str, str]]:
    """
    Черновой парсер судей: ищет строки с ролями и ФИО.
    Пример ролей: Главный судья, Линейный судья, Резервный судья.
    """
    out = []
    if not block:
        return out
    # вытащим все «ролевые» фразы
    role_lines = re.split(r"\n+", block)
    role_map = []
    for line in role_lines:
        ln = line.strip()
        if not ln:
            continue
        # роль + имя (иногда в соседних строках — хард-склейка)
        if re.search(r"(Главный судья|Линейный судья|Резервный( главный)? судья)", ln, flags=re.I):
            role_map.append({"role": ln, "name": ""})
        else:
            # если последняя запись без имени — приклеим ФИО сюда
            if role_map and not role_map[-1]["name"]:
                role_map[-1]["name"] = ln
            else:
                # просто возможное ФИО — добавим как судью без роли (fallback)
                if re.search(r"[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+", ln):
                    out.append({"role": "Судья", "name": ln})

    # нормализуем роли и имена
    for r in role_map:
        role_clean = re.sub(r".*?(Главный судья|Линейный судья|Резервный(?: главный)? судья).*", r"\1", r["role"], flags=re.I)
        name_clean = normalize_cyrillic(r["name"])
        if name_clean:
            out.append({"role": role_clean, "name": name_clean})
    # дедуп по имени
    seen = set()
    uniq = []
    for j in out:
        key = (j["role"], j["name"])
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(j)
    return uniq

def parse_goalies(block: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Черновой парсер вратарей. В протоколе обычно «Вратари» слева/справа (дом/гости).
    Мы разбиваем блок на две колоночные зоны по ключам 'Вратари' и пытаемся вытянуть ФИО/даты.
    """
    res = {"home": [], "away": []}
    if not block:
        return res
    # порежем по «Вратари»
    parts = re.split(r"\bВратари\b", block, flags=re.I)
    # parts[0] — заголовки до, дальше — секции; попытаемся взять первые две секции
    cols = [p.strip() for p in parts[1:3] if p.strip()]
    # Регекс для строк номеров/позиций/ФИО/дат в типичном формате
    row_re = re.compile(r"(?P<num>\d+)\s*\|\s*[ВV]\s*\|\s*(?P<name>[А-ЯЁA-Z][^0-9\|\n]+?)\s+[\(\[]?\d{1,2}\.\d{1,2}\.\d{2,4}|\]", re.I)
    for idx, col in enumerate(cols):
        side = "home" if idx == 0 else "away"
        # если OCR склеил, попробуем разбить по переносам
        lines = [normalize_cyrillic(x) for x in col.split("\n") if x.strip()]
        buf = " ".join(lines)
        # вытащить пары
        found = []
        # увеличим шанс: также искать просто ФИО без номеров
        fio_re = re.compile(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.)?\b")
        for m in fio_re.finditer(buf):
            nm = m.group(0).strip()
            if len(nm) >= 5:
                found.append({"name": nm})
        # дедуп
        seen = set()
        uniq = []
        for f in found:
            nm = f["name"]
            if nm in seen:
                continue
            seen.add(nm)
            uniq.append(f)
        res[side] = uniq[:3]  # обычно 2–3 вратаря
    return res

# ---------------------------- Pydantic модели ----------------------------

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
    pdf_bytes, final_url, tried = await fetch_pdf_with_fallback(pdf_url, match_id, season)
    if not pdf_bytes:
        return OCRResponse(ok=False, step="GET", status=404, match_id=match_id, season=season, tried=tried)

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
    """
    Черновой структурный парсинг:
      - refs: [{"role":"Главный судья","name":"Фамилия Имя"}, ...]
      - goalies: {"home":[{"name":"..."},...], "away":[...]}
      - lineups_raw: сырой блок состава (для твоего дальнейшего парсинга звеньев)
    """
    t0 = time.time()
    # 1) скачать PDF (с подменой documents->pdf при необходимости)
    pdf_bytes, final_url, tried = await fetch_pdf_with_fallback(pdf_url, match_id, season)
    if not pdf_bytes:
        return {"ok": False, "step": "GET", "status": 404, "match_id": match_id, "season": season, "tried": tried}

    # 2) OCR
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
    proc = [preprocess(p, scale=scale, bin_thresh=bin_thresh) for p in pages]
    raw_text = ocr_images(proc, lang="rus+eng")
    text = normalize_cyrillic(raw_text)

    # 3) Блоки
    keys = ["refs", "goalies", "lineups"] if target == "all" else [target]
    data: Dict[str, object] = {}
    for k in keys:
        blk = extract_block(text, k)
        if k == "refs":
            data["refs"] = parse_referees(blk)
        elif k == "goalies":
            data["goalies"] = parse_goalies(blk)
        elif k == "lineups":
            data["lineups_raw"] = blk  # сырым блоком (нужен для твоего будущего NLP)

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
