# main.py
# KHL PDF OCR Server — v1.4.0 (fast + fuzzy dictionaries)
# - /          : health
# - /ocr       : быстрый OCR (split на 2 колонки, dpi=130/scale=1.10), LRU-кэш 15 мин
# - /extract   : структура (refs[], goalies{home,away}, lineups_raw) + фаззи-исправление ФИО по словарям
# Ошибки -> JSON {"ok": false, "step": "...","error":"..."}

import os, re, time, hashlib, csv
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from rapidfuzz import process, fuzz

APP_VERSION = "1.4.0"
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

# ---------------------------- LRU-кэш OCR (на 15 минут) ----------------------------

class TTLCache(OrderedDict):
    def __init__(self, maxsize=64, ttl=900):
        super().__init__()
        self.maxsize = maxsize
        self.ttl = ttl  # seconds
    def _now(self): return time.time()
    def get_cached(self, key):
        item = super().get(key)
        if not item: return None
        ts, val = item
        if self._now() - ts > self.ttl:
            try: del self[key]
            except KeyError: pass
            return None
        self.move_to_end(key)
        return val
    def set_cached(self, key, val):
        super().__setitem__(key, (self._now(), val))
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)

OCR_CACHE = TTLCache(maxsize=64, ttl=900)

def cache_key_pdf(pdf_bytes: bytes, dpi: int, max_pages: int, scale: float, bin_thresh: int) -> str:
    h = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    return f"{h}:{dpi}:{max_pages}:{scale}:{bin_thresh}"

# ---------------------------- Словари ФИО (игроки/судьи) ----------------------------

PLAYERS_DICT: List[str] = []
REFEREES_DICT: List[str] = []

def _load_names_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    names = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            # поддерживаем и plain-текст (по строке), и CSV с колонкой name
            sniffer = csv.Sniffer()
            sample = f.read(1024)
            f.seek(0)
            is_csv = False
            try:
                is_csv = sniffer.has_header(sample)
            except Exception:
                is_csv = False
            if is_csv:
                reader = csv.DictReader(f)
                col = None
                # найдём подходящее поле
                for cand in ("name", "fio", "ФИО", "Name"):
                    if cand in reader.fieldnames:
                        col = cand
                        break
                if not col:  # fallback: возьмём первую колонку
                    col = reader.fieldnames[0]
                for row in reader:
                    val = (row.get(col) or "").strip()
                    if val:
                        names.append(val)
            else:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        names.append(line)
    except Exception:
        pass
    # нормализуем пробелы, первую букву прописную
    out = []
    for n in names:
        s = re.sub(r"\s+", " ", n).strip()
        if not s:
            continue
        parts = s.split()
        s = " ".join([parts[0].capitalize()] + [p.capitalize() for p in parts[1:]])
        out.append(s)
    return sorted(set(out))

def load_dictionaries():
    global PLAYERS_DICT, REFEREES_DICT
    PLAYERS_DICT = _load_names_from_file("data/players.csv")
    REFEREES_DICT = _load_names_from_file("data/referees.csv")

load_dictionaries()

def best_match_name(name: str, pool: List[str], threshold: int = 86) -> str:
    """Фаззи-исправление OCR-имени по словарю; если < threshold — вернём исходник."""
    cand, score, _ = process.extractOne(
        name, pool, scorer=fuzz.WRatio
    ) if pool and name else (None, 0, None)
    return cand if cand and score >= threshold else name

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
    if not text: return text
    s = text.translate(LAT_TO_CYR)
    # "БелоусовГеоргий" -> "Белоусов Георгий" (осторожно, не разрезать «СоставыКоманд»)
    s = re.sub(r"(?<![А-ЯЁа-яё])([А-ЯЁ][а-яё]{2,})([А-ЯЁ][а-яё]{2,})", r"\1 \2", s)
    # "ФамилияИ." -> "Фамилия И."
    s = re.sub(r"([А-ЯЁ][а-яё]+)([А-ЯЁ]\.)", r"\1 \2", s)
    for pat, repl in COMMON_FIXES:
        s = re.sub(pat, repl, s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---------------------------- OCR-препроцесс / OCR (быстро) ----------------------------

def preprocess(img, scale: float = 1.10, bin_thresh: int = 185):
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    img = ImageOps.autocontrast(img)
    img = img.convert("L")
    img = img.point(lambda x: 255 if x > bin_thresh else 0, mode="1")
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_one(im, lang="rus+eng"):
    cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(im, lang=lang, config=cfg)

def ocr_fast_halves(page_img) -> str:
    """Режем страницу пополам по вертикали — меньше мусора, быстрее."""
    w, h = page_img.size
    mid = w // 2
    left = page_img.crop((0, 0, mid, h))
    right = page_img.crop((mid, 0, w, h))
    return ocr_one(left) + "\n" + ocr_one(right)

def cache_key_pdf(pdf_bytes: bytes, dpi: int, max_pages: int, scale: float, bin_thresh: int) -> str:
    h = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    return f"{h}:{dpi}:{max_pages}:{scale}:{bin_thresh}"

def run_ocr_pipeline(pdf_bytes: bytes, dpi: int, max_pages: int, scale: float, bin_thresh: int) -> Tuple[str, int, Dict[str,float]]:
    key = cache_key_pdf(pdf_bytes, dpi, max_pages, scale, bin_thresh)
    cached = OCR_CACHE.get_cached(key)
    if cached:
        return cached["text"], cached["pages"], cached["metrics"]

    t0 = time.time()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
    t_pdf = time.time()

    out_blocks = []
    for p in pages:
        pp = preprocess(p, scale=scale, bin_thresh=bin_thresh)
        out_blocks.append(ocr_fast_halves(pp))
    t_ocr = time.time()

    text = normalize_cyrillic("\n".join(out_blocks))
    metrics = {
        "dur_raster_s": round(t_pdf - t0, 3),
        "dur_ocr_s": round(t_ocr - t_pdf, 3),
        "dur_total_s": round(t_ocr - t0, 3),
    }
    OCR_CACHE.set_cached(key, {"text": text, "pages": len(pages), "metrics": metrics})
    return text, len(pages), metrics

# ---------------------------- Скачивание PDF (с прогревом) ----------------------------

async def fetch_pdf_with_fallback(pdf_url: str, match_id: int, season: int) -> Tuple[Optional[bytes], Optional[str], List[str], Optional[str]]:
    tried: List[str] = []
    params = {"pdf_url": (pdf_url or "").strip(), "match_id": match_id, "season": season}
    if "khl.ru/documents/" in params["pdf_url"] and "/pdf/" not in params["pdf_url"]:
        params["pdf_url"] = f"https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf"

    last_error: Optional[str] = None
    timeout = httpx.Timeout(25.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=HEADERS) as client:
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
                if r.status_code == 200 and r.headers.get("content-type","").startswith("application/pdf"):
                    return r.content, url, tried, None
                else:
                    last_error = f"status:{r.status_code} ct:{r.headers.get('content-type','')}"
            except Exception as e:
                last_error = f"get:{type(e).__name__}:{e}"

        for url in list(tried):
            try:
                r = await client.get(url, headers={"User-Agent": HEADERS["User-Agent"], "Referer": f"https://www.khl.ru/game/{match_id}/"})
                if r.status_code == 200 and r.headers.get("content-type","").startswith("application/pdf"):
                    return r.content, url, tried, None
                else:
                    last_error = f"fallback_status:{r.status_code}"
            except Exception as e:
                last_error = f"fallback_get:{type(e).__name__}:{e}"

    return None, None, tried, last_error

# ---------------------------- Парсеры блоков (+ фаззи-правка) ----------------------------

SECTION_PATTERNS = {
    "goalies": r"(?:\bВратари\b[\s\S]{0,900})",
    "refs": r"(?:\bСудьи\b|Главный судья|Линейный судья|Резервный(?: главный)? судья)[\s\S]{0,900}",
    "lineups": r"(?:\bСоставы команд\b[\s\S]{0,5000})",
}

def extract_block(text: str, key: str) -> str:
    m = re.search(SECTION_PATTERNS[key], text, flags=re.IGNORECASE)
    return normalize_cyrillic(m.group(0) if m else "")

FIO_RE = re.compile(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.)?\b")

def clean_possible_name(s: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", s).strip()
    # выкинуть номера, позиции и даты
    s = re.sub(r"^\d+\s*\|\s*[ВЗН]\s*\|\s*", "", s)
    s = re.sub(r"\[\d{1,2}\.\d{1,2}\.\d{2,4}\]", "", s)
    s = re.sub(r"\(\d{1,2}\.\d{1,2}\.\d{2,4}\)", "", s)
    s = re.sub(r"\d{1,2}\.\d{1,2}\.\d{2,4}", "", s)
    s = s.strip("|·—-:;,. ").strip()
    if not s: return None
    m = FIO_RE.search(s)
    return m.group(0) if m else None

def parse_referees(block: str) -> List[Dict[str, str]]:
    """
    Ищем строки с ролями, затем берём следующие 1–2 строки как ФИО.
    """
    out = []
    if not block: return out
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    roles_idx = []
    for i, ln in enumerate(lines):
        if re.search(r"^(Главный судья|Линейный судья|Резервный(?: главный)? судья)\b", ln, flags=re.I):
            roles_idx.append((i, re.sub(r".*?(Главный судья|Линейный судья|Резервный(?: главный)? судья).*", r"\1", ln, flags=re.I)))

    for idx, role in roles_idx:
        # берём ближайшие 1-2 строки после роли
        for j in range(idx+1, min(idx+3, len(lines))):
            nm = clean_possible_name(lines[j])
            if nm:
                nm_fixed = best_match_name(nm, REFEREES_DICT, threshold=80)
                out.append({"role": role, "name": nm_fixed})
                break

    # дедуп
    seen, uniq = set(), []
    for j in out:
        k = (j["role"], j["name"])
        if k in seen: continue
        seen.add(k); uniq.append(j)
    return uniq

def parse_goalies(block: str) -> Dict[str, List[Dict[str, str]]]:
    res = {"home": [], "away": []}
    if not block: return res
    parts = re.split(r"\bВратари\b", block, flags=re.I)
    cols = [p.strip() for p in parts[1:3] if p.strip()]
    for idx, col in enumerate(cols):
        side = "home" if idx == 0 else "away"
        buf = " ".join(col.splitlines())
        found, seen = [], set()
        for m in FIO_RE.finditer(buf):
            nm = m.group(0).strip()
            nm = best_match_name(nm, PLAYERS_DICT, threshold=86)
            if len(nm) >= 5 and nm not in seen:
                seen.add(nm); found.append({"name": nm})
        res[side] = found[:3]
    return res

# ---------------------------- Модели ответа ----------------------------

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
    exists = {
        "players_csv": os.path.exists("data/players.csv"),
        "referees_csv": os.path.exists("data/referees.csv"),
        "players_loaded": len(PLAYERS_DICT),
        "refs_loaded": len(REFEREES_DICT),
    }
    return {"ok": True, "service": "khl-pdf-ocr", "version": APP_VERSION, "ready": True, "dicts": exists}

@app.get("/ocr", response_model=OCRResponse)
async def ocr_parse(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(DEFAULT_SEASON),
    dpi: int = Query(130, ge=120, le=360),
    max_pages: int = Query(1, ge=1, le=3),
    scale: float = Query(1.10, ge=1.0, le=2.0),
    bin_thresh: int = Query(185, ge=120, le=230),
):
    t0 = time.time()
    try:
        pdf_bytes, final_url, tried, last_err = await fetch_pdf_with_fallback(pdf_url, match_id, season)
        if not pdf_bytes:
            return OCRResponse(ok=False, step="GET", status=404, match_id=match_id, season=season, tried=tried, error=last_err)

        t_pdf = time.time()
        text, pages_cnt, metrics = run_ocr_pipeline(pdf_bytes, dpi, max_pages, scale, bin_thresh)
        t_done = time.time()

        snippet = re.sub(r"\s+", " ", text.strip())[:480]
        return OCRResponse(
            ok=True,
            match_id=match_id, season=season,
            source_pdf=final_url, pdf_len=len(pdf_bytes),
            dpi=dpi, pages_ocr=pages_cnt,
            dur_total_s=round(t_done - t0, 3),
            dur_download_s=round(t_pdf - t0, 3),
            dur_preproc_s=metrics["dur_raster_s"],
            dur_ocr_s=metrics["dur_ocr_s"],
            text_len=len(text), snippet=snippet
        )
    except Exception as e:
        return OCRResponse(ok=False, step="OCR", status=500, match_id=match_id, season=season, error=f"{type(e).__name__}: {e}")

@app.get("/extract")
async def extract_structured(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(DEFAULT_SEASON),
    dpi: int = Query(130, ge=120, le=360),
    max_pages: int = Query(1, ge=1, le=3),
    scale: float = Query(1.10, ge=1.0, le=2.0),
    bin_thresh: int = Query(185, ge=120, le=230),
    target: str = Query("all", description="refs|goalies|lineups|all"),
):
    t0 = time.time()
    try:
        pdf_bytes, final_url, tried, last_err = await fetch_pdf_with_fallback(pdf_url, match_id, season)
        if not pdf_bytes:
            return {"ok": False, "step": "GET", "status": 404, "match_id": match_id, "season": season, "tried": tried, "error": last_err}

        text, pages_cnt, metrics = run_ocr_pipeline(pdf_bytes, dpi, max_pages, scale, bin_thresh)

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
            "match_id": match_id, "season": season,
            "source_pdf": final_url, "pdf_len": len(pdf_bytes),
            "dpi": dpi, "pages_ocr": pages_cnt,
            "dur_total_s": round(time.time() - t0, 3),
            "dur_download_s": metrics["dur_raster_s"],
            "dur_preproc_s": 0.0,
            "dur_ocr_s": metrics["dur_ocr_s"],
            "text_len": len(text),
            "data": data,
        }
    except Exception as e:
        return {"ok": False, "step": "EXTRACT", "status": 500, "match_id": match_id, "season": season, "error": f"{type(e).__name__}: {e}"}
