import io
import os
import json
import time
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import httpx
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import regex as re

# --- rapidfuzz (опционально, если установлен) ---
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

APP_VERSION = "3.0.0"

# === Буферы словарей ===
PLAYERS_LIST: List[str] = []
REFS_LIST: List[str] = []

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def load_csv_basename(fname: str) -> List[str]:
    path = os.path.join(os.getcwd(), fname)
    if not os.path.exists(path):
        return []
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = norm_spaces(line.strip())
            if name:
                items.append(name)
    return items

def fuzzy_name(name: str, dict_list: List[str], cutoff: int = 90) -> str:
    """Нормализуем ФИО по словарю (если есть rapidfuzz и словарь)."""
    name = norm_spaces(name)
    if not name or not dict_list or not HAS_RAPIDFUZZ:
        return name
    match = process.extractOne(name, dict_list, scorer=fuzz.WRatio)
    if match and match[1] >= cutoff:
        return match[0]
    return name

# === HTTP ===
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru,en;q=0.9",
    "Referer": "https://www.khl.ru/",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

async def fetch_pdf(url: str) -> bytes:
    tried = []
    async with httpx.AsyncClient(timeout=20, headers=HTTP_HEADERS, http2=True, follow_redirects=True) as client:
        try:
            r = await client.get(url)
            tried.append(f"{url} [{r.status_code}]")
            r.raise_for_status()
            return r.content
        except Exception as e:
            tried.append(f"httpx err: {repr(e)}")
            raise RuntimeError("|".join(tried))

# === Текст-слой через PyMuPDF ===
def extract_text_pymupdf(pdf_bytes: bytes, max_pages: int = 1) -> str:
    text_parts: List[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = min(max_pages, len(doc)) if max_pages else len(doc)
        for p in range(pages):
            page = doc.load_page(p)
            # Сохраняем порядок блоков/колонок
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)

# === OCR fallback (по запросу) ===
def ocr_text(pdf_bytes: bytes, dpi: int = 200, max_pages: int = 1,
             scale: float = 1.5, bin_thresh: int = 190) -> str:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png", first_page=1, last_page=max_pages)
    out = []
    for img in pages:
        if scale and scale != 1.0:
            w, h = img.size
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        gray = img.convert("L")
        # бинаризация
        bw = gray.point(lambda x: 255 if x > bin_thresh else 0, "1")
        txt = pytesseract.image_to_string(
            bw,
            lang="rus+eng",
            config="--psm 6 --oem 1 --dpi 300 --tessedit_preserve_interword_spaces=1"
        )
        out.append(txt)
    return "\n".join(out)

# === Парсинг ===

NAME_TOKENS_RE = re.compile(r"[\p{L}\.\-ʼ’`]+", re.IGNORECASE)

GOALIES_SECTION_RE = re.compile(
    r"Вратари(.+?)(?:Звено|Главный тренер|Линейный|Главный судья|Команда|---RIGHT---|$)",
    re.IGNORECASE | re.DOTALL
)

GK_LINE_RE = re.compile(
    r"""^\s*
        (\d{1,2})              # номер
        \s+В\b                 # позиция В
        \s+([\p{L}\.\-ʼ’` ]+?) # ФИО (с возможными инициалами)
        (?:\s+([СРSR]))?       # опц флаг S/R/С/Р
        (?:\s+(\d{2}\.\d{2}\.\d{4}))? # опц ДР
        (?:\s+(\d{1,2}))?      # опц возраст
        \s*$""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE
)

LINEUP_LINE_RE = re.compile(
    r"""^\s*
        (\d{1,2})              # номер
        \s+(В|З|Н)\b           # позиция
        \s+([\p{L}\.\-ʼ’` ]+?) # ФИО
        (?:\s+[*АКA])?         # опц метки
        (?:\s+(\d{2}\.\д{2}\.\д{4}))? # (опечатка в шаблоне исключена ниже)
        \s*$""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE
)
# Исправим опечатку (в некоторых окружениях \д может не совпасть)
LINEUP_LINE_RE = re.compile(
    r"""^\s*
        (\d{1,2})              # номер
        \s+(В|З|Н)\b           # позиция
        \s+([\p{L}\.\-ʼ’` ]+?) # ФИО
        (?:\s+[*АКA])?         # опц метки
        (?:\s+(\d{2}\.\d{2}\.\d{4}))? # опц ДР
        (?:\s+(\d{1,2}))?      # опц возраст
        \s*$""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE
)

GK_FLAG_RE = re.compile(r"\b([СРSR])\b\.?$", re.IGNORECASE)

def split_capt(name: str) -> Tuple[str, str]:
    # капитан/ассистент встречаются отдельным столбцом, но иногда кидается к фамилии
    n = norm_spaces(name)
    capt = ""
    if re.search(r"\bК\b\.?$", n, re.IGNORECASE):
        n = norm_spaces(re.sub(r"\bК\b\.?$", "", n))
        capt = "K"
    elif re.search(r"\bA\b\.?$", n):
        n = norm_spaces(re.sub(r"\bA\b\.?$", "", n))
        capt = "A"
    return n, capt

def split_gk_flag(name: str) -> Tuple[str, Optional[str]]:
    n = norm_spaces(name)
    m = GK_FLAG_RE.search(n)
    if m:
        raw = m.group(1).upper()
        flag = "S" if raw in ("С", "S") else "R"
        return norm_spaces(GK_FLAG_RE.sub("", n)), flag
    return n, None

def gk_status_from_flag(flag: Optional[str]) -> Optional[str]:
    if not flag:
        return None
    if flag.upper() == "S":
        return "starter"
    if flag.upper() == "R":
        return "reserve"
    return None

def split_sides_by_goalies(text: str) -> Tuple[str, str]:
    """Пробуем поделить лист на левую/правую команду по первому и второму 'Вратари'."""
    idx = [m.start() for m in re.finditer(r"\bВратари\b", text)]
    if len(idx) >= 2:
        return text[idx[0]:idx[1]], text[idx[1]:]
    # fallback: делим по середине (не идеально, но лучше пусто, чем ничего)
    mid = len(text) // 2
    return text[:mid], text[mid:]

def parse_refs(text: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    lines = [norm_spaces(l) for l in text.splitlines()]
    for i, line in enumerate(lines):
        role = None
        if "Главный судья" in line:
            role = "Главный судья"
        elif "Линейный судья" in line:
            role = "Линейный судья"
        if not role:
            continue

        # Пытаемся вытащить имя из этой строки
        toks = NAME_TOKENS_RE.findall(line)
        toks = [t for t in toks if t.lower() not in ("главный", "судья", "линейный")]
        name = None
        if len(toks) >= 2:
            name = f"{toks[0]} {toks[1]}"

        # Если нет — смотрим 1–2 строки ниже
        if not name:
            for j in (i + 1, i + 2):
                if 0 <= j < len(lines):
                    ntoks = NAME_TOKENS_RE.findall(lines[j])
                    if len(ntoks) >= 2:
                        name = f"{ntoks[0]} {ntoks[1]}"
                        break
        if name:
            name = fuzzy_name(name, REFS_LIST)
            out.append({"role": role, "name": name})
    return out

def parse_goalies(side_text: str, side: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    m = GOALIES_SECTION_RE.search(side_text)
    if not m:
        return out
    sec = m.group(1)
    for mm in GK_LINE_RE.finditer(sec):
        raw_name = norm_spaces(mm.group(2))
        raw_name, flag = split_gk_flag(raw_name)
        name = fuzzy_name(raw_name, PLAYERS_LIST)
        status = gk_status_from_flag(flag) or "scratch"
        out.append({"name": name, "status": status})
    return out

def parse_lineups(side_text: str, side: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for mm in LINEUP_LINE_RE.finditer(side_text):
        num, pos, raw_name, dob, age = mm.group(1), mm.group(2), mm.group(3), mm.group(4), mm.group(5)
        raw_name, gk_flag = split_gk_flag(raw_name)
        name, capt = split_capt(raw_name)
        name = fuzzy_name(name, PLAYERS_LIST)
        item: Dict[str, str] = {
            "side": side,
            "num": num,
            "pos": pos,
            "name": name,
            "capt": capt,
        }
        if dob:
            item["dob"] = dob
        if age:
            item["age"] = age
        if pos == "В" and gk_flag:
            item["gk_flag"] = gk_flag
            status = gk_status_from_flag(gk_flag)
            if status:
                item["gk_status"] = status
        out.append(item)
    return out

# === FastAPI ===
app = FastAPI()

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "khl-pdf-ocr",
        "version": APP_VERSION,
        "ready": True,
        "dicts": {
            "players_csv": os.path.exists("players.csv"),
            "referees_csv": os.path.exists("referees.csv"),
            "players_loaded": len(PLAYERS_LIST),
            "refs_loaded": len(REFS_LIST),
        },
    }

@app.on_event("startup")
def _startup():
    global PLAYERS_LIST, REFS_LIST
    PLAYERS_LIST = load_csv_basename("players.csv")
    REFS_LIST = load_csv_basename("referees.csv")

def build_ok(match_id, season, source_pdf, data, pdf_len=None, dpi=None, pages_ocr=1, extra=None):
    resp = {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": source_pdf,
        "pdf_len": pdf_len,
        "dpi": dpi,
        "pages_ocr": pages_ocr,
        "data": data,
    }
    if extra:
        resp.update(extra)
    return resp

def build_err(match_id, season, step, status=None, tried=None, error=None):
    return {
        "ok": False,
        "match_id": match_id,
        "season": season,
        "step": step,
        "status": status,
        "tried": tried,
        "error": error,
    }

@app.get("/extract")
async def extract(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(...),
    target: str = Query("all", pattern="^(all|refs|goalies|lineups)$"),
    dpi: int = Query(130, ge=120, le=360),
    max_pages: int = Query(1, ge=1, le=5),
    force_ocr: int = Query(0, ge=0, le=1),
):
    tried = []
    try:
        pdf = await fetch_pdf(pdf_url)
        tried.append(f"{pdf_url} [200]")
    except Exception as e:
        return JSONResponse(build_err(match_id, season, "GET", 599, [str(e)]), status_code=200)

    pdf_len = len(pdf)
    # 1) Текст-слой
    text = ""
    try:
        text = extract_text_pymupdf(pdf, max_pages=max_pages)
    except Exception as e:
        text = ""

    # 2) OCR по запросу
    if force_ocr or not text or len(norm_spaces(text)) < 100:
        try:
            text = ocr_text(pdf, dpi=dpi, max_pages=max_pages)
        except Exception as e:
            return JSONResponse(build_err(match_id, season, "OCR", 500, tried, str(e)), status_code=200)

    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Разделим на две стороны
    left, right = split_sides_by_goalies(text)

    data: Dict[str, object] = {}
    if target in ("refs", "all"):
        data["refs"] = parse_refs(text)
    if target in ("goalies", "all"):
        data["goalies"] = {
            "home": parse_goalies(left, "home"),
            "away": parse_goalies(right, "away"),
        }
    if target in ("lineups", "all"):
        data["lineups"] = {
            "home": parse_lineups(left, "home"),
            "away": parse_lineups(right, "away"),
        }

    return JSONResponse(build_ok(match_id, season, pdf_url, data, pdf_len=pdf_len, dpi=dpi))

if __name__ == "__main__":
    # Для локального запуска; в Render это будет запускаться из Docker CMD
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)
