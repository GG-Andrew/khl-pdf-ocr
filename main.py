# main.py
from __future__ import annotations

import os
import re
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# fuzzy словари
from rapidfuzz import process, fuzz

APP_VERSION = "3.1.0"
app = FastAPI(title="KHL PDF OCR (column parser)", version=APP_VERSION)

# ---------- словари игроков/судей (опциональны, но мы их используем если есть) ----------
PLAYERS_CSV = os.getenv("PLAYERS_CSV", "players.csv")
REFEREES_CSV = os.getenv("REFEREES_CSV", "referees.csv")

def _load_list(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    vals: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().strip(",;")
            if t:
                vals.append(t)
    return vals

PLAYERS_LIST = _load_list(PLAYERS_CSV)
REFS_LIST = _load_list(REFEREES_CSV)

def fuzzy_name(name: str, pool: List[str], cutoff: int = 90) -> str:
    if not pool:
        return name
    cand = process.extractOne(name, pool, scorer=fuzz.token_set_ratio)
    if cand and cand[1] >= cutoff:
        return cand[0]
    return name

# ---------- утилы нормализации ----------
SPACE_RE = re.compile(r"[ \t\u00A0]+")
def norm_spaces(s: str) -> str:
    return SPACE_RE.sub(" ", s).strip()

CAPT_RE = re.compile(r"\b([AKАК])\b\.?$", re.IGNORECASE)
def split_capt(s: str) -> Tuple[str, str]:
    s = norm_spaces(s)
    m = CAPT_RE.search(s)
    if m:
        return norm_spaces(CAPT_RE.sub("", s)), m.group(1).upper().replace("К", "K").replace("А", "A")
    return s, ""

GK_FLAG_RE = re.compile(r"\b([СР])\b\.?$", re.IGNORECASE)  # С/Р (русские)
def split_gk_flag(name: str) -> Tuple[str, Optional[str]]:
    n = norm_spaces(name)
    m = GK_FLAG_RE.search(n)
    if m:
        flag = m.group(1).upper()
        flag = "S" if flag == "С" else "R"
        return norm_spaces(GK_FLAG_RE.sub("", n)), flag
    return n, None

def gk_status_from_flag(flag: Optional[str]) -> Optional[str]:
    if flag == "S":
        return "starter"
    if flag == "R":
        return "reserve"
    return None

# ---------- сетевой слой с обходом 403 ----------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

BASE_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Referer": "https://www.khl.ru/",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
}

async def fetch_pdf_bytes(url: str, timeout: float = 25.0) -> bytes:
    """
    1) прогреваем cookies на главной khl.ru
    2) выключаем HTTP/2 (http2=False)
    3) пробуем www.khl.ru и khl.ru
    """
    candidates = [url]
    # продублируем домен без www/с www
    if "://www.khl.ru/" in url:
        candidates.append(url.replace("://www.khl.ru/", "://khl.ru/"))
    elif "://khl.ru/" in url:
        candidates.append(url.replace("://khl.ru/", "://www.khl.ru/"))

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, http2=False, headers=BASE_HEADERS) as client:
        # cookie warm-up
        try:
            await client.get("https://www.khl.ru/", headers=BASE_HEADERS)
        except Exception:
            pass

        last_err: Optional[Exception] = None
        for u in candidates:
            try:
                r = await client.get(u, headers=BASE_HEADERS)
                r.raise_for_status()
                return r.content
            except Exception as e:
                last_err = e

        # если все кандидаты упали — поднимем ошибку
        if last_err:
            raise last_err
        raise RuntimeError("Unknown fetch error")

# ---------- разрезка на колонки (по координатам) ----------
def read_pdf_columns(pdf_bytes: bytes) -> Tuple[str, str]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count < 1:
            return "", ""
        page = doc[0]
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text, ...)
        W = float(page.rect.width)
        mid = W / 2.0

        left_blocks: List[Tuple[float, str]] = []
        right_blocks: List[Tuple[float, str]] = []

        for b in blocks:
            x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])
            if not txt or not txt.strip():
                continue
            if x1 <= mid - 8:
                left_blocks.append((y0, txt))
            elif x0 >= mid + 8:
                right_blocks.append((y0, txt))
            else:
                cx = (x0 + x1) / 2.0
                (left_blocks if cx < mid else right_blocks).append((y0, txt))

        left_text = "\n".join(t for _, t in sorted(left_blocks, key=lambda z: z[0]))
        right_text = "\n".join(t for _, t in sorted(right_blocks, key=lambda z: z[0]))
        return left_text, right_text

# ---------- парсеры ----------
# Судьи
NAME_TOKENS_RE = re.compile(r"[A-Za-zА-Яа-яЁё\.\-ʼ’`]+")
def parse_refs(side_text: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for line in side_text.splitlines():
        line = norm_spaces(line)
        if not line:
            continue
        role = None
        if "Главный судья" in line:
            role = "Главный судья"
        elif "Линейный судья" in line:
            role = "Линейный судья"
        if not role:
            continue
        tokens = NAME_TOKENS_RE.findall(line)
        tokens = [t for t in tokens if t.lower() not in ("главный", "судья", "линейный")]
        if len(tokens) >= 2:
            name = f"{tokens[0]} {tokens[1]}"
            name = fuzzy_name(name, REFS_LIST)
            out.append({"role": role, "name": name})
    return out

def dedup_refs(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    res = []
    for r in items:
        key = (r.get("role", ""), r.get("name", ""))
        if key in seen:
            continue
        seen.add(key)
        res.append(r)
    return res

# Вратари
GOALIES_SECTION_RE = re.compile(r"Вратари(.+?)(?:Звено|Главный тренер|Линейный|Главный судья|---RIGHT---|$)", re.IGNORECASE | re.DOTALL)
GK_LINE_RE = re.compile(
    r"^\s*(\d{1,2})\s+В\b[^\n]*?\b([A-Za-zА-Яа-яЁё\. \-ʼ’`]+?)(?:\s+([СР]))?\s+(?:\d{2}\.\d{2}\.\d{4})",
    re.MULTILINE
)

def parse_goalies(side_text: str, side: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    m = GOALIES_SECTION_RE.search(side_text)
    if not m:
        return out
    sec = m.group(1)

    for mm in GK_LINE_RE.finditer(sec):
        num = mm.group(1)
        raw_name = norm_spaces(mm.group(2))
        raw_name, flag = split_gk_flag(raw_name)
        name = fuzzy_name(raw_name, PLAYERS_LIST)
        status = gk_status_from_flag(flag) or "scratch"
        out.append({"name": name, "status": status})
    return out

# Составы
LINEUP_LINE_RE = re.compile(
    r"^\s*(\d{1,2})\s+(В|З|Н)\s+([A-Za-zА-Яа-яЁё\. \-ʼ’`]+?)\s+(?:\d{2}\.\d{2}\.\d{4})",
    re.MULTILINE
)

def parse_lineups(side_text: str, side: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for mm in LINEUP_LINE_RE.finditer(side_text):
        num, pos, raw_name = mm.group(1), mm.group(2), mm.group(3)
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
        # DOB/age (если в хвосте)
        tail = side_text[mm.end():mm.end()+32]
        mdate = re.search(r"\b(\d{2}\.\d{2}\.\d{4})\b(?:\s+(\d{1,2}))?", tail)
        if mdate:
            item["dob"] = mdate.group(1)
            if mdate.group(2):
                item["age"] = mdate.group(2)
        if pos == "В" and gk_flag:
            item["gk_flag"] = gk_flag
            status = gk_status_from_flag(gk_flag)
            if status:
                item["gk_status"] = status
        out.append(item)
    return out

# ---------- API ----------
@app.get("/")
async def root():
    return {
        "ok": True,
        "service": "khl-pdf-ocr",
        "version": APP_VERSION,
        "ready": True,
        "dicts": {
            "players_csv": os.path.exists(PLAYERS_CSV),
            "referees_csv": os.path.exists(REFEREES_CSV),
            "players_loaded": len(PLAYERS_LIST),
            "refs_loaded": len(REFS_LIST),
        },
    }

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/ocr")
async def ocr_endpoint(
    pdf_url: str = Query(...),
    match_id: int = Query(...),
    season: int = Query(...),
):
    try:
        pdf_bytes = await fetch_pdf_bytes(pdf_url)
    except Exception as e:
        return JSONResponse({
            "ok": False, "match_id": match_id, "season": season,
            "step": "GET", "status": 599, "error": str(e)
        })

    left, right = read_pdf_columns(pdf_bytes)
    snippet = norm_spaces((left + "\n" + right)[:700])
    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": pdf_url,
        "pdf_len": len(pdf_bytes),
        "pages_ocr": 1,
        "text_len": len(left) + len(right),
        "snippet": snippet,
    }

@app.get("/extract")
async def extract_endpoint(
    pdf_url: str = Query(...),
    match_id: int = Query(...),
    season: int = Query(...),
    target: str = Query("all", description="all|refs|goalies|lineups"),
):
    try:
        pdf_bytes = await fetch_pdf_bytes(pdf_url)
    except Exception as e:
        return JSONResponse({
            "ok": False, "match_id": match_id, "season": season,
            "step": "GET", "status": 599, "error": str(e)
        })

    left, right = read_pdf_columns(pdf_bytes)

    data: Dict[str, object] = {}
    if target in ("all", "refs"):
        refs = dedup_refs(parse_refs(left) + parse_refs(right))
        data["refs"] = refs
    if target in ("all", "goalies"):
        data["goalies"] = {
            "home": parse_goalies(left, side="home"),
            "away": parse_goalies(right, side="away"),
        }
    if target in ("all", "lineups"):
        data["lineups"] = {
            "home": parse_lineups(left, side="home"),
            "away": parse_lineups(right, side="away"),
        }

    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": pdf_url,
        "pdf_len": len(pdf_bytes),
        "dpi": 130,
        "pages_ocr": 1,
        "data": data,
    }
