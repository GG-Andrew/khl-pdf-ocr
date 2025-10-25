# main.py
# khl-pdf-ocr — колонковый парсер стартовых протоколов КХЛ
# Стек: FastAPI + httpx + PyMuPDF (fitz) + regex (+ rapidfuzz для нормализации, по возможности)

from __future__ import annotations

import io
import os
import re
import json
import asyncio
from typing import List, Dict, Tuple, Optional

import httpx
import fitz  # PyMuPDF
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# --- опционально: фуззи нормализация по CSV, если rapidfuzz доступен ---
try:
    from rapidfuzz import process, fuzz  # type: ignore
    HAVE_FUZZ = True
except Exception:
    HAVE_FUZZ = False


APP_VERSION = "3.0.0"
app = FastAPI(title="KHL PDF OCR (column parser)", version=APP_VERSION)

# -----------------------------
# Глобальные словари (опционально)
# -----------------------------
PLAYERS_CSV = os.getenv("PLAYERS_CSV", "players.csv")
REFEREES_CSV = os.getenv("REFEREES_CSV", "referees.csv")

PLAYERS_LIST: List[str] = []
REFS_LIST: List[str] = []

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

def load_dicts() -> None:
    global PLAYERS_LIST, REFS_LIST
    PLAYERS_LIST = _load_list(PLAYERS_CSV)
    REFS_LIST = _load_list(REFEREES_CSV)

load_dicts()

def fuzzy_name(name: str, pool: List[str], cutoff: int = 90) -> str:
    """Возвращает лучший матч из справочника, если rapidfuzz доступен и сходство высокое."""
    if not HAVE_FUZZ or not pool:
        return name
    cand = process.extractOne(name, pool, scorer=fuzz.token_set_ratio)
    if cand and cand[1] >= cutoff:
        return cand[0]
    return name

# -----------------------------
# Вспомогательные нормализации
# -----------------------------
SPACE_RE = re.compile(r"[ \t\u00A0]+")
def norm_spaces(s: str) -> str:
    return SPACE_RE.sub(" ", s).strip()

CAPT_RE = re.compile(r"\b([AKАК])\b\.?$", re.IGNORECASE)
def split_capt(s: str) -> Tuple[str, str]:
    """Выделяем литеру капитана в конце имени ('К'/'A'). Возвращаем (name, capt)."""
    s = norm_spaces(s)
    m = CAPT_RE.search(s)
    if m:
        return norm_spaces(CAPT_RE.sub("", s)), m.group(1).upper().replace("К", "K").replace("А", "A")
    return s, ""

GK_FLAG_RE = re.compile(r"\b([СР])\b\.?$", re.IGNORECASE)  # S/R по рус. верстке часто 'С'/'Р'
def split_gk_flag(name: str) -> Tuple[str, Optional[str]]:
    n = norm_spaces(name)
    m = GK_FLAG_RE.search(n)
    if m:
        flag = m.group(1).upper()
        flag = "S" if flag == "С" else "R"  # русские буквы
        return norm_spaces(GK_FLAG_RE.sub("", n)), flag
    return n, None

def gk_status_from_flag(flag: Optional[str]) -> Optional[str]:
    if flag == "S":
        return "starter"
    if flag == "R":
        return "reserve"
    return None

# -----------------------------
# Загрузка PDF и резка по колонкам
# -----------------------------
async def fetch_pdf_bytes(url: str, timeout: float = 25.0) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,*/*;q=0.9",
        "Referer": "https://www.khl.ru/",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, http2=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content

def read_pdf_columns(pdf_bytes: bytes) -> Tuple[str, str]:
    """Возвращает (left_text, right_text) для 1-й страницы PDF.
    Деление по x-координатам блоков (надёжнее маркеров)."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count < 1:
            return "", ""
        page = doc[0]
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text, block_no, ...)
        W = float(page.rect.width)
        mid = W / 2.0

        left_blocks: List[Tuple[float, str]] = []
        right_blocks: List[Tuple[float, str]] = []

        for b in blocks:
            x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])
            if not txt or not txt.strip():
                continue
            # Строгое разнесение по колонкам
            if x1 <= mid - 8:
                left_blocks.append((y0, txt))
            elif x0 >= mid + 8:
                right_blocks.append((y0, txt))
            else:
                # Попали в «щель» — отправляем по центру
                cx = (x0 + x1) / 2.0
                (left_blocks if cx < mid else right_blocks).append((y0, txt))

        left_text = "\n".join(t for _, t in sorted(left_blocks, key=lambda z: z[0]))
        right_text = "\n".join(t for _, t in sorted(right_blocks, key=lambda z: z[0]))
        return left_text, right_text

# -----------------------------
# Парсеры
# -----------------------------

# 1) Судьи
REFS_BLOCK_RE = re.compile(r"(Главный судья|Линейный судья)[^\n]*", re.IGNORECASE)
NAME_TOKENS_RE = re.compile(r"[A-Za-zА-Яа-яЁё\.\-ʼ’`]+")
def parse_refs(side_text: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for line in side_text.splitlines():
        line = norm_spaces(line)
        if not line:
            continue
        if "Главный судья" in line or "Линейный судья" in line:
            role = "Главный судья" if "Главный судья" in line else "Линейный судья"
            # имена могут быть на соседних строках — соберём токены
            tokens = NAME_TOKENS_RE.findall(line)
            # Уберём слова роли
            tokens = [t for t in tokens if t.lower() not in ("главный", "судья", "линейный")]
            if len(tokens) >= 2:
                # обычно "Фамилия Имя"
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

# 2) Вратари
GOALIES_SECTION_RE = re.compile(r"Вратари(.+?)(?:Звено|Главный тренер|Линейный|Главный судья|---RIGHT---|$)", re.IGNORECASE | re.DOTALL)
# варианты строки: "60 В Бочаров Иван С 18.05.1995 30"
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

# 3) Составы
# Пример строки: "27 З Коттон Алекс 12.05.2001 24" или "18 Н Кугрышев Дмитрий К 18.01.1990 35"
LINEUP_LINE_RE = re.compile(
    r"^\s*(\d{1,2})\s+(В|З|Н)\s+([A-Za-zА-Яа-яЁё\. \-ʼ’`]+?)\s+(?:\d{2}\.\d{2}\.\d{4})",
    re.MULTILINE
)

def parse_lineups(side_text: str, side: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    # берём всё поле (после "Составы" вверху обычно колонки уже сформированы)
    for mm in LINEUP_LINE_RE.finditer(side_text):
        num, pos, raw_name = mm.group(1), mm.group(2), mm.group(3)
        raw_name, gk_flag = split_gk_flag(raw_name)  # у полевых почти всегда None, но на вратарей сработает
        name, capt = split_capt(raw_name)
        name = fuzzy_name(name, PLAYERS_LIST)

        item = {
            "side": side,
            "num": num,
            "pos": pos,
            "name": name,
            "capt": capt,
        }
        # попробуем вытянуть DOB/возраст из хвоста строки
        tail = side_text[mm.end():mm.end()+30]
        mdate = re.search(r"\b(\d{2}\.\d{2}\.\d{4})\b(?:\s+(\d{1,2}))?", tail)
        if mdate:
            item["dob"] = mdate.group(1)
            if mdate.group(2):
                item["age"] = mdate.group(2)
        # если это вратарь в «Составы» — отметим флаг
        if pos == "В" and gk_flag:
            item["gk_flag"] = gk_flag
            item["gk_status"] = gk_status_from_flag(gk_flag)
        out.append(item)
    return out

# -----------------------------
# API
# -----------------------------
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
    """Просто вернём текст (левая/правая колонка) — для отладки."""
    try:
        pdf_bytes = await fetch_pdf_bytes(pdf_url)
    except Exception as e:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": 599, "error": str(e)})

    left, right = read_pdf_columns(pdf_bytes)
    snippet = norm_spaces((left + "\n" + right)[:600])
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
    """Основной парсер: режем PDF на 2 колонки и парсим блоки по-отдельности."""
    try:
        pdf_bytes = await fetch_pdf_bytes(pdf_url)
    except Exception as e:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": 599, "error": str(e)})

    left, right = read_pdf_columns(pdf_bytes)

    data: Dict[str, object] = {}
    if target in ("all", "refs"):
        r = dedup_refs(parse_refs(left) + parse_refs(right))
        data["refs"] = r
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
        "dpi": 130,  # неважно, мы не OCRим — используем текст-слой
        "pages_ocr": 1,
        "data": data,
    }
