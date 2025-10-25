# -*- coding: utf-8 -*-
# ============================================================
# khl-pdf-ocr — финальный main.py (единый файл)
# ============================================================

import re
import io
import unicodedata
from statistics import median
from typing import List, Dict, Any, Optional

import httpx
import fitz  # PyMuPDF
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

SERVICE_NAME = "khl-pdf-ocr"

# -----------------------------
#            УТИЛИТЫ
# -----------------------------

ROLE_ALIASES = {
    "Главный судья": "Главный судья",
    "Главные судьи": "Главный судья",
    "Линейный судья": "Линейный судья",
    "Линейные судьи": "Линейный судья",
}

def _norm_txt(s: str) -> str:
    s = s.replace("\xa0", " ").replace("  ", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("й", "й").replace("Й", "Й").replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", s).strip()

def find_roster_page(doc: fitz.Document) -> int:
    for i in range(len(doc)):
        t = _norm_txt(doc.load_page(i).get_text())
        if re.search(r"Составы|Составы команд", t, re.I):
            return i
    return 0  # fallback

def split_two_columns(words: List[List[Any]]):
    xs = sorted([(w[0] + w[2]) / 2.0 for w in words])  # центры слов
    if not xs:
        return [], []
    mid = median(xs)
    left = [w for w in words if ((w[0] + w[2]) / 2.0) <= mid]
    right = [w for w in words if ((w[0] + w[2]) / 2.0) > mid]
    if len(left) < 10 or len(right) < 10:
        page_w = max(w[2] for w in words)
        cut = page_w * 0.5
        left = [w for w in words if ((w[0] + w[2]) / 2.0) <= cut]
        right = [w for w in words if ((w[0] + w[2]) / 2.0) > cut]
    left.sort(key=lambda w: (w[1], w[0]))
    right.sort(key=lambda w: (w[1], w[0]))
    return left, right

def words_to_lines(words: List[List[Any]], y_tolerance: float = 3.0) -> List[str]:
    lines = []
    cur = []
    last_y = None
    for w in words:
        y = w[1]
        if last_y is None or abs(y - last_y) <= y_tolerance:
            cur.append(w)
        else:
            cur.sort(key=lambda z: z[0])
            line_txt = " ".join(_norm_txt(z[4]) for z in cur)
            if line_txt.strip():
                lines.append(line_txt)
            cur = [w]
        last_y = y
    if cur:
        cur.sort(key=lambda z: z[0])
        line_txt = " ".join(_norm_txt(z[4]) for z in cur)
        if line_txt.strip():
            lines.append(line_txt)
    return lines

ROSTER_ROW_RE = re.compile(
    r"^(?P<num>\d{1,3})\s+"
    r"(?P<pos>[ВЗН])\s+"
    r"(?P<name>[A-ЯЁA-Za-z\-’'\.]+(?:\s+[A-ЯЁA-Za-z\-’'\.]+){0,3})\s+"
    r"(?P<dob>\d{2}\.\d{2}\.\d{4})\s+"
    r"(?P<age>\d{1,2})(?:\s+(?P<tail>[SR]))?$"
)

def parse_roster_lines(lines: List[str], side: str) -> List[Dict[str, Any]]:
    out = []
    for ln in lines:
        raw = _norm_txt(ln)
        m = ROSTER_ROW_RE.match(raw)
        if not m:
            continue
        d = m.groupdict()
        tail = d.get("tail") or ""
        flag = "S" if tail.upper() == "S" else ("R" if tail.upper() == "R" else "")
        status = "starter" if flag == "S" else ("reserve" if flag == "R" else None)

        out.append(
            {
                "side": side,
                "num": d["num"],
                "pos": d["pos"],
                "name": d["name"].strip().strip("*").strip(),
                "capt": "",
                "dob": d["dob"],
                "age": d["age"],
                "gk_flag": flag,
                "gk_status": status,
            }
        )
    return out

REFS_BLOCK_RE = re.compile(r"(Судьи|Судейская бригада)(.*?)(\n\n|$)", re.S | re.I)
REF_LINE_RE = re.compile(
    r"(?P<role>Главн(ые|ый)\s+судья|Линейн(ые|ый)\s+судья|Линейный судья|Главный судья)\s*[:\-–]?\s*"
    r"(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\.]+){1,2})"
)

def extract_refs_from_text(full_text: str) -> List[Dict[str, str]]:
    full_text = _norm_txt(full_text)
    m = REFS_BLOCK_RE.search(full_text)
    if not m:
        return []
    block = m.group(0)
    refs = []
    for mm in REF_LINE_RE.finditer(block):
        role = ROLE_ALIASES.get(_norm_txt(mm.group("role")).replace("Линейн", "Линейн"), "Линейный судья")
        name = _norm_txt(mm.group("name"))
        if name and name not in ("Главный судья", "Линейный судья"):
            refs.append({"role": role, "name": name})
    uniq = []
    seen = set()
    for r in refs:
        k = (r["role"], r["name"])
        if k not in seen:
            uniq.append(r)
            seen.add(k)
    return uniq

# -----------------------------
#         ЗАГРУЗКА PDF
# -----------------------------

HEADERS = {
    "Referer": "https://www.khl.ru/online/",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/pdf,*/*;q=0.9",
}

URL_TEMPLATES = [
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-en.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-protocol-ru.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-official-ru.pdf",
    "https://khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf",
    "https://www.khl.ru/documents/{season}/{match_id}.pdf",
]

async def fetch_pdf(match_id: int, season: int, pdf_url: Optional[str] = None) -> (bytes, List[str], Optional[int]):
    tried = []
    status = None
    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS, timeout=30) as client:
        urls = [pdf_url] if pdf_url else [u.format(season=season, match_id=match_id) for u in URL_TEMPLATES]
        for u in urls:
            if not u:
                continue
            try:
                r = await client.get(u)
                status = r.status_code
                tried.append(f"{u} [{status}]")
                if r.status_code == 200 and r.headers.get("content-type", "").lower().startswith("application/pdf"):
                    return r.content, tried, 200
            except Exception as e:
                tried.append(f"httpx err: {repr(e)}")
    return b"", tried, status

# -----------------------------
#        ПАРСИНГ PDF
# -----------------------------

def parse_all(doc: fitz.Document) -> Dict[str, Any]:
    # 1) Судьи — берем по всему тексту страницы(страниц)
    full_text = "\n".join(doc.load_page(i).get_text() for i in range(len(doc)))
    refs = extract_refs_from_text(full_text)

    # 2) Регистр слов на странице составов, делим на 2 колонки
    p_index = find_roster_page(doc)
    page = doc.load_page(p_index)
    words = page.get_text("words")  # x0,y0,x1,y1,word,block_no,line_no,word_no
    left, right = split_two_columns(words)

    left_lines = words_to_lines(left)
    right_lines = words_to_lines(right)

    home = parse_roster_lines(left_lines, "home")
    away = parse_roster_lines(right_lines, "away")

    # 3) Вратари — фильтруем из составов по pos == 'В' + gk_status
    def collect_goalies(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out = []
        for it in items:
            if it.get("pos") == "В":
                st = it.get("gk_status") or ("starter" if it.get("gk_flag") == "S" else ("reserve" if it.get("gk_flag") == "R" else "unknown"))
                out.append({"name": it.get("name", ""), "status": st})
        return out

    goalies = {
        "home": collect_goalies(home),
        "away": collect_goalies(away),
    }

    return {
        "refs": refs,
        "goalies": goalies,
        "lineups": {"home": home, "away": away},
    }

# -----------------------------
#          FASTAPI
# -----------------------------

app = FastAPI(title=SERVICE_NAME)

@app.get("/")
def root():
    return {"ok": True, "service": SERVICE_NAME, "ready": True}

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": SERVICE_NAME, "ready": True}

@app.get("/extract")
async def extract(
    match_id: int = Query(...),
    season: int = Query(...),
    target: str = Query("all", regex="^(all|refs|goalies|lineups)$"),
    pdf_url: Optional[str] = None,
):
    pdf_bytes, tried, status = await fetch_pdf(match_id, season, pdf_url)
    if not pdf_bytes:
        return JSONResponse(
            {
                "ok": False,
                "match_id": match_id,
                "season": season,
                "step": "GET",
                "status": status,
                "tried": tried,
            },
            status_code=502,
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    data_all = parse_all(doc)

    payload = {"ok": True, "match_id": match_id, "season": season, "source_pdf": (pdf_url or ""), "data": {}}
    if target == "all":
        payload["data"] = data_all
    else:
        payload["data"][target] = data_all[target]

    return JSONResponse(payload)
