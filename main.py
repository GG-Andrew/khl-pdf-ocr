# main.py
import re
import io
import json
import unicodedata
from statistics import median
from typing import Optional, Dict, Any, List

import httpx
import fitz  # PyMuPDF
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# =========================
#  СЕРВИС/КОНФИГ
# =========================
SERVICE_NAME = "khl-pdf-ocr"
VERSION = "3.0.0"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.7",
    "Referer": "https://www.khl.ru/",
    "Origin": "https://www.khl.ru",
    "Connection": "keep-alive",
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

ROLE_ALIASES = {
    "Главный судья": "Главный судья",
    "Главные судьи": "Главный судья",
    "Линейный судья": "Линейный судья",
    "Линейные судьи": "Линейный судья",
}

# =========================
#  УТИЛИТЫ
# =========================
def _norm_txt(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("й", "й").replace("Й", "Й").replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", s).strip()

def words_to_lines(words, y_tol=3.0) -> List[str]:
    lines, cur, last_y = [], [], None
    for w in words:
        y = w[1]
        if last_y is None or abs(y - last_y) <= y_tol:
            cur.append(w)
        else:
            cur.sort(key=lambda z: z[0])
            t = " ".join(_norm_txt(z[4]) for z in cur)
            if t.strip():
                lines.append(t)
            cur = [w]
        last_y = y
    if cur:
        cur.sort(key=lambda z: z[0])
        t = " ".join(_norm_txt(z[4]) for z in cur)
        if t.strip():
            lines.append(t)
    return lines

def split_two_columns(words):
    if not words:
        return [], []
    xs = sorted(((w[0] + w[2]) / 2.0) for w in words)
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

ROSTER_ROW_RE = re.compile(
    r"^(?P<num>\d{1,3})\s+(?P<pos>[ВЗН])\s+(?P<name>[A-ЯЁA-Za-z\-’'\.]+(?:\s+[A-ЯЁA-Za-z\-’'\.]+){0,3})\s+(?P<dob>\d{2}\.\d{2}\.\d{4})\s+(?P<age>\d{1,2})(?:\s+(?P<flag>[SR]))?$"
)

def parse_roster_lines(lines: List[str], side: str):
    out = []
    for ln in lines:
        ln = _norm_txt(ln)
        m = ROSTER_ROW_RE.match(ln)
        if not m:
            continue
        d = m.groupdict()
        flag = (d.get("flag") or "").upper()
        status = "starter" if flag == "S" else ("reserve" if flag == "R" else None)
        out.append({
            "side": side,
            "num": d["num"],
            "pos": d["pos"],
            "name": d["name"].strip().strip("*"),
            "capt": "",
            "dob": d["dob"],
            "age": d["age"],
            "gk_flag": flag,
            "gk_status": status,
        })
    return out

REFS_BLOCK_RE = re.compile(r"(Судьи|Судейская бригада)(.*?)(\n\n|$)", re.S | re.I)
REF_LINE_RE = re.compile(
    r"(?P<role>Главн(ые|ый)\s+судья|Линейн(ые|ый)\s+судья|Линейный судья|Главный судья)\s*[:\-–]?\s*"
    r"(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\.]+){1,2})"
)

def extract_refs_from_text(full_text: str):
    full_text = _norm_txt(full_text)
    m = REFS_BLOCK_RE.search(full_text)
    if not m:
        # часто внизу просто 4 строки подряд — вытащим вольным поиском
        refs = []
        for mm in REFS_LINE_SCAN.finditer(full_text):
            role = ROLE_ALIASES.get(_norm_txt(mm.group("role")), "Линейный судья")
            name = _norm_txt(mm.group("name"))
            if name not in ("Главный судья", "Линейный судья"):
                refs.append({"role": role, "name": name})
        return _dedupe(refs)
    block = m.group(0)
    refs = []
    for mm in REF_LINE_RE.finditer(block):
        role = ROLE_ALIASES.get(_norm_txt(mm.group("role")), "Линейный судья")
        name = _norm_txt(mm.group("name"))
        if name not in ("Главный судья", "Линейный судья"):
            refs.append({"role": role, "name": name})
    return _dedupe(refs)

REFS_LINE_SCAN = re.compile(
    r"(?P<role>Главные судьи|Главный судья|Линейные судьи|Линейный судья)\s*[:\-–]?\s*(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\.]+){1,2})",
    re.I)

def _dedupe(items: List[Dict[str, Any]]):
    out, seen = [], set()
    for it in items:
        k = (it.get("role"), it.get("name"))
        if k not in seen:
            out.append(it)
            seen.add(k)
    return out

def collect_goalies(roster: List[Dict[str, Any]]):
    res = []
    for it in roster:
        if it["pos"] == "В":
            st = it["gk_status"] or ("starter" if it["gk_flag"] == "S" else ("reserve" if it["gk_flag"] == "R" else "unknown"))
            res.append({"name": it["name"], "status": st})
    return res

# =========================
#  ЗАГРУЗКА PDF + FALLBACK
# =========================
async def fetch_pdf(match_id: int, season: int, pdf_url: Optional[str] = None):
    tried, status = [], None
    urls = [pdf_url] if pdf_url else [u.format(season=season, match_id=match_id) for u in URL_TEMPLATES]

    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS, timeout=30, http2=True) as client:
        for u in urls:
            if not u:
                continue
            try:
                r = await client.get(u)
                status = r.status_code
                tried.append(f"{u} [{status}]")
                if r.status_code == 200 and r.headers.get("content-type", "").lower().startswith("application/pdf"):
                    return {"mode": "pdf", "content": r.content, "tried": tried, "status": 200, "url": u}
            except Exception as e:
                tried.append(f"httpx err: {repr(e)}")

    # fallback — плейнтекст через публичный ридер
    for u in urls:
        try:
            jin = f"https://r.jina.ai/http://{u.replace('https://','').replace('http://','')}"
            r = await httpx.AsyncClient(timeout=30).get(jin)
            tried.append(f"{jin} [{r.status_code}]")
            if r.status_code == 200 and r.text:
                return {"mode": "text", "content": r.text, "tried": tried, "status": 200, "url": u}
        except Exception as e:
            tried.append(f"jina err: {repr(e)}")

    return {"mode": "none", "content": b"", "tried": tried, "status": status, "url": None}

# =========================
#  ПАРСИНГ DOCUMENT
# =========================
def parse_all_from_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # 1) refs из целого текста
    full_text = []
    for i in range(len(doc)):
        full_text.append(doc.load_page(i).get_text())
    full_text = "\n".join(full_text)
    refs = extract_refs_from_text(full_text)

    # 2) составы из первой "составной" страницы
    page_idx = 0
    for i in range(len(doc)):
        t = _norm_txt(doc.load_page(i).get_text())
        if re.search(r"Составы|Составы команд", t, re.I):
            page_idx = i
            break

    p = doc.load_page(page_idx)
    words = p.get_text("words")  # (x0,y0,x1,y1,word,block_no,line_no,word_no)
    left, right = split_two_columns(words)
    left_lines = words_to_lines(left)
    right_lines = words_to_lines(right)

    home = parse_roster_lines(left_lines, "home")
    away = parse_roster_lines(right_lines, "away")

    goalies = {"home": collect_goalies(home), "away": collect_goalies(away)}
    return {"refs": refs, "goalies": goalies, "lineups": {"home": home, "away": away}}

def parse_all_from_text(txt: str) -> Dict[str, Any]:
    t = _norm_txt(txt)
    refs = extract_refs_from_text(t)
    parts = t.split("---RIGHT---")
    left_txt = parts[0] if parts else t
    right_txt = parts[1] if len(parts) > 1 else ""

    def extract_side(s: str, side: str):
        lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
        return parse_roster_lines(lines, side)

    home = extract_side(left_txt, "home")
    away = extract_side(right_txt, "away")
    goalies = {"home": collect_goalies(home), "away": collect_goalies(away)}
    return {"refs": refs, "goalies": goalies, "lineups": {"home": home, "away": away}}

# =========================
#  FASTAPI
# =========================
app = FastAPI(title=SERVICE_NAME, version=VERSION)

@app.get("/")
async def root():
    return {"ok": True, "service": SERVICE_NAME, "version": VERSION, "ready": True}

@app.get("/ocr")
async def ocr(
    match_id: int = Query(...),
    season: int = Query(...),
    pdf_url: Optional[str] = None,
):
    res = await fetch_pdf(match_id, season, pdf_url)
    if res["mode"] == "none":
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "GET",
                             "status": res["status"], "tried": res["tried"]}, status_code=502)

    if res["mode"] == "pdf":
        doc = fitz.open(stream=res["content"], filetype="pdf")
        text = "\n".join(doc.load_page(i).get_text() for i in range(len(doc)))
    else:
        text = res["content"]

    snippet = text[:300]
    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": res["url"],
        "pages_ocr": 1,
        "text_len": len(text),
        "snippet": snippet,
        "tried": res["tried"],
    }

@app.get("/extract")
async def extract(
    match_id: int = Query(...),
    season: int = Query(...),
    target: str = Query("all", pattern="^(all|refs|goalies|lineups)$"),
    pdf_url: Optional[str] = None,
):
    res = await fetch_pdf(match_id, season, pdf_url)
    if res["mode"] == "none":
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "GET",
                             "status": res["status"], "tried": res["tried"]}, status_code=502)

    if res["mode"] == "pdf":
        data_all = parse_all_from_pdf(res["content"])
    else:
        data_all = parse_all_from_text(res["content"])

    data = data_all if target == "all" else {target: data_all.get(target, {})}
    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": res["url"],
        "data": data,
        "tried": res["tried"],
    }
