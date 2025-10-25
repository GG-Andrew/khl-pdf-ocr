# main.py
import re
import unicodedata
import fitz  # PyMuPDF
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import requests
import tempfile
import os

app = FastAPI(title="KHL PDF OCR & Parser")

# ---------- UTILS ----------
ROLE_ALIASES = {
    "Главный судья": "Главный судья",
    "Главные судьи": "Главный судья",
    "Линейный судья": "Линейный судья",
    "Линейные судьи": "Линейный судья",
}

def _norm_txt(s: str) -> str:
    s = s.replace('\xa0',' ').replace('  ',' ')
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('й','й').replace('Й','Й').replace('ё','е').replace('Ё','Е')
    return re.sub(r'\s+', ' ', s).strip()

# ---------- PARSING ----------
ROSTER_ROW_RE = re.compile(
    r'^(?P<num>\d{1,3})\s+(?P<pos>[ВЗН])\s+(?P<name>[A-ЯЁA-Za-z\-’\'\.]+(?:\s+[A-ЯЁA-Za-z\-’\'\.]+){0,3})\s*(?P<flag>[SR]?)\s*(?P<dob>\d{2}\.\d{2}\.\d{4})?\s*(?P<age>\d{1,2})?$'
)

REFS_BLOCK_RE = re.compile(r'(Судьи|Судейская бригада)(.*?)(\n\n|$)', re.S|re.I)
REF_LINE_RE = re.compile(
    r'(?P<role>Главн(ые|ый)\s+судья|Линейн(ые|ый)\s+судья|Линейный судья|Главный судья)\s*[:\-–]?\s*'
    r'(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\.]+){0,2})'
)

def parse_roster_lines(text: str, side: str):
    lines = text.split('\n')
    out = []
    for ln in lines:
        ln = _norm_txt(ln)
        m = ROSTER_ROW_RE.match(ln)
        if not m:
            continue
        d = m.groupdict()
        out.append({
            "side": side,
            "num": d["num"],
            "pos": d["pos"],
            "name": d["name"].strip(),
            "capt": "",
            "dob": d.get("dob"),
            "age": d.get("age"),
            "gk_flag": "S" if d.get("flag")=="S" else ("R" if d.get("flag")=="R" else ""),
            "gk_status": "starter" if d.get("flag")=="S" else ("reserve" if d.get("flag")=="R" else None),
        })
    return out

def extract_refs_from_text(full_text: str):
    full_text = _norm_txt(full_text)
    m = REFS_BLOCK_RE.search(full_text)
    if not m:
        return []
    block = m.group(0)
    refs = []
    for mm in REF_LINE_RE.finditer(block):
        role = ROLE_ALIASES.get(_norm_txt(mm.group('role')), 'Линейный судья')
        name = _norm_txt(mm.group('name'))
        if name:
            refs.append({"role": role, "name": name})
    uniq = []
    seen = set()
    for r in refs:
        k = (r['role'], r['name'])
        if k not in seen:
            uniq.append(r); seen.add(k)
    return uniq

# ---------- FASTAPI MODELS ----------
class ExtractResponse(BaseModel):
    ok: bool
    match_id: int
    season: int
    source_pdf: str
    pdf_len: Optional[int]
    pages_ocr: Optional[int]
    text_len: Optional[int]
    snippet: Optional[str]
    data: dict
    tried: Optional[List[str]] = []

# ---------- MAIN ENDPOINT ----------
@app.get("/extract", response_model=ExtractResponse)
def extract(
    match_id: int = Query(...),
    season: int = Query(...),
    pdf_url: str = Query(...),
    target: str = Query("all"),
):
    tried = []
    try:
        # скачиваем PDF
        r = requests.get(pdf_url)
        tried.append(f"{pdf_url} [{r.status_code}]")
        r.raise_for_status()
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        pdf_len = os.path.getsize(pdf_path)

        # открываем PDF
        doc = fitz.open(pdf_path)
        pages_ocr = doc.page_count
        full_text = "\n".join([doc[i].get_text() for i in range(pages_ocr)])
        snippet = full_text[:500]

        # ---------- парсим состав ----------
        lineups_home = parse_roster_lines(full_text, "home")
        lineups_away = parse_roster_lines(full_text, "away")
        goalies_home = [p for p in lineups_home if p["pos"]=="В"]
        goalies_away = [p for p in lineups_away if p["pos"]=="В"]
        refs = extract_refs_from_text(full_text)

        data = {
            "lineups": {"home": lineups_home, "away": lineups_away},
            "goalies": {"home": goalies_home, "away": goalies_away},
            "refs": refs
        }

        return ExtractResponse(
            ok=True,
            match_id=match_id,
            season=season,
            source_pdf=pdf_url,
            pdf_len=pdf_len,
            pages_ocr=pages_ocr,
            text_len=len(full_text),
            snippet=snippet,
            data=data,
            tried=tried
        )
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
