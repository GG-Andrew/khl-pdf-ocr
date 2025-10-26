from fastapi import FastAPI, Query
from pydantic import BaseModel
import requests
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract
import re
import unicodedata
from statistics import median
from typing import List, Dict

app = FastAPI(title="KHL PDF OCR")

# --- ROLE ALIASES ---
ROLE_ALIASES = {
    "Главный судья": "Главный судья",
    "Главные судьи": "Главный судья",
    "Линейный судья": "Линейный судья",
    "Линейные судьи": "Линейный судья",
}

# --- UTILS ---
def _norm_txt(s: str) -> str:
    s = s.replace('\xa0',' ').replace('  ',' ')
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('й','й').replace('Й','Й').replace('ё','е').replace('Ё','Е')
    return re.sub(r'\s+', ' ', s).strip()

def extract_refs_from_text(full_text: str) -> List[Dict]:
    full_text = _norm_txt(full_text)
    refs = []
    REFS_BLOCK_RE = re.compile(r'(Судьи|Судейская бригада)(.*?)(\n\n|$)', re.S|re.I)
    REF_LINE_RE = re.compile(r'(?P<role>Главн(ые|ый)\s+судья|Линейн(ые|ый)\s+судья)\s*[:\-–]?\s*(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\.]+){1,2})')
    m = REFS_BLOCK_RE.search(full_text)
    if not m:
        return []
    block = m.group(0)
    for mm in REF_LINE_RE.finditer(block):
        role = ROLE_ALIASES.get(_norm_txt(mm.group('role')), 'Линейный судья')
        name = _norm_txt(mm.group('name'))
        if name and name not in ('Главный судья','Линейный судья'):
            refs.append({"role": role, "name": name})
    # dedupe
    uniq = []
    seen = set()
    for r in refs:
        k = (r['role'], r['name'])
        if k not in seen:
            uniq.append(r)
            seen.add(k)
    return uniq

# --- PDF/IMAGE OCR ---
def pdf_to_text(pdf_bytes: bytes) -> str:
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            t = page.get_text()
            if t.strip():
                text += t + "\n"
    except:
        # fallback OCR using pdf2image
        images = convert_from_bytes(pdf_bytes, dpi=150)
        for img in images:
            text += pytesseract.image_to_string(img, lang='rus+eng') + "\n"
    return text

# --- API MODEL ---
class OCRResponse(BaseModel):
    ok: bool
    match_id: int
    season: int
    source_pdf: str
    pages_ocr: int = 0
    text_len: int = 0
    snippet: str = ""
    data: Dict = {}

# --- ENDPOINT ---
@app.get("/extract", response_model=OCRResponse)
def extract(match_id: int = Query(...), pdf_url: str = Query(...), season: int = Query(...)):
    try:
        r = requests.get(pdf_url, timeout=15)
        r.raise_for_status()
        pdf_bytes = r.content
        text = pdf_to_text(pdf_bytes)
        pages_ocr = len(fitz.open(stream=pdf_bytes))
        snippet = text[:300].replace("\n"," ") if text else ""
        refs = extract_refs_from_text(text)
        return {
            "ok": True,
            "match_id": match_id,
            "season": season,
            "source_pdf": pdf_url,
            "pages_ocr": pages_ocr,
            "text_len": len(text),
            "snippet": snippet,
            "data": {
                "refs": refs,
                "goalies": {"home": [], "away": []},  # можно расширить
                "lineups": {"home": [], "away": []}   # можно расширить
            }
        }
    except requests.HTTPError as e:
        return {"ok": False, "match_id": match_id, "season": season, "step":"GET", "status": e.response.status_code, "tried":[pdf_url]}
    except Exception as e:
        return {"ok": False, "match_id": match_id, "season": season, "step":"PROCESS", "error": str(e)}

