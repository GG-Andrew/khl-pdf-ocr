# main.py
# KHL PDF OCR Server — v2.0.0 (text-layer first, OCR fallback)
# - /          : health (пути словарей)
# - /ocr       : сначала текст-слой (быстро), иначе OCR (медленно, как fallback)
# - /extract   : структура refs/goalies/lineups_raw с разбиением на колонки
#
# На большинстве протоколов КХЛ есть текст-слой → скорость < 1 c.
# Если текста нет, включается прежний OCR-конвейер.

import os, re, time, hashlib, csv
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel

# ---- PDF: текст-слой ----
import fitz  # PyMuPDF

# ---- OCR fallback ----
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps
import pytesseract

# ---- fuzzy fix ----
from rapidfuzz import process, fuzz

APP_VERSION = "2.0.0"
DEFAULT_SEASON = 1369

app = FastAPI(title="KHL PDF OCR Server", version=APP_VERSION)

# ---------------------------- HTTP ----------------------------

HEADERS = {
    "Referer": "https://www.khl.ru/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.9",
    "Accept-Language": "ru-RU,ru;q=0.9",
}

PDF_TEMPLATES = [
    "{pdf_url}",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-protocol-ru.pdf",
    "https://www.khl.ru/documents/{season}/{match_id}.pdf",
]

async def fetch_pdf_with_fallback(pdf_url: str, match_id: int, season: int):
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
            try:
                r = await client.get(url, headers={**HEADERS, "Referer": f"https://www.khl.ru/game/{match_id}/"})
                if r.status_code == 200 and r.headers.get("content-type","").startswith("application/pdf"):
                    return r.content, url, tried, None
                else:
                    last_error = f"status:{r.status_code} ct:{r.headers.get('content-type','')}"
            except Exception as e:
                last_error = f"get:{type(e).__name__}:{e}"

    return None, None, tried, last_error

# ---------------------------- Dictionaries ----------------------------

PLAYERS_DICT: List[str] = []
REFEREES_DICT: List[str] = []
DICT_SOURCES: Dict[str, str] = {"players": "", "referees": ""}

def _find_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def _load_names(path: Optional[str]) -> List[str]:
    if not path: return []
    names = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(1024); f.seek(0)
            try:
                sn = csv.Sniffer(); is_csv = sn.has_header(head)
            except Exception:
                is_csv = False
            if is_csv:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                use = None
                for c in ("name","fio","ФИО","Name"):
                    if c in cols: use = c; break
                if not use and cols: use = cols[0]
                for row in reader:
                    v = (row.get(use) or "").strip()
                    if v: names.append(v)
            else:
                for line in f:
                    s = line.strip()
                    if s: names.append(s)
    except Exception:
        pass
    out = []
    for n in names:
        s = re.sub(r"\s+"," ",n).strip()
        if not s: continue
        parts = s.split()
        s = " ".join([parts[0].capitalize()] + [p.capitalize() for p in parts[1:]])
        out.append(s)
    return sorted(set(out))

def load_dicts():
    global PLAYERS_DICT, REFEREES_DICT, DICT_SOURCES
    pp = _find_path(["players.csv","data/players.csv"])
    rr = _find_path(["referees.csv","data/referees.csv"])
    DICT_SOURCES["players"] = pp or ""
    DICT_SOURCES["referees"] = rr or ""
    PLAYERS_DICT = _load_names(pp)
    REFEREES_DICT = _load_names(rr)

load_dicts()

def best_match(name: str, pool: List[str], th: int = 75) -> str:
    if not name or not pool: return name
    cand, score, _ = process.extractOne(name, pool, scorer=fuzz.WRatio)
    return cand if cand and score >= th else name

# ---------------------------- Text normalization ----------------------------

LAT_TO_CYR = str.maketrans({
    "A":"А","B":"В","C":"С","E":"Е","H":"Н","K":"К","M":"М","O":"О","P":"Р","T":"Т","X":"Х","Y":"У",
    "a":"а","c":"с","e":"е","o":"о","p":"р","x":"х","y":"у",
})
FIO_RE = re.compile(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.)?\b")

def norm_text(s: str) -> str:
    if not s: return s
    s = s.translate(LAT_TO_CYR)
    s = re.sub(r"([А-ЯЁ][а-яё]{2,})([А-ЯЁ][а-яё]{2,})", r"\1 \2", s)
    s = re.sub(r"([А-ЯЁ][а-яё]+)([А-ЯЁ]\.)", r"\1 \2", s)
    s = re.sub(r"[ \t]+"," ",s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clean_name_chunk(s: str) -> Optional[str]:
    s = re.sub(r"\s+"," ", s or "").strip()
    s = re.sub(r"[\[\(]?\d{1,2}\.\d{1,2}\.\d{2,4}[\]\)]?","", s)
    s = s.strip("|·—-:;,. ")
    if not s: return None
    m = FIO_RE.search(s)
    return m.group(0) if m else None

# ---------------------------- TEXT-LAYER extraction (fast) ----------------------------

def extract_page_halves_text(pdf_bytes: bytes, dpi_for_bbox: int = 72) -> Tuple[str, str]:
    """
    Возвращает (left_text, right_text) первой страницы из текст-слоя.
    Если текста нет — вернёт ("","").
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        return "",""
    page = doc.load_page(0)
    # координаты страницы
    rect = page.rect  # (0,0,w,h) при 72dpi
    mid_x = rect.width / 2.0
    left_rect = fitz.Rect(0, 0, mid_x, rect.height)
    right_rect = fitz.Rect(mid_x, 0, rect.width, rect.height)

    def extract_rect(r: fitz.Rect) -> str:
        blocks = page.get_text("blocks", clip=r)
        # blocks: list of (x0,y0,x1,y1,"text", block_no, block_type)
        parts = []
        for b in sorted(blocks, key=lambda x: (x[1], x[0])):  # сверху-вниз, слева-направо
            txt = b[4]
            if txt:
                parts.append(txt)
        return "\n".join(parts)

    lt = norm_text(extract_rect(left_rect))
    rt = norm_text(extract_rect(right_rect))
    return lt, rt

# ---------------------------- OCR fallback ----------------------------

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

def ocr_halves_first_page(pdf_bytes: bytes, dpi=130, scale=1.10, bin_thresh=185) -> Tuple[str,str]:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
    if not pages: return "",""
    p = preprocess(pages[0], scale=scale, bin_thresh=bin_thresh)
    w, h = p.size
    mid = w // 2
    left = p.crop((0,0,mid,h))
    right = p.crop((mid,0,w,h))
    return norm_text(ocr_one(left)), norm_text(ocr_one(right))

# ---------------------------- Parsing (from halves text) ----------------------------

def parse_refs(left_text: str, right_text: str) -> List[Dict[str,str]]:
    lines = [l.strip() for l in (left_text + "\n" + right_text).split("\n") if l.strip()]
    roles = ["Главный судья","Линейный судья","Резервный главный судья","Резервный судья","Резервный линейный судья"]
    out = []
    for i, ln in enumerate(lines):
        for role in roles:
            if re.search(rf"^{role}\b", ln, flags=re.I):
                # имя в той же или следующих 2 строках
                for cand in [ln] + lines[i+1:i+3]:
                    nm = clean_name_chunk(cand)
                    if nm:
                        out.append({"role": role, "name": best_match(nm, REFEREES_DICT, 75)})
                        break
    # дедуп/ограничение
    seen, res = set(), []
    for j in out:
        k=(j["role"], j["name"])
        if k in seen: continue
        seen.add(k); res.append(j)
    return res[:6]

def parse_goalies(left_text: str, right_text: str) -> Dict[str,List[Dict[str,str]]]:
    def grab(text: str) -> List[Dict[str,str]]:
        out = []
        text = text.replace("│","|")
        # столбцовый вид: NN | В | Фамилия Имя ...
        for m in re.finditer(r"\b\d{1,2}\s*\|\s*[ВV]\s*\|\s*([А-ЯЁA-Z][^\n|]+)", text, flags=re.I):
            nm = clean_name_chunk(m.group(1))
            if nm:
                out.append({"name": best_match(nm, PLAYERS_DICT, 75)})
        # fallback: NN В Фамилия Имя ...
        if not out:
            for m in re.finditer(r"\b\d{1,2}\s*[ВV]\s+([А-ЯЁA-Z][^\n\d]{3,})", text, flags=re.I):
                nm = clean_name_chunk(m.group(1))
                if nm:
                    out.append({"name": best_match(nm, PLAYERS_DICT, 75)})
        # uniq+limit
        seen, res = set(), []
        for o in out:
            if o["name"] in seen: continue
            seen.add(o["name"]); res.append(o)
        return res[:3]
    return {"home": grab(left_text), "away": grab(right_text)}

# ---------------------------- Models ----------------------------

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

# ---------------------------- Endpoints ----------------------------

@app.get("/")
def root():
    return {
        "ok": True, "service":"khl-pdf-ocr", "version": APP_VERSION, "ready": True,
        "dicts": {
            "players_path": DICT_SOURCES["players"],
            "referees_path": DICT_SOURCES["referees"],
            "players_loaded": len(PLAYERS_DICT),
            "refs_loaded": len(REFEREES_DICT),
        }
    }

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
    pdf_bytes, final_url, tried, last_err = await fetch_pdf_with_fallback(pdf_url, match_id, season)
    if not pdf_bytes:
        return OCRResponse(ok=False, step="GET", status=404, match_id=match_id, season=season, tried=tried, error=last_err)
    t_pdf = time.time()

    # 1) быстрый текст-слой
    lt, rt = extract_page_halves_text(pdf_bytes)
    if (lt.strip() or rt.strip()):
        full = (lt + "\n" + rt).strip()
        return OCRResponse(
            ok=True, match_id=match_id, season=season,
            source_pdf=final_url, pdf_len=len(pdf_bytes),
            dpi=dpi, pages_ocr=1,
            dur_total_s=round(time.time()-t0,3),
            dur_download_s=round(t_pdf-t0,3),
            dur_preproc_s=0.0, dur_ocr_s=0.0,
            text_len=len(full), snippet=re.sub(r"\s+"," ", full)[:480]
        )

    # 2) fallback: OCR (медленно)
    p0 = time.time()
    ltxt, rtxt = ocr_halves_first_page(pdf_bytes, dpi=dpi, scale=scale, bin_thresh=bin_thresh)
    full = (ltxt + "\n" + rtxt).strip()
    return OCRResponse(
        ok=True, match_id=match_id, season=season,
        source_pdf=final_url, pdf_len=len(pdf_bytes),
        dpi=dpi, pages_ocr=1,
        dur_total_s=round(time.time()-t0,3),
        dur_download_s=round(t_pdf-t0,3),
        dur_preproc_s=round(p0-t_pdf,3), dur_ocr_s=round(time.time()-p0,3),
        text_len=len(full), snippet=re.sub(r"\s+"," ", full)[:480]
    )

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
    pdf_bytes, final_url, tried, last_err = await fetch_pdf_with_fallback(pdf_url, match_id, season)
    if not pdf_bytes:
        return {"ok": False, "step":"GET","status":404,"match_id":match_id,"season":season,"tried":tried,"error":last_err}

    # текст-слой сначала
    lt, rt = extract_page_halves_text(pdf_bytes)
    used_fallback = False
    if not (lt.strip() or rt.strip()):
        # OCR fallback
        used_fallback = True
        lt, rt = ocr_halves_first_page(pdf_bytes, dpi=dpi, scale=scale, bin_thresh=bin_thresh)

    data: Dict[str,object] = {}
    keys = ["refs","goalies","lineups"] if target == "all" else [target]
    for k in keys:
        if k == "refs":
            data["refs"] = parse_refs(lt, rt)
        elif k == "goalies":
            data["goalies"] = parse_goalies(lt, rt)
        elif k == "lineups":
            data["lineups_raw"] = lt + "\n---RIGHT---\n" + rt

    return {
        "ok": True,
        "match_id": match_id, "season": season,
        "source_pdf": final_url, "pdf_len": len(pdf_bytes),
        "dpi": dpi, "pages_ocr": 1,
        "dur_total_s": round(time.time()-t0,3),
        "used_fallback_ocr": used_fallback,
        "data": data,
    }
