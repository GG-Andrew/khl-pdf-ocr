# main.py
import os, io, re, csv, time, json
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

APP_VERSION = "2.3.0"

app = FastAPI(title="KHL PDF OCR Server", version=APP_VERSION)

# -------------------- ЛЕНИВЫЕ ИМПОРТЫ --------------------
_fitx = None
_pdf2img = None
_PIL = None
_tess = None

def _ensure_pymupdf():
    global _fitx
    if _fitx is None:
        import fitz as _fitx
    return _fitx

def _ensure_pdf2image():
    global _pdf2img
    if _pdf2img is None:
        from pdf2image import convert_from_bytes as _pdf2img
    return _pdf2img

def _ensure_pil():
    global _PIL
    if _PIL is None:
        from PIL import Image, ImageFilter, ImageOps
        _PIL = (Image, ImageFilter, ImageOps)
    return _PIL

def _ensure_tesseract():
    global _tess
    if _tess is None:
        import pytesseract as _tess
    return _tess

# -------------------- КОНСТАНТЫ --------------------------
DEFAULT_SEASON = 1369

HEADERS = {
    "Referer": "https://www.khl.ru/online/",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/pdf,*/*;q=0.9",
}

PDF_TEMPLATES = [
    # предпочтение текст-слою
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-en.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-protocol-ru.pdf",
    "https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-official-ru.pdf",
    "https://khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf",
    "https://www.khl.ru/documents/{season}/{match_id}.pdf",
]

# -------------------- СЛОВАРИ ----------------------------
PLAYERS_CSV = os.getenv("PLAYERS_CSV", "players.csv")
REFEREES_CSV = os.getenv("REFEREES_CSV", "referees.csv")
_dict_players: set = set()
_dict_refs: set = set()

def _load_csv_set(path: str) -> set:
    s = set()
    if not os.path.exists(path):
        return s
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            for cell in row:
                name = cell.strip()
                if name:
                    s.add(name)
    return s

# загрузим один раз (дёшево, не мешает /)
_dict_players = _load_csv_set(PLAYERS_CSV)
_dict_refs = _load_csv_set(REFEREES_CSV)

# -------------------- УТИЛЫ ------------------------------
def http_get(url: str, timeout: float = 20.0) -> Tuple[int, bytes]:
    # пробуем httpx, затем urllib
    try:
        import httpx
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=HEADERS) as c:
            r = c.get(url)
            return r.status_code, r.content
    except Exception:
        try:
            from urllib.request import Request, urlopen
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=timeout) as resp:
                return 200, resp.read()
        except Exception:
            return 0, b""

def fetch_pdf_with_fallback(match_id: int, season: int, pdf_url: Optional[str]) -> Tuple[Optional[bytes], List[str]]:
    tried = []
    if pdf_url:
        tried.append(pdf_url)
        code, data = http_get(pdf_url)
        if code == 200 and data[:4] == b"%PDF":
            return data, tried
    for tpl in PDF_TEMPLATES:
        u = tpl.format(match_id=match_id, season=season)
        tried.append(u)
        code, data = http_get(u)
        if code == 200 and data[:4] == b"%PDF":
            return data, tried
    return None, tried

def normalize_tail_y(text: str) -> str:
    # косметика: СергеИ → Сергей, НиколаИ → Николай, Виталии → Виталий
    text = re.sub(r"еи\b", "ей", text, flags=re.IGNORECASE)
    text = re.sub(r"аи\b", "ай", text, flags=re.IGNORECASE)
    text = re.sub(r"ии\b", "ий", text, flags=re.IGNORECASE)
    return text

def dict_fix(name: str, dictset: set) -> str:
    # если точное совпадение — вернём оригинал из словаря (правильный регистр/буквы)
    if name in dictset:
        return name
    # поиск по началу и без точек/инициалов
    base = re.sub(r"\.$", "", name).strip()
    for cand in dictset:
        if cand.startswith(base) or base in cand:
            return cand
    return name

# -------- ТЕКСТ-СЛОЙ: половины страницы, координаты -------
def extract_halves_text(pdf_bytes: bytes) -> Tuple[str, str]:
    fitz = _ensure_pymupdf()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    rect = page.rect
    mid_x = rect.width / 2
    left = fitz.Rect(rect.x0, rect.y0, mid_x, rect.y1)
    right = fitz.Rect(mid_x, rect.y0, rect.x1, rect.y1)
    tl = page.get_text("text", clip=left) or ""
    tr = page.get_text("text", clip=right) or ""
    doc.close()
    return tl, tr

# -------- OCR Fallback (1-я страница) ---------------------
def preprocess(pil_img):
    Image, ImageFilter, ImageOps = _ensure_pil()
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_first_page(pdf_bytes: bytes, dpi: int = 200, scale: float = 1.25, bin_thresh: int = 185) -> str:
    convert_from_bytes = _ensure_pdf2image()
    pytesseract = _ensure_tesseract()
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
    img = preprocess(pages[0])
    # бинаризация опционально
    if bin_thresh:
        import numpy as np
        arr = np.array(img)
        arr = (arr > bin_thresh) * 255
        from PIL import Image as _Image
        img = _Image.fromarray(arr.astype("uint8"))
    txt = pytesseract.image_to_string(img, lang="rus+eng", config="--oem 1 --psm 6 --dpi 300 --tessedit_preserve_interword_spaces=1")
    return txt

# -------- ПАРСИНГ БЛОКОВ ----------------------------------
RE_LINE = re.compile(
    r"^\s*(?P<num>\d{1,2})\s+"
    r"(?P<pos>[ВЗН])\s+"
    r"(?P<name>[A-Za-zА-ЯЁа-яё\.\-\'\s]+?)"
    r"(?:\s+(?P<capt>[AK]))?\s+"
    r"(?P<dob>\d{2}\.\d{2}\.\d{4})\s+"
    r"(?P<age>\d{1,2})\s*$"
)

def split_blocks(left: str, right: str) -> Dict[str, str]:
    # грубое деление на блоки
    def grab(block_name: str, text: str) -> str:
        m = re.search(block_name, text, flags=re.IGNORECASE)
        return text[m.start():] if m else ""
    return {
        "left": left,
        "right": right,
        "goalies_l": grab(r"Вратари", left),
        "goalies_r": grab(r"Вратари", right),
        "refs_l": grab(r"Главный судья", left),
        "refs_r": grab(r"Главный судья", right),
        "lineups_l": left,
        "lineups_r": right,
    }

def parse_goalies(block: str, side: str) -> List[Dict[str, str]]:
    res = []
    # берём 3 строки после слова "Вратари"
    m = re.search(r"Вратари(.+?)(?:Звено|Главный тренер|Линейный|$)", block, flags=re.S)
    if not m:
        return res
    lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
    for ln in lines[:6]:
        m2 = RE_LINE.search(ln)
        if not m2:
            continue
        name = m2.group("name").strip()
        # вытащим хвост S/R
        gk_flag = ""
        if name.endswith(" С"):
            name = name[:-2].strip(); gk_flag = "S"
        elif name.endswith(" Р"):
            name = name[:-2].strip(); gk_flag = "R"
        name = normalize_tail_y(name)
        name = dict_fix(name, _dict_players)
        status = "starter" if gk_flag == "S" else ("reserve" if gk_flag == "R" else "scratch")
        res.append({"name": name, "status": status})
    return res

def parse_refs(left: str, right: str) -> List[Dict[str, str]]:
    refs = []
    blob = (left + "\n" + right)
    # главные
    g = re.search(r"Главный судья.+?\n(.+?)\n(.+?)\n", blob, flags=re.S)
    if g:
        a = normalize_tail_y(g.group(1)).strip()
        b = normalize_tail_y(g.group(2)).strip()
        a = dict_fix(a, _dict_refs)
        b = dict_fix(b, _dict_refs)
        if a: refs.append({"role":"Главный судья","name":a})
        if b: refs.append({"role":"Главный судья","name":b})
    # линейные
    l = re.search(r"Линейный судья.+?\n(.+?)\n(.+?)\n", blob, flags=re.S)
    if l:
        a = normalize_tail_y(l.group(1)).strip()
        b = normalize_tail_y(l.group(2)).strip()
        a = dict_fix(a, _dict_refs)
        b = dict_fix(b, _dict_refs)
        if a: refs.append({"role":"Линейный судья","name":a})
        if b: refs.append({"role":"Линейный судья","name":b})
    return refs

def parse_lineups(left: str, right: str) -> Dict[str, List[Dict[str, Any]]]:
    out_home, out_away = [], []
    def collect(text: str, side: str, sink: List[Dict[str, Any]]):
        after = re.search(r"№\s+Поз\s+Фамилия,?\s*Имя.*?\n", text)
        start = after.end() if after else 0
        for ln in text[start:].splitlines():
            m = RE_LINE.search(ln)
            if not m: 
                continue
            num = m.group("num")
            pos = m.group("pos")
            name = m.group("name").strip()
            capt = (m.group("capt") or "").strip()
            dob  = m.group("dob")
            age  = m.group("age")
            gk_flag, gk_status = "", None
            if pos == "В":
                if name.endswith(" С"): name, gk_flag, gk_status = name[:-2].strip(), "S", "starter"
                elif name.endswith(" Р"): name, gk_flag, gk_status = name[:-2].strip(), "R", "reserve"
            name = normalize_tail_y(name)
            name = dict_fix(name, _dict_players)
            sink.append({
                "side": side, "num": num, "pos": pos, "name": name,
                "capt": capt, "dob": dob, "age": age,
                "gk_flag": gk_flag, "gk_status": gk_status
            })
    collect(left,  "home", out_home)
    collect(right, "away", out_away)
    return {"home": out_home, "away": out_away}

# -------------------- ENDPOINTS ---------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "khl-pdf-ocr",
        "version": APP_VERSION,
        "ready": True,
        "dicts": {
            "players_path": PLAYERS_CSV if os.path.exists(PLAYERS_CSV) else False,
            "referees_path": REFEREES_CSV if os.path.exists(REFEREES_CSV) else False,
            "players_loaded": len(_dict_players),
            "refs_loaded": len(_dict_refs),
        },
    }

@app.get("/ocr")
def ocr_parse(
    match_id: int = Query(...),
    pdf_url: Optional[str] = Query(None),
    season: int = Query(DEFAULT_SEASON),
    dpi: int = Query(200, ge=120, le=360),
    max_pages: int = Query(1, ge=1, le=5),
    scale: float = Query(1.25, ge=1.0, le=2.0),
    bin_thresh: int = Query(185, ge=120, le=230),
):
    t0 = time.time()
    pdf, tried = fetch_pdf_with_fallback(match_id, season, pdf_url)
    if not pdf:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": 404, "tried": tried})
    t1 = time.time()
    # пробуем текст-слой
    try:
        L, R = extract_halves_text(pdf)
        txt = (L + "\n---RIGHT---\n" + R).strip()
        if len(txt) > 100:
            return {
                "ok": True,
                "match_id": match_id,
                "season": season,
                "source_pdf": tried[-1],
                "pdf_len": len(pdf),
                "dpi": dpi,
                "pages_ocr": 1,
                "dur_total_s": round(time.time() - t0, 3),
                "dur_download_s": round(t1 - t0, 3),
                "dur_preproc_s": 0.0,
                "dur_ocr_s": 0.0,
                "text_len": len(txt),
                "snippet": txt[:400]
            }
    except Exception:
        pass
    # Fallback OCR
    t2 = time.time()
    txt = ocr_first_page(pdf, dpi=dpi, scale=scale, bin_thresh=bin_thresh)
    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": tried[-1],
        "pdf_len": len(pdf),
        "dpi": dpi,
        "pages_ocr": 1,
        "dur_total_s": round(time.time() - t0, 3),
        "dur_download_s": round(t1 - t0, 3),
        "dur_preproc_s": round(t2 - t1, 3),
        "dur_ocr_s": round(time.time() - t2, 3),
        "text_len": len(txt),
        "snippet": txt[:400]
    }

@app.get("/extract")
def extract(
    match_id: int = Query(...),
    pdf_url: Optional[str] = Query(None),
    season: int = Query(DEFAULT_SEASON),
    target: str = Query("all", pattern="^(refs|goalies|lineups|all)$"),
    dpi: int = 130,
):
    t0 = time.time()
    pdf, tried = fetch_pdf_with_fallback(match_id, season, pdf_url)
    if not pdf:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": 404, "tried": tried})
    # текст-слой
    L, R = extract_halves_text(pdf)
    blocks = split_blocks(L, R)
    data: Dict[str, Any] = {}

    if target in ("refs", "all"):
        data["refs"] = parse_refs(blocks["left"], blocks["right"])
    if target in ("goalies", "all"):
        home_g = parse_goalies(blocks["goalies_l"], "home")
        away_g = parse_goalies(blocks["goalies_r"], "away")
        data["goalies"] = {"home": home_g, "away": away_g}
    if target in ("lineups", "all"):
        data["lineups"] = parse_lineups(blocks["lineups_l"], blocks["lineups_r"])

    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": tried[-1],
        "pdf_len": len(pdf),
        "dpi": dpi,
        "pages_ocr": 1,
        "dur_total_s": round(time.time() - t0, 3),
        "data": data,
    }
