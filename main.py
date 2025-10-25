# main.py
import os, io, re, csv, time, json, unicodedata
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

APP_VERSION = "2.6.0"

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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "close",
}

PDF_TEMPLATES = [
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

_dict_players = _load_csv_set(PLAYERS_CSV)
_dict_refs = _load_csv_set(REFEREES_CSV)

# -------------------- HTTP ЗАГРУЗКА PDF -------------------
def http_get(url: str, timeout: float = 25.0) -> Tuple[int, bytes]:
    UA_POOL = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ]
    BASE = {
        "Referer": "https://www.khl.ru/online/",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "close",
    }
    for ua in UA_POOL:
        hdrs = dict(BASE)
        hdrs["User-Agent"] = ua
        # 1) httpx (HTTP/2)
        try:
            import httpx, random, time as _t
            with httpx.Client(follow_redirects=True, timeout=timeout, headers=hdrs, http2=True) as c:
                r = c.get(url)
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    return 200, r.content
                _t.sleep(0.5 + random.random() * 0.5)
        except Exception:
            pass
        # 2) urllib (HTTP/1.1)
        try:
            from urllib.request import Request, urlopen
            req = Request(url, headers=hdrs)
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if data[:4] == b"%PDF":
                    return 200, data
        except Exception:
            continue
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

# -------------------- ТЕКСТ-НОРМАЛИЗАЦИЯ ------------------
def _strip_combining(s: str) -> str:
    # удаляем все комбинирующие символы (в т.ч. ударение U+0301)
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if unicodedata.category(ch) != "Mn")

def clean_text(s: str) -> str:
    # NBSP/узкие пробелы/неразрывные дефисы -> обычные
    s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ").replace("\u2011", "-")
    s = _strip_combining(s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

def normalize_tail_y(text: str) -> str:
    # СергеИ -> Сергей, НиколаИ -> Николай, Виталии -> Виталий
    text = re.sub(r"еи\b", "ей", text, flags=re.IGNORECASE)
    text = re.sub(r"аи\b", "ай", text, flags=re.IGNORECASE)
    text = re.sub(r"ии\b", "ий", text, flags=re.IGNORECASE)
    return text

def dict_fix(name: str, dictset: set) -> str:
    if not name:
        return name
    if name in dictset:
        return name
    base = re.sub(r"\.$", "", name).strip()
    # грубый, но полезный автофикс: берём ближайший словарный по префиксу
    for cand in dictset:
        if cand.startswith(base) or base in cand:
            return cand
    return name

# -------------------- СБОРКА ТЕКСТА ИЗ WORDS ----------------
def words_to_text(page, clip) -> str:
    """Стабильно собирает текст в пределах clip по строкам."""
    words = page.get_text("words", clip=clip)  # [x0,y0,x1,y1, "word", block_no, line_no, word_no]
    # сортируем: сначала по y (строка), потом по x (порядок слов)
    words.sort(key=lambda w: (round(w[1], 1), w[0]))
    lines: List[List[str]] = []
    current_y = None
    buf: List[str] = []
    for w in words:
        y0 = round(w[1], 1)
        txt = w[4]
        if current_y is None:
            current_y = y0
            buf = [txt]
            continue
        # новый визуальный ряд, если вертикальный сдвиг заметный
        if abs(y0 - current_y) > 2.0:
            lines.append(buf)
            buf = [txt]
            current_y = y0
        else:
            buf.append(txt)
    if buf:
        lines.append(buf)
    # склеиваем слова пробелами, потом строки переводами
    out = "\n".join(" ".join(items) for items in lines)
    return out

def extract_halves_text(pdf_bytes: bytes) -> Tuple[str, str]:
    """Читает страницу как две независимые колонки по words -> строки."""
    fitz = _ensure_pymupdf()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    rect = page.rect
    mid_x = rect.width / 2.0
    # Иногда буквы подрезаются на самом краю — добавим по 2pt «воздуха»
    left = fitz.Rect(rect.x0 - 2, rect.y0, mid_x + 2, rect.y1)
    right = fitz.Rect(mid_x - 2, rect.y0, rect.x1 + 2, rect.y1)
    tl = words_to_text(page, left)
    tr = words_to_text(page, right)
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
    if bin_thresh:
        import numpy as np
        arr = np.array(img)
        arr = (arr > bin_thresh) * 255
        from PIL import Image as _Image
        img = _Image.fromarray(arr.astype("uint8"))
    txt = pytesseract.image_to_string(img, lang="rus+eng",
                                      config="--oem 1 --psm 6 --dpi 300 --tessedit_preserve_interword_spaces=1")
    return txt

# -------- ПАРСИНГ БЛОКОВ ----------------------------------
# заголовок таблицы бывает разный: «Фамилия Имя», «Фамилия, Имя», со звёздочкой и т.п.
HEADER_RE = re.compile(r"№\s+Поз\s+Фамилия,?\s*Имя(?:\s*\*)?\s+Д\.Р\.\s+Лет", re.I)

HEADER_RE = re.compile(
    r"№\s*Поз\s*Фамилия,?\s*Имя(?:\s*\*?)?\s*Д\.?\s*Р\.?\s*Лет",
    re.I
)

def split_blocks(left: str, right: str) -> Dict[str, str]:
    def grab(block_name: str, text: str) -> str:
        m = re.search(block_name, text, flags=re.I)
        return text[m.start():] if m else ""
    return {
        "left": left,
        "right": right,
        "goalies_l": grab(r"Вратари", left),
        "goalies_r": grab(r"Вратари", right),
        "lineups_l": left,
        "lineups_r": right,
    }

def parse_goalies(block: str, side: str) -> List[Dict[str, str]]:
    res = []
    m = re.search(r"Вратари(.+?)(?:Звено|Главный тренер|Линейный|№\s+Поз|$)", block, flags=re.S | re.I)
    if not m:
        return res
    lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
    for ln in lines[:6]:
        m2 = RE_LINE.search(ln)
        if not m2:
            continue
        name = m2.group("name").strip()
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
    blob = (left + "\n" + right).splitlines()
    blob = [clean_text(x).strip() for x in blob]
    refs: List[Dict[str, str]] = []

    # Собираем индексы заголовков
    idx_main = [i for i,s in enumerate(blob) if re.search(r"Главный\s+судья", s, re.I)]
    idx_line = [i for i,s in enumerate(blob) if re.search(r"Линейный\s+судья", s, re.I)]

    def take_two_after(idx_list, role):
        taken = 0
        for i in idx_list:
            # захватываем ближайшие 1-2 непустые строки после заголовка
            j = i + 1
            local = []
            while j < len(blob) and len(local) < 2:
                name = blob[j]
                if name and not re.search(r"(Главный|Линейный)\s+судья", name, re.I):
                    nm = normalize_tail_y(name)
                    nm = dict_fix(nm, _dict_refs)
                    if nm:
                        local.append(nm)
                j += 1
            for nm in local:
                refs.append({"role": role, "name": nm})
            taken += len(local)
        return taken

    take_two_after(idx_main, "Главный судья")
    take_two_after(idx_line, "Линейный судья")

    # де-дуп (иногда одинаковая строка встречается слева и справа)
    seen = set()
    uniq = []
    for r in refs:
        key = (r["role"], r["name"])
        if key not in seen:
            uniq.append(r); seen.add(key)
    return uniq


def parse_lineups_block(text: str, side: str) -> List[Dict[str, Any]]:
    sink: List[Dict[str, Any]] = []
    m = HEADER_RE.search(text)
    start = m.end() if m else 0
    for ln in text[start:].splitlines():
        m2 = RE_LINE.search(ln)
        if not m2:
            continue
        num = m2.group("num")
        pos = m2.group("pos")
        name = m2.group("name").strip()
        capt = (m2.group("capt") or "").strip()
        dob  = m2.group("dob")
        age  = m2.group("age")
        gk_flag, gk_status = "", None
        if pos == "В":
            if name.endswith(" С"): name, gk_flag, gk_status = name[:-2].strip(), "S", "starter"
            elif name.endswith(" Р"): name, gk_flag, gk_status = name[:-2].strip(), "R", "reserve"
        name = normalize_tail_y(name)
        name = dict_fix(name, _dict_players)
                # если метка капитана «A/K» прилипла в конец имени — вытащим
        if not capt:
            mcap = re.search(r"\s([AK])$", name)
            if mcap:
                capt = mcap.group(1)
                name = name[:mcap.start()].strip()

        sink.append({
            "side": side, "num": num, "pos": pos, "name": name,
            "capt": capt, "dob": dob, "age": age,
            "gk_flag": gk_flag, "gk_status": gk_status
        })
    return sink

def parse_lineups(left: str, right: str) -> Dict[str, List[Dict[str, Any]]]:
    out_home = parse_lineups_block(left,  "home")
    out_away = parse_lineups_block(right, "away")
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
    # Текст-слой
    try:
        L, R = extract_halves_text(pdf)
        L = clean_text(L); R = clean_text(R)
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

    # Текст-слой с устойчивой сборкой по колонкам
    L, R = extract_halves_text(pdf)
    L = clean_text(L); R = clean_text(R)

    data: Dict[str, Any] = {}
    # refs всегда можно парсить из колонок
    if target in ("refs", "all"):
        data["refs"] = parse_refs(L, R)

    # goalies / lineups
    if target in ("goalies", "all", "lineups"):
        blocks = split_blocks(L, R)

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
