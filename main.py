import os
import io
import re
import json
import time
import unicodedata
from statistics import median
from typing import List, Tuple, Dict, Any, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import httpx

# ---- optional OCR fallback ----
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# ---- optional fuzzy dictionaries (если нет rapidfuzz - код все равно работает) ----
try:
    from rapidfuzz import process, fuzz
    HAS_FUZZ = True
except Exception:
    HAS_FUZZ = False


app = FastAPI(title="khl-pdf-ocr", version="3.0.0")

# =========================
# Utilities / Normalization
# =========================

def _norm_txt(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("й", "й").replace("Й", "Й").replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", s).strip()

ROLE_ALIASES = {
    "Главный судья": "Главный судья",
    "Главные судьи": "Главный судья",
    "Линейный судья": "Линейный судья",
    "Линейные судьи": "Линейный судья",
}

# =========================
# Download PDF (with retry)
# =========================

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

async def fetch_pdf(pdf_url: str) -> bytes:
    tried = []
    headers = {
        "user-agent": UA,
        "accept": "application/pdf,*/*",
        "accept-language": "ru,en;q=0.9",
        "referer": "https://www.khl.ru/",
        "cache-control": "no-cache",
        "pragma": "no-cache",
    }
    urls = [pdf_url]
    for u in urls:
        # 1) http2 client
        try:
            async with httpx.AsyncClient(http2=True, timeout=20.0, headers=headers) as cl:
                r = await cl.get(u)
                tried.append(f"{u} [{r.status_code}]")
                if r.status_code == 200 and r.content:
                    return r.content
                # маленькая пауза и повтор plain http/1.1
        except Exception as e:
            tried.append(f"httpx err: {repr(e)}")

        # 2) http/1.1 client
        try:
            async with httpx.AsyncClient(http2=False, timeout=20.0, headers=headers) as cl:
                r = await cl.get(u)
                tried.append(f"{u} [{r.status_code}]")
                if r.status_code == 200 and r.content:
                    return r.content
        except Exception as e:
            tried.append(f"httpx err: {repr(e)}")

    raise RuntimeError(json.dumps({"step": "GET", "status": 403, "tried": tried}, ensure_ascii=False))

# =========================
# Text-layer & words helpers
# =========================

def get_page_text(doc: fitz.Document, page_index: int = 0) -> str:
    try:
        return doc.load_page(page_index).get_text()
    except Exception:
        return ""

def get_page_words(doc: fitz.Document, page_index: int = 0) -> List[Tuple]:
    # (x0,y0,x1,y1,"word",block_no,line_no,word_no)
    try:
        return doc.load_page(page_index).get_text("words")
    except Exception:
        return []

def find_roster_page(doc: fitz.Document) -> int:
    for i in range(len(doc)):
        t = _norm_txt(get_page_text(doc, i))
        if re.search(r"Составы|Составы команд", t, re.I):
            return i
    return 0

def split_two_columns(words: List[Tuple]) -> Tuple[List[Tuple], List[Tuple]]:
    xs = sorted(((w[0] + w[2]) / 2.0) for w in words)
    if not xs:
        return [], []
    mid = median(xs)
    left = [w for w in words if ((w[0] + w[2]) / 2.0) <= mid]
    right = [w for w in words if ((w[0] + w[2]) / 2.0) > mid]
    if len(left) < 10 or len(right) < 10:
        page_w = max((w[2] for w in words), default=0.0)
        cut = page_w * 0.5
        left = [w for w in words if ((w[0] + w[2]) / 2.0) <= cut]
        right = [w for w in words if ((w[0] + w[2]) / 2.0) > cut]
    left.sort(key=lambda w: (w[1], w[0]))
    right.sort(key=lambda w: (w[1], w[0]))
    return left, right

def words_to_lines(words: List[Tuple], y_tol: float = 3.0) -> List[str]:
    lines: List[str] = []
    cur: List[Tuple] = []
    last_y: Optional[float] = None
    for w in words:
        y = w[1]
        if last_y is None or abs(y - last_y) <= y_tol:
            cur.append(w)
        else:
            cur.sort(key=lambda z: z[0])
            line = " ".join(_norm_txt(z[4]) for z in cur).strip()
            if line:
                lines.append(line)
            cur = [w]
        last_y = y
    if cur:
        cur.sort(key=lambda z: z[0])
        line = " ".join(_norm_txt(z[4]) for z in cur).strip()
        if line:
            lines.append(line)
    return lines

# =========================
# Parsing: lineups / goalies
# =========================

ROSTER_ROW_RE = re.compile(
    r"^(?P<num>\d{1,3})\s+(?P<pos>[ВЗН])\s+(?P<name>[A-ЯЁA-Za-z\-’'\.]+(?:\s+[A-ЯЁA-Za-z\-’'\.]+){0,3})"
    r"(?:\s+(?P<capt>[АК]))?\s+(?P<dob>\d{2}\.\d{2}\.\d{4})\s+(?P<age>\d{1,2})(?:\s+(?P<gkflag>[SR]))?$"
)

def parse_roster_lines(lines: List[str], side: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ln in lines:
        ln_n = _norm_txt(ln)
        m = ROSTER_ROW_RE.match(ln_n)
        if not m:
            continue
        d = m.groupdict()
        gkflag = (d.get("gkflag") or "").strip()
        out.append({
            "side": side,
            "num": d["num"],
            "pos": d["pos"],
            "name": (d["name"] or "").strip().strip("*").strip(),
            "capt": (d.get("capt") or "").strip(),
            "dob": d["dob"],
            "age": d["age"],
            "gk_flag": gkflag,
            "gk_status": ("starter" if gkflag == "S" else ("reserve" if gkflag == "R" else None)),
        })
    return out

def extract_goalies(roster: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for r in roster:
        if r.get("pos") == "В":
            status = "starter" if r.get("gk_flag") == "S" else ("reserve" if r.get("gk_flag") == "R" else "scratch")
            out.append({"name": r.get("name", ""), "status": status})
    # дедупликация (иногда дубли при кривых строках)
    seen = set()
    uniq = []
    for g in out:
        k = (g["name"], g["status"])
        if k not in seen:
            uniq.append(g)
            seen.add(k)
    return uniq

# =========================
# Parsing: referees (from text)
# =========================

REFS_BLOCK_RE = re.compile(r"(Судьи|Судейская бригада)(.*?)(?:Обновлено|Главный тренер|$)", re.S | re.I)
REF_LINE_RE = re.compile(
    r"(?P<role>Главн(?:ые|ый)\s+судья|Линейн(?:ые|ый)\s+судья|Линейный судья|Главный судья)\s*[:\-–]?\s*"
    r"(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\.]+){1,2})"
)

def extract_refs_from_text(full_text: str) -> List[Dict[str, str]]:
    full_text = _norm_txt(full_text)
    m = REFS_BLOCK_RE.search(full_text)
    if not m:
        return []
    blk = m.group(0)
    refs = []
    for mm in REF_LINE_RE.finditer(blk):
        role_raw = _norm_txt(mm.group("role"))
        role = ROLE_ALIASES.get(role_raw.replace("Линейн", "Линейн"), "Линейный судья")
        name = _norm_txt(mm.group("name"))
        if name and name not in ("Главный судья", "Линейный судья"):
            refs.append({"role": role, "name": name})
    uniq, seen = [], set()
    for r in refs:
        k = (r["role"], r["name"])
        if k not in seen:
            uniq.append(r)
            seen.add(k)
    return uniq

# =========================
# Optional: CSV dictionaries for name fixes
# =========================

PLAYERS_DICT: List[str] = []
REFEREES_DICT: List[str] = []

def _load_csv_list(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = _norm_txt(line.strip().strip(","))
            if name:
                rows.append(name)
    return rows

def fuzzy_fix(name: str, dict_list: List[str]) -> str:
    if not name or not dict_list:
        return name
    if HAS_FUZZ:
        best = process.extractOne(
            name,
            dict_list,
            scorer=fuzz.WRatio,
            score_cutoff=90
        )
        if best:
            return best[0]
        return name
    # без rapidfuzz: простая эвристика по точной подстроке
    for canon in dict_list:
        if name in canon or canon in name:
            return canon
    return name

def post_fix_names_lineups(data: Dict[str, Any]) -> None:
    # Применяем словари к lineups и refs
    for side in ("home", "away"):
        for r in data.get("lineups", {}).get(side, []):
            r["name"] = fuzzy_fix(r.get("name", ""), PLAYERS_DICT)
    for r in data.get("refs", []):
        r["name"] = fuzzy_fix(r.get("name", ""), REFEREES_DICT)

# =========================
# OCR fallback with Tesseract
# =========================

def run_image_ocr(pdf_bytes: bytes, dpi: int = 200, scale: float = 1.25, bin_thresh: Optional[int] = 185) -> str:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png", thread_count=2)
    if not pages:
        return ""
    # только первая страница — именно там таблица стартов
    img: Image.Image = pages[0]
    if scale and abs(scale - 1.0) > 1e-3:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if bin_thresh is not None:
        gray = img.convert("L")
        bw = gray.point(lambda p: 255 if p > bin_thresh else 0)
    else:
        bw = img
    text = pytesseract.image_to_string(
        bw,
        lang="rus+eng",
        config="--oem 1 --psm 6 -c preserve_interword_spaces=1"
    )
    return text or ""

# =========================
# Startup: load dictionaries
# =========================

@app.on_event("startup")
def startup_event():
    global PLAYERS_DICT, REFEREES_DICT
    PLAYERS_DICT = _load_csv_list("players.csv")
    REFEREES_DICT = _load_csv_list("referees.csv")

# =========================
# Routes
# =========================

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "khl-pdf-ocr",
        "version": app.version,
        "ready": True,
        "dicts": {
            "players_loaded": len(PLAYERS_DICT),
            "refs_loaded": len(REFEREES_DICT),
        },
    }

@app.get("/ocr")
async def ocr_endpoint(
    match_id: int = Query(...),
    season: int = Query(...),
    pdf_url: str = Query(...),
    dpi: int = 130,
    scale: float = 1.0,
    bin_thresh: Optional[int] = None,
    fallback: int = 1,  # 1 = авто-fallback через tesseract, 0 = только текст-слой
):
    t0 = time.time()
    tried = []
    try:
        pdf_bytes = await fetch_pdf(pdf_url)
        tried.append(f"{pdf_url} [200]")
    except RuntimeError as e:
        err = json.loads(str(e))
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, **err}, status_code=200)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    p_idx = find_roster_page(doc)
    text = get_page_text(doc, p_idx) or ""
    used_fallback = False

    if not text.strip() and fallback:
        # OCR fallback
        text = run_image_ocr(pdf_bytes, dpi=max(dpi, 200), scale=max(scale, 1.25), bin_thresh=bin_thresh)
        used_fallback = True

    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": pdf_url,
        "pdf_len": len(pdf_bytes),
        "dpi": dpi,
        "pages_ocr": 1,
        "used_fallback_ocr": used_fallback,
        "text_len": len(text),
        "snippet": (text or "")[:400],
        "tried": tried,
        "dur_total_s": round(time.time() - t0, 3),
    }

@app.get("/extract")
async def extract_endpoint(
    match_id: int = Query(...),
    season: int = Query(...),
    pdf_url: str = Query(...),
    target: str = Query("all"),  # refs|goalies|lineups|all
    dpi: int = 130,
    fallback: int = 1,
):
    t0 = time.time()
    tried = []
    try:
        pdf_bytes = await fetch_pdf(pdf_url)
        tried.append(f"{pdf_url} [200]")
    except RuntimeError as e:
        err = json.loads(str(e))
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, **err}, status_code=200)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    p_idx = find_roster_page(doc)

    # пробуем текст-слой
    text = get_page_text(doc, p_idx) or ""
    words = get_page_words(doc, p_idx)

    used_fallback = False
    if (not text.strip() or len(words) == 0) and fallback:
        # делаем OCR и пытаемся построить строки без координат
        text = run_image_ocr(pdf_bytes, dpi=max(dpi, 200), scale=1.4, bin_thresh=185)
        used_fallback = True
        # хоть что-то парсим по простым строкам
        ocr_lines = [ln for ln in (_norm_txt(x) for x in text.splitlines()) if ln]
        # на OCR-ветке составы распарсить структурно сложно — вернем пустые lists, но refs попробуем
        refs = extract_refs_from_text(text)
        data = {"refs": refs, "goalies": {"home": [], "away": []}, "lineups": {"home": [], "away": []}}
        post_fix_names_lineups(data)
        res = {
            "ok": True,
            "match_id": match_id,
            "season": season,
            "source_pdf": pdf_url,
            "pdf_len": len(pdf_bytes),
            "dpi": dpi,
            "pages_ocr": 1,
            "used_fallback_ocr": used_fallback,
            "data": data,
            "tried": tried,
            "dur_total_s": round(time.time() - t0, 3),
        }
        if target != "all":
            # отфильтруем под запрос
            if target == "refs":
                res["data"] = {"refs": data["refs"]}
            elif target == "goalies":
                res["data"] = {"goalies": data["goalies"]}
            elif target == "lineups":
                res["data"] = {"lineups": data["lineups"]}
        return res

    # текст-слой + координаты: надежный парсинг
    left_w, right_w = split_two_columns(words)
    left_lines = words_to_lines(left_w)
    right_lines = words_to_lines(right_w)

    # вырезаем реальные "табличные" строки по нашему шаблону
    home_rows = parse_roster_lines(left_lines, "home")
    away_rows = parse_roster_lines(right_lines, "away")

    # судьи из текст-слоя (глобально по странице)
    refs = extract_refs_from_text(text)

    # вратари
    goalies_home = extract_goalies(home_rows)
    goalies_away = extract_goalies(away_rows)

    data: Dict[str, Any] = {
        "refs": refs,
        "goalies": {"home": goalies_home, "away": goalies_away},
        "lineups": {"home": home_rows, "away": away_rows},
    }
    post_fix_names_lineups(data)

    # таргет-фильтрация
    if target != "all":
        if target == "refs":
            data = {"refs": refs}
        elif target == "goalies":
            data = {"goalies": {"home": goalies_home, "away": goalies_away}}
        elif target == "lineups":
            data = {"lineups": {"home": home_rows, "away": away_rows}}

    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": pdf_url,
        "pdf_len": len(pdf_bytes),
        "dpi": dpi,
        "pages_ocr": 1,
        "used_fallback_ocr": used_fallback,
        "data": data,
        "tried": tried,
        "dur_total_s": round(time.time() - t0, 3),
    }
