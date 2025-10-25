# main.py
# KHL PDF OCR Server — v2.1.1 (text-layer first, coords-based lineups, safe name correction)
# Эндпойнты:
#   /                 — health (пути словарей)
#   /ocr              — берёт текст-слой; если его нет — OCR fallback
#   /extract          — refs + goalies + lineups{home,away} на основе координат PyMuPDF
#
# Зависимости (requirements.txt):
#   fastapi
#   uvicorn
#   httpx
#   pymupdf==1.24.4
#   rapidfuzz
#   pdf2image
#   pillow
#   pytesseract

import os, re, time, csv
from typing import List, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel
import fitz  # PyMuPDF

# OCR fallback
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps
import pytesseract

from rapidfuzz import process, fuzz

APP_VERSION = "2.1.1"
DEFAULT_SEASON = 1369

app = FastAPI(title="KHL PDF OCR Server", version=APP_VERSION)

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

# ==================== Dictionaries ====================

PLAYERS_DICT: List[str] = []
REFEREES_DICT: List[str] = []
DICT_SOURCES: Dict[str, str] = {"players": "", "referees": ""}
USE_DICTIONARIES = True  # можно временно выключить для тестов

def _find_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def _load_names(path: Optional[str]) -> List[str]:
    if not path:
        return []
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
    # нормализация регистра
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

def best_match(name: str, pool: List[str], th: int = 96) -> str:
    """
    Безопасная коррекция ФИО:
    - если словарь выключен/пустой → вернуть как есть;
    - правим только при очень высокой похожести (>=96) и
      совпадении первых букв фамилии и имени; длина почти совпадает.
    """
    if not USE_DICTIONARIES or not name or not pool:
        return name
    parts = name.split()
    if len(parts) < 2:
        return name
    cand, score, _ = process.extractOne(name, pool, scorer=fuzz.WRatio)
    if not cand or score < th:
        return name
    sp = cand.split()
    if len(sp) < 2:
        return name
    if parts[0][0].upper() == sp[0][0].upper() and parts[1][0].upper() == sp[1][0].upper():
        if abs(len(cand) - len(name)) <= 3:
            return cand
    return name

# ==================== HTTP fetch ====================

async def fetch_pdf_with_fallback(pdf_url: str, match_id: int, season: int):
    tried: List[str] = []
    params = {"pdf_url": (pdf_url or "").strip(), "match_id": match_id, "season": season}
    if "khl.ru/documents/" in params["pdf_url"] and "/pdf/" not in params["pdf_url"]:
        params["pdf_url"] = f"https://www.khl.ru/pdf/{season}/{match_id}/game-{match_id}-start-ru.pdf"
    last_err: Optional[str] = None
    timeout = httpx.Timeout(25.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=HEADERS) as client:
        try:
            await client.get("https://www.khl.ru/", headers=HEADERS)
            await client.get(f"https://www.khl.ru/game/{match_id}/", headers=HEADERS)
        except Exception as e:
            last_err = f"warmup:{e}"
        for tpl in PDF_TEMPLATES:
            url = tpl.format(**params).strip()
            if not url or url in tried: continue
            tried.append(url)
            try:
                r = await client.get(url, headers={**HEADERS, "Referer": f"https://www.khl.ru/game/{match_id}/"})
                if r.status_code == 200 and (r.headers.get("content-type","").startswith("application/pdf")):
                    return r.content, url, tried, None
                else:
                    last_err = f"status:{r.status_code} ct:{r.headers.get('content-type','')}"
            except Exception as e:
                last_err = f"get:{type(e).__name__}:{e}"
    return None, None, tried, last_err

# ==================== Text normalization ====================

LAT_TO_CYR = str.maketrans({
    "A":"А","B":"В","C":"С","E":"Е","H":"Н","K":"К","M":"М","O":"О","P":"Р","T":"Т","X":"Х","Y":"У",
    "a":"а","c":"с","e":"е","o":"о","p":"р","x":"х","y":"у",
})
FIO_RE = re.compile(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.)?\b")
DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
POS_RE = re.compile(r"^[ВЗН]$")

def norm(s: str) -> str:
    if not s: return s
    s = s.translate(LAT_TO_CYR)
    s = re.sub(r"([А-ЯЁ][а-яё]{2,})([А-ЯЁ][а-яё]{2,})", r"\1 \2", s)
    s = re.sub(r"[ \t]+"," ",s).strip()
    return s

# ==================== Fast text-layer extraction ====================

def extract_page_halves_text(pdf_bytes: bytes) -> Tuple[str, str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0: return "",""
    page = doc.load_page(0)
    rect = page.rect
    mid_x = rect.width/2
    left_rect, right_rect = fitz.Rect(0,0,mid_x,rect.height), fitz.Rect(mid_x,0,rect.width,rect.height)

    def blocks(r: fitz.Rect) -> str:
        bl = page.get_text("blocks", clip=r)
        parts = []
        for b in sorted(bl, key=lambda x: (x[1], x[0])):
            txt = b[4]
            if txt: parts.append(txt)
        return norm("\n".join(parts))

    return blocks(left_rect), blocks(right_rect)

def extract_words_by_half(pdf_bytes: bytes):
    """Список слов (x0,y0,x1,y1,text) по левой/правой половинам первой страницы."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0: return [],[]
    page = doc.load_page(0)
    rect = page.rect
    mid_x = rect.width/2
    L, R = [], []
    for x0,y0,x1,y1,word,block_no,line_no,word_no in page.get_text("words"):
        t = norm(word)
        if not t: continue
        if x1 <= mid_x: L.append((x0,y0,x1,y1,t))
        elif x0 >= mid_x: R.append((x0,y0,x1,y1,t))
        else:
            cx = (x0+x1)/2
            (L if cx<mid_x else R).append((x0,y0,x1,y1,t))
    L.sort(key=lambda w:(round(w[1],1), w[0]))
    R.sort(key=lambda w:(round(w[1],1), w[0]))
    return L, R

# ==================== Rows & Columns ====================

def group_rows(words, y_tol: float = 2.2):
    rows = []
    cur = []; cur_y = None
    for w in words:
        y = w[1]
        if cur_y is None:
            cur = [w]; cur_y = y
        elif abs(y - cur_y) <= y_tol:
            cur.append(w)
        else:
            rows.append(sorted(cur, key=lambda z:z[0]))
            cur = [w]; cur_y = y
    if cur: rows.append(sorted(cur, key=lambda z:z[0]))
    return rows

def detect_header_and_columns(rows) -> Optional[Dict[str,float]]:
    # Ищем строку с заголовком: № | Поз | Фамилия, Имя | Д.Р. | Лет
    for row in rows[:30]:
        texts = " ".join([w[4] for w in row])
        if ("Поз" in texts or "Поз." in texts) and ("Фамилия" in texts) and ("Лет" in texts):
            col = {}
            for w in row:
                t = w[4]
                if t in ("№","N","No"): col["num_x"] = w[0]
                if t.startswith("Поз"): col["pos_x"] = w[0]
                if "Фамилия" in t: col["name_x"] = w[0]
                if t.startswith("Д.") or t.startswith("Д.Р"): col["dob_x"] = w[0]
                if t == "Лет": col["age_x"] = w[0]
            if set(("num_x","pos_x","name_x","dob_x","age_x")).issubset(col.keys()):
                # правая граница колонки ФИО — начало следующей колонки (Д.Р.)
                col["name_r_x"] = col["dob_x"]
                return col
    return None


def row_to_player(row, cols, side: str) -> Optional[Dict[str,str]]:
    """
    Маппим слова по диапазонам X. Для колонки ФИО дополнительно вынимаем
    метки вратаря: 'С' (стартер) / 'Р' (резерв) – сохраняем в полях gk_flag/gk_status,
    но не включаем в name.
    """
    δ = 2.5
    name_l = cols["name_x"] - δ
    name_r = cols.get("name_r_x", cols["dob_x"]) - δ

    buckets = {"num": [], "pos": [], "name": [], "dob": [], "age": []}

    for x0,y0,x1,y1,t in row:
        # приоритет по содержимому
        if DATE_RE.match(t):
            buckets["dob"].append(t); 
            continue
        if re.fullmatch(r"\d{1,2}", t):
            if abs(x0 - cols["age_x"]) < abs(x0 - cols["name_x"]):
                buckets["age"].append(t); 
                continue
        if POS_RE.match(t) and x0 < name_l:
            if abs(x0 - cols["pos_x"]) <= abs(x0 - cols["num_x"]):
                buckets["pos"].append(t); 
                continue
        if re.fullmatch(r"\d{1,2}", t) and x0 < cols["pos_x"]:
            buckets["num"].append(t); 
            continue
        if name_l <= x0 < name_r:
            buckets["name"].append(t); 
            continue
        if cols["pos_x"] <= x0 < name_r:
            buckets["name"].append(t); 
            continue

    num = next((re.sub(r"\D","",w) for w in buckets["num"] if re.search(r"\d", w)), "")
    pos = next((w for w in buckets["pos"] if POS_RE.match(w)), "")

    # --- метки капитана и S/R для вратарей ---
    capt = ""
    gk_flag = ""  # "S"|"R" для вратарей
    for t in buckets["name"]:
        if t in ("А","A","К","K"):
            capt = "A" if t in ("А","A") else "K"
        # метка старт/резерв – допускаем русский/латинский символ
        if t in ("С","C","S"):
            gk_flag = "S"
        elif t in ("Р","P","R"):
            gk_flag = "R"

    # имя: исключаем служебные одиночные буквы (A/K/S/R и т.п.)
    IGNORE = {"А","A","К","K","С","C","S","Р","P","R","*","·",","}
    name_tokens = [t for t in buckets["name"] if t not in IGNORE and not re.fullmatch(r"[A-Za-zА-Яа-яЁё]\.?$", t)]
    raw_name = " ".join(name_tokens).strip(" ,*")
    raw_name = raw_name.replace(" ,", ",").replace(", ", " ")
    m = FIO_RE.search(raw_name)
    name = m.group(0) if m else raw_name
    # (опционально) убрать ударения:
    # name = strip_accents(name)
    name = best_match(name, PLAYERS_DICT)  # безопасная коррекция

    dob = next((w for w in buckets["dob"] if DATE_RE.match(w)), "")
    age = next((re.sub(r"\D","",w) for w in buckets["age"] if re.search(r"\d", w)), "")

    if not (num or pos or name):
        return None

    gk_status = None
    if pos == "В":
        gk_status = "starter" if gk_flag == "S" else ("reserve" if gk_flag == "R" else None)

    return {
        "side": side,
        "num": num,
        "pos": pos,
        "name": name,
        "capt": capt,
        "dob": dob,
        "age": age,
        "gk_flag": gk_flag if pos == "В" else "",
        "gk_status": gk_status,
    }


# ==================== Refs ====================

def parse_refs_from_text(lt: str, rt: str) -> List[Dict[str,str]]:
    lines = [l.strip() for l in (lt+"\n"+rt).split("\n") if l.strip()]
    roles = ["Главный судья","Линейный судья","Резервный главный судья","Резервный судья","Резервный линейный судья"]
    out = []
    for i, ln in enumerate(lines):
        for role in roles:
            if re.search(rf"^{role}\b", ln, flags=re.I):
                for cand in [ln] + lines[i+1:i+3]:
                    m = FIO_RE.search(cand)
                    if m:
                        out.append({"role": role, "name": best_match(m.group(0), REFEREES_DICT)})
                        break
    # дедуп
    seen, res = set(), []
    for j in out:
        k=(j["role"], j["name"])
        if k in seen: continue
        seen.add(k); res.append(j)
    return res

# ==================== OCR fallback ====================

def preprocess(img, scale: float = 1.10, bin_thresh: int = 185):
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    img = ImageOps.autocontrast(img).convert("L")
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
    w, h = p.size; mid = w//2
    left = p.crop((0,0,mid,h)); right = p.crop((mid,0,w,h))
    return norm(ocr_one(left)), norm(ocr_one(right))

# ==================== Models ====================

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

# ==================== Endpoints ====================

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

    # быстрый вариант: текст-слой
    lt, rt = extract_page_halves_text(pdf_bytes)
    if (lt.strip() or rt.strip()):
        full = (lt+"\n"+rt).strip()
        return OCRResponse(
            ok=True, match_id=match_id, season=season, source_pdf=final_url, pdf_len=len(pdf_bytes),
            dpi=dpi, pages_ocr=1,
            dur_total_s=round(time.time()-t0,3),
            dur_download_s=round(t_pdf-t0,3),
            dur_preproc_s=0.0, dur_ocr_s=0.0,
            text_len=len(full), snippet=re.sub(r"\s+"," ", full)[:480]
        )

    # fallback: OCR
    p0 = time.time()
    ltxt, rtxt = ocr_halves_first_page(pdf_bytes, dpi=dpi, scale=scale, bin_thresh=bin_thresh)
    full = (ltxt+"\n"+rtxt).strip()
    return OCRResponse(
        ok=True, match_id=match_id, season=season, source_pdf=final_url, pdf_len=len(pdf_bytes),
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

    # текст-слой
    lt, rt = extract_page_halves_text(pdf_bytes)
    L_words, R_words = extract_words_by_half(pdf_bytes)

    data: Dict[str,object] = {}
    keys = ["refs","goalies","lineups"] if target == "all" else [target]

    # lineups сразу считаем один раз (нужно и для goalies)
    home_players = away_players = None
    if "lineups" in keys or "goalies" in keys or target == "all":
        home_players = parse_lineups_struct(L_words, "home")
        away_players = parse_lineups_struct(R_words, "away")

    for k in keys:
        if k == "refs":
            data["refs"] = parse_refs_from_text(lt, rt)
        elif k == "lineups":
            data["lineups"] = {"home": home_players or [], "away": away_players or []}
        elif k == "goalies":
            data["goalies"] = {
                "home": [{"name": p["name"]} for p in (home_players or []) if p["pos"] == "В"][:3],
                "away": [{"name": p["name"]} for p in (away_players or []) if p["pos"] == "В"][:3],
            }

    return {
        "ok": True,
        "match_id": match_id, "season": season,
        "source_pdf": final_url, "pdf_len": len(pdf_bytes),
        "dpi": dpi, "pages_ocr": 1,
        "dur_total_s": round(time.time()-t0,3),
        "data": data,
    }
