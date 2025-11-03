# app.py
import os, re, time, json, unicodedata
from typing import List, Dict, Tuple
from flask import Flask, request, jsonify, Response
import requests
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
import pytesseract

# -------------------------------------------------
# Flask / base
# -------------------------------------------------
app = Flask(__name__)
app.json.ensure_ascii = False

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "application/pdf,*/*;q=0.9",
    "Referer": "https://www.khl.ru/",
})

PDF_PROXY_BASE = os.getenv("PDF_PROXY_BASE", "").rstrip("/")
TESS_LANG = "rus+eng"

# -------------------------------------------------
# Utils: PDF I/O
# -------------------------------------------------
def make_pdf_url(season: str, uid: str) -> str:
    path = f"{season}/{uid}/game-{uid}-start-ru.pdf"
    if PDF_PROXY_BASE:
        return f"{PDF_PROXY_BASE}/{path}"
    return f"https://www.khl.ru/pdf/{path}"

def http_get(url: str, timeout=30) -> bytes:
    r = SESSION.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content

def pdf_open(pdf_bytes: bytes) -> fitz.Document:
    return fitz.open(stream=pdf_bytes, filetype="pdf")

def pdf_page_image(doc: fitz.Document, pno=0, dpi=300) -> Image.Image:
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = doc.load_page(pno).get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

# -------------------------------------------------
# Text layer (words) helpers
# -------------------------------------------------
def page_words(doc: fitz.Document, pno=0) -> List[Tuple[float, float, str]]:
    page = doc.load_page(pno)
    words = page.get_text("words")  # x0,y0,x1,y1,text,block,line,word
    words_sorted = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
    return [(w[1], w[0], w[4]) for w in words_sorted]

def lines_from_words(words: List[Tuple[float, float, str]], tol=3.0) -> List[str]:
    rows: Dict[float, List[Tuple[float,str]]] = {}
    for y, x, t in words:
        key = None
        for ky in rows.keys():
            if abs(ky - y) <= tol:
                key = ky
                break
        if key is None:
            key = y
            rows[key] = []
        rows[key].append((x, t))
    out = []
    for ky in sorted(rows.keys()):
        items = sorted(rows[ky], key=lambda r: r[0])
        line = " ".join(t for _, t in items)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out

# -------------------------------------------------
# Common cleaners
# -------------------------------------------------
MONTHS = {
    'января':'01','февраля':'02','марта':'03','апреля':'04','мая':'05','июня':'06',
    'июля':'07','августа':'08','сентября':'09','октября':'10','ноября':'11','декабря':'12'
}

def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \u200e\u200f,.;:|")

def clean_role_words(name: str) -> str:
    if not name: return name
    name = re.sub(r"\b(Главн\w*|Линейн\w*|судья|судьи|Резервн\w*|Обновлено.*)\b", " ", name, flags=re.I)
    return _norm(name)

# -------------------------------------------------
# META (date/time/teams) from words-text
# -------------------------------------------------
TEAM_LINE_RE = re.compile(r'^\s*([А-ЯЁA-Z0-9«»"–\-\s\.]+?)\s{2,}([А-ЯЁA-Z0-9«»"–\-\s\.]+?)\s*$', re.MULTILINE)
DATE_RE = re.compile(r'(\d{1,2})\.(\d{2})\.(\d{4})|(\d{1,2})\s+([А-Яа-яЁё]+)\s+(\d{4})')
TIME_RE = re.compile(r'(\d{1,2}:\d{2})\s*MSK', re.IGNORECASE)

def parse_match_meta_from_words(full_text: str) -> Dict:
    t = unicodedata.normalize('NFKC', full_text)

    # date
    date_str = ""
    m = DATE_RE.search(t)
    if m:
        if m.group(1):
            dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
            date_str = f"{dd}.{mm}.{yyyy}"
        else:
            dd, mon, yyyy = m.group(4), m.group(5).lower(), m.group(6)
            mm = MONTHS.get(mon)
            date_str = f"{dd}.{mm}.{yyyy}" if mm else f"{dd} {mon} {yyyy}"

    # time
    time_msk = ""
    mt = TIME_RE.search(t)
    if mt:
        time_msk = mt.group(1)

    # teams near "Составы команд"
    head_idx = t.find("Составы команд")
    head_block = t if head_idx == -1 else "\n".join(t[head_idx:].splitlines()[:25])

    home = away = ""
    mm2 = TEAM_LINE_RE.search(head_block)
    if mm2:
        home, away = _norm(mm2.group(1)), _norm(mm2.group(2))
    else:
        # try two consequent lines
        lines = [ln.strip() for ln in head_block.splitlines() if ln.strip()]
        cand = [ln for ln in lines if not re.match(r'^(Составы|Начало матча|Матч|№)', ln, flags=re.I)]
        for i in range(len(cand)-1):
            a,b = cand[i], cand[i+1]
            if re.search(r'[А-ЯЁ]{3}', a) and re.search(r'[А-ЯЁ]{3}', b):
                home, away = _norm(a), _norm(b)
                break

    if home and not away and "  " in home:
        parts = re.split(r"\s{2,}", home)
        if len(parts) >= 2:
            home, away = _norm(parts[0]), _norm(parts[1])

    if home and not away:
        tokens = home.split()
        for cut in range(len(tokens)-1, 2, -1):
            left, right = " ".join(tokens[:cut]), " ".join(tokens[cut:])
            if re.search(r'[А-ЯЁ]{3}', left) and re.search(r'[А-ЯЁ]{3}', right):
                home, away = _norm(left), _norm(right)
                break

    return {"date": date_str, "time_msk": time_msk, "teams": {"home": home, "away": away}}

def words_plain_text(doc: fitz.Document) -> str:
    ws = page_words(doc, 0)
    lines = lines_from_words(ws)
    return "\n".join(lines)

# -------------------------------------------------
# OCR helpers
# -------------------------------------------------
def ocr_image_lines(img: Image.Image) -> List[str]:
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.4)
    gray = gray.filter(ImageFilter.SHARPEN)
    txt = pytesseract.image_to_string(gray, lang=TESS_LANG, config="--psm 6")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in txt.splitlines()]
    return [ln for ln in lines if ln]

def ocr_full_text(img: Image.Image) -> str:
    # --psm 6 хорошо для табличных блоков
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.35)
    gray = gray.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(gray, lang=TESS_LANG, config="--psm 6")

# -------------------------------------------------
# Refs (OCR-first with cleanup)
# -------------------------------------------------
def extract_refs_from_ocr_lines(lines: List[str]) -> Dict:
    # Найти заголовок и строку с ФИО
    hdr = -1
    for i, ln in enumerate(lines[:120]):
        if ("Главный судья" in ln) and ("Линейный судья" in ln):
            hdr = i
            break

    main, linesmen = [], []
    raw_line = ""
    if hdr != -1:
        # часто сразу следующая строка — все четыре ФИ
        for k in range(1, 4):
            j = hdr + k
            if j < len(lines):
                candidate = clean_role_words(lines[j])
                if re.search(r"[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+", candidate):
                    raw_line = candidate
                    break

    if raw_line:
        parts = [p for p in re.split(r"[,\|;]|\s+", raw_line) if p]
        # собрать биграммы Имя Фамилия
        cand = []
        for j in range(len(parts)-1):
            a,b = parts[j], parts[j+1]
            if all(re.match(r"^[А-ЯЁ][а-яё\-]+$", x) for x in (a,b)):
                cand.append(f"{a} {b}")
        # выбираем 4 штуки, первые 2 — главные, следующие 2 — линейные
        if len(cand) >= 4:
            main = cand[:2]
            linesmen = cand[2:4]

    return {"main": main, "linesmen": linesmen, "_raw": raw_line}

# -------------------------------------------------
# Goalies (OCR)
# -------------------------------------------------
GK_BLOCK_RE = re.compile(r'\bВратари\b', re.IGNORECASE)
GK_LINE_RE  = re.compile(
    r'(?P<num>\d{1,2})\s+В\b[^\S\r\n]*'
    r'(?P<name>[А-ЯЁ][а-яё\-]+(?:\s+[А-ЯЁ][а-яё\-]+){0,2})'
)
GK_FLAG_RE  = re.compile(r'\b([СР])\b')

def extract_goalies_from_ocr_text(txt: str) -> Dict:
    t = unicodedata.normalize('NFKC', txt)
    # Нарезаем на блоки по "Вратари"
    idx = [m.start() for m in GK_BLOCK_RE.finditer(t)]
    home, away = [], []

    def parse_block(part: str) -> List[Dict]:
        out = []
        for m in GK_LINE_RE.finditer(part):
            num  = int(m.group('num'))
            name = _norm(m.group('name'))
            # ближайшие 40 символов на предмет С/Р
            tail = part[m.end(): m.end()+50]
            flm = GK_FLAG_RE.search(tail)
            status = ''
            if flm:
                status = 'S' if flm.group(1) == 'С' else ('R' if flm.group(1) == 'Р' else '')
            out.append({"number": num, "name": name, "status": status})
        # дедуп
        uniq = {}
        for g in out:
            key = (g["number"], g["name"])
            if key not in uniq:
                uniq[key] = g
        return list(uniq.values())

    if idx:
        parts = []
        for i, st in enumerate(idx):
            en = idx[i+1] if i+1 < len(idx) else len(t)
            parts.append(t[st:en])
        parsed = [parse_block(p) for p in parts]
        if len(parsed) == 1:
            home = parsed[0]
        else:
            home, away = parsed[0], parsed[1]
    else:
        # один массив, без деления — положим всё хозяевам
        home = parse_block(t)

    return {"home": home, "away": away}

# -------------------------------------------------
# EXTRACT MODES
# -------------------------------------------------
def extract_words_meta(doc: fitz.Document) -> Dict:
    plain = words_plain_text(doc)
    meta = parse_match_meta_from_words(plain)
    return {"ok": True, "engine": "words", "match": meta}

def extract_refs(doc: fitz.Document, ocr_text_top: str = None, debug=False) -> Dict:
    if ocr_text_top is None:
        img = pdf_page_image(doc, 0, dpi=300)
        h = img.height
        crop = img.crop((0, 0, img.width, int(h*0.35)))
        lines = ocr_image_lines(crop)
    else:
        lines = [ln for ln in ocr_text_top.splitlines() if ln.strip()]

    refs = extract_refs_from_ocr_lines(lines)
    res = {"ok": True, "engine": "ocr-refs", "referees": {"main": refs["main"], "linesmen": refs["linesmen"]}}
    if debug:
        res["_debug"] = {"raw": refs.get("_raw","")}
    return res

def extract_goalies(doc: fitz.Document, ocr_full: str = None, debug=False) -> Dict:
    if ocr_full is None:
        img = pdf_page_image(doc, 0, dpi=300)
        full = ocr_full_text(img)
    else:
        full = ocr_full
    g = extract_goalies_from_ocr_text(full)
    res = {"ok": True, "engine": "gk", "goalies": g}
    if debug:
        res["_debug"] = {"home_n": len(g["home"]), "away_n": len(g["away"])}
    return res

def extract_all(doc: fitz.Document, season: str, uid: str, debug=False) -> Dict:
    t0 = time.time()
    # 1) words → meta
    meta = extract_words_meta(doc).get("match", {"teams":{"home":"","away":""}, "date":"", "time_msk":""})

    # 2) OCR один раз
    img = pdf_page_image(doc, 0, dpi=300)
    h = img.height
    crop_top = img.crop((0, 0, img.width, int(h*0.35)))
    ocr_top = "\n".join(ocr_image_lines(crop_top))
    ocr_full = ocr_full_text(img)

    # 3) refs / goalies
    refs = extract_refs(doc, ocr_text_top=ocr_top, debug=debug)
    gk   = extract_goalies(doc, ocr_full=ocr_full, debug=debug)

    out = {
        "ok": True,
        "engine": "all",
        "match": {"season": season, "uid": uid, **meta},
        "referees": refs.get("referees", {"main": [], "linesmen": []}),
        "goalies": gk.get("goalies", {"home": [], "away": []}),
        "duration_s": round(time.time() - t0, 3),
    }
    if debug:
        out["_debug"] = {"ocr_top_len": len(ocr_top), "ocr_full_len": len(ocr_full)}
    return out

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "engine": "ready"})

@app.get("/extract")
def extract_route():
    """
    /extract?season=1369&uid=897689&mode=all|refs|goalies|words[&debug=1]
    """
    season = (request.args.get("season") or "").strip()
    uid    = (request.args.get("uid") or "").strip()
    mode   = (request.args.get("mode") or "all").strip().lower()
    debug  = (request.args.get("debug") in ("1","true","yes"))

    if not (season and uid):
        return jsonify({"ok": False, "error": "season or uid missing"}), 400

    url = make_pdf_url(season, uid)
    try:
        pdf_bytes = http_get(url, timeout=30)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", 0)
        return jsonify({"ok": False, "error": f"http {code}", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": "download_error", "detail": str(e)}), 502

    try:
        doc = pdf_open(pdf_bytes)
    except Exception as e:
        return jsonify({"ok": False, "error": "pdf_open_error", "detail": str(e)}), 500

    t0 = time.time()
    try:
        if mode == "words":
            res = extract_words_meta(doc)
        elif mode in ("refs","referees"):
            res = extract_refs(doc, ocr_text_top=None, debug=debug)
        elif mode in ("gk","goalies"):
            res = extract_goalies(doc, ocr_full=None, debug=debug)
        else:
            res = extract_all(doc, season, uid, debug=debug)

        res["source_url"] = url
        if "duration_s" not in res:
            res["duration_s"] = round(time.time() - t0, 3)
        return Response(json.dumps(res, ensure_ascii=False), mimetype="application/json")
    except Exception as e:
        return jsonify({"ok": False, "error": "extract_error", "detail": str(e)}), 500

# -------------------------------------------------
# Local run
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
