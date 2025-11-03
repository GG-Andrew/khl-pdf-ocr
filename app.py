import os, re, json, time, logging
from typing import List, Tuple, Dict, Optional
from flask import Flask, request, jsonify, Response
import requests
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
import pytesseract

# ------------------------
# Flask
# ------------------------
app = Flask(__name__)
app.json.ensure_ascii = False
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("khl-pdf-ocr")

# ------------------------
# HTTP session
# ------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.9",
    "Referer": "https://www.khl.ru/",
})

PDF_PROXY_BASE = os.getenv("PDF_PROXY_BASE", "").rstrip("/")
TESS_LANG = "rus+eng"

# ------------------------
# Helpers
# ------------------------
def make_pdf_url(season: str, uid: str) -> str:
    path = f"{season}/{uid}/game-{uid}-start-ru.pdf"
    return f"{PDF_PROXY_BASE}/{path}" if PDF_PROXY_BASE else f"https://www.khl.ru/pdf/{path}"

def http_get(url: str, timeout=25) -> bytes:
    r = SESSION.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content

def pdf_to_pix(doc: fitz.Document, pno=0, dpi=300) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc.load_page(pno).get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_image_lines(img: Image.Image) -> List[str]:
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.4)
    gray = gray.filter(ImageFilter.SHARPEN)
    txt = pytesseract.image_to_string(gray, lang=TESS_LANG, config="--psm 6")
    return [re.sub(r"\s+", " ", ln).strip() for ln in txt.splitlines() if ln.strip()]

# ---- text extraction ----
def words_with_boxes(doc: fitz.Document, pno=0) -> List[Tuple[float,float,float,float,str]]:
    # x0,y0,x1,y1,text,block,line,word
    return doc.load_page(pno).get_text("words")

def lines_from_words_with_pos(words, y_tol=3.0) -> List[Tuple[float,float,str]]:
    """
    Склеиваем слова в строки; возвращаем (y, mean_x, text)
    """
    rows: Dict[float, List[Tuple[float,str]]] = {}
    for x0,y0,x1,y1,t, *_ in words:
        key = None
        for ky in rows.keys():
            if abs(ky - y0) <= y_tol:
                key = ky; break
        if key is None:
            key = y0
            rows[key] = []
        rows[key].append((x0, t))
    out = []
    for ky in sorted(rows.keys()):
        items = sorted(rows[ky], key=lambda r: r[0])
        text = re.sub(r"\s+", " ", " ".join(t for _,t in items)).strip()
        if text:
            mean_x = sum(x for x,_ in items) / len(items)
            out.append((ky, mean_x, text))
    return out

# ------------------------
# Parsers
# ------------------------
def parse_match_meta(lines: List[Tuple[float,float,str]]) -> Dict:
    meta = {"date":"", "time_msk":"", "teams":{"home":"","away":""}}
    top = [txt for _,_,txt in lines[:200]]

    # дата: 25.10.2025 или 25 октября 2025
    for ln in top:
        m = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", ln)
        if m: meta["date"]=m.group(0); break
        m = re.search(r"\b\d{1,2}\s+[А-Яа-яё]+\s+20\d{2}", ln)
        if m: meta["date"]=m.group(0).replace(" г.","").strip(); break

    # время
    for ln in top:
        m = re.search(r"\b([01]\d|2[0-3]):[0-5]\d\b", ln)
        if m: meta["time_msk"]=m.group(0); break

    # команды — берем две самые длинные КАПС-строки до таблиц (№/Поз/Вратари)
    caps = []
    for _,_,ln in lines:
        if "Вратари" in ln or "Поз" in ln or "№" == ln.strip():
            break
        if re.search(r"[А-ЯЁ]{3,}", ln) and len(ln) >= 10:
            caps.append(ln)
    caps = sorted(set(caps), key=lambda s: -len(s))
    if caps: meta["teams"]["home"] = caps[0]
    if len(caps) > 1: meta["teams"]["away"] = caps[1]
    return meta

def parse_referees(lines: List[Tuple[float,float,str]]) -> Tuple[List[str], List[str], Dict]:
    dbg = {}
    # ищем строку с заголовком
    header_i = -1
    for i,(_,_,txt) in enumerate(lines[:120]):
        if ("Главный судья" in txt) and ("Линейный судья" in txt):
            header_i = i; break
    if header_i != -1 and header_i+1 < len(lines):
        _,_,raw = lines[header_i+1]; dbg["raw"]=raw
        parts = [p for p in re.split(r"[,\|;]|\s+", raw) if p]
        cand = []
        for j in range(len(parts)-1):
            a,b = parts[j], parts[j+1]
            if all(re.match(r"^[А-ЯЁ][а-яё\-]+$", x) for x in (a,b)):
                cand.append(a+" "+b)
        if len(cand) >= 4:
            return cand[:2], cand[2:4], dbg
    return [], [], {"note":"ref header not found"}

NAME_RX = re.compile(r"^[А-ЯЁ][а-яё\-]+ [А-ЯЁ][а-яё\-]+(?: [А-ЯЁ][а-яё\-]+)?")

def parse_goalies_fast(lines: List[Tuple[float,float,str]], page_width: float) -> Dict:
    """
    Быстрый парсер без OCR:
    - находим строки 'Вратари' (обычно две: левая/правая колонка);
    - по каждой колонке берём следующие 10-18 строк, пока не встретим 'Звено';
    - вытаскиваем имена + флаги С/Р.
    """
    ys = []
    for i,(y,x,txt) in enumerate(lines):
        if txt.strip().startswith("Вратари"):
            ys.append((i,y,x))

    home, away = [], []
    mid = page_width/2 if page_width else 500  # грубая середина

    def collect(start_i: int, want_left: bool) -> List[Dict]:
        acc = []
        for j in range(start_i+1, min(start_i+20, len(lines))):
            _, x, t = lines[j]
            if t.startswith("Звено"): break
            if want_left and x > mid: break
            if (not want_left) and x < mid: break
            m = NAME_RX.search(t)
            if m:
                flag = "C" if re.search(r"\bС\b", t) else ("R" if re.search(r"\bР\b", t) else "")
                acc.append({"name": m.group(0), "flag": flag})
        return acc

    for i, y, x in ys:
        if x < mid and not home:
            home = collect(i, True)
        elif x >= mid and not away:
            away = collect(i, False)

    return {"home": home, "away": away}

def parse_goalies_with_ocr(doc: fitz.Document, lines: List[Tuple[float,float,str]]) -> Dict:
    """Узкий OCR-fallback (не жрёт время/CPU)."""
    # если в тексте «Вратари» не нашли — ищем OCR верхней трети
    img = pdf_to_pix(doc, 0, dpi=300)
    h, w = img.height, img.width
    crop = img.crop((0, int(h*0.18), w, int(h*0.48)))
    ocr = ocr_image_lines(crop)
    # считаем, что левая/правая колонка разделены словом "Вратари"
    # вытаскиваем первые 2–3 ФИО слева и справа
    left, right = [], []
    seen = False
    for t in ocr[:120]:
        if t.startswith("Вратари"):
            seen = True
            continue
        if not seen: continue
        if t.startswith("Звено"): break
        m = NAME_RX.search(t)
        if m:
            # грубо: первые 2 имя — лев, следующие 2 — прав (как горячий запас)
            (left if len(left) < 3 else right).append({"name": m.group(0), "flag": ""})
        if len(left) >= 3 and len(right) >= 3: break
    return {"home": left, "away": right}

# ------------------------
# Extractors
# ------------------------
def extract_words(doc: fitz.Document) -> Dict:
    words = words_with_boxes(doc, 0)
    lines = lines_from_words_with_pos(words)
    return {"ok": True, "engine": "words", "match": parse_match_meta(lines)}

def extract_refs(doc: fitz.Document, debug=False) -> Dict:
    words = words_with_boxes(doc, 0)
    lines = lines_from_words_with_pos(words)
    main, linesmen, dbg = parse_referees(lines)
    out = {"ok": True, "engine": "ocr-refs", "referees": {"main": main, "linesmen": linesmen}}
    if debug: out["_debug"] = dbg
    return out

def extract_goalies(doc: fitz.Document, force_ocr=False, debug=False) -> Dict:
    words = words_with_boxes(doc, 0)
    lines = lines_from_words_with_pos(words)
    page_w = doc.load_page(0).rect.width
    g = parse_goalies_fast(lines, page_w)
    if not g["home"] and not g["away"] and force_ocr:
        try:
            g = parse_goalies_with_ocr(doc, lines)
        except Exception as e:
            if debug: return {"ok": True, "engine": "gk", "goalies": g, "_debug":{"ocr_error": str(e)}}
    out = {"ok": True, "engine": "gk", "goalies": g}
    if debug: out["_debug"] = {"fast": True, "force_ocr": force_ocr}
    return out

def extract_all(doc: fitz.Document, season: str, uid: str, force_ocr=False, debug=False) -> Dict:
    t0 = time.time()
    meta = extract_words(doc).get("match", {"teams":{"home":"","away":""}})
    refs = extract_refs(doc, debug=debug)
    gk = extract_goalies(doc, force_ocr=force_ocr, debug=debug)
    out = {
        "ok": True,
        "engine": "all",
        "match": {"season": season, "uid": uid, **meta},
        "referees": refs.get("referees", {"main":[], "linesmen":[]}),
        "goalies": gk.get("goalies", {"home":[], "away":[]}),
        "duration_s": round(time.time()-t0, 3),
    }
    if debug: out["_debug"] = {"meta": meta}
    return out

# ------------------------
# Routes
# ------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "engine": "ready"})

@app.get("/extract")
def extract():
    season = (request.args.get("season") or "").strip()
    uid    = (request.args.get("uid") or "").strip()
    mode   = (request.args.get("mode") or "all").strip().lower()
    debug  = request.args.get("debug") in ("1","true","yes")
    force_ocr = request.args.get("ocr") in ("1","true","yes")

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
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return jsonify({"ok": False, "error": "pdf_open_error", "detail": str(e)}), 500

    t0 = time.time()
    try:
        if mode == "refs":
            res = extract_refs(doc, debug=debug)
        elif mode in ("gk","goalies"):
            res = extract_goalies(doc, force_ocr=force_ocr, debug=debug)
        elif mode == "words":
            res = extract_words(doc)
        else:
            res = extract_all(doc, season, uid, force_ocr=force_ocr, debug=debug)
        res["source_url"] = url
        if "duration_s" not in res:
            res["duration_s"] = round(time.time()-t0, 3)
        return Response(json.dumps(res, ensure_ascii=False), mimetype="application/json")
    except Exception as e:
        return jsonify({"ok": False, "error": "extract_error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")), debug=False)
