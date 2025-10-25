# main.py
# KHL-OCR_Core_v1.0.0 (Render) — v2.6.0r4
# FIX: layout-based extract for "Вратари"/"Судьи" по координатам (PyMuPDF words)
#      устойчиво собирает ФИО (≥2 слов) и gk_status из С/Р/(С)/(Р)/scratch.

import os, re, time, unicodedata
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

# === ENV / defaults ===
DEFAULT_DPI = int(os.getenv("OCR_DPI", "300"))
DEFAULT_SCALE = float(os.getenv("OCR_SCALE", "1.6"))
DEFAULT_BIN_THRESH = int(os.getenv("OCR_BIN_THRESH", "185"))
DEFAULT_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "2"))

PLAYERS_CSV = os.getenv("PLAYERS_CSV")      # опц.: CSV игроков (для фаззи)
REFEREES_CSV = os.getenv("REFEREES_CSV")    # опц.: CSV судей (для фаззи)

# === utils: normalize ===
def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
def _normalize_spaces(s: str) -> str:
    s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ").replace("\u00ad", "-")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()
def _clean_text(s: str) -> str:
    return _normalize_spaces(_strip_accents(s))

# === robust PDF fetch ===
def fetch_pdf_bytes(url: str) -> Tuple[bytes, List[str]]:
    tried: List[str] = []
    base = (url or "").trim() if hasattr(str, "trim") else (url or "").strip()
    variants: List[str] = []
    if base:
        variants.append(base)
        if "www.khl.ru" in base:
            variants.append(base.replace("www.khl.ru", "khl.ru"))
        m = re.search(r"/pdf/(\d{4})/(\d{6})/game-(\d+)-start-ru\.pdf$", base)
        if m:
            season, _mid6, mid = m.groups()
            variants.append(f"https://www.khl.ru/documents/{season}/{mid}.pdf")
            variants.append(f"https://www.khl.ru/documents/{season}/game-{mid}-start-ru.pdf")
        if base.endswith("-start-ru.pdf"):
            variants.append(base.replace("-start-ru.pdf", "-start.pdf"))

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/127.0.0.1 Safari/537.36"),
        "Referer": "https://www.khl.ru/",
        "Accept": "application/pdf,application/*;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru,en;q=0.9",
        "Connection": "keep-alive",
    }

    # httpx без http2
    try:
        import httpx
        with httpx.Client(http2=False, timeout=30.0, headers=headers, follow_redirects=True) as client:
            for v in variants:
                try:
                    r = client.get(v)
                    tried.append(f"{v} [{r.status_code}]")
                    ctype = (r.headers.get("content-type") or "").lower()
                    if r.status_code == 200 and "pdf" in ctype:
                        return r.content, tried
                except Exception as e:
                    tried.append(f"{v} [httpx err: {e}]")
    except Exception as e:
        tried.append(f"httpx unavailable: {e}")

    # urllib fallback
    if variants:
        try:
            import urllib.request
            req = urllib.request.Request(variants[0], headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                tried.append(f"{variants[0]} [urllib {getattr(resp,'status',200)}]")
                return data, tried
        except Exception as e:
            tried.append(f"{variants[0]} [urllib err: {e}]")
    return b"", tried

# === imaging / OCR ===
def _binarize(img: Image.Image, threshold: int) -> Image.Image:
    return img.convert("L").point(lambda p: 255 if p > threshold else 0, mode="1").convert("L")
def _preprocess_for_ocr(pix: "fitz.Pixmap", scale: float, bin_thresh: int) -> Image.Image:
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if scale and scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.25)
    return _binarize(img, bin_thresh)

# === words→columns text (для snippet/резерва) ===
def _words_to_lines_by_columns(doc: "fitz.Document", max_pages: int) -> Tuple[str, int]:
    pages_to_process = min(len(doc), max_pages)
    all_lines: List[str] = []
    for i in range(pages_to_process):
        page = doc.load_page(i)
        words = page.get_text("words")  # [x0,y0,x1,y1,text,block,line,word]
        if not words:
            continue
        tokens, xs = [], []
        for (x0, y0, x1, y1, wtext, *_rest) in words:
            t = _clean_text(wtext)
            if not t:
                continue
            tokens.append((x0, y0, x1, y1, t))
            xs.append(x0)
        if not tokens:
            continue
        xs.sort()
        midx = xs[len(xs)//2]
        left  = [t for t in tokens if t[0] <= midx]
        right = [t for t in tokens if t[0] >  midx]

        def build_lines(tok_list):
            if not tok_list: return []
            tok_list.sort(key=lambda z: (round(z[1]/8), z[0]))
            lines, cur_y, buf = [], None, []
            for (x0,y0,x1,y1,t) in tok_list:
                if cur_y is None:
                    cur_y, buf = y1, [t]
                else:
                    if abs(y0-cur_y) > 6:
                        if buf: lines.append(" ".join(buf))
                        buf, cur_y = [t], y1
                    else:
                        buf.append(t); cur_y = max(cur_y,y1)
            if buf: lines.append(" ".join(buf))
            return [_normalize_spaces(ln) for ln in lines if ln.strip()]

        all_lines += build_lines(left)
        all_lines += build_lines(right)
    return "\n".join(all_lines).strip(), pages_to_process

def pdf_to_text_pref_words(pdf_bytes: bytes, dpi: int, scale: float, bin_thresh: int, max_pages: int) -> Tuple[str, int]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text, used = _words_to_lines_by_columns(doc, max_pages=max_pages)
        if len(text) >= 300:
            return text, used
        # OCR fallback
        ocr_parts: List[str] = []
        pages_to_process = min(len(doc), max_pages)
        for i in range(pages_to_process):
            page = doc.load_page(i)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = _preprocess_for_ocr(pix, scale=scale, bin_thresh=bin_thresh)
            txt = pytesseract.image_to_string(img, lang="rus+eng", config="--oem 1 --psm 4 -c preserve_interword_spaces=1")
            ocr_parts.append(_clean_text(txt))
        return ("\n".join(ocr_parts).strip() or text), pages_to_process

# === LAYOUT extractors by coordinates ===
NAME_SEQ = r"[А-ЯЁA-Z][а-яёa-z'\-\.]{2,}(?:\s+[А-ЯЁA-Z][а-яёa-z'\-\.]{2,}){1,2}"

def _median_x(tokens: List[Tuple[float,float,float,float,str]]) -> float:
    xs = sorted([t[0] for t in tokens])
    return xs[len(xs)//2] if xs else 0.0

def _find_anchor_y(tokens, anchor_re: re.Pattern, column: str, midx: float) -> Optional[float]:
    # column: "left"/"right"
    col = [t for t in tokens if (t[0] <= midx if column=="left" else t[0] > midx)]
    for (x0,y0,x1,y1,t) in sorted(col, key=lambda z: (z[1], z[0])):
        if anchor_re.search(t):
            return y1  # нижняя граница строки якоря
    return None

def _collect_names_below(tokens, start_y: float, column: str, midx: float, height: float = 500.0) -> List[str]:
    col = [t for t in tokens if (t[0] <= midx if column=="left" else t[0] > midx)]
    # берём слова в прямоугольнике: y ∈ (start_y, start_y+height)
    region = [t for t in col if (t[1] > start_y and t[1] < start_y + height)]
    # группировка в строки по Y
    region.sort(key=lambda z: (round(z[1]/8), z[0]))
    lines, cur_y, buf = [], None, []
    for (x0,y0,x1,y1,t) in region:
        if cur_y is None:
            cur_y, buf = y1, [t]
        else:
            if abs(y0-cur_y) > 6:
                if buf: lines.append(" ".join(buf))
                buf, cur_y = [t], y1
            else:
                buf.append(t); cur_y = max(cur_y,y1)
    if buf: lines.append(" ".join(buf))
    lines = [_normalize_spaces(ln) for ln in lines if ln.strip()]
    # из каждой строки достаём ФИО ≥2 слов
    names = []
    for ln in lines:
        # выкинуть одиночные С/Р метки, чтобы не ломали имя
        ln2 = re.sub(r"(^|\s)[СРCP](?=\s|$)|\([СРCP]\)", " ", ln, flags=re.I)
        m = re.search(NAME_SEQ, ln2)
        if m:
            names.append(m.group(0))
    return names

def _detect_gk_status(line: str) -> Optional[str]:
    if re.search(r"(^|\W)[СC](\W|$)|\((?:С|C)\)", line): return "starter"
    if re.search(r"(^|\W)[РP](\W|$)|\((?:Р|P)\)", line): return "reserve"
    if re.search(r"scratch", line, re.I):               return "scratch"
    return None

def extract_goalies_layout(pdf_bytes: bytes, max_pages: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    res = {"home": [], "away": []}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = min(len(doc), max_pages)
        for i in range(pages):
            page = doc.load_page(i)
            words = page.get_text("words")
            if not words: continue
            tokens = []
            for (x0,y0,x1,y1,t, *_rest) in words:
                tt = _clean_text(t)
                if tt: tokens.append((x0,y0,x1,y1,tt))
            if not tokens: continue
            midx = _median_x(tokens)

            # Якорь "Вратари"
            re_gk = re.compile(r"^Вратари$", re.I)
            y_home = _find_anchor_y(tokens, re_gk, "left",  midx)
            y_away = _find_anchor_y(tokens, re_gk, "right", midx)

            # Собираем кандидатов имён и статус по строке
            def pack(column: str, y_anchor: Optional[float]) -> List[Dict[str,Any]]:
                out: List[Dict[str,Any]] = []
                if y_anchor is None: return out
                # Соберём строки в зоне интереса и одновременно держим «сырые строки» для статуса
                col = [t for t in tokens if (t[0] <= midx if column=="left" else t[0] > midx)]
                region = [t for t in col if (t[1] > y_anchor and t[1] < y_anchor + 500.0)]
                region.sort(key=lambda z: (round(z[1]/8), z[0]))
                lines, cur_y, buf = [], None, []
                for (x0,y0,x1,y1,t) in region:
                    if cur_y is None:
                        cur_y, buf = y1, [t]
                    else:
                        if abs(y0-cur_y) > 6:
                            if buf: lines.append(" ".join(buf))
                            buf, cur_y = [t], y1
                        else:
                            buf.append(t); cur_y = max(cur_y,y1)
                if buf: lines.append(" ".join(buf))
                lines = [_normalize_spaces(ln) for ln in lines if ln.strip()]

                for ln in lines[:12]:  # ограничим разумным числом строк
                    # имя (≥2 слова)
                    ln2 = re.sub(r"(^|\s)[СРCP](?=\s|$)|\([СРCP]\)", " ", ln, flags=re.I)
                    m = re.search(NAME_SEQ, ln2)
                    if not m: 
                        continue
                    name = m.group(0)
                    status = _detect_gk_status(ln) or "unknown"
                    out.append({"name": name, "gk_status": status})
                return out

            res["home"] += pack("left",  y_home)
            res["away"] += pack("right", y_away)

    # Дедуп по имени + статусу, сохраняем порядок
    def dedup(lst):
        seen, out = set(), []
        for it in lst:
            key = (it["name"].lower(), it["gk_status"])
            if key not in seen:
                seen.add(key); out.append(it)
        return out[:6]  # ограничим до первых валидных записей
    res["home"] = dedup(res["home"])
    res["away"] = dedup(res["away"])
    return res

def extract_refs_layout(pdf_bytes: bytes, max_pages: int = 2) -> List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = min(len(doc), max_pages)
        for i in range(pages):
            page = doc.load_page(i)
            words = page.get_text("words")
            if not words: continue
            tokens = []
            for (x0,y0,x1,y1,t, *_rest) in words:
                tt = _clean_text(t)
                if tt: tokens.append((x0,y0,x1,y1,tt))
            if not tokens: continue
            midx = _median_x(tokens)
            re_refs = re.compile(r"^Судьи$|^Referees$", re.I)
            # Можно искать в любой колонке — берём обе
            for col in ("left","right"):
                y = _find_anchor_y(tokens, re_refs, col, midx)
                if y is None: continue
                lines = _collect_names_below(tokens, y, col, midx, height=280.0)
                for ln in lines[:6]:
                    role = "Unknown"
                    if re.search(r"лайнсмен|linesman", ln, re.I): role = "Linesman"
                    if re.search(r"судья|referee", ln, re.I):     role = "Referee"
                    out.append({"name": ln, "role": role})
    # dedup by name/role
    seen, res = set(), []
    for r in out:
        k = (r["name"].lower(), r["role"])
        if k not in seen:
            seen.add(k); res.append(r)
    return res

# === (опционально) фаззи нормализация по словарю игроков для киперов
def _load_dict(csv_path: str) -> List[str]:
    if not csv_path or not os.path.exists(csv_path): return []
    try:
        import csv
        vals = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                for cell in row:
                    cell = _clean_text(cell)
                    if re.search(NAME_SEQ, cell):
                        vals.append(cell); break
        return vals
    except Exception:
        return []
def _fuzzy_fix_goalies(data: Dict[str,Any]) -> None:
    try:
        from rapidfuzz import process, fuzz
    except Exception:
        return
    dict_players = _load_dict(PLAYERS_CSV)
    if not dict_players: return
    for side in ("home","away"):
        for gk in data.get("goalies",{}).get(side, []):
            nm = gk.get("name"); 
            if not nm: continue
            best = process.extractOne(nm, dict_players, scorer=fuzz.WRatio, score_cutoff=87)
            if best: gk["name"] = best[0]

# === FastAPI ===
app = FastAPI(title="KHL PDF OCR", version="2.6.0r4")

@app.get("/")
def health():
    return {"ok": True, "service": "khl-pdf-ocr", "version": "2.6.0r4"}

@app.get("/ocr")
@app.post("/ocr")
def ocr_endpoint(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(1369),
    dpi: int = Query(DEFAULT_DPI),
    scale: float = Query(DEFAULT_SCALE),
    bin_thresh: int = Query(DEFAULT_BIN_THRESH),
    max_pages: int = Query(DEFAULT_MAX_PAGES),
    body: Optional[Dict[str, Any]] = Body(None)
):
    t0 = time.perf_counter()
    if body:
        match_id = body.get("match_id", match_id)
        pdf_url = body.get("pdf_url", pdf_url)
        season  = body.get("season", season)
        dpi     = body.get("dpi", dpi)
        scale   = body.get("scale", scale)
        bin_thresh = body.get("bin_thresh", bin_thresh)
        max_pages  = body.get("max_pages", max_pages)

    b, tried = fetch_pdf_bytes(pdf_url or "")
    if not b:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": 404, "tried": tried}, status_code=404)

    text, pages_used = pdf_to_text_pref_words(b, dpi=dpi, scale=scale, bin_thresh=bin_thresh, max_pages=max_pages)
    dur = round(time.perf_counter() - t0, 3)
    return {
        "ok": True, "match_id": match_id, "season": season, "source_pdf": pdf_url,
        "pdf_len": len(b), "dpi": dpi, "pages_ocr": pages_used, "dur_total_s": dur,
        "text_len": len(text), "snippet": text[:800], "tried": tried
    }

@app.get("/extract")
@app.post("/extract")
def extract_endpoint(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(1369),
    target: str = Query("all"),
    dpi: int = Query(DEFAULT_DPI),
    scale: float = Query(DEFAULT_SCALE),
    bin_thresh: int = Query(DEFAULT_BIN_THRESH),
    max_pages: int = Query(DEFAULT_MAX_PAGES),
    body: Optional[Dict[str, Any]] = Body(None)
):
    t0 = time.perf_counter()
    if body:
        match_id = body.get("match_id", match_id)
        pdf_url  = body.get("pdf_url", pdf_url)
        season   = body.get("season", season)
        target   = body.get("target", target)
        dpi      = body.get("dpi", dpi)
        scale    = body.get("scale", scale)
        bin_thresh = body.get("bin_thresh", bin_thresh)
        max_pages  = body.get("max_pages", max_pages)

    b, tried = fetch_pdf_bytes(pdf_url or "")
    if not b:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": 404, "tried": tried}, status_code=404)

    # 1) layout-based извлечение goalies/refs (по координатам)
    data: Dict[str, Any] = {}
    if target in ("goalies","all"):
        data["goalies"] = extract_goalies_layout(b, max_pages=max_pages)
    if target in ("refs","all"):
        data["refs"]    = extract_refs_layout(b, max_pages=max_pages)

    # 2) lineups (оставим пока текстовый, можно тоже перевести на layout при необходимости)
    if target in ("lineups","all"):
        # простой текстовый fallback (если нужно — переведём на layout)
        text, _ = pdf_to_text_pref_words(b, dpi=dpi, scale=scale, bin_thresh=bin_thresh, max_pages=max_pages)
        LINE_RE = re.compile(r"(?P<num>\d{1,2})\s*\|\s*(?P<pos>[ВДНFG])\s*\|\s*(?P<name>"+NAME_SEQ+r")", re.U)
        lines = text.splitlines()
        buckets, cur = [], []
        pos_map = {"В":"G","Д":"D","Н":"F","F":"F","G":"G","D":"D"}
        for ln in lines:
            m = LINE_RE.search(ln)
            if m:
                num = int(m.group("num"))
                pos = pos_map.get(m.group("pos"), m.group("pos"))
                name = " ".join([p for p in m.groups()[2:] if p])
                cur.append({"num": num, "pos": pos, "name": name})
            elif cur:
                buckets.append(cur); cur = []
        if cur: buckets.append(cur)
        data["lineups"] = {"home": buckets[0] if buckets else [], "away": (buckets[1] if len(buckets)>1 else [])}

    # 3) фаззи-нормализация киперов по словарю игроков (если PLAYERS_CSV задан)
    if data.get("goalies"):
        _fuzzy_fix_goalies(data)

    dur = round(time.perf_counter() - t0, 3)
    return {
        "ok": True, "match_id": match_id, "season": season, "source_pdf": pdf_url,
        "pdf_len": len(b), "dpi": dpi, "pages_ocr": max_pages, "dur_total_s": dur,
        "data": data, "tried": tried
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
