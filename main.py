# main.py
# KHL-OCR_Core_v1.0.0 (Render) — v2.6.0r3
# Фокус фикса: корректная сборка ФИО киперов (goalies) + устойчивые статусы С/Р/ scratch.

import os
import re
import time
import unicodedata
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

PLAYERS_CSV = os.getenv("PLAYERS_CSV")      # опц.: путь к CSV игроков
REFEREES_CSV = os.getenv("REFEREES_CSV")    # опц.: путь к CSV судей

# === utils: normalize ===
def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _normalize_spaces(s: str) -> str:
    s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ").replace("\u00ad", "-")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def _clean_text(s: str) -> str:
    return _normalize_spaces(_strip_accents(s))

# === robust fetch pdf ===
def fetch_pdf_bytes(url: str) -> Tuple[bytes, List[str]]:
    tried: List[str] = []
    base = (url or "").strip()
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

    # httpx (без http2)
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

# === words→columns → lines ===
def _words_to_lines_by_columns(doc: "fitz.Document", max_pages: int) -> Tuple[str, int]:
    pages_to_process = min(len(doc), max_pages)
    all_lines: List[str] = []

    for i in range(pages_to_process):
        page = doc.load_page(i)
        words = page.get_text("words")  # [x0,y0,x1,y1,text,block,line,word]
        if not words:
            continue

        tokens = []
        xs = []
        for (x0, y0, x1, y1, wtext, _bno, _lno, _wno) in words:
            t = _clean_text(wtext)
            if not t:
                continue
            tokens.append((x0, y0, x1, y1, t))
            xs.append(x0)
        if not tokens:
            continue

        xs_sorted = sorted(xs)
        midx = xs_sorted[len(xs_sorted) // 2]

        left = [t for t in tokens if t[0] <= midx]
        right = [t for t in tokens if t[0] > midx]

        def build_lines(tok_list: List[Tuple[float, float, float, float, str]]) -> List[str]:
            if not tok_list:
                return []
            tok_list.sort(key=lambda z: (round(z[1] / 8), z[0]))  # кластеризация по Y, затем X
            lines: List[str] = []
            cur_y = None
            buf: List[str] = []
            for (x0, y0, x1, y1, t) in tok_list:
                if cur_y is None:
                    cur_y = y1
                    buf = [t]
                else:
                    if abs(y0 - cur_y) > 6:
                        if buf:
                            lines.append(" ".join(buf))
                        buf = [t]
                        cur_y = y1
                    else:
                        buf.append(t)
                        cur_y = max(cur_y, y1)
            if buf:
                lines.append(" ".join(buf))
            return [_normalize_spaces(ln) for ln in lines if ln.strip()]

        left_lines = build_lines(left)
        right_lines = build_lines(right)

        if left_lines:
            all_lines.extend(left_lines)
        if right_lines:
            all_lines.extend(right_lines)

    return "\n".join(all_lines).strip(), pages_to_process

def pdf_to_text_pref_words(pdf_bytes: bytes, dpi: int, scale: float, bin_thresh: int, max_pages: int) -> Tuple[str, int]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text, used = _words_to_lines_by_columns(doc, max_pages=max_pages)
        if len(text) >= 300:
            return text, used
        # OCR fallback (psm=4 для колонок)
        ocr_parts: List[str] = []
        pages_to_process = min(len(doc), max_pages)
        for i in range(pages_to_process):
            page = doc.load_page(i)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = _preprocess_for_ocr(pix, scale=scale, bin_thresh=bin_thresh)
            txt = pytesseract.image_to_string(
                img, lang="rus+eng",
                config="--oem 1 --psm 4 -c preserve_interword_spaces=1"
            )
            ocr_parts.append(_clean_text(txt))
        ocr_text = "\n".join(ocr_parts).strip()
        return (ocr_text or text), pages_to_process

# === structured extractors ===
# Жёсткий паттерн: минимум ДВА слова в имени (Фамилия Имя [Отчество])
NAME_SEQ = r"([А-ЯЁA-Z][а-яёa-z'\-\.]{2,})(?:\s+([А-ЯЁA-Z][а-яёa-z'\-\.]{2,})){1,2}"

def extract_refs_from_text(text: str) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    m = re.search(r"(Судьи|Referees)[:\s]*\n?(.{0,300})", text, flags=re.I | re.S)
    if m:
        chunk = _normalize_spaces(m.group(2))
        parts = re.split(r"[;,\u2013\-•]+", chunk)
        for c in parts:
            c = c.strip()
            if len(c) < 3:
                continue
            for nm in re.findall(NAME_SEQ, c):
                name = " ".join([x for x in nm if x])
                role = "Unknown"
                if re.search(r"лайнсмен|linesman", c, re.I):
                    role = "Linesman"
                if re.search(r"судья|referee", c, re.I):
                    role = "Referee"
                refs.append({"name": name, "role": role})
    # dedup
    out, seen = [], set()
    for r in refs:
        k = (r["name"].lower(), r["role"])
        if k not in seen:
            out.append(r); seen.add(k)
    return out

def _detect_gk_status_and_name(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Из строки блока 'Вратари' достаём:
    - ФИО (минимум 2 слова, NAME_SEQ)
    - статус: starter/reserve/scratch (по 'С','Р','(С)','(Р)','scratch')
    """
    s = _normalize_spaces(line)
    # отрежем левую табличную часть, если есть
    s = re.sub(r"^\d{1,2}\s*\|\s*[ВДНFG]\s*\|\s*", " ", s)
    # статус
    status = None
    if re.search(r"(^|\W)[СC](\W|$)|\((?:С|C)\)", s):
        status = "starter"
    elif re.search(r"(^|\W)[РP](\W|$)|\((?:Р|P)\)", s):
        status = "reserve"
    if re.search(r"scratch", s, re.I):
        status = status or "scratch"
    # выкинуть одиночные маркеры статуса из текста, чтобы не мешали имени
    s2 = re.sub(r"(^|\s)[СРCP](?=\s|$)|\([СРCP]\)", " ", s)
    # взять ДЛИННУЮ последовательность имени (≥2 слова)
    m = re.search(NAME_SEQ, s2)
    if not m:
        return None, status
    name = " ".join([x for x in m.groups() if x])
    return name, (status or "unknown")

def extract_goalies_from_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Берём два подряд идущих блока 'Вратари' и внутри каждой строки
    ищем ФИО как минимум из двух слов (склейка гарантирована).
    """
    res = {"home": [], "away": []}
    blocks = list(re.finditer(r"Вратари[^\n]*\n(.{0,500})", text, flags=re.I | re.S))

    def parse_block(block_text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
        for ln in lines[:10]:
            name, gk_status = _detect_gk_status_and_name(ln)
            if name:
                out.append({"name": name, "gk_status": gk_status or "unknown"})
        return out

    if blocks:
        res["home"] = parse_block(blocks[0].group(1))
        if len(blocks) > 1:
            res["away"] = parse_block(blocks[1].group(1))
    return res

def extract_lineups_from_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Базовый разбор табличных строк состава: "№ | Поз | ФИО ..."
    """
    LINE_RE = re.compile(r"(?P<num>\d{1,2})\s*\|\s*(?P<pos>[ВДНFG])\s*\|\s*(?P<name>"+NAME_SEQ+r")", re.U)
    chunks = text.splitlines()
    buckets: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []

    pos_map = {"В":"G","Д":"D","Н":"F","F":"F","G":"G","D":"D"}

    for ln in chunks:
        m = LINE_RE.search(ln)
        if m:
            num = int(m.group("num"))
            # name как полная последовательность
            name_parts = [x for x in m.groups()[1:] if x]  # пропускаем num/pos
            # m.groups(): (num,pos, w1, w2, w3)
            # проще пересобрать имя из конца:
            nm = " ".join(name_parts[-3:]).strip()
            pos = pos_map.get(m.group("pos"), m.group("pos"))
            cur.append({"num": num, "pos": pos, "name": nm})
        elif cur:
            buckets.append(cur); cur = []
    if cur:
        buckets.append(cur)

    lineup = {"home": [], "away": []}
    if buckets:
        lineup["home"] = buckets[0]
        if len(buckets) > 1:
            lineup["away"] = buckets[1]
    return lineup

# === optional: fuzzy normalization by CSV (rapidfuzz) ===
def _load_dict(csv_path: str) -> List[str]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    try:
        import csv
        vals = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                # берём первую «похожую на ФИО» колонку
                for cell in row:
                    cell = _clean_text(cell)
                    if re.search(NAME_SEQ, cell):
                        vals.append(cell)
                        break
        return vals
    except Exception:
        return []

def _fuzzy_fix_names(data: Dict[str, Any]) -> None:
    try:
        from rapidfuzz import process, fuzz
    except Exception:
        return
    dict_players = _load_dict(PLAYERS_CSV)
    if not dict_players:
        return
    for side in ("home", "away"):
        for gk in data.get("goalies", {}).get(side, []):
            nm = gk.get("name")
            if not nm:
                continue
            best = process.extractOne(nm, dict_players, scorer=fuzz.WRatio, score_cutoff=85)
            if best:
                gk["name"] = best[0]

# === FastAPI ===
app = FastAPI(title="KHL PDF OCR", version="2.6.0r3")

@app.get("/")
def health():
    return {"ok": True, "service": "khl-pdf-ocr", "version": "2.6.0r3"}

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
        season = body.get("season", season)
        dpi = body.get("dpi", dpi)
        scale = body.get("scale", scale)
        bin_thresh = body.get("bin_thresh", bin_thresh)
        max_pages = body.get("max_pages", max_pages)

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
        pdf_url = body.get("pdf_url", pdf_url)
        season = body.get("season", season)
        target = body.get("target", target)
        dpi = body.get("dpi", dpi)
        scale = body.get("scale", scale)
        bin_thresh = body.get("bin_thresh", bin_thresh)
        max_pages = body.get("max_pages", max_pages)

    b, tried = fetch_pdf_bytes(pdf_url or "")
    if not b:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": 404, "tried": tried}, status_code=404)

    text, pages_used = pdf_to_text_pref_words(b, dpi=dpi, scale=scale, bin_thresh=bin_thresh, max_pages=max_pages)

    data: Dict[str, Any] = {}
    if target in ("goalies", "all"):
        data["goalies"] = extract_goalies_from_text(text)
    if target in ("refs", "all"):
        data["refs"] = extract_refs_from_text(text)
    if target in ("lineups", "all"):
        data["lineups"] = extract_lineups_from_text(text)

    # опциональная фаззи-нормализация ФИО киперов по словарю (если задан PLAYERS_CSV)
    if data.get("goalies"):
        _fuzzy_fix_names(data)

    dur = round(time.perf_counter() - t0, 3)
    return {
        "ok": True, "match_id": match_id, "season": season, "source_pdf": pdf_url,
        "pdf_len": len(b), "dpi": dpi, "pages_ocr": pages_used, "dur_total_s": dur,
        "data": data, "tried": tried
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
