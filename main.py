# main.py
# KHL-OCR_Core_v1.0.0 (Render) — v2.7.0
# Извлекаем: goalies (текстовый FSM) + refs (расширенные якоря) + lineups (FSM по заголовкам "No|Поз|Фамилия, Имя")

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

PLAYERS_CSV = os.getenv("PLAYERS_CSV")
REFEREES_CSV = os.getenv("REFEREES_CSV")

# === normalize ===
def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
def _normalize_spaces(s: str) -> str:
    s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ").replace("\u00ad", "-")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()
def _clean_text(s: str) -> str:
    return _normalize_spaces(_strip_accents(s))

# === fetch ===
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
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win32; x32) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/127.0.0.1 Safari/537.36"),
        "Referer": "https://www.khl.ru/",
        "Accept": "application/pdf,application/*;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru,en;q=0.9",
        "Connection": "keep-alive",
    }

    try:
        import httpx
        with httpx.Client(http2=False, timeout=30.0, headers=headers, follow_redirects=True) as client:
            for v in variants:
                try:
                    r = client.get(v)
                    tried.append(f"{v} [{r.status_code}]")
                    if r.status_code == 200 and "pdf" in (r.headers.get("content-type","").lower()):
                        return r.content, tried
                except Exception as e:
                    tried.append(f"{v} [httpx err: {e}]")
    except Exception as e:
        tried.append(f"httpx unavailable: {e}")

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

# === imaging/OCR ===
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

# === words->lines (left then right) ===
def _words_to_lines_by_columns(doc: "fitz.Document", max_pages: int) -> Tuple[str, int]:
    pages_to_process = min(len(doc), max_pages)
    all_lines: List[str] = []
    for i in range(pages_to_process):
        page = doc.load_page(i)
        words = page.get_text("words")
        if not words: continue
        tokens, xs = [], []
        for (x0, y0, x1, y1, wtext, *_rest) in words:
            t = _clean_text(wtext)
            if not t: continue
            tokens.append((x0, y0, x1, y1, t)); xs.append(x0)
        if not tokens: continue
        xs.sort(); midx = xs[len(xs)//2]
        left  = [t for t in tokens if t[0] <= midx]
        right = [t for t in tokens if t[0] >  midx]

        def build_lines(tok_list):
            if not tok_list: return []
            tok_list.sort(key=lambda z: (round(z[1]/8), z[0]))
            lines, cur_y, buf = [], None, []
            for (x0,y0,x1,y1,t) in tok_list:
                if cur_y is None: cur_y, buf = y1, [t]
                else:
                    if abs(y0-cur_y) > 6:
                        if buf: lines.append(" ".join(buf))
                        buf, cur_y = [t], y1
                    else: buf.append(t); cur_y = max(cur_y,y1)
            if buf: lines.append(" ".join(buf))
            return [_normalize_spaces(ln) for ln in lines if ln.strip()]

        all_lines += build_lines(left)
        all_lines += build_lines(right)
    return "\n".join(all_lines).strip(), pages_to_process

def pdf_to_text_pref_words(pdf_bytes: bytes, dpi: int, scale: float, bin_thresh: int, max_pages: int) -> Tuple[str, int]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text, used = _words_to_lines_by_columns(doc, max_pages=max_pages)
        if len(text) >= 300: return text, used
        # fallback OCR (редко)
        ocr_parts: List[str] = []
        pages = min(len(doc), max_pages)
        for i in range(pages):
            page = doc.load_page(i)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = _preprocess_for_ocr(pix, scale=scale, bin_thresh=bin_thresh)
            txt = pytesseract.image_to_string(img, lang="rus+eng",
                   config="--oem 1 --psm 4 -c preserve_interword_spaces=1")
            ocr_parts.append(_clean_text(txt))
        return ("\n".join(ocr_parts).strip() or text), pages

# === parsers ===
NAME_WORD = r"[А-ЯЁ][а-яё'\-\.]{2,}"
NAME_SEQ  = rf"{NAME_WORD}(?:\s+{NAME_WORD}){{1,2}}"
DATE_RE   = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
ROW_START_RE = re.compile(r"^\s*(?:\d{1,2}\s+)?[ВДНFG]\b")
STATUS_S_RE  = re.compile(r"(^|\W)[СC](\W|$)|\((?:С|C)\)", re.I)
STATUS_R_RE  = re.compile(r"(^|\W)[РP](\W|$)|\((?:Р|P)\)", re.I)

def _status_from_line(s: str) -> Optional[str]:
    if STATUS_S_RE.search(s): return "starter"
    if STATUS_R_RE.search(s): return "reserve"
    if re.search(r"scratch", s, re.I): return "scratch"
    return None

# --- GOALIES (текст) ---
def _extract_goalies_from_text_block(lines: List[str]) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = []
    name_buf: List[str] = []
    status_buf: Optional[str] = None
    def flush():
        nonlocal name_buf, status_buf
        if len(name_buf) >= 2:
            nm = " ".join(name_buf[:3])
            res.append({"name": nm, "gk_status": status_buf or "unknown"})
        name_buf, status_buf = [], None

    for ln in lines:
        ln = _normalize_spaces(ln)
        if not ln: continue
        if ROW_START_RE.search(ln) and len(name_buf) >= 2:
            flush()
        st = _status_from_line(ln)
        if st: status_buf = status_buf or st
        words = [w for w in re.findall(NAME_WORD, ln)
                 if w.lower() not in ("вратари","вратарь","звено")]
        words = [w for w in words if not re.fullmatch(r"[А-ЯЁ]\.", w)]
        if DATE_RE.search(ln) and len(name_buf) >= 2:
            flush(); continue
        for w in words:
            if len(name_buf) < 3: name_buf.append(w)
    if len(name_buf) >= 2: flush()
    return res[:6]

def extract_goalies_from_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    lines = [ln.strip() for ln in text.splitlines()]
    idxs = [i for i, ln in enumerate(lines)
            if re.search(r"(?i)^вратари|^вратарь", ln)]
    res = {"home": [], "away": []}
    if not idxs: return res
    def slice_block(start_idx: int) -> List[str]:
        stop = re.compile(r"(?i)звено|составы|матч|lineup|защитники|нападающие|РОСТЕР")
        blk: List[str] = []
        for ln in lines[start_idx+1 : start_idx+40]:
            if stop.search(ln): break
            blk.append(ln)
        return blk
    home_blk = slice_block(idxs[0]); home = _extract_goalies_from_text_block(home_blk)
    away = []
    if len(idxs) > 1:
        away_blk = slice_block(idxs[1]); away = _extract_goalies_from_text_block(away_blk)
    if not away and home:
        tail = lines[idxs[0]+1+len(home_blk) : idxs[0]+1+len(home_blk)+50]
        away  = _extract_goalies_from_text_block(tail)
    return {"home": home, "away": away}

# --- REFS (текст, расширенные якоря + общий поиск) ---
REF_ANCHORS = re.compile(r"(?im)^(Судьи|Судьи матча|Официальные лица|Referees)\s*$")
def extract_refs_from_text(text: str) -> List[Dict[str, str]]:
    lines = [ln.strip() for ln in text.splitlines()]
    refs: List[Dict[str,str]] = []

    # 1) По якорю
    for i, ln in enumerate(lines):
        if REF_ANCHORS.fullmatch(ln):
            for s in lines[i+1 : i+15]:
                s1 = _normalize_spaces(s)
                nm = re.findall(NAME_SEQ, s1)
                for n in nm:
                    role = "Unknown"
                    if re.search(r"лайнсмен|linesman", s1, re.I): role = "Linesman"
                    if re.search(r"судья|referee",   s1, re.I):   role = "Referee"
                    refs.append({"name": n, "role": role})
            break

    # 2) Если пусто — общий поиск паттернов типа “ФИО — судья/лайнсмен”
    if not refs:
        for s in lines:
            s1 = _normalize_spaces(s)
            nm = re.findall(NAME_SEQ, s1)
            if not nm: continue
            if re.search(r"судья|referee|лайнсмен|linesman", s1, re.I):
                for n in nm:
                    role = "Unknown"
                    if re.search(r"лайнсмен|linesman", s1, re.I): role = "Linesman"
                    if re.search(r"судья|referee",   s1, re.I):   role = "Referee"
                    refs.append({"name": n, "role": role})

    # dedup
    out, seen = [], set()
    for r in refs:
        k = (r["name"].lower(), r["role"])
        if k not in seen:
            out.append(r); seen.add(k)
    return out[:8]

# --- LINEUPS (текст, FSM вокруг заголовков таблицы) ---
HEADER_RE = re.compile(r"(?i)\b(No|№)\b.*\bПоз\b.*\bФамилия\b")
POS_MAP   = {"В":"G","Д":"D","З":"D","Н":"F","F":"F","G":"G","D":"D"}

def _parse_lineup_table(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Разбираем строки после заголовка таблицы.
    Состояние: ждём номер/позицию → собираем имя (2–3 слова) → фиксим запись по сигналам
    (дата, новый номер/позиция, метки 'Звено', 'Матч' и т.п.).
    """
    out: List[Dict[str,Any]] = []
    cur: Dict[str, Any] = {"num": None, "pos": None, "name_parts": []}

    def flush():
        nonlocal cur
        if cur["pos"] and len(cur["name_parts"]) >= 2:
            name = " ".join(cur["name_parts"][:3])
            out.append({"num": cur["num"], "pos": cur["pos"], "name": name})
        cur = {"num": None, "pos": None, "name_parts": []}

    for raw in lines:
        ln = _normalize_spaces(raw)
        if not ln: 
            continue

        # новый ряд — если видим начало формата
        m_numpos = re.match(r"^\s*(?P<num>\d{1,2})?\s*(?P<pos>[ВДНFG])\b", ln)
        if m_numpos:
            if cur["name_parts"]: flush()
            num = m_numpos.group("num")
            pos = m_numpos.group("pos")
            cur["num"] = int(num) if num else None
            cur["pos"] = POS_MAP.get(pos, pos)

        # накапливаем имя
        words = [w for w in re.findall(NAME_WORD, ln)
                 if w.lower() not in ("вратари","вратарь","звено","составы","матч","ладa","лaдa")]
        words = [w for w in words if not re.fullmatch(r"[А-ЯЁ]\.", w)]
        for w in words:
            if len(cur["name_parts"]) < 3:
                cur["name_parts"].append(w)

        # триггеры конца строки
        if DATE_RE.search(ln) and cur["name_parts"]:
            flush(); continue
        if re.search(r"(?i)\bзвено\b|^Составы|^Матч|^LINEUP", ln) and cur["name_parts"]:
            flush(); continue

    if cur["name_parts"]:
        flush()

    # подчистка: убрать None num/pos, если совсем пустые
    clean = []
    for r in out:
        rr = {"name": r["name"], "pos": r["pos"], "num": r["num"] if isinstance(r["num"], int) else None}
        clean.append(rr)
    # ограничим до разумного числа
    return clean[:30]

def extract_lineups_from_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    lines = [ln.strip() for ln in text.splitlines()]

    # Находим все индексы заголовков таблиц состава
    header_idxs = [i for i, ln in enumerate(lines) if HEADER_RE.search(ln)]
    # Если не нашли — попробуем мягче
    if not header_idxs:
        header_idxs = [i for i, ln in enumerate(lines) if ("Поз" in ln and "Фамилия" in ln)]

    home, away = [], []
    if header_idxs:
        # первая таблица — дом
        start = header_idxs[0]
        # ищем следующий заголовок (гости) или ограничитель
        next_start = None
        for j in range(start+1, min(start+80, len(lines))):
            if HEADER_RE.search(lines[j]) or re.search(r"(?i)^вратари|^матч|^судьи", lines[j]):
                next_start = j; break
        block1 = lines[start+1 : (next_start if next_start else start+80)]
        home = _parse_lineup_table(block1)

        # вторая таблица — гости (если есть)
        if next_start:
            start2 = next_start
            next2 = None
            for j in range(start2+1, min(start2+80, len(lines))):
                if HEADER_RE.search(lines[j]) or re.search(r"(?i)^вратари|^матч|^судьи", lines[j]):
                    next2 = j; break
            block2 = lines[start2+1 : (next2 if next2 else start2+80)]
            away = _parse_lineup_table(block2)

    return {"home": home, "away": away}

# === optional: fuzzy for goalies ===
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
            nm = gk.get("name")
            if not nm: continue
            best = process.extractOne(nm, dict_players, scorer=fuzz.WRatio, score_cutoff=87)
            if best: gk["name"] = best[0]

# === FastAPI ===
app = FastAPI(title="KHL PDF OCR", version="2.7.0")

@app.get("/")
def health():
    return {"ok": True, "service": "khl-pdf-ocr", "version": "2.7.0"}

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
    try:
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
    except Exception as e:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "OCR", "status": 500, "error": str(e)}, status_code=500)

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
    try:
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

        text, _ = pdf_to_text_pref_words(b, dpi=dpi, scale=scale, bin_thresh=bin_thresh, max_pages=max_pages)
        data: Dict[str, Any] = {}

        if target in ("goalies","all"):
            gk = extract_goalies_from_text(text)
            data["goalies"] = gk
            data["diag_goalies"] = "text" if (gk.get("home") or gk.get("away")) else "nothing_found_text"

        if target in ("refs","all"):
            data["refs"] = extract_refs_from_text(text)

        if target in ("lineups","all"):
            data["lineups"] = extract_lineups_from_text(text)

        if data.get("goalies"):
            _fuzzy_fix_goalies(data)

        dur = round(time.perf_counter() - t0, 3)
        return {
            "ok": True, "match_id": match_id, "season": season, "source_pdf": pdf_url,
            "pdf_len": len(b), "dpi": dpi, "pages_ocr": max_pages, "dur_total_s": dur,
            "data": data, "tried": tried
        }
    except Exception as e:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season, "step": "EXTRACT", "status": 500, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
