# main.py
# KHL-OCR_Core_v1.0.0 (Render) — v3.0.0
# Refs: поддержка "Фамилия И.О."
# Lineups: layout-парсер по колонкам (лев/прав), строгие якоря и фильтры шума.

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

# === words->lines (для OCR fallback и текстовых парсеров) ===
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
        # fallback OCR
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

# === общие паттерны ===
NAME_WORD   = r"[А-ЯЁ][а-яё'\-\.]{2,}"
NAME_SEQ    = rf"{NAME_WORD}(?:\s+{NAME_WORD}){{1,2}}"
# Фамилия + инициалы: "Иванов И.И." или "Иванов И. И."
SURNAME_INIT = r"[А-ЯЁ][а-яё'\-]{2,}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\."
DATE_RE   = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
POS_MAP   = {"В":"G","Д":"D","З":"D","Н":"F","F":"F","G":"G","D":"D"}
# позиция только в начале строки (после необязательного номера) или как отдельный токен/разделитель
POS_START_RE = re.compile(r"^\s*(?:(?P<num>\d{1,2})\s*(?:[|:;,\-–]|\s{2,})\s*)?(?P<pos>[ВДЗНFG])(?:\b|(?=[|:;,\-–]|\s{2,}))", re.I)

STATUS_S_RE  = re.compile(r"(^|\W)[СC](\W|$)|\((?:С|C)\)", re.I)
STATUS_R_RE  = re.compile(r"(^|\W)[РP](\W|$)|\((?:Р|P)\)", re.I)

ARENA_NOISE_RE = re.compile(r"(?i)\b(Арена|Дворец|Ледовый|Начало|Матч|Октября|Ноября|Декабря|Стадион|ЛДС)\b")
STOP_TOKENS_RE = re.compile(r"(?i)вратари|судьи|referees|официальные лица|матч")

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
        if DATE_RE.search(ln) and len(name_buf) >= 2:
            flush(); continue
        st = _status_from_line(ln)
        if st: status_buf = status_buf or st

        words = [w for w in re.findall(NAME_WORD, ln)
                 if w.lower() not in ("вратари","вратарь","звено","составы","referees","судьи")]
        words = [w for w in words if not re.fullmatch(r"[А-ЯЁ]\.", w)]
        if POS_START_RE.search(ln) and len(name_buf) >= 2:
            flush()
        for w in words:
            if len(name_buf) < 3: name_buf.append(w)

    if len(name_buf) >= 2: flush()
    return res[:6]

def extract_goalies_from_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    lines = [ln.strip() for ln in text.splitlines()]
    idxs = [i for i, ln in enumerate(lines) if re.search(r"(?i)^вратари|^вратарь", ln)]
    res = {"home": [], "away": []}
    if not idxs: return res
    def slice_block(start_idx: int) -> List[str]:
        blk: List[str] = []
        for ln in lines[start_idx+1 : start_idx+150]:
            if STOP_TOKENS_RE.search(ln): break
            blk.append(ln)
        return blk
    home_blk = slice_block(idxs[0]); home = _extract_goalies_from_text_block(home_blk)
    away = []
    if len(idxs) > 1:
        away_blk = slice_block(idxs[1]); away = _extract_goalies_from_text_block(away_blk)
    if not away and home:
        tail = lines[idxs[0]+1+len(home_blk) : idxs[0]+1+len(home_blk)+100]
        away  = _extract_goalies_from_text_block(tail)
    return {"home": home, "away": away}

# --- REFS (поддержка Фамилия И.О.) ---
REF_HEAD = re.compile(
    r"(?im)^(Судьи(?:\s*матча)?|Официальные лица|Главные судьи|Главный судья|Линейные судьи|Referees|Officials|Linesmen)\s*:?\s*$"
)
def extract_refs_from_text(text: str) -> List[Dict[str, str]]:
    lines = [ln.strip() for ln in text.splitlines()]
    refs: List[Dict[str, str]] = []

    def collect_from_chunk(chunk: str):
        chunk = _normalize_spaces(chunk)
        parts = re.split(r"[;,•\-\u2013]+|\s{2,}", chunk)
        for p in parts:
            p = _normalize_spaces(p)
            if not p: continue
            names = []
            names += re.findall(NAME_SEQ, p)         # Иванов Сергей
            names += re.findall(SURNAME_INIT, p)     # Иванов С.С.
            if not names: continue
            role = "Unknown"
            if re.search(r"лайнсмен|linesman", p, re.I): role = "Linesman"
            if re.search(r"судья|referee|official", p, re.I): role = "Referee"
            for n in names:
                refs.append({"name": n, "role": role})

    # 1) По якорям
    for i, ln in enumerate(lines):
        if REF_HEAD.fullmatch(ln):
            chunk = " ".join(lines[i+1:i+25])  # расширенное окно
            collect_from_chunk(chunk)
            break

    # 2) Глобальный fallback
    if not refs:
        for s in lines:
            if not re.search(r"судья|referee|лайнсмен|linesman|official", s, re.I):
                continue
            collect_from_chunk(s)

    # dedup
    out, seen = [], set()
    for r in refs:
        k = (r["name"].lower(), r["role"])
        if k not in seen:
            out.append(r); seen.add(k)
    return out[:8]

# --- LINEUPS (LAYOUT по колонкам) ---
HEADER_STRICT_RE = re.compile(r"(?i)\b(?:No|№)\b.*\bПоз\b.*\bФамилия\b")
HEADER_SOFT_RE   = re.compile(r"(?i)\bПоз\b.*\bФамилия\b")

def _group_lines(tokens: List[Tuple[float,float,float,float,str]]) -> List[List[Tuple[float,float,float,float,str]]]:
    """Группируем токены в строки по близости y."""
    if not tokens: return []
    tokens = sorted(tokens, key=lambda z: (round(z[1]/6), z[0]))
    rows = []
    cur_y = None
    cur = []
    for (x0,y0,x1,y1,t) in tokens:
        if cur_y is None:
            cur_y = y1; cur = [(x0,y0,x1,y1,t)]
        else:
            if abs(y0 - cur_y) > 6:
                rows.append(cur)
                cur = [(x0,y0,x1,y1,t)]
                cur_y = y1
            else:
                cur.append((x0,y0,x1,y1,t))
                cur_y = max(cur_y, y1)
    if cur: rows.append(cur)
    return rows

def _row_text(row: List[Tuple[float,float,float,float,str]]) -> str:
    row = sorted(row, key=lambda z: z[0])
    return _normalize_spaces(" ".join([t for *_xy, t in row]))

def _parse_row_to_player(row_txt: str) -> Optional[Dict[str, Any]]:
    # отсекаем шум "Арена/Начало/Матч/дата"
    if ARENA_NOISE_RE.search(row_txt) or DATE_RE.search(row_txt):
        return None
    mps = POS_START_RE.match(row_txt)
    if not mps:
        return None
    num = mps.group("num")
    pos = mps.group("pos").upper()
    after = row_txt[mps.end():].strip()
    # имя: допускаем "Фамилия Имя [Отчество]" или "Фамилия И.О."
    m_name = re.search(rf"({NAME_SEQ}|{SURNAME_INIT})", after)
    if not m_name:
        return None
    name = m_name.group(0)
    return {
        "num": int(num) if num else None,
        "pos": POS_MAP.get(pos, pos),
        "name": name
    }

def extract_lineups_layout(pdf_bytes: bytes, max_pages: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    out = {"home": [], "away": []}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = min(len(doc), max_pages)
        for p in range(pages):
            page = doc.load_page(p)
            words = page.get_text("words")
            if not words: continue
            tokens = []
            for (x0,y0,x1,y1,t, *_rest) in words:
                tt = _clean_text(t)
                if tt: tokens.append((x0,y0,x1,y1,tt))
            if not tokens: continue
            xs = sorted([t[0] for t in tokens])
            midx = xs[len(xs)//2] if xs else 0.0
            left  = [t for t in tokens if t[0] <= midx]
            right = [t for t in tokens if t[0] >  midx]

            def extract_side(col_tokens):
                rows = _group_lines(col_tokens)
                # найдём строку заголовка
                hdr_idx = None
                for i, row in enumerate(rows):
                    txt = _row_text(row)
                    if HEADER_STRICT_RE.search(txt) or HEADER_SOFT_RE.search(txt):
                        hdr_idx = i; break
                if hdr_idx is None:
                    return []
                data_rows = rows[hdr_idx+1:]
                # режем по стоп-якорям
                clean_rows = []
                for r in data_rows:
                    t = _row_text(r)
                    if STOP_TOKENS_RE.search(t):
                        break
                    clean_rows.append(r)
                # в игрока
                players = []
                for r in clean_rows:
                    pt = _row_text(r)
                    pl = _parse_row_to_player(pt)
                    if pl: players.append(pl)
                return players

            home_side = extract_side(left)
            away_side = extract_side(right)
            # дополняем, но не дублируем
            if home_side: out["home"] += home_side
            if away_side: out["away"] += away_side

    # dedup/trim
    def dedup(lst):
        seen = set()
        res = []
        for it in lst:
            k = (it.get("num"), it["pos"], it["name"].lower())
            if k in seen: continue
            seen.add(k); res.append(it)
        return res[:30]
    out["home"] = dedup(out["home"])
    out["away"] = dedup(out["away"])
    return out

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
app = FastAPI(title="KHL PDF OCR", version="3.0.0")

@app.get("/")
def health():
    return {"ok": True, "service": "khl-pdf-ocr", "version": "3.0.0"}

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
            return JSONResponse({"ok": False, "match_id": match_id, "season": season,
                                 "step": "GET", "status": 404, "tried": tried}, status_code=404)

        text, pages_used = pdf_to_text_pref_words(b, dpi=dpi, scale=scale,
                                                  bin_thresh=bin_thresh, max_pages=max_pages)
        dur = round(time.perf_counter() - t0, 3)
        return {
            "ok": True, "match_id": match_id, "season": season, "source_pdf": pdf_url,
            "pdf_len": len(b), "dpi": dpi, "pages_ocr": pages_used, "dur_total_s": dur,
            "text_len": len(text), "snippet": text[:800], "tried": tried
        }
    except Exception as e:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season,
                             "step": "OCR", "status": 500, "error": str(e)}, status_code=500)

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
            return JSONResponse({"ok": False, "match_id": match_id, "season": season,
                                 "step": "GET", "status": 404, "tried": tried}, status_code=404)

        text, _ = pdf_to_text_pref_words(b, dpi=dpi, scale=scale,
                                         bin_thresh=bin_thresh, max_pages=max_pages)

        data: Dict[str, Any] = {}

        if target in ("goalies","all"):
            gk = extract_goalies_from_text(text)
            data["goalies"] = gk
            data["diag_goalies"] = "text" if (gk.get("home") or gk.get("away")) else "nothing_found_text"

        if target in ("refs","all"):
            data["refs"] = extract_refs_from_text(text)

        if target in ("lineups","all"):
            data["lineups"] = extract_lineups_layout(b, max_pages=max_pages)

        if data.get("goalies"):
            _fuzzy_fix_goalies(data)

        dur = round(time.perf_counter() - t0, 3)
        return {
            "ok": True, "match_id": match_id, "season": season, "source_pdf": pdf_url,
            "pdf_len": len(b), "dpi": dpi, "pages_ocr": max_pages, "dur_total_s": dur,
            "data": data, "tried": tried
        }
    except Exception as e:
        return JSONResponse({"ok": False, "match_id": match_id, "season": season,
                             "step": "EXTRACT", "status": 500, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
