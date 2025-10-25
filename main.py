# main.py
# KHL-OCR_Core_v1.0.0 (Render) — v2.6.0r2
# ✔ PyMuPDF get_text("words") с разбиением на левую/правую колонку
# ✔ Очистка: ударения/диакритики, NBSP, двойные пробелы, тонкие дефисы
# ✔ Табличные регексы (goalies/lineups/refs)
# ✔ Киперы: сохраняем букву "С/Р" во входной строке, а статус кладём в gk_status; name без буквы
# ✔ Fallback OCR (pytesseract rus+eng) только если «живого» текста недостаточно
# ✔ fetch_pdf_bytes с браузерными заголовками, httpx(http2=False), urllib fallback, tried[]
# ✔ /ocr и /extract возвращают tried[]

import os
import re
import io
import sys
import time
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

# -------- Defaults / ENV --------
DEFAULT_DPI = int(os.getenv("OCR_DPI", "300"))
DEFAULT_SCALE = float(os.getenv("OCR_SCALE", "1.6"))
DEFAULT_BIN_THRESH = int(os.getenv("OCR_BIN_THRESH", "185"))
DEFAULT_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "2"))
PLAYERS_CSV = os.getenv("PLAYERS_CSV")  # опционально
REFEREES_CSV = os.getenv("REFEREES_CSV")  # опционально

# -------- Helpers --------
def _strip_accents(s: str) -> str:
    # Убираем ударения/диакритику
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _normalize_spaces(s: str) -> str:
    s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def _clean_text(s: str) -> str:
    s = _strip_accents(s)
    s = s.replace("\u00ad", "-")  # soft hyphen
    s = _normalize_spaces(s)
    return s

# -------- PDF fetcher --------
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
            season, mid6, mid = m.groups()
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

    # httpx (без http2, чтобы не требовать h2)
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

# -------- Imaging / OCR --------
def _binarize(img: Image.Image, threshold: int) -> Image.Image:
    return img.convert("L").point(lambda p: 255 if p > threshold else 0, mode="1").convert("L")

def _preprocess_for_ocr(pix: "fitz.Pixmap", scale: float, bin_thresh: int) -> Image.Image:
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if scale and scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.25)
    img = _binarize(img, bin_thresh)
    return img

# -------- Text extraction (words + columns) --------
def _words_to_lines_by_columns(doc: "fitz.Document", max_pages: int) -> Tuple[str, int]:
    """
    Разбираем через get_text("words"), бьём на 2 колонки (лев/прав) по медиане X,
    внутри колонки сортируем по Y/X и собираем строки.
    Возвращает (text, pages_used). Если текста получилось мало — вернём его как есть.
    """
    pages_to_process = min(len(doc), max_pages)
    all_lines: List[str] = []

    for i in range(pages_to_process):
        page = doc.load_page(i)
        words = page.get_text("words")  # [x0,y0,x1,y1,text,block,line,word]
        if not words:
            continue

        # нормализуем текст токенов
        tokens = []
        xs = []
        for (x0, y0, x1, y1, wtext, bno, lno, wno) in words:
            t = _clean_text(wtext)
            if not t:
                continue
            tokens.append((x0, y0, x1, y1, t))
            xs.append(x0)

        if not tokens:
            continue

        # медианный X для грубого разделения колонок
        xs_sorted = sorted(xs)
        midx = xs_sorted[len(xs_sorted) // 2]

        left = [t for t in tokens if t[0] <= midx]
        right = [t for t in tokens if t[0] > midx]

        def build_lines(tok_list: List[Tuple[float, float, float, float, str]]) -> List[str]:
            if not tok_list:
                return []
            tok_list.sort(key=lambda z: (round(z[1] / 8), z[0]))  # кластеризация по Y, потом по X
            lines = []
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
            # финальная чистка
            lines = [_normalize_spaces(ln) for ln in lines if ln and ln.strip()]
            return lines

        left_lines = build_lines(left)
        right_lines = build_lines(right)

        # Сохраняем порядок: левая колонка затем правая
        if left_lines:
            all_lines.extend(left_lines)
        if right_lines:
            all_lines.extend(right_lines)

    text = "\n".join(all_lines).strip()
    return text, pages_to_process

def pdf_to_text_pref_words(pdf_bytes: bytes, dpi: int, scale: float, bin_thresh: int, max_pages: int) -> Tuple[str, int]:
    """
    1) Пытаемся собрать текст через words+columns
    2) Если слишком мало текста (< 300 символов), делаем OCR tesseract (psm=4)
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text, used = _words_to_lines_by_columns(doc, max_pages=max_pages)
        if len(text) >= 300:
            return text, used

        # OCR fallback постранично
        ocr_parts: List[str] = []
        pages_to_process = min(len(doc), max_pages)
        for i in range(pages_to_process):
            page = doc.load_page(i)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = _preprocess_for_ocr(pix, scale=scale, bin_thresh=bin_thresh)
            txt = pytesseract.image_to_string(
                img,
                lang="rus+eng",
                config="--oem 1 --psm 4 -c preserve_interword_spaces=1"
            )
            ocr_parts.append(_clean_text(txt))
        ocr_text = "\n".join(ocr_parts).strip()
        return (ocr_text or text), pages_to_process

# -------- Structured extractors --------
NAME_RE = r"[А-ЯЁA-Z][а-яёa-z'\-\.]+(?:\s+[А-ЯЁA-Z][а-яёa-z'\-\.]+){0,2}"

def extract_refs_from_text(text: str) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    # Ищем блок с ключевыми словами
    m = re.search(r"(Судьи|Referees)[:\s]*\n?(.{0,300})", text, flags=re.I | re.S)
    if m:
        chunk = _normalize_spaces(m.group(2))
        parts = re.split(r"[;,\u2013\-•]+", chunk)
        for c in parts:
            c = c.strip()
            if len(c) < 3:
                continue
            nm = re.findall(NAME_RE, c)
            for n in nm:
                role = "Unknown"
                if re.search(r"лайнсмен|linesman", c, re.I):
                    role = "Linesman"
                if re.search(r"судья|referee", c, re.I):
                    role = "Referee"
                refs.append({"name": n, "role": role})
    # dedup
    out, seen = [], set()
    for r in refs:
        k = (r["name"].lower(), r["role"])
        if k not in seen:
            out.append(r)
            seen.add(k)
    return out

def _detect_gk_status(raw: str) -> Tuple[str, Optional[str]]:
    """
    Оставляем исходную строку имени (чтобы не терять символы),
    но возвращаем (name_without_flag, gk_status) на основе 'С'/'Р'/'scratch'
    """
    s = raw
    status = None
    if re.search(r"(^|\W)[СC](\W|$)", s):
        status = "starter"
    elif re.search(r"(^|\W)[РP](\W|$)", s):
        status = "reserve"
    if re.search(r"scratch", s, re.I):
        status = "scratch" if status is None else status
    # выкинем одиночные С/Р в скобках/отдельным токеном
    name = re.sub(r"(^|\s)[СРCP](?=\s|$)|\([СРCP]\)", " ", s)
    name = _normalize_spaces(name)
    return name, status

def extract_goalies_from_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    res = {"home": [], "away": []}
    # Выцепим два соседних блока "Вратари"
    blocks = list(re.finditer(r"Вратари[^\n]*\n(.{0,400})", text, flags=re.I | re.S))
    def parse_block(block_text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
        for ln in lines[:8]:
            # варианты формата: "60 | В | Бочаров Иван  С 18.05.1995"
            # или "Фукале Зак (С)" и пр.
            # вырежем левую "табличную" часть с номером/позициями
            clean = re.sub(r"^\d{1,2}\s*\|\s*[ВДНFG]\s*\|\s*", "", ln)
            # заберём ФИО
            m = re.search(NAME_RE, clean)
            if not m:
                continue
            raw_name = m.group(0)
            name, gk_status = _detect_gk_status(clean)
            # ещё раз уточним имя после удаления буквы статуса
            m2 = re.search(NAME_RE, name)
            if not m2:
                continue
            final_name = m2.group(0)
            out.append({"name": final_name, "gk_status": gk_status or "unknown"})
        return out

    if blocks:
        res["home"] = parse_block(blocks[0].group(1))
        if len(blocks) > 1:
            res["away"] = parse_block(blocks[1].group(1))
    return res

def extract_lineups_from_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Базовый парсер ростеров через табличные строки "№ | Поз | ФИО ..."
    """
    LINE_RE = re.compile(
        r"(?P<num>\d{1,2})\s*\|\s*(?P<pos>[ВДНFG])\s*\|\s*(?P<name>"+NAME_RE+r")",
        re.U
    )
    # Разделим документ на крупные блоки по ключевым заголовкам
    chunks = re.split(r"Составы команд|СОСТАВЫ|РОСТЕР|LINEUP|ЛАДА|АК БАРС|ДИНАМО|СПАРТАК|ТРАКТОР|СИБИРЬ|НЕФТЕХИМИК|СОЧИ|КУНЬЛУНЬ|АВТОМОБИЛИСТ|АМУР|МЕТАЛЛУРГ|ТОРПЕДО|ЛОКОМОТИВ|САЛАВАТ|ЙУЛАЕВ|БАРЫС|СЕВЕРСТАЛЬ",
                      text, flags=re.I)
    buckets: List[List[Dict[str, Any]]] = []
    for ch in chunks:
        lst: List[Dict[str, Any]] = []
        for m in LINE_RE.finditer(ch):
            d = m.groupdict()
            pos_map = {"В":"G","Д":"D","Н":"F","F":"F","G":"G","D":"D"}
            lst.append({
                "num": int(d["num"]),
                "pos": pos_map.get(d["pos"], d["pos"]),
                "name": _normalize_spaces(d["name"])
            })
        if lst:
            buckets.append(lst)

    lineup = {"home": [], "away": []}
    if buckets:
        lineup["home"] = buckets[0]
        if len(buckets) > 1:
            lineup["away"] = buckets[1]
    return lineup

# -------- FastAPI --------
app = FastAPI(title="KHL PDF OCR", version="2.6.0r2")

@app.get("/")
def health():
    return {"ok": True, "service": "khl-pdf-ocr", "version": "2.6.0r2"}

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
        return JSONResponse({
            "ok": False, "match_id": match_id, "season": season,
            "step": "GET", "status": 404, "tried": tried
        }, status_code=404)

    text, pages_used = pdf_to_text_pref_words(b, dpi=dpi, scale=scale, bin_thresh=bin_thresh, max_pages=max_pages)
    dur = round(time.perf_counter() - t0, 3)
    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": pdf_url,
        "pdf_len": len(b),
        "dpi": dpi,
        "pages_ocr": pages_used,
        "dur_total_s": dur,
        "text_len": len(text),
        "snippet": text[:800],
        "tried": tried
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
        return JSONResponse({
            "ok": False, "match_id": match_id, "season": season,
            "step": "GET", "status": 404, "tried": tried
        }, status_code=404)

    text, pages_used = pdf_to_text_pref_words(b, dpi=dpi, scale=scale, bin_thresh=bin_thresh, max_pages=max_pages)

    data: Dict[str, Any] = {}
    if target in ("refs", "all"):
        data["refs"] = extract_refs_from_text(text)
    if target in ("goalies", "all"):
        data["goalies"] = extract_goalies_from_text(text)
    if target in ("lineups", "all"):
        data["lineups"] = extract_lineups_from_text(text)

    dur = round(time.perf_counter() - t0, 3)
    return {
        "ok": True,
        "match_id": match_id,
        "season": season,
        "source_pdf": pdf_url,
        "pdf_len": len(b),
        "dpi": dpi,
        "pages_ocr": pages_used,
        "dur_total_s": dur,
        "data": data,
        "tried": tried
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
