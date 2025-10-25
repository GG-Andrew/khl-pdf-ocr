import sys
import unicodedata
from typing import Dict, List, Tuple, Any

import httpx
import fitz  # PyMuPDF
import regex as re
from fastapi import FastAPI, Query

app = FastAPI(title="KHL PDF OCR Server", version="3.0.0")

# =========================
# НОРМАЛИЗАЦИЯ ТЕКСТА
# =========================
_ACCENT_TABLE = dict.fromkeys(
    c for c in range(sys.maxunicode)
    if unicodedata.category(chr(c)) == "Mn"
)

def _strip_accents(s: str) -> str:
    # NFKD -> убираем диакритику -> NFC
    return unicodedata.normalize("NFC", unicodedata.normalize("NFKD", s).translate(_ACCENT_TABLE))

def _latin_lookalikes_to_cyr(s: str) -> str:
    # Минимально нужные замены латиницы на визуально близкие кирсимволы
    return (s.replace("A","А").replace("a","а")
             .replace("B","В").replace("E","Е").replace("e","е")
             .replace("K","К").replace("M","М").replace("H","Н")
             .replace("O","О").replace("o","о").replace("P","Р")
             .replace("C","С").replace("c","с").replace("T","Т")
             .replace("X","Х"))

def normalize_khl_text(s: str) -> str:
    if not s:
        return s
    s = _strip_accents(s)
    s = s.replace("Ё", "Е").replace("ё", "е")
    s = _latin_lookalikes_to_cyr(s)
    # выравниваем вертикальные разделители
    s = re.sub(r"[ \t]*\|[ \t]*", " | ", s)
    # схлопываем пробелы
    s = re.sub(r"[^\S\r\n]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    # аккуратнее с переносами: оставим явные блоки
    s = s.replace("\r", "")
    return s.strip()

# =========================
# ЗАГРУЗКА PDF
# =========================
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/127.0 Safari/537.36")

def fetch_pdf_bytes(url: str) -> Tuple[bytes, List[str], int]:
    tried: List[str] = []
    code = 0
    headers = {
        "User-Agent": _UA,
        "Referer": "https://www.khl.ru/",
        "Accept": "application/pdf,application/*;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru,en;q=0.9",
        "Connection": "keep-alive",
    }
    variants = [url]
    if "www.khl.ru" in url:
        variants.append(url.replace("www.khl.ru", "khl.ru"))
    if url.endswith("-start-ru.pdf"):
        variants.extend([
            url.replace("-start-ru.pdf", "-start.pdf"),
            url.replace("-start-ru.pdf", "-start-en.pdf"),
            url.replace("-start-ru.pdf", "-protocol-ru.pdf"),
            url.replace("-start-ru.pdf", "-official-ru.pdf"),
        ])
    # на всякий случай http
    if url.startswith("https://"):
        variants.append(url.replace("https://", "http://"))

    try:
        # ВАЖНО: http2=False, чтобы не требовать пакет h2
        with httpx.Client(http2=False, timeout=30.0, headers=headers, follow_redirects=True) as client:
            for v in variants:
                r = client.get(v)
                code = r.status_code
                tried.append(f"{v} [{code}]")
                if code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
                    return r.content, tried, 200
    except Exception as e:
        tried.append(f"httpx err: {e!r}")
        return b"", tried, 599

    return b"", tried, code or 520

# =========================
# ТЕКСТ ИЗ PDF (первая страница)
# =========================
def text_from_pdf_first_page(pdf: bytes) -> str:
    doc = fitz.open(stream=pdf, filetype="pdf")
    if doc.page_count == 0:
        return ""
    page = doc[0]
    # sort=True для стабильного порядка строк
    return page.get_text("text", sort=True) or ""

# =========================
# ПАРСЕРЫ
# =========================
def parse_refs(t: str) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []

    # Главные судьи
    for m in re.finditer(r"(Главный судья)\s+([А-ЯЁA-Z][а-яёa-z]+)\s+([А-ЯЁA-Z][а-яёa-z]+)", t):
        refs.append({"role": m.group(1), "name": f"{m.group(2)} {m.group(3)}"})
    # Линейные судьи
    for m in re.finditer(r"(Линейный судья)\s+([А-ЯЁA-Z][а-яёa-z]+)\s+([А-ЯЁA-Z][а-яёa-z]+)", t):
        refs.append({"role": m.group(1), "name": f"{m.group(2)} {m.group(3)}"})

    # Удалим дубликаты, сохраняя порядок
    seen = set()
    uniq = []
    for r in refs:
        key = (r["role"], r["name"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq

def parse_goalies(t: str) -> Dict[str, List[Dict[str, str]]]:
    out = {"home": [], "away": []}

    # Грубое деление на левую/правую части.
    # Часто в текст-слое есть "---RIGHT---". Если нет — попробуем расколоть по словам типа названия второй команды.
    parts = re.split(r"\n+---RIGHT---\n+", t)
    if len(parts) == 1:
        parts = re.split(r"\nДИНАМО|^ДИНАМО| СПАРТАК| ЦСКА| СКА| ЛОКОМОТИВ| АВАНГАРД| СИБИРЬ| СЕВЕРСТАЛЬ| АДМИРАЛ| АВТОМОБИЛИСТ| АК БАРС| АМУР| АТЛАНТ| БАРЫС| ВИТЯЗЬ| НЕФТЕХИМИК| МЕТАЛЛУРГ", t, maxsplit=1, flags=re.M)
    left_text = parts[0]
    right_text = parts[1] if len(parts) > 1 else ""

    def grab(side_text: str, side: str):
        # Вратари блок
        blk = re.search(r"Вратари(.*?)(Звено|Главный тренер|Линейный судья|$)", side_text, flags=re.S)
        if not blk:
            return
        chunk = blk.group(1)
        # Формат с вертикалями: 60 | В | Фамилия Имя [С/Р]
        for m in re.finditer(r"\b(\d{1,2})\s*(?:\|\s*)?В(?:\s*\|)?\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([СР]))?", chunk):
            name = f"{m.group(2)} {m.group(3)}"
            status = {"С": "starter", "Р": "reserve"}.get((m.group(4) or "").upper(), "scratch")
            cand = {"name": name, "status": status}
            if cand not in out[side]:
                out[side].append(cand)

    grab(left_text, "home")
    grab(right_text, "away")
    return out

def parse_lineups(t: str) -> Dict[str, List[Dict[str, Any]]]:
    out = {"home": [], "away": []}
    # Деляем на левую/правую колонку
    parts = re.split(r"\n+---RIGHT---\n+", t)
    if len(parts) == 1:
        parts = re.split(r"\nДИНАМО|^ДИНАМО| СПАРТАК| ЦСКА| СКА| ЛОКОМОТИВ| АВАНГАРД| СИБИРЬ| СЕВЕРСТАЛЬ| АДМИРАЛ| АВТОМОБИЛИСТ| АК БАРС| АМУР| АТЛАНТ| БАРЫС| ВИТЯЗЬ| НЕФТЕХИМИК| МЕТАЛЛУРГ", t, maxsplit=1, flags=re.M)
    left_text = parts[0]
    right_text = parts[1] if len(parts) > 1 else ""

    row_re = re.compile(
        r"\b(?P<num>\d{1,2})\s*(?:\|\s*)?(?P<pos>[ВЗН])(?:\s*\|)?\s+"
        r"(?P<last>[А-ЯЁ][а-яё]+)\s+(?P<first>[А-ЯЁ][а-яё\.]+)"
        r"(?:\s+(?P<capt>[АК]))?"
        r"(?:\s*(?:\*\s*)?)"
        r"(?:\s*(?P<dob>\d{2}\.\d{2}\.\d{4}))?"
        r"(?:\s+(?P<age>\d{2}))?"
    )

    def collect(side_text: str, side: str):
        # ограничимся секциями Звено/Составы до "Главный тренер" — чтобы не ловить судей
        msec = re.search(r"(Составы|Звено|Вратари)(.*?)(Главный тренер|Линейный судья|$)", side_text, flags=re.S)
        body = msec.group(2) if msec else side_text
        for m in row_re.finditer(body):
            d = m.groupdict()
            name = f"{d['last']} {d['first'].rstrip('.')}"
            item: Dict[str, Any] = {
                "side": side,
                "num": d["num"],
                "pos": d["pos"],
                "name": name,
                "capt": (d.get("capt") or "").upper(),
            }
            if d.get("dob"):
                item["dob"] = d["dob"]
            if d.get("age"):
                item["age"] = d["age"]
            if item["pos"] == "В":
                # флаги киперов иногда прилипают к имени
                if item["name"].endswith(" С"):
                    item["gk_flag"] = "S"
                    item["gk_status"] = "starter"
                    item["name"] = item["name"][:-2]
                elif item["name"].endswith(" Р"):
                    item["gk_flag"] = "R"
                    item["gk_status"] = "reserve"
                    item["name"] = item["name"][:-2]
            out[side].append(item)

    collect(left_text, "home")
    collect(right_text, "away")
    return out

# =========================
# ЭНДПОИНТЫ
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "khl-pdf-ocr", "ready": True}

@app.get("/ocr")
def ocr_parse(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    dpi: int = Query(130, ge=120, le=360),
    max_pages: int = Query(1, ge=1, le=5),
    scale: float = Query(1.3, ge=1.0, le=2.0),
    bin_thresh: int = Query(185, ge=120, le=230),
):
    # Совместимый по форме ответ, но без OCR (используем текст-слой)
    try:
        pdf, tried, code = fetch_pdf_bytes(pdf_url)
        if not pdf:
            return {"ok": False, "match_id": match_id, "step": "GET", "status": code or 404, "tried": tried}
        raw = text_from_pdf_first_page(pdf)
        norm = normalize_khl_text(raw)
        return {
            "ok": True,
            "match_id": match_id,
            "pdf_len": len(pdf),
            "dpi": dpi,
            "pages_ocr": 1,
            "dur_total_s": 0.0,
            "dur_ocr_s": 0.0,
            "text_len": len(norm),
            "snippet": norm[:600],
            "tried": tried,
        }
    except Exception as e:
        return {"ok": False, "match_id": match_id, "step": "OCR", "status": 500, "error": str(e)}

@app.get("/extract")
def extract(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(...),
    target: str = Query("all")
):
    try:
        pdf, tried, code = fetch_pdf_bytes(pdf_url)
        if not pdf:
            return {"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": code or 404, "tried": tried}

        raw = text_from_pdf_first_page(pdf)
        norm = normalize_khl_text(raw)

        data: Dict[str, Any] = {}
        if target in ("refs", "all"):
            data["refs"] = parse_refs(norm)
        if target in ("goalies", "all"):
            data["goalies"] = parse_goalies(norm)
        if target in ("lineups", "all"):
            data["lineups"] = parse_lineups(norm)

        return {
            "ok": True,
            "match_id": match_id,
            "season": season,
            "source_pdf": pdf_url,
            "pdf_len": len(pdf),
            "dpi": 130,
            "pages_ocr": 1,
            "dur_total_s": 0.0,
            "data": data,
            "tried": tried
        }
    except Exception as e:
        return {"ok": False, "match_id": match_id, "season": season, "step": "EXTRACT", "status": 500, "error": str(e)}

@app.get("/debug_text")
def debug_text(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(...)
):
    try:
        pdf, tried, code = fetch_pdf_bytes(pdf_url)
        if not pdf:
            return {"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": code or 404, "tried": tried}
        raw = text_from_pdf_first_page(pdf)
        norm = normalize_khl_text(raw)
        markers = {
            "has_vratari": ("Вратари" in norm),
            "has_zveno": ("Звено" in norm or "Составы" in norm),
            "has_refs": ("Главный судья" in norm or "Линейный судья" in norm),
        }
        return {"ok": True, "match_id": match_id, "season": season, "snippet": norm[:900], "markers": markers, "tried": tried}
    except Exception as e:
        return {"ok": False, "match_id": match_id, "season": season, "step": "DEBUG", "status": 500, "error": str(e)}
