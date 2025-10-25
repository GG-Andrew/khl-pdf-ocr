import sys
import io
import unicodedata
from typing import Dict, List, Tuple, Any

import httpx
import fitz  # PyMuPDF
import regex as re
from fastapi import FastAPI, Query

app = FastAPI(title="KHL PDF OCR Server", version="3.0.0")

# ---------------------------
# НОРМАЛИЗАЦИЯ ТЕКСТА (снимаем ударения/диакритику, латиницу-псевдокириллицу)
# ---------------------------
ACCENT_TABLE = dict.fromkeys(
    c for c in range(sys.maxunicode)
    if unicodedata.category(chr(c)) == "Mn"
)

def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFKD", s).translate(ACCENT_TABLE))

def normalize_khl_text(s: str) -> str:
    if not s:
        return s
    s = strip_accents(s)
    s = s.replace("Ё", "Е").replace("ё", "е")
    # латиница-псевдокириллица → кириллица (минимально нужный набор)
    s = (s.replace("A","А").replace("B","В").replace("E","Е").replace("K","К")
           .replace("M","М").replace("H","Н").replace("O","О").replace("P","Р")
           .replace("C","С").replace("T","Т").replace("X","Х"))
    # вычистим пробелы/разделители
    s = re.sub(r"[ \t]*\|[ \t]*", " | ", s)
    s = re.sub(r"[^\S\r\n]+", " ", s)  # множественные пробелы → один
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

# ---------------------------
# ЗАГРУЗКА PDF
# ---------------------------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/127.0.0.1 Safari/537.36")

def fetch_pdf_bytes(url: str) -> Tuple[bytes, List[str], int]:
    tried = []
    code = 0
    headers = {
        "User-Agent": UA,
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
    variants.append(url.replace("https://", "http://"))

    try:
        with httpx.Client(http2=True, timeout=30.0, headers=headers, follow_redirects=True) as client:
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

# ---------------------------
# ИЗВЛЕЧЕНИЕ ТЕКСТ-СЛОЯ (без OCR — быстро и стабильно)
# ---------------------------
def text_from_pdf_first_page(pdf: bytes) -> str:
    doc = fitz.open(stream=pdf, filetype="pdf")
    if doc.page_count == 0:
        return ""
    # старт-протокол — всегда первая страница
    page = doc[0]
    # sort=True — стабилизирует порядок строк
    return page.get_text("text", sort=True) or ""

# ---------------------------
# ПАРСЕРЫ
# ---------------------------
def parse_refs(t: str) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    # Главные судьи
    for m in re.finditer(r"(Главный судья)\s+([А-ЯЁA-Z][а-яёa-z]+)\s+([А-ЯЁA-Z][а-яёa-z]+)", t):
        refs.append({"role": m.group(1), "name": f"{m.group(2)} {m.group(3)}"})
    # Линейные судьи
    for m in re.finditer(r"(Линейный судья)\s+([А-ЯЁA-Z][а-яёa-z]+)\s+([А-ЯЁA-Z][а-яёa-z]+)", t):
        refs.append({"role": m.group(1), "name": f"{m.group(2)} {m.group(3)}"})
    return refs

def parse_goalies(t: str) -> Dict[str, List[Dict[str, str]]]:
    out = {"home": [], "away": []}

    # Выделим левый/правый блоки примерно (названия команд могут отличаться, берём «Вратари» с контекстом)
    left = re.search(r"(ЛАДА|АК БАРС|АВТОМОБИЛИСТ|.*?)(?s).*?Вратари(.*?)(Звено|Главный тренер|---RIGHT---)", t)
    right = re.search(r"(ДИНАМО|СКА|СПАРТАК|.*?)(?s).*?Вратари(.*?)(Звено|Главный тренер|Резервный|$)", t)

    def grab(block, side):
        if not block:
            return
        chunk = block.group(2)
        # № В Фамилия Имя [опц. метка С/Р]
        for m in re.finditer(r"\b(\d{1,2})\s*[| ]\s*В\s*[| ]\s*([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([СР]))?", chunk):
            name = f"{m.group(2)} {m.group(3)}"
            status = {"С": "starter", "Р": "reserve"}.get((m.group(4) or "").upper(), "scratch")
            out[side].append({"name": name, "status": status})

        # fallback: без вертикалей
        for m in re.finditer(r"\b(\d{1,2})\s+В\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([СР]))?", chunk):
            name = f"{m.group(2)} {m.group(3)}"
            status = {"С": "starter", "Р": "reserve"}.get((m.group(4) or "").upper(), "scratch")
            if {"name": name, "status": status} not in out[side]:
                out[side].append({"name": name, "status": status})

    grab(left, "home")
    grab(right, "away")
    return out

def parse_lineups(t: str) -> Dict[str, List[Dict[str, Any]]]:
    out = {"home": [], "away": []}
    # общий стабильный шаблон строк
    row_re = re.compile(
        r"\b(?P<num>\d{1,2})\s*[| ]\s*(?P<pos>[ВЗН])\s*[| ]\s*(?P<last>[А-ЯЁ][а-яё]+)\s+(?P<first>[А-ЯЁ][а-яё\.]+)"
        r"(?:\s+(?P<capt>[АК]))?(?:\s+\*\s*)?"
        r"(?:\s*(?P<dob>\d{2}\.\d{2}\.\d{4}))?(?:\s+(?P<age>\d{2}))?",
    )

    # разделим текст на левую/правую часть по "---RIGHT---" (если есть)
    parts = re.split(r"\n+---RIGHT---\n+", t)
    left_text = parts[0]
    right_text = parts[1] if len(parts) > 1 else ""

    def collect(side_text: str, side: str):
        for m in row_re.finditer(side_text):
            d = m.groupdict()
            item = {
                "side": side,
                "num": d["num"],
                "pos": d["pos"],
                "name": f"{d['last']} {d['first'].rstrip('.')}",
                "capt": (d.get("capt") or "").upper(),
            }
            if d.get("dob"):
                item["dob"] = d["dob"]
            if d.get("age"):
                item["age"] = d["age"]
            # признак кипера по статусной букве (если в имени прилепилось)
            if item["pos"] == "В":
                # вытащим возможную букву С/Р из конца строки (иногда остаётся в имени)
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

# ---------------------------
# ЭНДПОИНТЫ
# ---------------------------
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
    # Здесь сохраняем совместимость формата ответа,
    # но извлекаем только текст-слой без OCR (быстро и стабильно)
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
        # вместо 500 — всегда JSON с ошибкой
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
            "has_zveno": ("Звено" in norm),
            "has_refs": ("Главный судья" in norm or "Линейный судья" in norm),
        }
        return {"ok": True, "match_id": match_id, "season": season, "snippet": norm[:800], "markers": markers, "tried": tried}
    except Exception as e:
        return {"ok": False, "match_id": match_id, "season": season, "step": "DEBUG", "status": 500, "error": str(e)}
