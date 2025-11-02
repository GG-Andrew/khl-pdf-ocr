import io, re, time, json
from typing import List, Dict, Any, Tuple, Optional

import httpx
import fitz  # PyMuPDF
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# ---- FastAPI app
app = FastAPI(title="KHL PDF Extractor", version="1.0.0")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/119.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "*/*",
    "Referer": "https://www.khl.ru/",
    "Connection": "keep-alive",
}

# ---------- helpers

def khl_pdf_url(season: int, uid: int) -> str:
    return f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"

async def fetch_bytes(url: str, timeout: float = 15.0) -> bytes:
    async with httpx.AsyncClient(headers=HEADERS, timeout=timeout, follow_redirects=True) as cli:
        r = await cli.get(url)
        r.raise_for_status()
        return r.content

def words_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Текстовый слой через PyMuPDF: пословно (x0,y0,x1,y1,text,page)."""
    doc = fitz.open("pdf", pdf_bytes)
    words = []
    for pno in range(len(doc)):
        page = doc[pno]
        for (x0, y0, x1, y1, text, block_no, line_no, word_no) in page.get_text("words"):
            words.append({"page": pno, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})
    return words

def quick_text(pdf_bytes: bytes) -> str:
    """Грубый текстовый слой для fallback’ов."""
    doc = fitz.open("pdf", pdf_bytes)
    return "\n".join(page.get_text() or "" for page in doc)

def ocr_page_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    return convert_from_bytes(pdf_bytes, dpi=dpi)

def ocr_text(pdf_bytes: bytes, dpi: int = 300) -> str:
    imgs = ocr_page_images(pdf_bytes, dpi=dpi)
    blocks = []
    for im in imgs[:2]:  # хватит первых страниц
        txt = pytesseract.image_to_string(im, lang="rus+eng", config="--psm 6 --oem 1")
        blocks.append(txt)
    return "\n".join(blocks)

# ---------- parsing utils

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s, flags=re.M).strip(" ,.;:—-")

def clean_role_words(name: str) -> str:
    if not name:
        return name
    name = re.sub(r"\b(Главн\w*|Линейн\w*|Линейн\w*|судья|судьи|Резервн\w*)\b", " ", name, flags=re.I)
    return normalize_spaces(name)

def split_columns(words: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Первую страницу делим по вертикали на левую/правую колонку."""
    if not words:
        return [], []
    page0 = [w for w in words if w["page"] == 0 and w["text"].strip()]
    if not page0:
        return [], []
    xs = [w["x0"] for w in page0]
    mid = (min(xs) + max(xs)) / 2.0
    left = [w for w in page0 if w["x0"] < mid]
    right = [w for w in page0 if w["x0"] >= mid]
    left_sorted = " ".join(w["text"] for w in sorted(left, key=lambda z: (round(z["y0"]), z["x0"])))
    right_sorted = " ".join(w["text"] for w in sorted(right, key=lambda z: (round(z["y0"]), z["x0"])))
    return left_sorted.splitlines(), right_sorted.splitlines()

def parse_team_block(lines: List[str]) -> Dict[str, Any]:
    # Название команды — первая крупная строка со словами в ВЕРХНЕМ регистре
    joined = "\n".join(lines)
    team_match = re.search(r"([А-ЯЁA-Z][А-ЯЁA-Z \.-]{6,})\n", joined)
    team = normalize_spaces(team_match.group(1)) if team_match else ""

    # Вратари: строки, где встречается одиночная буква "В" (позиция), или слово "Вратарь"
    goalies = []
    for m in re.finditer(r"(?:^|\n)\s*(\d{1,2})\s*[ВВ]\s+([A-Za-zА-Яа-яЁё.\- ]+)", joined):
        num = m.group(1)
        name = normalize_spaces(m.group(2))
        if name:
            goalies.append({"number": num, "name": name, "gk_status": ""})

    # Попробовать отметить starter/reserve по маркерам "С" / "Р"
    if len(goalies) >= 1:
        goalies[0]["gk_status"] = goalies[0].get("gk_status") or "starter"
    if len(goalies) >= 2 and not goalies[1]["gk_status"]:
        goalies[1]["gk_status"] = "reserve"

    # Звенья 1..4: ловим блоки "Звено N" и тянем 5 последующих игроков
    lines_map = {"1": [], "2": [], "3": [], "4": []}
    for zv in ("1", "2", "3", "4"):
        pat = re.compile(rf"Звено\s*{zv}(.+?)(?:Звено\s*\d|Главн|Линейн|$)", re.S | re.I)
        mm = pat.search(joined)
        if not mm:
            continue
        chunk = mm.group(1)
        # Игроки: номер + фамилия/инициалы, позиция D/F иногда перед номером
        for im in re.finditer(r"(?P<num>\d{1,2})\s*(?P<name>[A-Za-zА-Яа-яЁё.\- ]+)", chunk):
            num = im.group("num")
            name = normalize_spaces(im.group("name"))
            if not name:
                continue
            pos = "F"
            if re.search(r"\bЗ\b|\bD\b| защитник |^З\s", name, re.I):
                pos = "D"
            lines_map[zv].append({"pos": pos, "number": num, "name": name})
            if len(lines_map[zv]) >= 5:  # больше не тянем
                break

    return {"team": team, "goalies": goalies, "lines": lines_map, "bench": []}

def parse_referees(text: str) -> Dict[str, List[str]]:
    block = re.search(r"Судьи.+?(?:Обновлено|$)", text, re.S | re.I)
    main, linesmen = [], []
    if block:
        t = block.group(0)
        # Главные (2 фамилии)
        for m in re.finditer(r"(Главн\w+\s+судья[^A-Za-zА-Яа-яЁё]{0,10})([A-Za-zА-Яа-яЁё.\- ]+)", t, re.I):
            name = clean_role_words(m.group(2))
            if name and name not in main:
                main.append(name)
        # Линейные (2 фамилии)
        for m in re.finditer(r"(Линейн\w+\s+судья[^A-Za-zА-Яа-яЁё]{0,10})([A-Za-zА-Яа-яЁё.\- ]+)", t, re.I):
            name = clean_role_words(m.group(2))
            if name and name not in linesmen:
                linesmen.append(name)
    # если метки не нашлись, попробуем простую табличку нижнего блока
    if not main and not linesmen:
        for m in re.finditer(r"(Главн\w+\s+судья|Линейн\w+\s+судья)\s+([A-Za-zА-Яа-яЁё.\- ]+)", text, re.I):
            role = m.group(1)
            name = clean_role_words(m.group(2))
            if re.search(r"Главн", role, re.I):
                if name not in main:
                    main.append(name)
            else:
                if name not in linesmen:
                    linesmen.append(name)
    return {
        "main": main[:2],
        "linesmen": linesmen[:2]
    }

# ---------- core extraction

def extract_from_words(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    left, right = split_columns(words)
    home = parse_team_block(left)
    away = parse_team_block(right)
    # общий текст для судей
    doc_text = "\n".join([" ".join([w["text"] for w in words if w["page"] == p]) for p in sorted({w["page"] for w in words})])
    refs = parse_referees(doc_text)
    return {"home": home, "away": away}, refs

def extract(pdf_bytes: bytes) -> Tuple[Dict[str, Any], Dict[str, List[str]], str]:
    """Пробуем PyMuPDF, при пустоте падать в OCR."""
    engine = "words"
    words = words_from_pdf(pdf_bytes)
    if not words or sum(1 for w in words if w["text"].strip()) < 50:
        engine = "ocr"
        text = ocr_text(pdf_bytes, dpi=300)
        # для OCR парсим всё «прямым текстом» (без координат)
        refs = parse_referees(text)
        # обе команды — грубо: разделим по «Звено 1»
        pages = text.split("\n")
        left_text = "\n".join(pages)  # у OCR нет колоночной уверенности — вернём пустые звенья
        stub = {"team": "", "goalies": [], "lines": {"1": [], "2": [], "3": [], "4": []}, "bench": []}
        data = {"home": stub, "away": stub}
        return data, refs, engine
    data, refs = extract_from_words(words)
    return data, refs, engine

# ---------- endpoints

@app.get("/health")
def health():
    return {"ok": True, "engine": "ready"}

@app.get("/extract")
async def extract_endpoint(
    uid: Optional[int] = Query(None),
    season: Optional[int] = Query(None),
    url: Optional[str] = Query(None),
):
    t0 = time.time()
    if not url:
        if not (uid and season):
            return JSONResponse({"ok": False, "error": "params", "details": "pass ?url=… or ?season=…&uid=…"}, status_code=400)
        url = khl_pdf_url(season, uid)

    try:
        pdf_bytes = await fetch_bytes(url)
    except httpx.HTTPError as e:
        return JSONResponse({"ok": False, "error": "fetch_failed", "details": str(e), "source_url": url}, status_code=502)

    try:
        data, refs, engine = extract(pdf_bytes)
    except Exception as e:
        # последняя попытка — чистый OCR
        try:
            text = ocr_text(pdf_bytes, dpi=300)
            data = {"home": {"team": "", "goalies": [], "lines": {"1": [], "2": [], "3": [], "4": []}, "bench": []},
                    "away": {"team": "", "goalies": [], "lines": {"1": [], "2": [], "3": [], "4": []}, "bench": []}}
            refs = parse_referees(text)
            engine = "ocr"
        except Exception as e2:
            return JSONResponse({"ok": False, "error": "parse_failed", "details": f"{e} / {e2}", "source_url": url}, status_code=500)

    resp = {
        "ok": True,
        "engine": engine,
        "data": data,
        "referees": refs,
        "referee_entries": (
            [{"role": "Главный судья", "name": n} for n in refs.get("main", [])] +
            [{"role": "Линейный судья", "name": n} for n in refs.get("linesmen", [])]
        ),
        "source_url": url,
        "duration_s": round(time.time() - t0, 3)
    }
    return JSONResponse(resp)
