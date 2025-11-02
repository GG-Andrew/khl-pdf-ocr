# app.py — KHL PDF OCR/WORDS API (Render)
# эндпоинты:
#  GET /health
#  GET /ocr?url=<pdf-url> | /ocr?season=1369&uid=897689
#  GET /extract?season=1369&uid=897689&mode=words|ocr&dpi=300

import os
import io
import re
import time
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, request, jsonify, Response

import requests
import cloudscraper

# pdf → text (вариант 1: words)
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTChar, LTAnno, LTTextLine

# pdf → OCR (вариант 2: ocr)
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

app = Flask(__name__)

# -----------------------------
# 1) Грузим PDF с khl.ru с обходом 403
# -----------------------------
def fetch_pdf_bytes(season: str, uid: str) -> bytes:
    """
    Тянем PDF с khl.ru, обходя 403 через Cloudflare:
    1) cloudscraper с хромовскими заголовками
    2) fallback на requests
    3) поддержка PROXY_URL (если нужно)
    """
    uid = str(uid).strip()
    season = str(season).strip()

    urls = [
        f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf",
        f"https://khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf",
    ]
    referer = f"https://www.khl.ru/game/{season}/{uid}/preview/"

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": referer,
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    proxy_url = os.getenv("PROXY_URL", "").strip()
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    # 1) cloudscraper
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    for u in urls:
        try:
            r = scraper.get(u, headers=headers, timeout=25, proxies=proxies)
            if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
                return r.content
            if r.status_code in (403, 409, 429, 503):
                time.sleep(1.5)
                r = scraper.get(u, headers=headers, timeout=25, proxies=proxies)
                if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
                    return r.content
        except Exception:
            pass  # пойдём на fallback

    # 2) fallback: requests
    for u in urls:
        try:
            r = requests.get(u, headers=headers, timeout=25, proxies=proxies)
            if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
                return r.content
        except Exception:
            continue

    raise RuntimeError("http 403: khl.ru заблокировал загрузку PDF (после всех попыток)")

# -----------------------------
# 2) WORDS: аккуратное чтение текстового слоя
# -----------------------------
def page_words(pdf_bytes: bytes) -> List[str]:
    out: List[str] = []
    for page_layout in extract_pages(io.BytesIO(pdf_bytes)):
        lines: List[str] = []
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                for obj in element:
                    if isinstance(obj, LTTextLine):
                        text = obj.get_text()
                        text = text.replace("\xa0", " ").strip()
                        if text:
                            lines.append(text)
        out.extend(lines)
    return out

# -----------------------------
# 3) OCR (как fallback)
# -----------------------------
def ocr_text(pdf_bytes: bytes, dpi: int = 300) -> str:
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    chunks: List[str] = []
    for img in images[:2]:  # достаточно первых 2 страниц
        txt = pytesseract.image_to_string(img, lang="rus+eng", config="--psm 6")
        chunks.append(txt)
    return "\n".join(chunks)

# -----------------------------
# 4) Черновой парсинг из words/ocr в структуру (минимум — чтобы ты мог видеть прогресс)
#    Это базовый каркас. Тонкую логику можно донастроить после.
# -----------------------------
POS_MAP = {"З": "D", "Н": "F", "В": "G"}
SECT_TITLES = ("Звено 1", "Звено 2", "Звено 3", "Звено 4")

def rough_extract_struct(lines: List[str]) -> Dict[str, Any]:
    data = {
        "home": {"team": "", "goalies": [], "lines": {"1": [], "2": [], "3": [], "4": []}, "bench": []},
        "away": {"team": "", "goalies": [], "lines": {"1": [], "2": [], "3": [], "4": []}, "bench": []},
    }
    refs = {"main": [], "linesmen": []}

    # очень простой хук: выдёргиваем судей из большого блока внизу
    joined = " ".join(lines)
    m = re.search(r"(Главн\w+\s+суд[ьяи].+)$", joined, flags=re.I)
    if m:
        block = m.group(1)
        # главные
        mains = re.findall(r"(?:Главн\w+\s+суд[ьяи])\s*[:\-]?\s*([А-ЯЁA-Z][^,;]+)", block, flags=re.I)
        for x in mains:
            name = re.sub(r"\s{2,}", " ", x).strip().strip(" .,:;")
            if name:
                refs["main"].append(name)
        # линейные
        linesm = re.findall(r"(?:Линейн\w+\s+суд[ьяи])\s*[:\-]?\s*([А-ЯЁA-Z][^,;]+)", block, flags=re.I)
        for x in linesm:
            name = re.sub(r"\s{2,}", " ", x).strip().strip(" .,:;")
            if name:
                refs["linesmen"].append(name)

    # команды
    for s in lines[:15]:
        if "НЕФТЕХИМИК" in s.upper():
            data["home"]["team"] = "НЕФТЕХИМИК НИЖНЕКАМСК"
        if "САЛАВАТ ЮЛАЕВ" in s.upper():
            data["away"]["team"] = "САЛАВАТ ЮЛАЕВ УФА"

    # вратари — ищем блок сверху (3 строки, где "В " и номер)
    for s in lines[:30]:
        ms = re.match(r"^\s*(\d{1,2})\s+В\s+([А-ЯЁа-яёA-Za-z\.\- ]+)", s)
        if ms:
            num = ms.group(1)
            name = ms.group(2).strip()
            if "Самонов" in name or "Вязов" in name or "Мозгов" in name:
                data["away"]["goalies"].append({"number": num, "name": name, "gk_status": ""})
            else:
                data["home"]["goalies"].append({"number": num, "name": name, "gk_status": ""})

    # звенья — черновик
    cur = "home"
    cur_line = "1"
    for s in lines:
        if "Звено 1" in s: cur_line = "1"; continue
        if "Звено 2" in s: cur_line = "2"; continue
        if "Звено 3" in s: cur_line = "3"; continue
        if "Звено 4" in s: cur_line = "4"; continue
        # переключение на away примерно по середине листа
        if "Главный тренер" in s and cur == "home":
            cur = "away"
            cur_line = "1"
            continue

        m = re.match(r"^\s*(\d{1,2})\s+([ЗНВ])\s+(.+?)\s*$", s)
        if m and cur_line in ("1", "2", "3", "4"):
            num = m.group(1)
            pos = POS_MAP.get(m.group(2), "F")
            name = m.group(3)
            name = re.sub(r"\s{2,}", " ", name).strip().strip(" .,:;")
            data[cur]["lines"][cur_line].append({"name": name, "pos": pos, "number": num})

    return data, refs

# -----------------------------
# 5) Helpers
# -----------------------------
def get_pdf_from_query() -> bytes:
    """
    Поддерживает:
      - ?url=... (прямой URL)
      - ?season=1369&uid=897689 (khl-шаблон)
    """
    url = request.args.get("url")
    season = request.args.get("season")
    uid = request.args.get("uid")

    if url:
        # если пришёл прямой url — тянем его (с теми же заголовками/обходом)
        # пробуем угадать season/uid, иначе просто качаем как есть
        m = re.search(r"/pdf/(\d{4})/(\d+)/game-\2-start-ru\.pdf", url)
        if m:
            return fetch_pdf_bytes(m.group(1), m.group(2))
        # общий даунлоад (через cloudscraper)
        headers = {"User-Agent": "Mozilla/5.0"}
        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, headers=headers, timeout=25)
        if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
            return r.content
        raise RuntimeError(f"http {r.status_code}: cannot fetch url")
    if season and uid:
        return fetch_pdf_bytes(season, uid)
    raise RuntimeError("param 'url' or 'season&uid' required")

# -----------------------------
# 6) Endpoints
# -----------------------------
@app.get("/health")
def health() -> Response:
    return jsonify({"ok": True, "engine": "ready"})

@app.get("/ocr")
def route_ocr() -> Response:
    try:
        pdf = get_pdf_from_query()
        dpi = int(request.args.get("dpi", "300"))
        text = ocr_text(pdf, dpi=dpi)
        return jsonify({"ok": True, "engine": "ocr", "text_len": len(text), "snippet": text[:500]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.get("/extract")
def route_extract() -> Response:
    """
    mode:
      - words (по умолчанию) — парсит текстовый слой
      - ocr — OCR fallback
    """
    mode = request.args.get("mode", "words").lower()
    try:
        pdf = get_pdf_from_query()
        if mode == "ocr":
            text = ocr_text(pdf, dpi=int(request.args.get("dpi", "300")))
            lines = [x.strip() for x in text.replace("\xa0", " ").splitlines() if x.strip()]
        else:
            lines = page_words(pdf)

        data, refs = rough_extract_struct(lines)
        return jsonify({
            "ok": True,
            "engine": mode,
            "data": data,
            "referees": refs,
            "source_url": request.args.get("url") or (
                f"https://www.khl.ru/pdf/{request.args.get('season')}/{request.args.get('uid')}/game-{request.args.get('uid')}-start-ru.pdf"
                if request.args.get("season") and request.args.get("uid") else None
            )
        })
    except Exception as e:
        code = 400
        msg = str(e)
        if "403" in msg:
            code = 403
        return jsonify({"ok": False, "error": msg}), code

# -----------------------------
# 7) gunicorn entry
# -----------------------------
if __name__ == "__main__":
    # локальный запуск
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
