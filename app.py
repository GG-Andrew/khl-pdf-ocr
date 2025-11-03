import os, io, time, re, json
from typing import Dict, Any, List

import requests
try:
    import cloudscraper
except Exception:
    cloudscraper = None

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps
import pytesseract

from flask import Flask, request, Response, jsonify

# ----------------------------
# CONFIG
# ----------------------------
# Прокси-воркер Cloudflare, который уже отдаёт PDF без 403:
UPSTREAM_PREFIX = os.getenv("UPSTREAM_PREFIX", "https://pdf2.palladiumgames2d.workers.dev")
# Таймаут сети
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "25"))

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

app = Flask(__name__)


# ----------------------------
# UTILS
# ----------------------------
def j(payload: Dict[str, Any], code: int = 200) -> Response:
    return Response(json.dumps(payload, ensure_ascii=False), status=code, mimetype="application/json")


def build_pdf_url(season: str, uid: str) -> str:
    """
    Всегда формируем URL в формате воркера:
    https://<worker>/khlpdf/{season}/{uid}/game-{uid}-start-ru.pdf
    """
    return f"{UPSTREAM_PREFIX}/khlpdf/{season}/{uid}/game-{uid}-start-ru.pdf"


def _make_session():
    if cloudscraper:
        s = cloudscraper.create_scraper()
    else:
        s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "application/pdf",
        "Referer": "https://www.khl.ru/",
    })
    return s


def fetch_pdf_bytes(season: str, uid: str) -> bytes:
    url = build_pdf_url(season, uid)
    s = _make_session()
    r = s.get(url, timeout=HTTP_TIMEOUT)
    if r.status_code != 200 or (not r.content or r.headers.get("Content-Type", "").lower().find("pdf") == -1):
        raise RuntimeError(f"http {r.status_code}")
    return r.content


# ----------------------------
# WORDS META (без OCR)
# ----------------------------
_date_re = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")
_time_re = re.compile(r"\b(\d{2})[:\.](\d{2})\b")


def parse_match_meta_words(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Быстро вытаскиваем:
      - HOME / AWAY (большими буквами, верхняя часть)
      - дату (DD.MM.YYYY)
      - время (HH:MM)
    Только по words/text слоям, без Tesseract.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    # Текст (обычный и words)
    text = page.get_text("text") or ""
    words = page.get_text("words") or []  # [x0, y0, x1, y1, "word", block_no, line_no, word_no]

    # Дата
    date = None
    for m in _date_re.finditer(text):
        dd, mm, yyyy = m.groups()
        try:
            y = int(yyyy)
            if y >= 2010:  # отсечём очевидный мусорный "2000" и т.п.
                date = f"{dd}.{mm}.{yyyy}"
                break
        except Exception:
            pass

    # Время
    time_msk = None
    for m in _time_re.finditer(text):
        hh, mm = m.groups()
        if 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59:
            time_msk = f"{hh}:{mm}"
            break

    # Команды (ищем две верхние строки ALL-CAPS кириллицей)
    def is_caps_cyr(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        # хотя бы 6 символов кириллицей и почти все заглавные
        letters = re.findall(r"[А-ЯЁA-Z]", s)
        return len(letters) >= 6 and (sum(ch.isupper() for ch in letters) / len(letters) > 0.8)

    # Берём верхние 30% страницы
    _, y0, _, y1 = page.rect
    cut = y0 + 0.30 * (y1 - y0)
    lines_map: Dict[int, List[str]] = {}
    for x0, wy0, x1, wy1, w, *_ in words:
        if wy1 <= cut and w.strip():
            ln = int(round(wy0 / 8.0))  # грубая "квантизация" строк
            lines_map.setdefault(ln, []).append(w)

    top_lines = [" ".join(lines_map[k]) for k in sorted(lines_map.keys())]
    caps = [ln.strip() for ln in top_lines if is_caps_cyr(ln)]
    home = caps[0] if len(caps) > 0 else ""
    away = caps[1] if len(caps) > 1 else ""

    doc.close()
    return {
        "date": date,
        "time_msk": time_msk,
        "teams": {"home": home, "away": away},
    }


# ----------------------------
# OCR REFS (ускорённый ROI в шапке)
# ----------------------------
def ocr_refs_first_page(pdf_bytes: bytes, dpi: int = 240, roi_bottom: float = 0.35, lang: str = "rus+eng") -> Dict[str, Any]:
    """
    OCR только верхней зоны (roi_bottom доля по высоте).
    Обычно 6–15 сек на Render.
    Возвращает словарь:
      {"main": [...], "linesmen": [...], "_raw_lines": [...]}
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # ROI: верхняя часть страницы
    H = img.height
    crop_h = max(80, int(H * roi_bottom))
    img = img.crop((0, 0, img.width, crop_h))

    # Препроцессинг без тяжёлых фильтров
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    arr = np.array(g)
    bin_img = Image.fromarray((arr > 180).astype("uint8") * 255)

    # OCR
    cfg = "--oem 1 --psm 6"
    raw_text = pytesseract.image_to_string(bin_img, lang=lang, config=cfg)
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    # Парс имён
    raw_join = " ".join(lines)
    # Вырежем служебные слова (Главный/Линейный/судья/Резервный/Обновлено)
    cleaned = re.sub(r"\b(Главн\w*|Линейн\w*|судьи?|Резервн\w*|Обновлено[:\s\d\.]*)\b", " ", raw_join, flags=re.I)
    # ФИО (1–3 слова), допускаем «Алексей»/«А.» и т.п.
    fio = re.findall(r"[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\.]+){1,2}", cleaned)

    # На большинстве протоколов: 2 главных, 2 линейных
    main = fio[:2]
    linesmen = fio[2:4]

    doc.close()
    return {"main": main, "linesmen": linesmen, "_raw_lines": lines}


# ----------------------------
# PLACEHOLDER: goalies (заглушка)
# ----------------------------
def extract_goalies_placeholder(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Заглушка: возвращает пустые списки.
    На следующем шаге подключим точный парсер таблицы составов.
    """
    return {"home": [], "away": []}


# ----------------------------
# API
# ----------------------------
@app.route("/health")
def health():
    return j({"ok": True, "engine": "ready"})


@app.route("/extract")
def extract():
    season = (request.args.get("season") or "").strip()
    uid = (request.args.get("uid") or "").strip()
    mode = (request.args.get("mode") or "refs").strip().lower()
    debug = request.args.get("debug", "0") in ("1", "true", "yes")

    if not season or not uid:
        return j({"ok": False, "error": "params required: season, uid"}, 400)

    # быстрая проверка health без скачивания
    if mode == "ping":
        return j({"ok": True, "engine": "ping"})

    url = build_pdf_url(season, uid)
    t0 = time.time()

    try:
        pdf_bytes = fetch_pdf_bytes(season, uid)
    except Exception as e:
        return j({"ok": False, "error": f"http fetch failed: {e}"}, 502)

    out: Dict[str, Any] = {"ok": True, "source_url": url}
    warns: List[str] = []

    # Параметры ускорения OCR
    fast = request.args.get("fast", "1") in ("1", "true", "yes")
    dpi = int(request.args.get("dpi") or (220 if fast else 260))
    roi_bottom = float(request.args.get("roi_bottom") or (0.33 if fast else 0.40))
    lang = request.args.get("lang") or "rus+eng"

    # words
    if mode in ("words", "all"):
        try:
            meta = parse_match_meta_words(pdf_bytes)
            out["match"] = {"season": season, "uid": uid, **meta}
        except Exception as e:
            warns.append(f"words-meta: {e}")

    # refs
    if mode in ("refs", "all"):
        try:
            refs = ocr_refs_first_page(pdf_bytes, dpi=dpi, roi_bottom=roi_bottom, lang=lang)
            out["referees"] = {"main": refs["main"], "linesmen": refs["linesmen"]}
            if debug:
                out["_debug"] = {"raw_lines": refs.get("_raw_lines", [])}
        except Exception as e:
            warns.append(f"ocr-refs: {e}")

    # goalies (пока заглушка)
    if mode in ("goalies", "all"):
        try:
            out["goalies"] = extract_goalies_placeholder(pdf_bytes)
        except Exception as e:
            warns.append(f"goalies: {e}")

    if warns:
        out["warnings"] = warns

    out["engine"] = mode
    out["duration_s"] = round(time.time() - t0, 3)
    return j(out)
