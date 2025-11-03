# app.py
import io
import os
import re
import json
import time
import math
import base64
import logging
from typing import List, Dict, Tuple, Optional

from flask import Flask, request, jsonify, Response

import requests
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
import pytesseract

# ------------------------
# Flask base
# ------------------------
app = Flask(__name__)
app.json.ensure_ascii = False  # русские символы как есть
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("khl-pdf-ocr")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.9",
    "Referer": "https://www.khl.ru/",
})

PDF_PROXY_BASE = os.getenv("PDF_PROXY_BASE", "").rstrip("/")
TESS_LANG = "rus+eng"  # tesseract languages

# ------------------------
# Utils
# ------------------------
def make_pdf_url(season: str, uid: str) -> str:
    """
    Возвращает URL PDF с учётом Cloudflare Worker-прокси, если задан.
    Ожидается, что PDF_PROXY_BASE = https://pdf2....workers.dev/khlpdf
    Тогда итог: {PDF_PROXY_BASE}/{season}/{uid}/game-{uid}-start-ru.pdf
    """
    path = f"{season}/{uid}/game-{uid}-start-ru.pdf"
    if PDF_PROXY_BASE:
        return f"{PDF_PROXY_BASE}/{path}"
    return f"https://www.khl.ru/pdf/{path}"

def http_get(url: str, timeout=25) -> bytes:
    r = SESSION.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content

def pdf_to_pix(doc: fitz.Document, pno=0, dpi=300) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc.load_page(pno).get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def ocr_image_lines(img: Image.Image) -> List[str]:
    # лёгкая предобработка
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.4)
    gray = gray.filter(ImageFilter.SHARPEN)
    txt = pytesseract.image_to_string(gray, lang=TESS_LANG, config="--psm 6")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in txt.splitlines()]
    return [ln for ln in lines if ln]

def text_words(doc: fitz.Document, pno=0) -> List[Tuple[float, float, str]]:
    """
    Возвращает список (y, x, text) слов с первой страницы,
    отсортированный сверху-вниз слева-направо.
    """
    page = doc.load_page(pno)
    words = page.get_text("words")  # x0, y0, x1, y1, "text", block_no, line_no, word_no
    words_sorted = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
    out = [(w[1], w[0], w[4]) for w in words_sorted]
    return out

def lines_from_words(words: List[Tuple[float, float, str]], tolerance=3.0) -> List[str]:
    """
    Группирует слова в строки по координате y.
    """
    rows: Dict[float, List[Tuple[float, str]]] = {}
    for y, x, t in words:
        key = None
        for ky in rows.keys():
            if abs(ky - y) <= tolerance:
                key = ky
                break
        if key is None:
            key = y
            rows[key] = []
        rows[key].append((x, t))
    lines = []
    for ky in sorted(rows.keys()):
        items = sorted(rows[ky], key=lambda r: r[0])
        line = " ".join(t for _, t in items).strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return lines

def find_ref_lines(lines: List[str]) -> Tuple[List[str], List[str], Dict]:
    """
    Находит блок судей. Стратегия:
    - ищем строку, где встречаются слова 'Главный судья' и 'Линейный судья'
    - следующая строка содержит 4 фамилии (2 главных, 2 линейных) — допускаем слипание,
      разбираем по пробелам и эвристикам (капитализация, количество).
    Если не нашли — OCR fallback.
    """
    debug = {}
    header_idx = -1
    for i, ln in enumerate(lines[:80]):  # только верх страницы
        if ("Главный судья" in ln) and ("Линейный судья" in ln):
            header_idx = i
            break
    if header_idx != -1 and header_idx + 1 < len(lines):
        ref_line = lines[header_idx + 1]
        debug["raw_ref_line"] = ref_line
        # очистка от мусора
        ref_line = re.sub(r"Обновлено.*", "", ref_line, flags=re.I).strip()
        # разбиение на слова
        parts = [p for p in re.split(r"[,\|;]+|\s{1,}", ref_line) if p]
        # Склеиваем Имя Фамилия
        # Эвристика: фамилии и имена — 2 слова подряд с заглавной буквы
        names: List[str] = []
        buf = []
        for p in parts:
            if re.match(r"^[A-ЯЁ][а-яё\-]+$", p):
                buf.append(p)
                if len(buf) == 2:
                    names.append(" ".join(buf))
                    buf = []
            else:
                buf = []
        # Если слов слишком много/мало — fallback: словарь биграмм по окну
        if len(names) < 4 and len(parts) >= 4:
            alt = []
            for j in range(len(parts) - 1):
                big = parts[j:j+2]
                if all(re.match(r"^[A-ЯЁ][а-яё\-]+$", x) for x in big):
                    alt.append(" ".join(big))
            if len(alt) >= 4:
                names = alt[:4]
        main = names[:2]
        linesmen = names[2:4]
        return main, linesmen, debug
    return [], [], {"note": "ref header not found"}

def find_match_meta(lines: List[str]) -> Dict:
    """
    Пытаемся из текстового слоя (без OCR) вытащить:
    - Дата (любой формат dd.mm.yyyy или с русским месяцем)
    - Время (HH:MM)
    - Названия команд вверху (две большие строки капсом)
    """
    meta = {"date": "", "time_msk": "", "teams": {"home": "", "away": ""}}
    # дата
    for ln in lines[:120]:
        if re.search(r"\d{2}\.\d{2}\.\d{4}", ln):
            meta["date"] = re.search(r"\d{2}\.\d{2}\.\d{4}", ln).group(0)
            break
        # например "25 октября 2025" / "25 октября 2025 г."
        m = re.search(r"\b\d{1,2}\s+[А-Яа-яё]+\s+20\d{2}", ln)
        if m:
            meta["date"] = m.group(0).replace(" г.", "").strip()
            break
    # время
    for ln in lines[:160]:
        m = re.search(r"\b([01]\d|2[0-3]):[0-5]\d\b", ln)
        if m:
            meta["time_msk"] = m.group(0)
            break
    # команды: берём две «самые длинные» строки капсом до слова "Вратари"
    team_candidates = []
    for ln in lines[:200]:
        if "Вратари" in ln:
            break
        # строка капсом (много заглавных русских)
        if re.search(r"[А-ЯЁ]{3,}", ln) and len(ln) >= 10:
            team_candidates.append(ln)
    team_candidates = sorted(team_candidates, key=lambda s: -len(s))
    if team_candidates:
        meta["teams"]["home"] = team_candidates[0]
    if len(team_candidates) > 1:
        # попробуем найти вторую отличную по содержанию
        for s in team_candidates[1:]:
            if s != meta["teams"]["home"]:
                meta["teams"]["away"] = s
                break
    return meta

def extract_goalies_lines(all_lines: List[str]) -> Dict[str, List[Dict]]:
    """
    Эвристический парсер вратарей. Ищем блоки 'Вратари' в двух колонках.
    Собираем список имён до первого 'Звено' или пока не упремся в явные столбцы.
    Возвращаем {"home":[{name, flag}], "away":[...]}
    """
    idxs = [i for i, ln in enumerate(all_lines) if ln.strip().startswith("Вратари")]
    # ожидаем два появления "Вратари" (левая/правая колонка)
    home, away = [], []
    def collect(start_idx):
        acc = []
        for j in range(start_idx+1, min(start_idx+40, len(all_lines))):
            t = all_lines[j]
            if t.startswith("Звено"):  # конец блока вратарей
                break
            # имя формата "Фамилия Имя" (может с отч.)
            if re.search(r"^[А-ЯЁ][а-яё\-]+ [А-ЯЁ][а-яё\-]+", t):
                nm = re.search(r"^[А-ЯЁ][а-яё\-]+ [А-ЯЁ][а-яё\-]+(?: [А-ЯЁ][а-яё\-]+)?", t).group(0)
                # статус С/Р
                flag = ""
                if re.search(r"\bС\b", t):
                    flag = "C"  # starter
                elif re.search(r"\bР\b", t):
                    flag = "R"  # reserve
                acc.append({"name": nm, "flag": flag})
        return acc
    if idxs:
        home = collect(idxs[0])
    if len(idxs) > 1:
        away = collect(idxs[1])
    return {"home": home, "away": away}

# ------------------------
# extract modes
# ------------------------
def extract_words(doc: fitz.Document) -> Dict:
    words = text_words(doc, 0)
    lines = lines_from_words(words)
    meta = find_match_meta(lines)
    return {"ok": True, "engine": "words", "match": meta}

def extract_refs(doc: fitz.Document, debug=False) -> Dict:
    # сначала пробуем текстовый слой
    words = text_words(doc, 0)
    lines = lines_from_words(words)
    main, linesmen, dbg = find_ref_lines(lines)

    if not main or not linesmen:
        # fallback OCR маленького верхнего прямоугольника
        try:
            img = pdf_to_pix(doc, 0, dpi=300)
            # кроп верхней полосы
            h = img.height
            crop = img.crop((0, 0, img.width, int(h * 0.33)))
            ocr_lines = ocr_image_lines(crop)
            # ищем по OCR
            header = -1
            for i, ln in enumerate(ocr_lines[:120]):
                if ("Главный судья" in ln) and ("Линейный судья" in ln):
                    header = i
                    break
            if header != -1 and header + 1 < len(ocr_lines):
                txt = ocr_lines[header + 1]
                parts = [p for p in re.split(r"[,|;]|\s+", txt) if p]
                cand = []
                for j in range(len(parts) - 1):
                    a, b = parts[j], parts[j+1]
                    if all(re.match(r"^[A-ЯЁ][а-яё\-]+$", x) for x in (a, b)):
                        cand.append(a + " " + b)
                if len(cand) >= 4:
                    main = cand[:2]
                    linesmen = cand[2:4]
                    dbg["ocr_ref_line"] = txt
        except Exception as e:
            dbg["ocr_error"] = str(e)

    result = {"ok": True, "engine": "ocr-refs", "referees": {"main": main, "linesmen": linesmen}}
    if debug:
        result["_debug"] = {}
        if "raw_ref_line" in dbg:
            result["_debug"]["raw_ref_line"] = dbg["raw_ref_line"]
        if "ocr_ref_line" in dbg:
            result["_debug"]["ocr_ref_line"] = dbg["ocr_ref_line"]
        if "note" in dbg:
            result["_debug"]["note"] = dbg["note"]
    return result

def extract_goalies(doc: fitz.Document, debug=False) -> Dict:
    words = text_words(doc, 0)
    lines = lines_from_words(words)
    g = extract_goalies_lines(lines)
    # fallback: если пусто — OCR лев+прав блоки под заголовком
    if not g["home"] and not g["away"]:
        try:
            img = pdf_to_pix(doc, 0, dpi=300)
            # чуть ниже шапки
            h = img.height
            crop = img.crop((0, int(h * 0.15), img.width, int(h * 0.55)))
            ocr_lines = ocr_image_lines(crop)
            gg = extract_goalies_lines(ocr_lines)
            # переносим статус С/Р если нашёлся
            if gg["home"] or gg["away"]:
                g = gg
        except Exception as e:
            if debug:
                return {"ok": True, "engine": "gk", "goalies": g, "_debug": {"error": str(e)}}
    res = {"ok": True, "engine": "gk", "goalies": g}
    if debug:
        res["_debug"] = {"lines_used": len(lines)}
    return res

def extract_all(doc: fitz.Document, season: str, uid: str, debug=False) -> Dict:
    t0 = time.time()
    # words/meta
    meta = extract_words(doc)
    meta_match = meta.get("match", {"teams": {"home": "", "away": ""}})
    # refs
    refs = extract_refs(doc, debug=debug)
    # goalies
    gk = extract_goalies(doc, debug=debug)
    out = {
        "ok": True,
        "engine": "all",
        "match": {"season": season, "uid": uid, **meta_match},
        "referees": refs.get("referees", {"main": [], "linesmen": []}),
        "goalies": gk.get("goalies", {"home": [], "away": []}),
        "duration_s": round(time.time() - t0, 3),
    }
    if debug:
        out["_debug"] = {}
        if meta.get("match"):
            out["_debug"]["match"] = meta["match"]
    return out

# ------------------------
# Routes
# ------------------------
@app.get("/health")
def health():
    try:
        return jsonify({"ok": True, "engine": "ready"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/extract")
def extract():
    """
    /extract?season=1369&uid=897689&mode=all|refs|goalies|words[&debug=1]
    """
    season = (request.args.get("season") or "").strip()
    uid = (request.args.get("uid") or "").strip()
    mode = (request.args.get("mode") or "all").strip().lower()
    debug = (request.args.get("debug") in ("1", "true", "yes"))

    if not (season and uid):
        return jsonify({"ok": False, "error": "season or uid missing"}), 400

    url = make_pdf_url(season, uid)
    try:
        pdf_bytes = http_get(url, timeout=30)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", 0)
        return jsonify({"ok": False, "error": f"http {code}", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": "download_error", "detail": str(e)}), 502

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return jsonify({"ok": False, "error": "pdf_open_error", "detail": str(e)}), 500

    t0 = time.time()
    try:
        if mode == "refs":
            res = extract_refs(doc, debug=debug)
        elif mode in ("gk", "goalies"):
            res = extract_goalies(doc, debug=debug)
        elif mode == "words":
            res = extract_words(doc)
        else:
            res = extract_all(doc, season, uid, debug=debug)
        res["source_url"] = url
        # duration (если не было внутри)
        if "duration_s" not in res:
            res["duration_s"] = round(time.time() - t0, 3)
        return Response(json.dumps(res, ensure_ascii=False), mimetype="application/json")
    except Exception as e:
        return jsonify({"ok": False, "error": "extract_error", "detail": str(e)}), 500

# ------------------------
# local run
# ------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
