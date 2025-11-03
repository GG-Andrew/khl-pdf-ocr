# app.py
import os
import re
import json
import time
import logging
import shutil
from typing import List, Dict, Tuple

import requests
from flask import Flask, request, jsonify, Response

from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF

# ---- OCR (безопасная инициализация)
try:
    import pytesseract  # type: ignore
    HAS_TESSERACT = shutil.which("tesseract") is not None
except Exception:
    pytesseract = None  # type: ignore
    HAS_TESSERACT = False

# ------------------------
# Flask base
# ------------------------
app = Flask(__name__)
app.json.ensure_ascii = False  # русские символы как есть
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("khl-pdf-ocr")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "application/pdf,*/*;q=0.9",
    "Referer": "https://www.khl.ru/",
})

# Ожидается БАЗА С /khlpdf на конце, например:
# PDF_PROXY_BASE=https://pdf2.palladiumgames2d.workers.dev/khlpdf
PDF_PROXY_BASE = os.getenv("PDF_PROXY_BASE", "").rstrip("/")
TESS_LANG = "rus+eng"

# ------------------------
# Utils
# ------------------------
def make_pdf_url(season: str, uid: str) -> str:
    """
    Возвращает URL PDF с учётом Cloudflare Worker-прокси, если задан.
    При базе https://.../khlpdf итог:
        {BASE}/{season}/{uid}/game-{uid}-start-ru.pdf
    """
    path = f"{season}/{uid}/game-{uid}-start-ru.pdf"
    if PDF_PROXY_BASE:
        return f"{PDF_PROXY_BASE}/{path}"
    return f"https://www.khl.ru/pdf/{path}"


def http_get(url: str, timeout=30) -> bytes:
    r = SESSION.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content


def pdf_to_pix(doc: fitz.Document, pno=0, dpi=300) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc.load_page(pno).get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def _ocr_image_lines(img: Image.Image) -> List[str]:
    """Безопасный OCR: если tesseract не установлен — возвращает []."""
    if not HAS_TESSERACT or pytesseract is None:
        return []
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
    words = page.get_text("words")  # x0,y0,x1,y1,text,block,line,word
    words_sorted = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
    return [(w[1], w[0], w[4]) for w in words_sorted]


def lines_from_words(words: List[Tuple[float, float, str]], tolerance=3.0) -> List[str]:
    """Группирует слова в строки по координате y."""
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


def _group_words_by_lines_and_cols(words, y_tol=3.0):
    """
    Разделяет слова на левую/правую колонку по медиане X.
    Возврат:
      lines_all: все строки,
      cols: {"left": [(y,x,t), ...], "right": [...]}
    """
    if not words:
        return [], {"left": [], "right": []}
    xs = sorted(w[1] for w in words)
    mid_x = xs[len(xs)//2] if xs else 9999
    left = [w for w in words if w[1] <= mid_x]
    right = [w for w in words if w[1] > mid_x]
    return lines_from_words(words, tolerance=y_tol), {"left": left, "right": right}

# ------------------------
# Match meta (команды/дата/время)
# ------------------------
def find_match_meta(lines: List[str], cols_words: Dict[str, List[Tuple[float,float,str]]]) -> Dict:
    """
    Дата/время — из общих lines.
    Команды — из верхней четверти каждой колонки отдельно (КАПС-строки).
    """
    meta = {"date": "", "time_msk": "", "teams": {"home": "", "away": ""}}

    # дата
    for ln in lines[:160]:
        m = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", ln)
        if m:
            meta["date"] = m.group(0)
            break
        m2 = re.search(r"\b\d{1,2}\s+[А-Яа-яё]+\s+20\d{2}", ln)
        if m2:
            meta["date"] = m2.group(0).replace(" г.", "").strip()
            break

    # время
    for ln in lines[:200]:
        m = re.search(r"\b([01]\d|2[0-3]):[0-5]\d\b", ln)
        if m:
            meta["time_msk"] = m.group(0)
            break

    def top_caps_from(words_part):
        if not words_part:
            return ""
        ys = [w[0] for w in words_part]
        y_min, y_max = min(ys), max(ys)
        y_cut = y_min + 0.25 * (y_max - y_min)
        top = [w for w in words_part if w[0] <= y_cut]
        all_lines = lines_from_words(top, tolerance=3.0)
        cand = [ln for ln in all_lines if re.search(r"[А-ЯЁ]{3,}", ln) and len(ln) >= 8]
        if not cand:
            return ""
        cand = sorted(cand, key=len, reverse=True)
        return re.sub(r"\s{2,}", " ", cand[0]).strip()

    meta["teams"]["home"] = top_caps_from(cols_words["left"])
    meta["teams"]["away"] = top_caps_from(cols_words["right"])
    return meta


def extract_words(doc: fitz.Document) -> Dict:
    words = text_words(doc, 0)
    lines_all, cols = _group_words_by_lines_and_cols(words)
    meta = find_match_meta(lines_all, cols)
    return {"ok": True, "engine": "words", "match": meta}

# ------------------------
# Referees
# ------------------------
def find_ref_lines(lines: List[str]) -> Tuple[List[str], List[str], Dict]:
    """
    Ищем заголовок с 'Главный судья' и 'Линейный судья', следующая строка — ФИО.
    """
    debug = {}
    header_idx = -1
    for i, ln in enumerate(lines[:100]):
        if ("Главный судья" in ln) and ("Линейный судья" in ln):
            header_idx = i
            break
    if header_idx != -1 and header_idx + 1 < len(lines):
        ref_line = lines[header_idx + 1]
        debug["raw_ref_line"] = ref_line
        ref_line = re.sub(r"Обновлено.*", "", ref_line, flags=re.I).strip()
        parts = [p for p in re.split(r"[,\|;]+|\s+", ref_line) if p]

        names: List[str] = []
        buf: List[str] = []
        for p in parts:
            if re.match(r"^[А-ЯЁ][а-яё\-]+$", p):
                buf.append(p)
                if len(buf) == 2:
                    names.append(" ".join(buf))
                    buf = []
            else:
                buf = []

        if len(names) < 4 and len(parts) >= 4:
            alt = []
            for j in range(len(parts) - 1):
                a, b = parts[j], parts[j+1]
                if all(re.match(r"^[А-ЯЁ][а-яё\-]+$", x) for x in (a, b)):
                    alt.append(a + " " + b)
            if len(alt) >= 4:
                names = alt[:4]

        main = names[:2]
        linesmen = names[2:4]
        return main, linesmen, debug

    return [], [], {"note": "ref header not found"}


def extract_refs(doc: fitz.Document, debug=False) -> Dict:
    words = text_words(doc, 0)
    lines = lines_from_words(words)
    main, linesmen, dbg = find_ref_lines(lines)

    # OCR fallback верхней трети листа (мягко)
    if (not main or not linesmen) and HAS_TESSERACT:
        try:
            img = pdf_to_pix(doc, 0, dpi=300)
            crop = img.crop((0, 0, img.width, int(img.height * 0.33)))
            ocr_lines = _ocr_image_lines(crop)
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
                    if all(re.match(r"^[А-ЯЁ][а-яё\-]+$", x) for x in (a, b)):
                        cand.append(f"{a} {b}")
                if len(cand) >= 4:
                    main = cand[:2]
                    linesmen = cand[2:4]
                    dbg["ocr_ref_line"] = txt
        except Exception as e:
            dbg["ocr_error"] = str(e)

    res = {"ok": True, "engine": "ocr-refs", "referees": {"main": main, "linesmen": linesmen}}
    if debug:
        res["_debug"] = dbg
    return res

# ------------------------
# Goalies
# ------------------------
def _collect_goalies_from_column(words_part: List[Tuple[float, float, str]]) -> List[Dict]:
    """Ищем 'Вратари' в колонке и ниже читаем ФИО + флаги С/Р в той же строке."""
    if not words_part:
        return []
    lines = lines_from_words(words_part, tolerance=3.0)
    idx = -1
    for i, ln in enumerate(lines[:120]):
        if ln.strip().startswith("Вратари"):
            idx = i
            break
    if idx == -1:
        return []

    out = []
    for ln in lines[idx+1: idx+40]:
        if ln.startswith("Звено") or re.search(r"\bЗвено\s*\d", ln):
            break
        m = re.search(r"([А-ЯЁ][а-яё\-]+ [А-ЯЁ][а-яё\-]+(?: [А-ЯЁ][а-яё\-]+)?)", ln)
        if not m:
            continue
        name = m.group(1).strip()
        flag = ""
        if re.search(r"\bС\b", ln):
            flag = "C"
        elif re.search(r"\bР\b", ln):
            flag = "R"
        out.append({"name": name, "flag": flag})
    return out


def extract_goalies(doc: fitz.Document, debug=False) -> Dict:
    words = text_words(doc, 0)
    _, cols = _group_words_by_lines_and_cols(words)
    home = _collect_goalies_from_column(cols["left"])
    away = _collect_goalies_from_column(cols["right"])

    dbg = {}
    # OCR fallback обеих колонок только если вообще пусто
    if not home and not away and HAS_TESSERACT:
        try:
            img = pdf_to_pix(doc, 0, dpi=300)
            h = img.height
            left_img = img.crop((0, 0, img.width // 2, int(h * 0.6)))
            right_img = img.crop((img.width // 2, 0, img.width, int(h * 0.6)))
            l_lines = _ocr_image_lines(left_img)
            r_lines = _ocr_image_lines(right_img)

            def from_ocr(lines):
                if not lines:
                    return []
                # Грубая пост-обработка: ищем ФИО и С/Р в строке
                out = []
                for ln in lines:
                    m = re.search(r"([А-ЯЁ][а-яё\-]+ [А-ЯЁ][а-яё\-]+(?: [А-ЯЁ][а-яё\-]+)?)", ln)
                    if not m:
                        continue
                    name = m.group(1).strip()
                    flag = "C" if re.search(r"\bС\b", ln) else ("R" if re.search(r"\bР\b", ln) else "")
                    out.append({"name": name, "flag": flag})
                return out

            home = from_ocr(l_lines)
            away = from_ocr(r_lines)
            dbg["fallback"] = "ocr"
        except Exception as e:
            dbg["ocr_error"] = str(e)

    res = {"ok": True, "engine": "gk", "goalies": {"home": home, "away": away}}
    if debug:
        res["_debug"] = dbg
    return res

# ------------------------
# ALL
# ------------------------
def extract_all(doc: fitz.Document, season: str, uid: str, debug=False) -> Dict:
    t0 = time.time()
    meta = extract_words(doc)
    meta_match = meta.get("match", {"teams": {"home": "", "away": ""}})

    refs = extract_refs(doc, debug=debug)
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
        out["_debug"] = {"has_tesseract": HAS_TESSERACT}
    return out

# ------------------------
# Routes
# ------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "engine": "ready", "tesseract": HAS_TESSERACT})


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
        return jsonify({"ok": False, "error": f"http {code}", "detail": str(e), "source_url": url}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": "download_error", "detail": str(e), "source_url": url}), 502

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return jsonify({"ok": False, "error": "pdf_open_error", "detail": str(e), "source_url": url}), 500

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
        if "duration_s" not in res:
            res["duration_s"] = round(time.time() - t0, 3)
        return Response(json.dumps(res, ensure_ascii=False), mimetype="application/json")
    except Exception as e:
        return jsonify({"ok": False, "error": "extract_error", "detail": str(e), "source_url": url}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
