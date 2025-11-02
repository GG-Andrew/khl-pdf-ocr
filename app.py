import io, os, re, time
from typing import Any, Dict, List

from flask import Flask, request, jsonify
import cloudscraper
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

app = Flask(__name__)

# ---------------- HTTP загрузка PDF (обход 403 khl.ru) ----------------
def fetch_pdf_bytes(season: str, uid: str, timeout: int = 20) -> bytes:
    pdf_url = f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"
    preview = f"https://www.khl.ru/game/{season}/{uid}/preview/"
    s = cloudscraper.create_scraper(browser={"browser": "firefox", "platform": "windows"})
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) "
                       "Gecko/20100101 Firefox/132.0"),
        "Accept": "application/pdf,*/*;q=0.8",
        "Accept-Language": "ru,en;q=0.9",
        "Referer": preview,
        "Origin": "https://www.khl.ru",
        "Connection": "keep-alive",
    }
    last = None
    for i in range(3):
        try:
            s.head(pdf_url, headers=headers, timeout=timeout, allow_redirects=True)
            r = s.get(pdf_url, headers=headers, timeout=timeout, allow_redirects=True)
            last = r
            if r.status_code == 200 and r.content and r.headers.get("Content-Type","").startswith("application/pdf"):
                return r.content
            if r.status_code == 403:
                headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                                         "Chrome/119.0.0.0 Safari/537.36")
                time.sleep(0.7)
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"http {getattr(last,'status_code', 'error')}")

# ---------------- Базовые извлекатели ----------------
def extract_words(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for pno in range(len(doc)):
        page = doc[pno]
        words = page.get_text("words")
        words.sort(key=lambda w: (w[1], w[0]))
        for (x0, y0, x1, y1, w, b, l, n) in words:
            out.append({"page": pno+1, "x0": round(x0,2), "y0": round(y0,2),
                        "x1": round(x1,2), "y1": round(y1,2),
                        "w": w, "block": b, "line": l, "idx": n})
    return out

def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(p.get_text() for p in doc)

# ---------------- Вырезка области + OCR судей ----------------
REF_ROLE_RX = re.compile(r"\b(Главн\w*\s+судья|Линейн\w*\s+судья|Резервн\w+)\b", re.I)
NAME_CLEAN_RX = re.compile(r"\b(Главн\w*|Линейн\w*|Резервн\w*|судья|судьи|Обновлено.*)\b", re.I)
MULTI_SPACE = re.compile(r"\s{2,}")

def page_image(pdf_bytes: bytes, page_index: int, zoom: float = 2.0) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_referees(pdf_bytes: bytes) -> Dict[str, List[str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    last_idx = len(doc) - 1
    page = doc[last_idx]
    # пробуем сначала найти прямоугольник блока судей по словам "судья"
    words = page.get_text("words")
    cand = [w for w in words if REF_ROLE_RX.search(w[4] or "")]
    if cand:
        y0 = min(w[1] for w in cand) - 40
        y1 = max(w[3] for w in cand) + 40
        y0 = max(y0, 0); y1 = min(y1, page.rect.height)
        rect = fitz.Rect(0, y0, page.rect.width, y1)
    else:
        # эвристика: нижняя четверть страницы
        h = page.rect.height
        rect = fitz.Rect(0, h*0.72, page.rect.width, h)

    # рендерим только выбранную область
    mat = fitz.Matrix(3.0, 3.0)  # ~300dpi
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    ocr_text = pytesseract.image_to_string(img, lang="rus+eng", config="--oem 1 --psm 6")
    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]

    # собираем роли -> имена
    main, linesmen = [], []
    current_role = None
    for raw in lines:
        low = raw.lower()
        if "главн" in low and "суд" in low:
            current_role = "main"
            continue
        if "линей" in low and "суд" in low:
            current_role = "linesmen"
            continue
        # чистим хвосты "Резервный ..."
        clean = NAME_CLEAN_RX.sub(" ", raw)
        clean = MULTI_SPACE.sub(" ", clean).strip(" ,.;:—-")
        if not clean:
            continue
        # фильтруем явные мусорные слова
        if len(clean) < 3 or clean.lower().startswith("обновлено"):
            continue
        # эвристика: имена обычно из 2 слов
        # отрезаем возможные лишние буквы в начале (К, Гл и т.п.)
        clean = re.sub(r"^[КГЛР]\.?$", "", clean).strip()
        if not clean:
            continue
        if current_role == "main":
            if clean not in main:
                main.append(clean)
        elif current_role == "linesmen":
            if clean not in linesmen:
                linesmen.append(clean)

    # страховка: если роли не поймались — вытащим все имена в одну строку и порежем по 2-3 слову
    if not main and not linesmen and lines:
        blob = " ".join(lines)
        blob = NAME_CLEAN_RX.sub(" ", blob)
        # простая выборка «Имя Фамилия»
        cand = re.findall(r"[А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z\.]+){1,2}", blob)
        for i, nm in enumerate(cand):
            (main if i < 2 else linesmen).append(nm.strip())

    # ограничим до здравого минимума
    main = main[:2]
    linesmen = linesmen[:4]

    return {"main": main, "linesmen": linesmen}

# ---------------- Черновой парсер звеньев (как раньше) ----------------
def draft_parse_lineups(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not words:
        return {"home": {"team": "", "goalies": [], "lines": {}},
                "away": {"team": "", "goalies": [], "lines": {}}}

    xs = [w["x0"] for w in words]
    mid = (min(xs) + max(xs)) / 2
    left = [w for w in words if w["x0"] <= mid]
    right = [w for w in words if w["x0"] > mid]

    def group(column: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
        by_y: Dict[int, List[Dict[str, Any]]] = {}
        for w in column:
            y = int(round(w["y0"]))
            by_y.setdefault(y, []).append(w)
        lines = {"1": [], "2": [], "3": [], "4": []}
        current = "1"
        for y in sorted(by_y):
            row = sorted(by_y[y], key=lambda z: z["x0"])
            t = " ".join(x["w"] for x in row)
            if "Звено" in t:
                for k in ("1","2","3","4"):
                    if f"Звено {k}" in t:
                        current = k
                        break
                continue
            m = re.search(r"(^|\s)(\d{1,2})\s+([ЗДDFGFНН])?\s*([A-Za-zА-Яа-яЁё\-\.'’]+(?:\s+[A-Za-zА-Яа-яЁё\-\.'’]+)?)", t)
            if m:
                num = m.group(2)
                pos_raw = (m.group(3) or "").upper()
                pos = "D" if any(x in pos_raw for x in ["Д","З","D"]) else ("F" if any(x in pos_raw for x in ["F","Н"]) else "")
                name = m.group(4).strip(" .,-;:")
                lines.setdefault(current, []).append({"number": num, "pos": pos, "name": name})
        return lines

    return {
        "home": {"team": "", "goalies": [], "lines": group(left), "bench": []},
        "away": {"team": "", "goalies": [], "lines": group(right), "bench": []},
    }

# ---------------- API ----------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "engine": "ready"})

@app.get("/extract")
def extract():
    season = request.args.get("season","").strip()
    uid = request.args.get("uid","").strip()
    mode = (request.args.get("mode","words") or "words").lower()
    if not season or not uid:
        return jsonify({"ok": False, "error": "params 'season' and 'uid' required"}), 400
    t0 = time.time()
    try:
        pdf = fetch_pdf_bytes(season, uid)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502

    src = f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"
    try:
        if mode == "text":
            text = extract_text(pdf)
            return jsonify({"ok": True, "engine": "text", "text_len": len(text),
                            "text": text, "source_url": src, "duration_s": round(time.time()-t0,3)})
        if mode == "draft":
            words = extract_words(pdf)
            data = draft_parse_lineups(words)
            return jsonify({"ok": True, "engine": "draft", "data": data,
                            "source_url": src, "duration_s": round(time.time()-t0,3)})
        if mode == "refs":
            refs = ocr_referees(pdf)
            return jsonify({"ok": True, "engine": "ocr-refs", "referees": refs,
                            "source_url": src, "duration_s": round(time.time()-t0,3)})
        # default: words
        words = extract_words(pdf)
        return jsonify({"ok": True, "engine": "words", "words_total": len(words),
                        "words": words[:5000], "source_url": src, "duration_s": round(time.time()-t0,3)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/ocr")
def ocr_all():
    season = request.args.get("season","").strip()
    uid = request.args.get("uid","").strip()
    if not season or not uid:
        return jsonify({"ok": False, "error": "params 'season' and 'uid' required"}), 400
    t0 = time.time()
    pdf = fetch_pdf_bytes(season, uid)
    # полный OCR последней страницы (на случай очень кривых файлов)
    img = page_image(pdf, -1, zoom=3.0)
    text = pytesseract.image_to_string(img, lang="rus+eng", config="--oem 1 --psm 6")
    return jsonify({"ok": True, "engine": "ocr", "text_len": len(text), "text": text,
                    "duration_s": round(time.time()-t0,3)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","10000")))
