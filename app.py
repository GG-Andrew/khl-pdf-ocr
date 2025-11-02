# app.py
import os, io, re, time
from typing import List, Dict
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import cloudscraper
from flask import Flask, request
from flask.json.provider import DefaultJSONProvider

# --------- Flask JSON: не экранировать кириллицу ----------
class UTF8JSON(DefaultJSONProvider):
    ensure_ascii = False

app = Flask(__name__)
app.json = UTF8JSON(app)

# --------- HTTP загрузка PDF с нужными заголовками ----------
SCRAPER = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

def khl_pdf_url(season: str, uid: str) -> str:
    return f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"

def fetch_pdf_bytes(url: str) -> bytes:
    headers = {
        "Referer": "https://www.khl.ru",
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/121.0 Safari/537.36"),
        "Accept": "application/pdf,*/*",
    }
    r = SCRAPER.get(url, headers=headers, timeout=25)
    if r.status_code != 200 or "pdf" not in (r.headers.get("content-type","").lower()):
        raise RuntimeError(f"http {r.status_code}")
    return r.content

# --------- Вспомогалки для OCR судей ----------
ROLE_RX = re.compile(r"(главн\w*\s*судья|линейн\w*\s*судья|резервн\w*)", re.I)

def _split_into_name_pairs(line: str) -> List[str]:
    """
    Делим строку вида: 'Морозов Сергей Васильев Алексей Седов Егор Шишло Дмитрий'
    на пары по 2 слова: ['Морозов Сергей', 'Васильев Алексей', 'Седов Егор', 'Шишло Дмитрий'].
    Берём только слова, начинающиеся с заглавной буквы (рус/лат), чтобы отсеять мусор.
    """
    tokens = [t for t in re.split(r"\s+", line.strip()) if t]
    def is_name_word(w: str) -> bool:
        return bool(re.match(r"^[А-ЯЁA-Z][а-яёa-z\.]+$", w))
    words = [w for w in tokens if is_name_word(w)]
    pairs = []
    i = 0
    while i + 1 < len(words):
        pairs.append(f"{words[i]} {words[i+1]}")
        i += 2
    return pairs

def ocr_referees(pdf_bytes: bytes, return_raw: bool=False) -> Dict[str, List[str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[-1]  # судьи обычно внизу последней страницы

    # находим вертикальный диапазон блока по словам 'судья'
    words = page.get_text("words")
    marks = [w for w in words if ROLE_RX.search(w[4] or "")]
    if marks:
        y0 = max(min(w[1] for w in marks) - 50, 0)
        y1 = min(max(w[3] for w in marks) + 50, page.rect.height)
    else:
        h = page.rect.height
        y0, y1 = h * 0.70, h * 0.98

    # рендерим ROI с повышенным DPI
    rect = fitz.Rect(0, y0, page.rect.width, y1)
    mat = fitz.Matrix(3.5, 3.5)  # ~350 dpi
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # важная опция — сохраняем интервалы
    cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    raw_text = pytesseract.image_to_string(img, lang="rus+ang+eng", config=cfg)  # ang+eng на всякий
    lines = [re.sub(r"[ \t]+", " ", L.strip()) for L in raw_text.splitlines() if L.strip()]

    # 1) ищем строку с чистыми именами (без слов 'судья', 'резервный', 'обновлено')
    name_line = None
    for L in lines:
        Llow = L.lower()
        if ("суд" in Llow) or ("резерв" in Llow) or ("обновлен" in Llow):
            continue
        # эвристика: в строке должно быть хотя бы 4 слова с заглавной
        caps = re.findall(r"\b[А-ЯЁA-Z][а-яёa-z\.]+", L)
        if len(caps) >= 4:
            name_line = L
            break

    main: List[str] = []
    linesmen: List[str] = []

    if name_line:
        pairs = _split_into_name_pairs(name_line)
        # первые 2 — главные, остальные — линейные
        main = pairs[:2]
        linesmen = pairs[2:6]
    else:
        # фоллбэк: пробуем собрать имена из всех строк
        blob = " ".join([L for L in lines if "суд" not in L.lower()])
        pairs = _split_into_name_pairs(blob)
        main = pairs[:2]
        linesmen = pairs[2:6]

    # очистка на всякий
    main = [re.sub(r"[ ,.;:]+$", "", x) for x in main]
    linesmen = [re.sub(r"[ ,.;:]+$", "", x) for x in linesmen]

    out = {"main": main, "linesmen": linesmen}
    if return_raw:
        out["_raw_lines"] = lines
    return out

# --------- HTTP endpoints ----------
@app.get("/health")
def health():
    return app.response_class(
        response=app.json.dumps({"ok": True, "engine": "ready"}),
        status=200,
        mimetype="application/json; charset=utf-8"
    )

@app.get("/extract")
def extract():
    t0 = time.time()
    season = request.args.get("season")
    uid = request.args.get("uid")
    mode = (request.args.get("mode") or "refs").lower()

    if not season or not uid:
        return app.response_class(
            response=app.json.dumps({"ok": False, "error": "params 'season' and 'uid' required"}),
            status=400, mimetype="application/json; charset=utf-8"
        )

    url = khl_pdf_url(season, uid)
    try:
        pdf = fetch_pdf_bytes(url)
    except Exception as e:
        return app.response_class(
            response=app.json.dumps({"ok": False, "error": str(e), "detail": f"{e}", "source_url": url}),
            status=502, mimetype="application/json; charset=utf-8"
        )

    if mode == "refs":
        refs = ocr_referees(pdf, return_raw=bool(request.args.get("debug")))
        out = {
            "ok": True,
            "engine": "ocr-refs",
            "referees": refs,
            "source_url": url,
            "duration_s": round(time.time() - t0, 3)
        }
        return app.response_class(
            response=app.json.dumps(out),
            status=200, mimetype="application/json; charset=utf-8"
        )

    return app.response_class(
        response=app.json.dumps({
            "ok": False,
            "error": f"mode '{mode}' not implemented",
            "source_url": url,
            "duration_s": round(time.time() - t0, 3)
        }),
        status=400, mimetype="application/json; charset=utf-8"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
