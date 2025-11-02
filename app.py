# app.py
import io, re, time, os
from typing import List, Dict, Tuple
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
import cloudscraper

from flask import Flask, request, Response
from flask.json.provider import DefaultJSONProvider

# ---------- Flask JSON: не экранировать кириллицу ----------
class UTF8JSON(DefaultJSONProvider):
    ensure_ascii = False
app = Flask(__name__)
app.json = UTF8JSON(app)

# ---------- HTTP загрузка PDF (с нужным Referer/UA) ----------
SCRAPER = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

def khl_pdf_url(season: str, uid: str) -> str:
    return f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"

def fetch_pdf_bytes(url: str) -> bytes:
    # важен Referer на страницу игры — иначе 403
    headers = {
        "Referer": "https://www.khl.ru",
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/121.0 Safari/537.36"),
        "Accept": "application/pdf,*/*",
    }
    r = SCRAPER.get(url, headers=headers, timeout=25)
    if r.status_code != 200 or r.headers.get("content-type", "").lower().find("pdf") < 0:
        raise RuntimeError(f"http {r.status_code}")
    return r.content

# ---------- OCR судей (улучшенный) ----------
REF_ROLE_RX = re.compile(r"(главн\w*\s*судья|линейн\w*\s*судья|резервн\w*)", re.I)
NAME_TOKEN_RX = re.compile(r"[А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z\.]+){1,2}")

def ocr_referees(pdf_bytes: bytes, return_raw: bool=False) -> Dict[str, List[str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[-1]  # блок судей обычно на последней странице
    words = page.get_text("words")

    marks = [w for w in words if REF_ROLE_RX.search((w[4] or ""))]
    if marks:
        y0 = max(min(w[1] for w in marks) - 50, 0)
        y1 = min(max(w[3] for w in marks) + 50, page.rect.height)
    else:
        h = page.rect.height
        y0, y1 = h * 0.70, h * 0.98

    rect = fitz.Rect(0, y0, page.rect.width, y1)
    mat = fitz.Matrix(3.5, 3.5)  # ~350 dpi
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # сохраняем интервалы, чтобы имена не «слипались»
    cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    raw_text = pytesseract.image_to_string(img, lang="rus+eng", config=cfg)
    lines = [re.sub(r"[ \t]+", " ", L.strip()) for L in raw_text.splitlines() if L.strip()]

    main: List[str] = []
    linesmen: List[str] = []

    # 1) По ролям
    current = None
    for L in lines:
        L_low = L.lower()
        if "главн" in L_low and "суд" in L_low:
            current = "main"
            L = re.sub(r"(?i)главн\w*\s*судья[:\-]?\s*", " ", L).strip()
        elif "линей" in L_low and "суд" in L_low:
            current = "linesmen"
            L = re.sub(r"(?i)линейн\w*\s*судья[:\-]?\s*", " ", L).strip()

        if NAME_TOKEN_RX.search(L):
            names = NAME_TOKEN_RX.findall(L)
            if current == "main":
                for n in names:
                    if n not in main:
                        main.append(n)
            elif current == "linesmen":
                for n in names:
                    if n not in linesmen:
                        linesmen.append(n)

    # 2) Фоллбэк: без ролей — режем весь блок на имена
    if not main and not linesmen:
        blob = " ".join(lines)
        blob = re.sub(r"(?i)\b(главн\w*|линейн\w*|резервн\w*|судья|судьи|обновлено.*)\b", " ", blob)
        cand = NAME_TOKEN_RX.findall(blob)
        main = [c for c in cand[:2]]
        linesmen = [c for c in cand[2:6]]

    # косметика
    main = [re.sub(r"[ ,.;:]+$", "", x) for x in main][:2]
    linesmen = [re.sub(r"[ ,.;:]+$", "", x) for x in linesmen][:4]

    out = {"main": main, "linesmen": linesmen}
    if return_raw:
        out["_raw_lines"] = lines
    return out

# ---------- HTTP endpoints ----------
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

    # Заглушка для других режимов, чтобы не рушить API
    out = {
        "ok": False,
        "error": f"mode '{mode}' not implemented in this build",
        "source_url": url,
        "duration_s": round(time.time() - t0, 3)
    }
    return app.response_class(
        response=app.json.dumps(out),
        status=400, mimetype="application/json; charset=utf-8"
    )

# Render запускает через gunicorn (см. Start Command)
if __name__ == "__main__":
    # локальный запуск для отладки (python app.py)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
