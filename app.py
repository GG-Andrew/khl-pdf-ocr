import os, io, re, json, time
from flask import Flask, request, Response
import requests
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

app = Flask(__name__)

# --- конфиг ---
PDF_PROXY_BASE = os.getenv("PDF_PROXY_BASE", "").rstrip("/")
KHL_PDF = "https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"

KHL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.8",
    "Referer": "https://www.khl.ru/",
    "Origin": "https://www.khl.ru",
}

def j(data: dict, code: int = 200):
    # ВСЕГДА чистый UTF-8 без \uXXXX
    return Response(json.dumps(data, ensure_ascii=False), status=code,
                    mimetype="application/json; charset=utf-8")

def source_url_for(season: str, uid: str) -> str:
    raw = KHL_PDF.format(season=season, uid=uid)
    if PDF_PROXY_BASE:
        # Worker, который уже умеет ходить к khl.ru с нужными заголовками
        return f"{PDF_PROXY_BASE}/khlpdf/{season}/{uid}/game-{uid}-start-ru.pdf"
    return raw

def fetch_pdf(url: str) -> bytes:
    # если идём через Worker — без лишних заголовков
    if "workers.dev" in url or ".pages.dev" in url:
        r = requests.get(url, timeout=30)
    else:
        r = requests.get(url, headers=KHL_HEADERS, timeout=30)
    r.raise_for_status()
    if "application/pdf" not in r.headers.get("Content-Type", "") and r.content[:4] != b"%PDF":
        raise RuntimeError("upstream is not a PDF")
    return r.content

def pdf_text_lines(pdf_bytes: bytes) -> list[str]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = []
        for p in doc:
            text.append(p.get_text("text"))
    full = "\n".join(text)
    # нормализация
    full = full.replace("\r", "")
    # иногда встречаются двойные пробелы
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in full.split("\n")]
    return [ln for ln in lines if ln]

# --- парсинг «шапки» матча (дата/время/команды) из текстового слоя ---
def parse_match_meta(lines: list[str], season: str, uid: str) -> dict:
    date = ""
    time_msk = ""
    home = ""
    away = ""

    # дата «25.10.2025» или «25 октября 2025»
    for ln in lines:
        m = re.search(r"(\d{2}\.\d{2}\.\d{4})", ln)
        if m:
            date = m.group(1)
            break
        m = re.search(r"(\d{2})\s*[а-яА-Я]+?\s*(\d{4})", ln)
        if m and not date:
            d, y = m.group(1), m.group(2)
            date = f"{d}.??.{y}"

    # время
    for ln in lines:
        m = re.search(r"(\d{1,2}[:.]\d{2})\s*(MSK|МСК|ET|МСК\)|MSK\))?", ln, re.I)
        if m:
            t = m.group(1).replace(".", ":")
            time_msk = t if len(t) == 5 else ("0"+t)
            break

    # команды: ищем строку, где подряд ИДЁТ НАЗВАНИЕ ДОМА + ГОСТЕЙ (часто так в KHL PDF)
    # подстрахуемся — берём самую длинную CAPS-строку кириллицей
    cand = sorted(
        [ln for ln in lines if re.search(r"[А-ЯЁ]{3,}", ln)],
        key=len, reverse=True
    )
    if cand:
        big = cand[0]
        # пробуем разбить по двум и более пробелам или « – »/«-»
        if " – " in big:
            home, away = [s.strip() for s in big.split(" – ", 1)]
        elif " - " in big:
            home, away = [s.strip() for s in big.split(" - ", 1)]
        else:
            # запасной вариант: делим пополам по словам
            parts = big.split()
            mid = len(parts)//2
            home = " ".join(parts[:mid])
            away = " ".join(parts[mid:])

    return {
        "season": season,
        "uid": uid,
        "date": date,
        "time_msk": time_msk,
        "teams": {"home": home, "away": away}
    }

# --- парсинг судей ---
def parse_referees(lines: list[str]) -> dict:
    # ищем блоки «Главный судья …» и «Линейный судья …»
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text)

    mains, linesmen = [], []

    # Частые паттерны в KHL PDF
    # Пример: "Главный судья Главный судья Линейный судья Линейный судья Морозов Сергей Васильев Алексей Седов Егор Шишло Дмитрий"
    # Забираем ИМЕНА как пары "Фамилия Имя"
    # Сначала попробуем аккуратно вырезать «хвост» после слов «Главный судья … Линейный судья …»
    tail = re.search(r"(Главный судья.*?)(Обновлено|Составы|Матч №|ЛД «|$)", text)
    if tail:
        block = tail.group(1)
    else:
        block = text

    # Вынем все сочетания "Фамилия Имя"
    pairs = re.findall(r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\b", block)
    # Часто порядок: [2 главных] + [2 линейных]
    # Но OCR иногда "склеивает". Поэтому возьмём первые 2 как главных, следующие 2 как линейных.
    if pairs:
        names = [f"{a} {b}" for a, b in pairs]
        if len(names) >= 2:
            mains = names[:2]
        if len(names) >= 4:
            linesmen = names[2:4]

    return {"main": mains, "linesmen": linesmen}

# --- OCR для таблиц (вратари и т.п.) как задел; пока отдаём [], чтобы не ломать пайплайн ---
def try_ocr_goalies(pdf_bytes: bytes) -> dict:
    # Заглушка: вернём пустые списки. (Включим по готовности правил)
    return {"home": [], "away": []}

# --------- ROUTES ---------

@app.get("/health")
def health():
    return j({"ok": True, "engine": "ready"})

@app.get("/extract")
def extract():
    t0 = time.time()
    season = request.args.get("season", "").strip()
    uid = request.args.get("uid", "").strip()
    mode = (request.args.get("mode", "words")).strip().lower()
    debug = request.args.get("debug") is not None

    if not season or not uid:
        return j({"ok": False, "error": "params 'season' and 'uid' required"}, 400)

    url = source_url_for(season, uid)
    try:
        pdf_bytes = fetch_pdf(url)
    except Exception as e:
        return j({"ok": False, "error": f"http  {getattr(e, 'response', None) and getattr(e.response,'status_code', '') or '' or 'error'}",
                  "detail": str(e)}, 502)

    lines = pdf_text_lines(pdf_bytes)

    if mode == "words":
        out = {
            "ok": True,
            "source_url": url,
            "match": parse_match_meta(lines, season, uid),
            "engine": "words",
        }
        out["duration_s"] = round(time.time() - t0, 3)
        if debug: out["_debug"] = {"lines": lines[:80]}
        return j(out)

    if mode == "refs":
        refs = parse_referees(lines)
        out = {
            "ok": True,
            "source_url": url,
            "referees": refs,
            "engine": "ocr-refs",
        }
        out["duration_s"] = round(time.time() - t0, 3)
        if debug: out["_debug"] = {"raw_lines": lines[:120]}
        return j(out)

    if mode == "goalies":
        gk = try_ocr_goalies(pdf_bytes)
        out = {
            "ok": True,
            "source_url": url,
            "goalies": gk,
            "engine": "gk",
        }
        out["duration_s"] = round(time.time() - t0, 3)
        return j(out)

    if mode == "all":
        meta = parse_match_meta(lines, season, uid)
        refs = parse_referees(lines)
        gk = try_ocr_goalies(pdf_bytes)
        out = {
            "ok": True,
            "source_url": url,
            "match": meta,
            "referees": refs,
            "goalies": gk,
            "engine": "all",
        }
        out["duration_s"] = round(time.time() - t0, 3)
        if debug: out["_debug"] = {"raw_lines": lines[:120]}
        return j(out)

    return j({"ok": False, "error": "unknown mode"}, 400)

if __name__ == "__main__":
    # локальный запуск:  python app.py
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
