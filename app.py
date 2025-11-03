# app.py
# KHL PDF -> JSON (refs + meta)
# безопасный сервер: без tesseract, устойчив к ошибкам, с прокси khl-pdf worker

import os
import io
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, jsonify, request
import requests

# PyMuPDF (fitz) — только для чтения текстового слоя
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# ------------ Config ------------
KHL_PDF_PROXY = os.getenv("KHL_PDF_PROXY", "").rstrip("/")  # например: https://pdf2.palladiumgames2d.workers.dev
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/121.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
    "Accept-Language": "ru,en;q=0.9",
    "Referer": "https://www.khl.ru/",
}

# ------------ App ------------
app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"ok": True, "engine": "ready"})

# ------------ Helpers ------------

def pdf_url(season: str, uid: str) -> str:
    return f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"

def proxied_pdf_url(season: str, uid: str) -> str:
    assert KHL_PDF_PROXY, "proxy not configured"
    # твой воркер ждёт путь /khlpdf/<season>/<uid>/game-<uid>-start-ru.pdf
    return f"{KHL_PDF_PROXY}/khlpdf/{season}/{uid}/game-{uid}-start-ru.pdf"

def load_pdf_bytes(season: str, uid: str) -> Tuple[bytes, str]:
    """
    Возвращает (pdf_bytes, source_url). Сначала пробуем воркер, иначе напрямую.
    """
    last_err = None

    # 1) Пробуем прокси (если задан)
    if KHL_PDF_PROXY:
        try:
            u = proxied_pdf_url(season, uid)
            r = requests.get(u, timeout=25)
            if r.ok and r.content.startswith(b"%PDF"):
                return r.content, u
            last_err = f"proxy status={r.status_code}"
        except Exception as e:
            last_err = f"proxy error: {e}"

    # 2) Прямая загрузка
    try:
        u2 = pdf_url(season, uid)
        r = requests.get(u2, headers=DEFAULT_HEADERS, timeout=25)
        if r.ok and r.content.startswith(b"%PDF"):
            return r.content, u2
        last_err = f"direct status={r.status_code}"
    except Exception as e:
        last_err = f"direct error: {e}"

    raise RuntimeError(f"unable to fetch pdf ({last_err})")

# ---- text extract with fitz ----

def extract_text_pages(pdf_bytes: bytes, max_pages: int = 2) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF not available")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = min(len(doc), max_pages)
    chunks = []
    for i in range(pages):
        page = doc.load_page(i)
        # текстовый слой (без OCR)
        chunks.append(page.get_text("text"))
    return "\n".join(chunks)

# ---- parsing ----

ROLE_WORDS = r"(Главн\w*\s+судья|Линейн\w*\s+судья|Резервн\w+)"
DATE_RE = r"(\d{2}\.\d{2}\.\d{4})"
TIME_RE = r"(\d{2}[:\.]\d{2})"

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s, flags=re.M).strip()

def split_russian_names(line: str) -> List[str]:
    """
    Грубый сплит списка ФИО в одну строку.
    Делим по шаблонам 'Фамилия Имя' + возможное Отчество.
    """
    # берём последовательности из 2–3 слов на кириллице с заглавной
    tokens = re.findall(r"[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}", line)
    return [normalize_spaces(x) for x in tokens]

@dataclass
class ParsedRefs:
    main: List[str]
    linesmen: List[str]
    raw_lines: List[str]

def parse_referees_from_text(text: str) -> ParsedRefs:
    """
    Ищем блок судей. В PDF обычно есть строки из серии:
      Главный судья ... Линейный судья ...
      <список фамилий в одной строке>
    """
    lines = [normalize_spaces(x) for x in text.splitlines() if normalize_spaces(x)]
    raw = []
    main: List[str] = []
    linesmen: List[str] = []

    # 1) найдём строку с ролями
    role_idx = None
    for i, ln in enumerate(lines):
        if re.search(ROLE_WORDS, ln, flags=re.I):
            role_idx = i
            raw.append(ln)
            break

    # 2) следующая(ие) строки — список фамилий
    if role_idx is not None:
        # соберём 1-2 следующие строки в пул
        pool = []
        for j in range(role_idx + 1, min(role_idx + 3, len(lines))):
            if re.search(ROLE_WORDS, lines[j], flags=re.I):
                # новая «шапка» — прекращаем
                break
            pool.append(lines[j])
            raw.append(lines[j])

        merged = " ".join(pool)

        # распилим на имена
        names = split_russian_names(merged)

        # эвристика распределения: сначала идут 2 главных, затем 2 линейных (или 1 и 1)
        # если нашли >= 4 — делим пополам 2+2; если 3 — 2 главных + 1 линейный; если 2 — 1+1, и т.д.
        if len(names) >= 4:
            main = names[:2]
            linesmen = names[2:4]
        elif len(names) == 3:
            main = names[:2]
            linesmen = names[2:]
        elif len(names) == 2:
            # чаще всего это главный + линейный (бывает не полная сетка)
            main = [names[0]]
            linesmen = [names[1]]
        elif len(names) == 1:
            # хоть что-то
            main = [names[0]]

    return ParsedRefs(main=main, linesmen=linesmen, raw_lines=raw)

def parse_meta_from_text(text: str) -> Dict[str, Any]:
    """
    Пытаемся вытащить дату/время и команды (верх PDF).
    Это эвристики: мы берём первые ~30 строк и ловим
    - ДАТА: 25.10.2025 (или 25.10.2025 19:00)
    - Команды: две *большими* фразами (КИРИЛЛИЦА/КАПС)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = lines[:30]

    date_m = None
    time_m = None
    for ln in head:
        m1 = re.search(DATE_RE, ln)
        m2 = re.search(TIME_RE, ln)
        if m1 and not date_m:
            date_m = m1.group(1)
        if m2 and not time_m:
            t = m2.group(1).replace(".", ":")
            if re.match(r"^\d{2}:\d{2}$", t):
                time_m = t
        if date_m and time_m:
            break

    # команды — две «громкие» строки (кириллица, ≥ 2 слова, много заглавных)
    caps = []
    for ln in head:
        if re.search(r"[А-ЯЁ]{2,}", ln) and len(ln) >= 6 and "судья" not in ln.lower():
            # фильтруем «обновлено», «главный судья» и т.д.
            if re.search(ROLE_WORDS, ln, flags=re.I):
                continue
            caps.append(normalize_spaces(ln))
    teams = []
    # возьмём уникальные подрядные крупные строки
    for c in caps:
        if not teams or (teams and teams[-1] != c):
            teams.append(c)
        if len(teams) >= 2:
            break

    meta = {
        "date": date_m,
        "time_msk": time_m,
        "teams": {"home": teams[0] if len(teams) >= 1 else None,
                  "away": teams[1] if len(teams) >= 2 else None},
    }
    return meta

# ------------ Endpoint ------------

@app.route("/extract")
def extract():
    t0 = time.time()
    season = request.args.get("season", "").strip()
    uid = request.args.get("uid", "").strip()
    mode = (request.args.get("mode", "refs")).strip().lower()

    if not (season and uid):
        return jsonify({"ok": False, "error": "params 'season' and 'uid' required"}), 400

    try:
        pdf_bytes, src = load_pdf_bytes(season, uid)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502

    # сейчас реализуем refs (+ meta). words/goalies можно докинуть позже.
    if mode not in ("refs", "all"):
        return jsonify({"ok": False, "error": "mode must be 'refs' or 'all' (refs supported now)"}), 400

    try:
        text = extract_text_pages(pdf_bytes, max_pages=2)
    except Exception as e:
        return jsonify({"ok": False, "error": f"text-extract failed: {e}", "source_url": src}), 500

    refs = parse_referees_from_text(text)
    meta = parse_meta_from_text(text)

    out: Dict[str, Any] = {
        "ok": True,
        "engine": "words-refs",
        "duration_s": round(time.time() - t0, 3),
        "source_url": src,
        "match": {
            "season": season,
            "uid": uid,
            "date": meta.get("date"),
            "time_msk": meta.get("time_msk"),
            "teams": meta.get("teams"),
        },
        "referees": {
            "main": refs.main,
            "linesmen": refs.linesmen,
        },
        "_debug": {
            "raw_lines": refs.raw_lines
        }
    }
    return jsonify(out)

# ------------ Local run ------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
