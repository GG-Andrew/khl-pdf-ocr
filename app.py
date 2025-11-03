import os
import io
import re
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import requests
import cloudscraper
from flask import Flask, request

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# ------------------------------
# Flask + JSON UTF-8 без \uXXXX
# ------------------------------
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

def j(data: dict, status: int = 200):
    return app.response_class(
        json.dumps(data, ensure_ascii=False),
        status=status,
        mimetype="application/json; charset=utf-8",
    )

# ------------------------------
# Конфигурация загрузки PDF
# ------------------------------
KHL_PDF_PROXY = os.getenv("KHL_PDF_PROXY", "").rstrip("/")
RAW_HEADERS = os.getenv("KHL_PDF_HEADERS", "").strip()
EXTRA_HEADERS = {}
if RAW_HEADERS:
    try:
        EXTRA_HEADERS = json.loads(RAW_HEADERS)
    except Exception:
        EXTRA_HEADERS = {"Referer": "https://www.khl.ru"}
else:
    EXTRA_HEADERS = {"Referer": "https://www.khl.ru"}

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
)

def build_pdf_url(season: str, uid: str) -> str:
    return f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"

def build_proxy_url(season: str, uid: str) -> Optional[str]:
    if not KHL_PDF_PROXY:
        return None
    return f"{KHL_PDF_PROXY}/{season}/{uid}/game-{uid}-start-ru.pdf"

def fetch_pdf_bytes(season: str, uid: str) -> bytes:
    """
    1) Пытаемся cloudscraper напрямую (часто хватает)
    2) При 403 / неуспехе — через Cloudflare Worker-прокси
    """
    url = build_pdf_url(season, uid)

    # direct try
    try:
        scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
        r = scraper.get(url, headers={"User-Agent": UA, **EXTRA_HEADERS}, timeout=20)
        if r.status_code == 200 and r.content and r.content.startswith(b"%PDF"):
            return r.content
        # если не PDF или 4xx — упадём в прокси
    except Exception:
        pass

    # proxy try
    purl = build_proxy_url(season, uid)
    if purl:
        r2 = requests.get(purl, headers={"User-Agent": UA}, timeout=25)
        if r2.status_code == 200 and r2.content and r2.content.startswith(b"%PDF"):
            return r2.content
        raise RuntimeError(f"proxy fetch failed: {r2.status_code}")
    raise RuntimeError("direct fetch failed (and no proxy)")

# ------------------------------
# Извлечение МЕТА (date/time/teams) через words
# ------------------------------
RE_DATE = re.compile(r"(\d{2}\.\d{2}\.\d{4})")
RE_TIME = re.compile(r"\b([01]\d|2[0-3]):[0-5]\d\b")

def parse_match_meta_words(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Извлекаем быстрой «текстовой» выборкой:
    - teams (две верхние «капсом» строки около шапки)
    - date/time_msk (любые первые встреченные в верхней половине)
    Сильно упрощённо, но на твоих протоколах даёт стабильный результат.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    # берем полный текст и первые 80 строк — этого хватает
    full_text = page.get_text("text")
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    head = lines[:80]

    # Дата/время
    date = ""
    time_msk = ""
    for ln in head:
        if not date:
            m = RE_DATE.search(ln)
            if m:
                date = m.group(1)
        if not time_msk:
            m = RE_TIME.search(ln)
            if m:
                time_msk = m.group(1)
        if date and time_msk:
            break

    # Команды: ищем две длинные UPPERCASE-строки рядом со словом «СОСТАВ»
    # или рядом с «Гости/Хозяева». Набор эвристик:
    team_home = ""
    team_away = ""

    # пробуем по ключам
    idx_comp = -1
    keys = ["СОСТАВ КОМАНДЫ", "СОСТАВ", "СОСТАВЫ", "СОСТАВ КОМАНД"]
    for i, ln in enumerate(head):
        if any(k in ln.upper() for k in keys):
            idx_comp = i
            break

    def looks_team(s: str) -> bool:
        s2 = s.replace("Ё", "Е")
        # длинные капс-строки на кириллице без лишних знаков
        return (len(s2) >= 8) and (s2 == s2.upper()) and re.search(r"[А-Я]", s2)

    candidates: List[str] = []
    if idx_comp != -1:
        # смотрим окно +-10 строк
        lo = max(0, idx_comp - 10)
        hi = min(len(head), idx_comp + 10)
        for ln in head[lo:hi]:
            if looks_team(ln):
                candidates.append(ln)
    else:
        # fallback — возьмём первые 20 «капсовых» строк в шапке
        for ln in head[:40]:
            if looks_team(ln):
                candidates.append(ln)

    # оставим уникальные в порядке появления
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    if len(uniq) >= 2:
        # чаще всего первая — хозяева, вторая — гости
        team_home, team_away = uniq[0], uniq[1]

    doc.close()
    return {
        "date": date,
        "time_msk": time_msk,
        "teams": {"home": team_home, "away": team_away},
    }

# ------------------------------
# OCR судьи (надёжный блок)
# ------------------------------
ROLE_WORDS = (
    r"(Главн\w*\s+судья|Линейн\w*\s+судья|Резервн\w+(\s+главн\w+)?(\s+линейн\w+)?\s*судья)"
)

def ocr_refs_first_page(pdf_bytes: bytes, dpi: int = 300) -> Dict[str, Any]:
    """
    Рендерим первую страницу, OCR rus+eng, вытаскиваем блок судей.
    """
    t0 = time.time()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    text = pytesseract.image_to_string(img, lang="rus+eng")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Найдём строку, где перечисляются роли, и следующую, где имена
    # Часто встречается «Главный судья Главный судья Линейный судья Линейный судья»
    roles_line_idx = -1
    for i, ln in enumerate(lines):
        if re.search(ROLE_WORDS, ln, flags=re.I):
            roles_line_idx = i
            break

    refs_main: List[str] = []
    refs_lines: List[str] = []
    raw_lines_dbg: List[str] = []

    if roles_line_idx != -1:
        # Имён обычно на следующей строке (иногда через одну)
        name_line_idx = roles_line_idx + 1
        if name_line_idx < len(lines):
            raw_names = lines[name_line_idx]
            raw_lines_dbg = [lines[roles_line_idx], raw_names]
            # Разбиваем по пробелам и склеиваем парами Фамилия Имя
            tokens = [t for t in re.split(r"[,\s]+", raw_names) if t]
            # склейка «Фамилия Имя» — грубая, но эффективная
            pairs: List[str] = []
            buf = []
            for t in tokens:
                buf.append(t)
                if len(buf) == 2:
                    pairs.append(" ".join(buf))
                    buf = []
            # если не кратно двум — добрособираем
            if buf:
                pairs.append(" ".join(buf))

            # теперь распределим по ролям: 2 главных + 2 линейных обычно
            # Сначала вытащим все роли (по словам 'Главн' и 'Линейн') из роли-строки:
            role_counts = re.findall(r"(Главн\w+|Линейн\w+)", lines[roles_line_idx], flags=re.I)
            # попытаемся раздать в порядке
            m, l = [], []
            idx = 0
            for role in role_counts:
                if idx >= len(pairs):
                    break
                if role.lower().startswith("главн"):
                    m.append(pairs[idx])
                else:
                    l.append(pairs[idx])
                idx += 1
            # если не хватило — добрасываем остатками
            while idx < len(pairs) and (len(m) < 2 or len(l) < 2):
                (m if len(m) < 2 else l).append(pairs[idx])
                idx += 1

            refs_main = m
            refs_lines = l

    doc.close()

    # Чистим «Резервный …» и мусорные слова, нормализуем пробелы
    def clean_name(s: str) -> str:
        s = re.sub(r"\b(Резервн\w*|судья|судьи)\b", " ", s, flags=re.I)
        s = re.sub(r"\s{2,}", " ", s).strip(" ,.;:")
        # Случаи типа «Алексей Седов Егор» — оставим пары «Фамилия Имя»
        # Если три токена — попробуем выбрать 2 «похожих» на ФИ:
        toks = s.split()
        if len(toks) >= 3:
            # простая эвристика — возьмём первые два
            s = " ".join(toks[:2])
        return s

    refs_main = [clean_name(x) for x in refs_main if x]
    refs_lines = [clean_name(x) for x in refs_lines if x]

    # финальный ремонт: убрать дубли и пустые
    def uniq_keep(seq: List[str]) -> List[str]:
        out, seen = [], set()
        for x in seq:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    refs_main = uniq_keep(refs_main)
    refs_lines = uniq_keep(refs_lines)

    return {
        "main": refs_main,
        "linesmen": refs_lines,
        "_raw_lines": raw_lines_dbg,
        "_duration_s": round(time.time() - t0, 3),
    }

# ------------------------------
# Flask endpoints
# ------------------------------
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

    url = build_pdf_url(season, uid)
    t0 = time.time()
    try:
        pdf_bytes = fetch_pdf_bytes(season, uid)
    except Exception as e:
        return j({"ok": False, "error": f"http fetch failed: {e}"}, 502)

    out: Dict[str, Any] = {"ok": True, "source_url": url}

    # words: матч-мета
    if mode in ("words", "all"):
        try:
            meta = parse_match_meta_words(pdf_bytes)
            out.setdefault("match", {"season": season, "uid": uid})
            out["match"].update(meta)
            out["engine"] = "words" if mode == "words" else out.get("engine", "all")
        except Exception as e:
            out.setdefault("warnings", []).append(f"words-meta: {e}")

    # refs: судьи
    if mode in ("refs", "all"):
        try:
            refs = ocr_refs_first_page(pdf_bytes, dpi=300)
            out["referees"] = {"main": refs["main"], "linesmen": refs["linesmen"]}
            out["engine"] = "ocr-refs" if mode == "refs" else out.get("engine", "all")
            if debug:
                out["_debug"] = {"raw_lines": refs.get("_raw_lines", [])}
        except Exception as e:
            return j({"ok": False, "error": f"ocr-refs failed: {e}"}, 500)

    # только refs?
    if mode == "refs":
        pass
    elif mode == "words":
        out.setdefault("match", {"season": season, "uid": uid})
    elif mode == "all":
        out.setdefault("match", {"season": season, "uid": uid})
    else:
        return j({"ok": False, "error": "mode must be refs|words|all"}, 400)

    out["duration_s"] = round(time.time() - t0, 3)
    return j(out)

# ------------------------------
# gunicorn entrypoint
# ------------------------------
if __name__ == "__main__":
    # для локального запуска: python app.py
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
