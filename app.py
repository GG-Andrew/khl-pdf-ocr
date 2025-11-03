# app.py
import os, io, re, json, time
import fitz  # PyMuPDF
import requests
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify

try:
    import pytesseract
    HAS_TESS = True
except Exception:
    HAS_TESS = False

# -------------------- Config --------------------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

PDF_PROXY_BASE = os.getenv("PDF_PROXY_BASE", "").rstrip("/")
DEFAULT_LANG = "rus+eng"

app = Flask(__name__)

# -------------------- Helpers --------------------
def build_pdf_url(season: str, uid: str) -> str:
    return f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"

def build_preview_ref(season: str, uid: str) -> str:
    return f"https://www.khl.ru/game/{season}/{uid}/preview/"

def fetch_pdf_bytes(season: str, uid: str):
    """Берём PDF через воркер если задан, иначе прямым запросом с реферером."""
    src_pdf = build_pdf_url(season, uid)

    if PDF_PROXY_BASE:
        proxy = f"{PDF_PROXY_BASE}/{season}/{uid}/game-{uid}-start-ru.pdf"
        r = requests.get(proxy, timeout=30)
        r.raise_for_status()
        return r.content, proxy

    headers = {
        "User-Agent": UA,
        "Referer": build_preview_ref(season, uid),
        "Accept": "application/pdf,*/*",
        "Accept-Language": "ru,en;q=0.9",
        "Connection": "close",
    }
    r = requests.get(src_pdf, headers=headers, timeout=30)
    r.raise_for_status()
    return r.content, src_pdf

def text_words_by_columns(pdf_bytes: bytes, page_no: int = 0, split_x: float | None = None):
    """
    Возвращает (left_lines, right_lines, all_words)
    all_words — список кортежей (x0,y0,x1,y1,word).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_no)
    words = page.get_text("words")  # [x0,y0,x1,y1,"text", block_no, line_no, word_no]
    words = sorted(words, key=lambda w: (w[1], w[0]))  # sort by y, then x

    # Определим середину страницы, если не задано
    if split_x is None:
        split_x = page.rect.width / 2

    left = [w for w in words if w[0] < split_x]
    right = [w for w in words if w[0] >= split_x]

    def words_to_lines(ws, tol=3.0):
        out = []
        cur = []
        last_y = None
        for x0,y0,x1,y1,txt,*_ in ws:
            if last_y is None or abs(y0 - last_y) <= tol:
                cur.append((x0, txt))
                last_y = y0
            else:
                cur.sort(key=lambda r: r[0])
                out.append(" ".join(t for _,t in cur).strip())
                cur = [(x0, txt)]
                last_y = y0
        if cur:
            cur.sort(key=lambda r: r[0])
            out.append(" ".join(t for _,t in cur).strip())
        return out

    return words_to_lines(left), words_to_lines(right), words

def norm(s: str) -> str:
    return re.sub(r"\s{2,}", " ", s or "").strip(" ,.;:\u00A0")

def clean_role_words(name: str) -> str:
    if not name: return name
    name = re.sub(r"\b(Главн\w*|Линейн\w*|судья|судьи|Резервн\w*)\b", " ", name, flags=re.I)
    return norm(name)

# -------------------- Parsing: meta --------------------
META_TEAM_RX = r"[А-ЯA-Z][а-яa-zё\- ]{2,}"
DATE_RX = r"(\d{2}\.\d{2}\.\d{4})"
TIME_RX = r"(\d{2}:\d{2})"

def parse_meta_from_lines(lines_all):
    meta = {"date": None, "time": None, "home": None, "away": None}
    # Ищем дату/время
    text = " ".join(lines_all)
    m = re.search(DATE_RX, text)
    if m: meta["date"] = m.group(1)
    t = re.search(TIME_RX, text)
    if t: meta["time"] = t.group(1)

    # Команды — часто встречается в заголовке вида "НЕФТЕХИМИК НИЖНЕКАМСК – САЛАВАТ ЮЛАЕВ УФА"
    # Пробуем найти через дефис/длинное тире
    team_line = None
    for ln in lines_all[:20]:
        if "–" in ln or "-" in ln:
            team_line = ln
            break
    if team_line:
        parts = re.split(r"–|-", team_line)
        if len(parts) >= 2:
            meta["home"] = norm(parts[0])
            meta["away"] = norm(parts[1])
    return meta

# -------------------- Parsing: referees --------------------
def parse_refs_by_words(lines_all):
    """
    Пробуем вытащить судей из текстового слоя.
    Ищем блок, где встречаются «Главный судья»/«Линейный судья».
    """
    joined = "\n".join(lines_all)
    block = None
    for i, ln in enumerate(lines_all):
        if re.search(r"Главн\w*\s+судья", ln, re.I):
            block = " ".join(lines_all[i:i+4])
            break
    if not block:
        # иной вариант: всё одной строкой
        m = re.search(r"(Главн\w* судья.*)", joined, re.I)
        if m:
            block = m.group(1)

    if not block:
        return {"main": [], "linesmen": []}

    # Отрезаем служебные куски
    block = re.sub(r"Обновлено.*", " ", block, flags=re.I)
    # Пытаемся разложить по ролям
    names = []
    # часто имена идут подряд: "Морозов Сергей Васильев Алексей Седов Егор Шишло Дмитрий"
    # разделим на возможные ФИО-пары (простая эвристика "Слово Слово")
    tokens = block.split()
    pairs = []
    i = 0
    while i+1 < len(tokens):
        w1, w2 = tokens[i], tokens[i+1]
        if re.match(r"^[А-ЯA-ZЁ][а-яa-zё\-]+$", w1) and re.match(r"^[А-ЯA-ZЁ][а-яa-zё\-]+$", w2):
            pairs.append(f"{w1} {w2}")
            i += 2
        else:
            i += 1

    # И теперь разделяем по ролям, опираясь на ключевые слова вокруг.
    main, linesmen = [], []
    if "Главный судья" in block and "Линейный судья" in block:
        # допустим первые 2 пары — главные, следующие 2 — линейные (типовой кейс)
        # если меньше — подрежем
        main = pairs[:2]
        linesmen = pairs[2:4]
    else:
        # fallback: считаем первых двоих главными
        main = pairs[:2]
        linesmen = pairs[2:4]

    main = [clean_role_words(x) for x in main if x]
    linesmen = [clean_role_words(x) for x in linesmen if x]
    return {"main": main, "linesmen": linesmen}

def parse_refs_by_ocr(pdf_bytes: bytes):
    """OCR только шапку (первая страница)."""
    if not HAS_TESS:
        return {"main": [], "linesmen": []}

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)

    txt = pytesseract.image_to_string(img, lang=DEFAULT_LANG, config="--psm 6")
    txt = txt.replace("\xa0", " ")
    lines = [norm(x) for x in txt.splitlines() if norm(x)]

    # Ищем строку с ролями и следующую с ФИО
    raw = []
    for i, ln in enumerate(lines):
        if re.search(r"Главн\w*\s+судья", ln, re.I) or re.search(r"Линейн\w*\s+судья", ln, re.I):
            raw.extend(lines[i:i+4])
            break
    if not raw:
        return {"main": [], "linesmen": []}

    block = " ".join(raw)
    block = re.sub(r"Обновлено.*", " ", block, flags=re.I)

    # выделим пары "Слово Слово"
    tokens = block.split()
    pairs = []
    i = 0
    while i+1 < len(tokens):
        w1, w2 = tokens[i], tokens[i+1]
        if re.match(r"^[А-ЯA-ZЁ][а-яa-zё\-]+$", w1) and re.match(r"^[А-ЯA-ZЁ][а-яa-zё\-]+$", w2):
            pairs.append(f"{w1} {w2}")
            i += 2
        else:
            i += 1

    main = pairs[:2]
    linesmen = pairs[2:4]
    main = [clean_role_words(x) for x in main if x]
    linesmen = [clean_role_words(x) for x in linesmen if x]
    return {"main": main, "linesmen": linesmen}

def parse_referees(pdf_bytes: bytes, lines_all: list[str], want_debug: bool = False):
    start = time.time()
    refs = parse_refs_by_words(lines_all)
    engine = "words"
    if len(refs.get("main", [])) < 1 or len(refs.get("linesmen", [])) < 1:
        ocr_refs = parse_refs_by_ocr(pdf_bytes)
        if ocr_refs["main"] or ocr_refs["linesmen"]:
            refs = ocr_refs
            engine = "ocr-refs"
    if want_debug:
        refs["_raw_lines"] = lines_all[:60]
    return refs, engine, time.time() - start

# -------------------- Parsing: lines (home/away) --------------------
POS_MAP = {"З": "D", "Д": "D", "Н": "F", "F": "F", "D": "D", "В": "G", "G": "G"}

def split_teams_blocks(left_lines, right_lines):
    """Простая эвристика: левый блок — хозяева, правый — гости (как в большинстве PDF)."""
    return {"home": left_lines, "away": right_lines}

def row_to_player(row: str):
    # ожидаем шаблон: "<pos> <Фамилия ...> <Номер>"
    # Примеры: "З Хлыстов Никита 7", "Н Жафяров Дамир 88"
    m = re.search(r"^([ЗДНDFVG])\s+(.+?)\s+(\d{1,2})$", row.strip())
    if not m:
        # иногда номер перед фамилией: "7 Хлыстов Никита З"
        m2 = re.search(r"^(\d{1,2})\s+(.+?)\s+([ЗДНDFVG])$", row.strip())
        if not m2:
            return None
        num, name, pos = m2.group(1), m2.group(2), m2.group(3)
    else:
        pos, name, num = m.group(1), m.group(2), m.group(3)

    pos = POS_MAP.get(pos, pos)
    return {"pos": pos, "name": norm(name), "number": num}

def parse_team_lines(team_lines: list[str]):
    """
    Ищем якоря 'Звено 1/2/3/4' и собираем 5 игроков после каждого.
    Это твой «engine":"words".
    """
    out = {"goalies": [], "lines": {"1": [], "2": [], "3": [], "4": []}, "bench": []}

    # вратари: строки, где явно «Вратар»
    for ln in team_lines:
        if re.search(r"Вратар", ln, re.I):
            # выцепим пары "Фамилия И." / "Фамилия Имя" и номер
            # пример: "Вратари: Иванов Иван 30 Петров Петр 60"
            nums = re.findall(r"([А-ЯA-ZЁ][а-яa-zё\-]+ [А-ЯA-ZЁ][а-яa-zё\-]+)\s+(\d{1,2})", ln)
            for nm, num in nums:
                out["goalies"].append({"name": norm(nm), "number": num})
            break

    # звенья
    idx = 0
    while idx < len(team_lines):
        ln = team_lines[idx]
        z = re.search(r"Звено\s*(\d)", ln, re.I)
        if z:
            zi = z.group(1)
            players = []
            j = idx + 1
            while j < len(team_lines) and len(players) < 5:
                p = row_to_player(team_lines[j])
                if p: players.append(p)
                j += 1
            out["lines"][zi] = players
            idx = j
        else:
            idx += 1

    return out

def parse_lines(pdf_bytes: bytes):
    left, right, words = text_words_by_columns(pdf_bytes)
    lines_all = left + right
    meta = parse_meta_from_lines(lines_all)

    teams_blocks = split_teams_blocks(left, right)
    home_parsed = parse_team_lines(teams_blocks["home"])
    away_parsed = parse_team_lines(teams_blocks["away"])

    data = {
        "home": {"team": meta.get("home") or "", **home_parsed},
        "away": {"team": meta.get("away") or "", **away_parsed},
    }
    return data, meta, lines_all

# -------------------- Flask endpoints --------------------
@app.route("/health")
def health():
    return jsonify({"ok": True, "engine": "ready"})

@app.route("/extract")
def extract():
    season = request.args.get("season", "").strip()
    uid = request.args.get("uid", "").strip()
    mode = request.args.get("mode", "all").strip().lower()
    debug = request.args.get("debug", "0") == "1"

    if not season or not uid:
        return jsonify({"ok": False, "error": "season & uid required"}), 400

    t0 = time.time()
    try:
        pdf_bytes, used_url = fetch_pdf_bytes(season, uid)
    except Exception as e:
        return jsonify({"ok": False, "error": f"http {getattr(e, 'response', None) and e.response.status_code or 'err'}"}), 502

    # базовый разбор линий/меты (нужен большинству режимов)
    lines_data, meta, all_lines = parse_lines(pdf_bytes)

    result = {
        "ok": True,
        "engine": "words",
        "season": season,
        "uid": uid,
        "meta": {
            "date": meta.get("date"),
            "time": meta.get("time"),
            "home": meta.get("home"),
            "away": meta.get("away"),
        },
        "data": lines_data,
        "referees": {"main": [], "linesmen": []},
        "source_url": used_url,
    }

    if mode in ("refs", "all"):
        refs, eng, dur = parse_referees(pdf_bytes, all_lines, want_debug=debug)
        result["referees"] = refs
        result["engine"] = eng

    if debug:
        result["_debug"] = {"lines_sample": all_lines[:80]}

    result["duration_s"] = round(time.time() - t0, 3)
    return jsonify(result)


if __name__ == "__main__":
    # для локального запуска (Render использует gunicorn)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
