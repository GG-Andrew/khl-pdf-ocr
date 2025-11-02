# app.py
import os, io, time, re, json
from datetime import datetime
from flask import Flask, request, jsonify, Response
import requests
import cloudscraper
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# ----------------------------- Config -----------------------------
app = Flask(__name__)

# Render/Heroku style port
PORT = int(os.environ.get("PORT", "8000"))

# Tesseract (опционально, для OCR-судей)
# На Render обычно пути уже корректные; ниже — безопасные дефолты.
TESSDATA_PREFIX = os.environ.get("TESSDATA_PREFIX") or "/usr/share/tesseract-ocr/5/tessdata"
os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

# HTTP
SCRAPER = cloudscraper.create_scraper(
    browser={"custom": "chrome"},
    delay=0.2
)

# Русские месяцы для дат
MONTHS_RU = {
    "января":"01","февраля":"02","марта":"03","апреля":"04","мая":"05","июня":"06",
    "июля":"07","августа":"08","сентября":"09","октября":"10","ноября":"11","декабря":"12"
}

# ----------------------- Utils: HTTP/PDF fetch --------------------
def pdf_url(season: str, uid: str) -> str:
    return f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"

def khl_referer(season: str, uid: str) -> str:
    return f"https://www.khl.ru/game/{season}/{uid}/preview/"

def fetch_pdf_bytes(season: str, uid: str) -> bytes:
    url = pdf_url(season, uid)
    ref = khl_referer(season, uid)
    headers = {
        "Referer": ref,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    # cloudscraper иногда проседает на pdf — подстрахуем через requests
    try:
        r = SCRAPER.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return r.content
    except Exception as e:
        # fallback
        rr = requests.get(url, headers=headers, timeout=20)
        rr.raise_for_status()
        return rr.content

# ----------------------- PyMuPDF helpers --------------------------
def extract_words(pdf_bytes: bytes):
    """Возвращает список слов со структурами x0,x1,y0,y1,text (первая страница)."""
    words = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return words
        page = doc.load_page(0)
        for w in page.get_text("words"):  # (x0, y0, x1, y1, "text", block_no, line_no, word_no)
            words.append({
                "x0": float(w[0]), "y0": float(w[1]),
                "x1": float(w[2]), "y1": float(w[3]),
                "text": str(w[4])
            })
    return words

def extract_fulltext(pdf_bytes: bytes):
    """Весь текст страницы 0 — строками."""
    lines = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return lines
        page = doc.load_page(0)
        raw = page.get_text("text")
        for ln in raw.splitlines():
            t = ln.strip()
            if t:
                lines.append(t)
    return lines

def page_png_for_ocr(pdf_bytes: bytes, zoom=2.0):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes))

# --------------------- Lines grouping / columns -------------------
def split_columns_by_median(words):
    """Делит слова на левый/правый столбец по медиане X-центров и группирует в строки."""
    if not words:
        return [], []
    centers = sorted([(w["x0"] + w["x1"]) / 2.0 for w in words])
    mid = centers[len(centers)//2]

    left = [w for w in words if (w["x0"] + w["x1"]) / 2.0 <= mid]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2.0 > mid]

    def to_lines(ws, y_tol=3.0):
        ws_sorted = sorted(ws, key=lambda z: (round((z["y0"]+z["y1"])/2.0, 1), z["x0"]))
        lines = []
        cur_y = None
        cur_line = []
        for w in ws_sorted:
            cy = round((w["y0"] + w["y1"]) / 2.0, 1)
            if cur_y is None or abs(cy - cur_y) > y_tol:
                if cur_line:
                    lines.append([t["text"] for t in cur_line])
                cur_line = [w]
                cur_y = cy
            else:
                cur_line.append(w)
        if cur_line:
            lines.append([t["text"] for t in cur_line])
        # Преобразуем в простые токены
        return [[t["text"] for t in line] for line in lines]

    return to_lines(left), to_lines(right)

# -------------------------- Header parse --------------------------
_SKIP_CAPS = {"СОСТАВЫ КОМАНД", "СОСТАВЫ", "КОМАНД", "СОСТАВ", "ИГРОКИ"}

def pick_teams_from_lines(lines):
    """Ищем две ВЕРХНИЕ all-caps строки как имена команд (без служебных слов)."""
    caps = []
    for ln in lines[:40]:
        up = ln.strip().upper()
        if not up:
            continue
        if any(ch.isalpha() for ch in up):
            # пусть это «крупные» строки
            if re.fullmatch(r"[A-ZА-ЯЁ0-9\.\- ()]+", up):
                if up not in _SKIP_CAPS and len(up) >= 8:
                    caps.append(up)
    # эвристика: две самые длинные разные
    caps = sorted(set(caps), key=lambda s: (-len(s), s))
    if len(caps) >= 2:
        return caps[0], caps[1]
    return "", ""

def parse_russian_date(lines):
    """
    Ищем дату формата '25 октября 2025 г.' и, если есть, рядом время '19:00'.
    Возвращаем {"date":"YYYY-MM-DD","time":"HH:MM"} (может быть пустое).
    """
    txt = " ".join(lines)
    # день месяц(словом) год
    m = re.search(r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})\s*г?\.?", txt, flags=re.I)
    date_iso, tm = "", ""
    if m:
        d, mon, y = m.group(1), m.group(2).lower(), m.group(3)
        mm = MONTHS_RU.get(mon, "01")
        date_iso = f"{y}-{mm}-{int(d):02d}"
        mt = re.search(r"(\d{1,2}:\d{2})", txt)
        if mt:
            tm = mt.group(1)
    return {"date": date_iso, "time": tm}

# -------------------------- Referees (OCR) ------------------------
ROLE_WORDS = {"главный", "главные", "линейный", "линейные", "судья", "судьи",
              "резервный", "резервные", "обновлено", "обновлёно", "обновлено:"}

SKIP_TOK = {"р.", "p.", "г.", "г", "р", "p"}

def clean_role_words(name: str) -> str:
    if not name:
        return name
    name = re.sub(r"\b(Главн\w*|Линейн\w*|судья|судьи|Резервн\w*|Обновлен\w*)\b", " ", name, flags=re.I)
    name = re.sub(r"\s{2,}", " ", name).strip(" ,.;:")
    return name

def group_fio(tokens):
    """Группирует как 'Фамилия Имя' по 2 токена, отбрасывая служебные."""
    out = []
    cur = []
    for t in tokens:
        tt = t.strip(".,;:() ").replace("ё", "ё")
        if not tt:
            continue
        low = tt.lower()
        if low in ROLE_WORDS or low in SKIP_TOK:
            continue
        if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", tt):
            continue
        cur.append(tt)
        if len(cur) == 2:
            out.append(" ".join(cur))
            cur = []
    # если нечётное — отбросим хвост
    return out

def parse_referees_ocr(pdf_bytes: bytes, debug=False):
    img = page_png_for_ocr(pdf_bytes, zoom=2.0)
    # Tesseract без указания языков всё равно что-то вернёт (eng+rus если есть), но Render обычно имеет rus+eng.
    text = pytesseract.image_to_string(img, lang=os.environ.get("TESS_LANGS", "rus+eng"))
    # Возьмём 3-4 строки вокруг ключевых слов
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # найдём строку с ролями и следующую строку с ФИО
    roles_idx = None
    for i, ln in enumerate(lines):
        if "судья" in ln.lower():
            roles_idx = i
            break
    raw_names_line = ""
    # часто ФИО на следующей строке
    if roles_idx is not None and roles_idx + 1 < len(lines):
        raw_names_line = lines[roles_idx+1]
        # иногда растянуто на две строки
        if roles_idx + 2 < len(lines) and len(lines[roles_idx+2]) > 6:
            raw_names_line += " " + lines[roles_idx+2]

    raw_names_line = clean_role_words(raw_names_line)
    tokens = re.split(r"\s+", raw_names_line)
    fio = group_fio(tokens)

    # Назначим первые два — главные, вторые два — линейные
    main = fio[:2]
    linesmen = fio[2:4]

    result = {"main": main, "linesmen": linesmen}
    if debug:
        result["_raw_lines"] = lines
    return result

# ------------------------- Rosters / Lines ------------------------
def looks_pos(tok):
    t = tok.strip(" .").upper()
    return t in {"D", "F", "З", "Н"}  # защитник/нападающий

def looks_number(tok):
    return tok.isdigit() and (1 <= len(tok) <= 2 or tok == "00")

def normalize_name_tokens(tokens):
    # склеиваем Фамилию + Имя: берём первые 2 осмысленных слова
    cleaned = []
    for t in tokens:
        t = t.strip(" .,:;()")
        if not t:
            continue
        if t.lower() in {"з", "н", "d", "f"}:
            continue
        cleaned.append(t)
    # если первый короткий, а второй длинный — берём длинный
    out = []
    i = 0
    while i < len(cleaned):
        w = cleaned[i]
        if len(w) <= 3 and i+1 < len(cleaned) and len(cleaned[i+1]) >= 4:
            out.append(cleaned[i+1])
            i += 2
        else:
            out.append(w)
            i += 1
        if len(out) >= 2:
            break
    return " ".join(out)

def parse_lines_side(lines):
    """
    Ждём блоки по 5 игроков в линии. Формат токенов в строке:
    <POS> <Фамилия> <Имя> <номер>
    Иногда номер идёт раньше, поэтому аккуратно собираем.
    """
    out = {"1": [], "2": [], "3": [], "4": []}
    cur_key = "1"
    buf = []

    def flush_line():
        nonlocal buf, cur_key
        if len(buf) >= 5:
            out[cur_key] = buf[:5]
            buf = []
            # следующий ключ
            if cur_key == "1": cur_key = "2"
            elif cur_key == "2": cur_key = "3"
            elif cur_key == "3": cur_key = "4"

    for toks in lines:
        if not toks: 
            continue
        # упрощаем
        tokens = [t.strip() for t in toks if t.strip()]
        # ищем POS, NAME, NUMBER
        pos = None; number = None
        name = ""
        # попытаемся найти POS в первых 2-3 токенах
        for j in range(min(3, len(tokens))):
            if looks_pos(tokens[j]):
                pos = tokens[j].upper()
                # имя — между этим POS и номером
                break
        # номер — последний/предпоследний токен
        cand_nums = [t for t in tokens[-2:] if looks_number(t)]
        if cand_nums:
            number = cand_nums[0]
        # имя — всё между pos и number
        name_tokens = []
        started = False if pos else True
        for t in tokens:
            if not started and looks_pos(t):
                started = True
                continue
            if started:
                if number is not None and t == number:
                    break
                name_tokens.append(t)
        nm = normalize_name_tokens(name_tokens)
        if pos in {"З","D"}: pos_std = "D"
        elif pos in {"Н","F"}: pos_std = "F"
        else: pos_std = "F"

        if nm and number:
            buf.append({"pos": pos_std, "number": number, "name": nm})
            if len(buf) == 5:
                flush_line()

    # Если не набрали 4 пятёрки — вернём, что собрали
    return out

# --------------------------- Goalies parse ------------------------
def token_is_status(tok):
    t = tok.strip().strip("()").upper()
    if t in {"C", "С"}:
        return "starter"
    if t in {"R", "Р"}:
        return "reserve"
    return ""

def parse_goalies_side(lines):
    """
    Ищем: <номер> [В|Вр] <Фамилия> <Имя> [C|С|R|Р]
    """
    out = []
    for toks in lines:
        tokens = [t.strip() for t in toks if t.strip()]
        if not tokens:
            continue

        # номер среди первых трёх
        num_idx = None
        for j in range(min(3, len(tokens))):
            if looks_number(tokens[j]):
                num_idx = j
                break
        if num_idx is None:
            continue

        # позиция 'В' / 'Вр' рядом
        pos_idx = None
        for j in range(num_idx+1, min(num_idx+3, len(tokens))):
            t = tokens[j].lower().strip(" .")
            if t in {"в", "вр", "в."}:
                pos_idx = j
                break
        name_start = (pos_idx+1) if pos_idx is not None else (num_idx+1)

        # статус в хвосте
        status = ""
        for tt in reversed(tokens[-2:]):
            ss = token_is_status(tt)
            if ss:
                status = ss
                break

        number = tokens[num_idx]
        name = normalize_name_tokens(tokens[name_start:name_start+4])
        if name:
            out.append({"number": number, "name": name, "gk_status": status})
    return out

# ------------------------------ API -------------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "engine": "ready"})

@app.get("/extract")
def extract():
    t0 = time.time()
    season = (request.args.get("season") or "").strip()
    uid = (request.args.get("uid") or "").strip()
    mode = (request.args.get("mode") or "all").strip().lower()
    debug = request.args.get("debug") == "1"

    if not season or not uid:
        return jsonify({"ok": False, "error": "params 'season' and 'uid' required"}), 400

    try:
        pdf_bytes = fetch_pdf_bytes(season, uid)
    except Exception as e:
        return jsonify({"ok": False, "error": "http 403" if "403" in str(e) else str(e)}), 403 if "403" in str(e) else 500

    resp = {
        "ok": True,
        "engine": mode,
        "source_url": pdf_url(season, uid)
    }

    # --- header: teams/date
    lines_text = extract_fulltext(pdf_bytes)
    team_home, team_away = pick_teams_from_lines(lines_text)
    dt = parse_russian_date(lines_text)
    resp["match"] = {
        "season": season,
        "uid": uid,
        "home_team": team_home,
        "away_team": team_away,
        "date": dt.get("date", ""),
        "time": dt.get("time", "")
    }

    # универсальные заготовки
    resp.setdefault("data", {}).setdefault("home", {}).setdefault("lines", {"1": [], "2": [], "3": [], "4": []})
    resp.setdefault("data", {}).setdefault("away", {}).setdefault("lines", {"1": [], "2": [], "3": [], "4": []})
    resp["data"]["home"].setdefault("goalies", [])
    resp["data"]["away"].setdefault("goalies", [])
    resp["data"]["home"].setdefault("bench", [])
    resp["data"]["away"].setdefault("bench", [])
    resp.setdefault("referees", {"main": [], "linesmen": []})

    # предварительно получим слова/столбцы (нужно в нескольких режимах)
    words = None
    left_lines = right_lines = None
    if mode in {"words", "goalies", "all"}:
        words = extract_words(pdf_bytes)
        left_lines, right_lines = split_columns_by_median(words)

    # --- words: линии игроков (полевые)
    if mode in {"words", "all"}:
        home_lines = parse_lines_side(left_lines or [])
        away_lines = parse_lines_side(right_lines or [])
        resp["data"]["home"]["lines"] = home_lines
        resp["data"]["away"]["lines"] = away_lines

    # --- goalies
    if mode in {"goalies", "all"}:
        home_goalies = parse_goalies_side(left_lines or [])
        away_goalies = parse_goalies_side(right_lines or [])
        # иногда столбцы меняются местами — если домашние пустые, а гостевые не пустые, попробуем свапнуть
        if not home_goalies and away_goalies:
            home_goalies, away_goalies = away_goalies, home_goalies
        resp["data"]["home"]["goalies"] = home_goalies
        resp["data"]["away"]["goalies"] = away_goalies

    # --- referees (OCR)
    if mode in {"refs", "all"}:
        refs = parse_referees_ocr(pdf_bytes, debug=debug)
        resp["referees"] = {
            "main": refs.get("main", []),
            "linesmen": refs.get("linesmen", [])
        }
        if debug:
            resp.setdefault("_debug", {})["_raw_ref_lines"] = refs.get("_raw_lines", [])

    resp["duration_s"] = round(time.time() - t0, 3)
    return jsonify(resp)

# ---------------------------- Main --------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
