import os, io, re, json, time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import requests
from flask import Flask, request, jsonify
import fitz  # PyMuPDF
from PIL import Image

try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

UA = "BETON-KHL OCR/2.1 (+https://results.beton-khl.ru)"
TIMEOUT = 20
PDF_HEADERS = {
    "User-Agent": UA,
    "Accept": "*/*",
    "Accept-Language": "ru,en;q=0.9",
    "Connection": "keep-alive",
}
REF_PREVIEW_TMPL = "https://www.khl.ru/game/{season}/{uid}/preview/"

PLAYERS_MASTER = "players_master.csv"
REFEREES_MASTER = "referees_master.csv"

app = Flask(__name__)

# ========== helpers: load canonical maps ==========
def load_master(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    mp: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = re.split(r"[;,]\s*", ln)
            if len(parts) >= 2:
                raw, canon = parts[0].strip(), parts[1].strip()
                if raw and canon:
                    mp[raw.lower()] = canon
    return mp

PLAYERS_MAP = load_master(PLAYERS_MASTER)
REFS_MAP    = load_master(REFEREES_MASTER)

def canonize(name: str, mapping: Dict[str,str]) -> str:
    key = (name or "").lower().strip()
    return mapping.get(key, name)

# ========== HTTP ==========
def fetch_pdf(url: str, season: int|None=None, uid: int|None=None) -> bytes:
    headers = PDF_HEADERS.copy()
    if season and uid:
        ref = REF_PREVIEW_TMPL.format(season=season, uid=uid)
    else:
        m = re.search(r"/(\d{4})/(\d{6})/", url)
        ref = REF_PREVIEW_TMPL.format(season=m.group(1), uid=m.group(2)) if m else "https://www.khl.ru/"
    headers["Referer"] = ref

    r = requests.get(ref, headers={"User-Agent": UA}, timeout=8)
    time.sleep(0.4)
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    if r.status_code == 403:
        time.sleep(0.6)
        r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content

# ========== low-level text extraction ==========
@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

def extract_words(pdf_bytes: bytes) -> Tuple[List[Word], Tuple[float,float]]:
    words: List[Word] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return [], (0.0,0.0)
        page = doc.load_page(0)
        w, h = page.rect.width, page.rect.height
        for b in page.get_text("words"):
            words.append(Word(b[0], b[1], b[2], b[3], b[4]))
    return words, (w, h)

def words_to_lines(words: List[Word], y_tol: float=2.2) -> List[str]:
    """Группируем слова по строкам по y, потом сортируем по x, склеиваем."""
    if not words: return []
    ws = sorted(words, key=lambda w:(w.y0, w.x0))
    lines: List[List[Word]] = []
    cur: List[Word] = []
    last_y = None
    for w in ws:
        if last_y is None or abs(w.y0 - last_y) <= y_tol:
            cur.append(w)
            last_y = w.y0 if last_y is None else (last_y + (w.y0-last_y)*0.0)
        else:
            lines.append(sorted(cur, key=lambda k:k.x0))
            cur = [w]
            last_y = w.y0
    if cur:
        lines.append(sorted(cur, key=lambda k:k.x0))
    out = []
    for ln in lines:
        out.append(" ".join(tok.text for tok in ln))
    return out

def split_columns_words(words: List[Word], page_w: float) -> Tuple[List[Word], List[Word]]:
    if not words: return [], []
    mid = page_w/2.0
    left = [w for w in words if (w.x0+w.x1)/2.0 <= mid]
    right= [w for w in words if (w.x0+w.x1)/2.0  > mid]
    return left, right

def page_to_image(pdf_bytes: bytes, dpi: int=300) -> Image.Image:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(0)
        mat = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

def ocr_block(img: Image.Image, lang: str="rus+eng") -> str:
    if not TESS_AVAILABLE: return ""
    return pytesseract.image_to_string(img, lang=lang, config="--psm 6")

# ========== parsing ==========
NAME_RE = r"[А-ЯЁ][а-яё]+(?:[-\s][А-ЯЁа-яё\.]+){0,3}"
NUM_RE  = r"\d{1,2}"
ROLE_LET = {"З":"D","Н":"F","В":"G"}

def clean_space(s: str) -> str:
    return re.sub(r"\s{2,}", " ", s).strip(" \t.;,:")

def clean_roles_from_name(s: str) -> str:
    s = re.sub(r"\b(Главн\w*|Линейн\w*|судья|судьи|Резервн\w*|Обновлено[:\s]*\d{1,2}\.\d{1,2}\.\d{4}.*)$", " ", s, flags=re.I)
    s = re.sub(r"\b(Обновлено|РЕЗЕРВНЫЙ|Резервный)\b", " ", s, flags=re.I)
    s = clean_space(s)
    return s

def parse_referees_from_text(txt: str) -> Tuple[List[str], List[str]]:
    t = txt.replace("\n", " ")
    t = clean_space(t)
    mains, lines = [], []

    # 1) Явные блоки
    for block, bucket in [
        (r"Главн\w+\s+суд[ьяи]:?\s*([^:]+?)(?=Линейн\w+\s+суд|$)", mains),
        (r"Линейн\w+\s+суд[ьи]:?\s*([^:]+)", lines)
    ]:
        for m in re.finditer(block, t, flags=re.I):
            chunk = m.group(1)
            parts = re.split(r"[;,]|\s{2,}", chunk)
            for p in parts:
                name = clean_roles_from_name(p)
                if re.match(r"^[А-ЯЁA-Z]", name) and len(name.split()) >= 2:
                    bucket.append(name)

    # 2) Если пусто — резервная маска «... судья ... Фам Имя»
    if not mains and not lines:
        for m in re.finditer(r"(?:суд[ьяи]\s*[:\-]?\s*)("+NAME_RE+r"(?:\s+"+NAME_RE+r")?)", t, flags=re.I):
            nm = clean_roles_from_name(m.group(1))
            if nm:
                mains.append(nm)

    # Канонизация + дедуп
    def uniq(seq):
        out=[]; seen=set()
        for x in seq:
            x = canonize(x, REFS_MAP)
            if x and x not in seen:
                seen.add(x); out.append(x)
        return out
    return uniq(mains), uniq(lines)

def parse_team_name(col_lines: List[str]) -> str:
    # Берём первую длинную капсовую фразу, не содержащую «СОСТАВ»
    caps = []
    for ln in col_lines[:10]:
        m = re.search(r"([А-ЯЁA-Z][А-ЯЁA-Z\s\-\.\(\)]{6,})", ln)
        if m:
            cand = clean_space(m.group(1))
            if "СОСТАВ" not in cand and "КОМАНД" not in cand:
                caps.append(cand)
    if caps:
        # Самая длинная
        return max(caps, key=len)
    return ""

def scan_goalies_tokens(col_lines: List[str]) -> List[Dict[str,str]]:
    """
    Ищем подряд токены формата:
    (опц.) 'В'  <№>  <Фам Имя ...>   пока не встретим новое 'З/Н/В' + № или заголовок звена.
    """
    text = " \n".join(col_lines)
    # Обрежем до блока «Вратар» если он есть (не обязательно в PDF)
    m = re.search(r"Вратар\w*[:\s]+(.+)$", text, flags=re.I|re.S)
    if m:
        text = m.group(1)

    toks = re.split(r"\s+", text)
    out: List[Dict[str,str]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        # старт записи: (В)? + число
        if (t in ("В", "Вратарь", "Вратари")) or re.fullmatch(NUM_RE, t):
            # если первая буква роль, следующая должна быть номер
            j = i
            role = ""
            if t in ("В", "Вратарь", "Вратари"):
                j += 1
                if j>=len(toks) or not re.fullmatch(NUM_RE, toks[j]):
                    i += 1; continue
                role = "G"
            else:
                role = ""  # может быть З/Н без «В»
            # номер
            num = toks[j]
            if not re.fullmatch(NUM_RE, num):
                i += 1; continue
            j += 1
            # имя — собираем, пока не встретим «З/Н/В + число», «<цифра> звено», запятую-разделитель и т.п.
            name_parts = []
            while j < len(toks):
                nxt = toks[j]
                # стоп-условия
                if nxt in ("З","Н","В") and (j+1)<len(toks) and re.fullmatch(NUM_RE, toks[j+1]): break
                if re.fullmatch(r"[1-4]", nxt) and (j+1)<len(toks) and "звен" in toks[j+1].lower(): break
                if re.fullmatch(NUM_RE, nxt) and name_parts: break
                # не тащим служебные куски
                if nxt.lower() in {"обновлено","резервный","резервная","резерв"}: break
                name_parts.append(nxt)
                j += 1
            name = clean_space(" ".join(name_parts))
            name = re.sub(r"^[ЗНВ]\s+", "", name)  # если прилипла буква роли
            name = canonize(name, PLAYERS_MAP)
            # фильтр по виду имени
            if re.search(r"[А-ЯЁа-яё]{3,}", name):
                out.append({"number": num, "name": name, "gk_status": ""})
            i = j
            continue
        i += 1
    return out

def scan_lines_block(col_lines: List[str]) -> Dict[str, List[Dict[str,str]]]:
    text = " \n".join(col_lines)
    # Упростим маркеры звеньев
    text = re.sub(r"Первое\s*звено", "1 звено", text, flags=re.I)
    text = re.sub(r"Второе\s*звено", "2 звено", text, flags=re.I)
    text = re.sub(r"Третье\s*звено", "3 звено", text, flags=re.I)
    text = re.sub(r"Четвёртое\s*звено|Четвертое\s*звено", "4 звено", text, flags=re.I)

    lines = {"1":[], "2":[], "3":[], "4":[]}

    def grab(no: int) -> str:
        pat = rf"\b{no}\s*звено[:]?\s*(.+?)(?=\b{no+1}\s*звено|$)"
        m = re.search(pat, text, flags=re.I|re.S)
        return m.group(1) if m else ""

    def parse_members(seg: str) -> List[Dict[str,str]]:
        # режем по запятым/точкам с пробелами/двойным пробелам
        raw = re.split(r"[;,]\s*|\s{2,}", seg)
        out=[]
        for p in raw:
            p = clean_space(p)
            if not p: continue
            # варианты:
            # "З 24 Сериков Артём"  |  "24 Сериков Артём"  | "Н 71 Белозёров Андрей"
            m = re.match(rf"(?:(З|Н|В)\s+)?({NUM_RE})\s+({NAME_RE})", p)
            if not m:
                # fallback: иногда позиция/номер склеены или номер в конце
                m = re.match(rf"(?:(З|Н|В)\s+)?({NAME_RE})\s+({NUM_RE})", p)
                if m:
                    role = m.group(1) or ""
                    name = canonize(m.group(2), PLAYERS_MAP)
                    num  = m.group(3)
                    pos  = ROLE_LET.get(role or "", "F")
                    out.append({"name":name, "pos":pos, "number":num})
                continue
            role = m.group(1) or ""
            num  = m.group(2)
            name = canonize(m.group(3), PLAYERS_MAP)
            pos  = ROLE_LET.get(role or "", "F")
            out.append({"name":name, "pos":pos, "number":num})
        return out

    for i in (1,2,3,4):
        seg = grab(i)
        if seg:
            lines[str(i)] = parse_members(seg)
    return lines

@dataclass
class Roster:
    team: str
    goalies: List[Dict[str,str]]
    lines: Dict[str, List[Dict[str,str]]]
    bench: List[Dict[str,str]]

def parse_roster_from_column(col_words: List[Word]) -> Roster:
    col_lines = words_to_lines(col_words, y_tol=2.4)
    team = parse_team_name(col_lines)
    goalies = scan_goalies_tokens(col_lines)
    lines = scan_lines_block(col_lines)
    return Roster(team=team or "", goalies=goalies, lines=lines, bench=[])

# ========== main pipeline ==========
def analyze_pdf(pdf_bytes: bytes, mode: str="auto", dpi: int=300, debug: bool=False) -> Dict[str, Any]:
    ocr_used = False
    words, (pw, ph) = extract_words(pdf_bytes)

    if mode not in ("auto","words","ocr"):
        mode = "auto"

    if mode == "ocr" or (mode=="auto" and not words):
        img = page_to_image(pdf_bytes, dpi=dpi)
        ocr_used = True
        full_txt = ocr_block(img, "rus+eng")
        # чтобы сохранить логику «двух колонок» при OCR — режем картинку пополам
        w, h = img.size
        left_img, right_img = img.crop((0,0,w//2,h)), img.crop((w//2,0,w,h))
        left_txt  = ocr_block(left_img,  "rus+eng")
        right_txt = ocr_block(right_img, "rus+eng")
        # подменяем words «фиктивными» строками — ниже мы используем только текстовые парсеры
        left_words, right_words = [], []
        left_lines  = [ln for ln in left_txt.splitlines() if ln.strip()]
        right_lines = [ln for ln in right_txt.splitlines() if ln.strip()]
        left_text_join  = "\n".join(left_lines)
        right_text_join = "\n".join(right_lines)
        # судьи — из полного текста
        mains, linesmen = parse_referees_from_text(full_txt)
        # составы — из левой/правой «полу-картинки»
        # примитивный разбор: конвертим строки в фиктивные Word (для переиспользования парсеров)
        def fake_words(lines):
            ww=[]; y=0.0
            for ln in lines:
                x = 0.0
                for tok in ln.split():
                    ww.append(Word(x, y, x+len(tok)*6, y+8, tok))
                    x += (len(tok)+1)*6
                y += 10.0
            return ww
        left_wf, right_wf = fake_words(left_lines), fake_words(right_lines)
        home = parse_roster_from_column(left_wf)
        away = parse_roster_from_column(right_wf)

        # проставим статусы киперов
        def mark(gl):
            if not gl: return
            if len(gl)==1: gl[0]["gk_status"]="starter"
            else:
                gl[0]["gk_status"]="starter"; gl[-1]["gk_status"]="reserve"

        mark(home.goalies); mark(away.goalies)

        result = {
            "ok": True,
            "engine": "ocr",
            "data": {
                "home":{"team":home.team,"goalies":home.goalies,"lines":home.lines,"bench":[]},
                "away":{"team":away.team,"goalies":away.goalies,"lines":away.lines,"bench":[]},
            },
            "referees":{"main":mains,"linesmen":linesmen},
            "referee_entries":(
                [{"role":"Главный судья","name":n} for n in mains] +
                [{"role":"Линейный судья","name":n} for n in linesmen]
            )
        }
        if debug:
            result["_debug"]={"left_text":left_text_join, "right_text":right_text_join, "full_text":full_txt[:5000]}
        return result

    # words / auto
    left_words, right_words = split_columns_words(words, pw)
    # Судьи — из всего текста слов
    full_text = " ".join(w.text for w in words)
    mains, linesmen = parse_referees_from_text(full_text)

    # Составы
    home = parse_roster_from_column(left_words)
    away = parse_roster_from_column(right_words)

    # эвристика статусов киперов
    def mark(gl):
        if not gl: return
        if len(gl)==1: gl[0]["gk_status"]="starter"
        else:
            gl[0]["gk_status"]="starter"; gl[-1]["gk_status"]="reserve"
    mark(home.goalies); mark(away.goalies)

    result = {
        "ok": True,
        "engine": "words" if not ocr_used else "ocr",
        "data": {
            "home":{"team":home.team,"goalies":home.goalies,"lines":home.lines,"bench":[]},
            "away":{"team":away.team,"goalies":away.goalies,"lines":away.lines,"bench":[]},
        },
        "referees":{"main":mains,"linesmen":linesmen},
        "referee_entries":(
            [{"role":"Главный судья","name":n} for n in mains] +
            [{"role":"Линейный судья","name":n} for n in linesmen]
        )
    }
    if debug:
        result["_debug"]={
            "left_lines": words_to_lines(left_words),
            "right_lines": words_to_lines(right_words),
            "full_text": full_text[:5000]
        }
    return result

# ========== HTTP ==========
@app.get("/health")
def health():
    return jsonify({"ok": True, "engine": "ready"})

@app.get("/extract")
def extract():
    url = request.args.get("url", "").strip()
    season = request.args.get("season", "").strip()
    uid = request.args.get("uid", "").strip()
    mode = request.args.get("mode", "auto").strip().lower()
    dpi  = int(request.args.get("dpi", "300"))
    debug= request.args.get("debug","0") in ("1","true","yes","on")

    try:
        if url:
            pdf_url = url
            s = re.search(r"/(\d{4})/(\d{6})/", url)
            s_int = int(s.group(1)) if s else None
            u_int = int(s.group(2)) if s else None
        else:
            if not (season and uid):
                return jsonify({"ok": False, "error": "pass url=... or season&uid"}), 400
            s_int = int(season); u_int = int(uid)
            pdf_url = f"https://www.khl.ru/pdf/{s_int}/{u_int}/game-{u_int}-start-ru.pdf"

        t0 = time.time()
        pdf = fetch_pdf(pdf_url, s_int, u_int)
        data = analyze_pdf(pdf, mode=mode, dpi=dpi, debug=debug)
        data["source_url"] = pdf_url
        data["duration_s"] = round(time.time()-t0, 3)
        return jsonify(data)

    except requests.HTTPError as e:
        return jsonify({"ok": False, "error": f"http {e.response.status_code}", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": "parse-failed", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
