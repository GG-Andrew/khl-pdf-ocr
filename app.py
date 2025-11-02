import os, io, re, json, time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import requests
from flask import Flask, request, jsonify
import fitz  # PyMuPDF
from PIL import Image

# --- OCR fallback (tesseract) ---
try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

# ====== Конфиг ======
UA = "BETON-KHL OCR/2.0 (+https://results.beton-khl.ru)"
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

# ====== Справочники (опционально) ======
def load_master(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    mp: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):  # csv простого вида: raw;canon
                continue
            parts = re.split(r"[;,]\s*", ln)
            if len(parts) >= 2:
                raw = parts[0].strip()
                canon = parts[1].strip()
                if raw and canon:
                    mp[raw.lower()] = canon
    return mp

PLAYERS_MAP = load_master(PLAYERS_MASTER)
REFS_MAP    = load_master(REFEREES_MASTER)

def canonize(name: str, mapping: Dict[str,str]) -> str:
    if not name: return name
    key = name.lower().strip()
    return mapping.get(key, name)

# ====== HTTP ======
def fetch_pdf(url: str, season: int|None=None, uid: int|None=None) -> bytes:
    headers = PDF_HEADERS.copy()
    # Тёплый прогрев Referer
    if season and uid:
        ref = REF_PREVIEW_TMPL.format(season=season, uid=uid)
    else:
        # попытаемся вытащить season/uid из самой ссылки
        m = re.search(r"/(\d{4})/(\d{6})/", url)
        ref = REF_PREVIEW_TMPL.format(season=m.group(1), uid=m.group(2)) if m else "https://www.khl.ru/"
    headers["Referer"] = ref

    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    if r.status_code == 403:
        # ещё раз после явного GET превью
        try:
            requests.get(ref, headers={"User-Agent": UA}, timeout=8)
            time.sleep(0.8)
        except Exception:
            pass
        r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content

# ====== low-level utils ======
def extract_words(pdf_bytes: bytes) -> List[dict]:
    """Возвращает список слов (bbox + text) первой страницы."""
    words: List[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return words
        page = doc.load_page(0)
        for b in page.get_text("words"):
            # b: (x0,y0,x1,y1,"text", block_no, line_no, word_no)
            words.append({
                "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                "text": b[4]
            })
    return words

def page_to_image(pdf_bytes: bytes, dpi: int=300) -> Image.Image:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(0)
        mat = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

def ocr_image(img: Image.Image, lang: str="rus+eng") -> str:
    if not TESS_AVAILABLE:
        return ""
    # Tesseract в контейнере обычно уже знает tessdata prefix
    return pytesseract.image_to_string(img, lang=lang, config="--psm 6")

# ====== high-level parsing ======
@dataclass
class Roster:
    team: str = ""
    goalies: List[Dict[str,str]] = None
    lines: Dict[str, List[Dict[str,str]]] = None
    bench: List[Dict[str,str]] = None

def split_columns(words: List[dict]) -> Tuple[List[str], List[str]]:
    """Разделяем на левую/правую колонку по median X."""
    if not words:
        return [], []
    xs = sorted([(w["x0"]+w["x1"])/2.0 for w in words])
    median_x = xs[len(xs)//2]
    left = " ".join([w["text"] for w in words if (w["x0"]+w["x1"])/2.0 <= median_x])
    right= " ".join([w["text"] for w in words if (w["x0"]+w["x1"])/2.0 >  median_x])
    # Уберём двойные пробелы
    left = re.sub(r"\s{2,}", " ", left)
    right= re.sub(r"\s{2,}", " ", right)
    return left, right

NAME_RE = r"[А-ЯЁ][а-яё]+(?:[-\s][А-ЯЁа-яё\.]+){0,3}"
NUM_RE  = r"\d{1,2}"
ROLE_RE = r"(?:Вратар[ьи]|Вратари|Голкипер[ы]|З|Н|Нападающ[ие]|Защитник[и])"
SKIP_TOK = {"p.", "р.", "р", "p", "г."}

def clean_role_words(s: str) -> str:
    s = re.sub(r"\b(Главн\w*|Линейн\w*|судья|судьи|Резервн\w*)\b", " ", s, flags=re.I)
    return re.sub(r"\s{2,}", " ", s).strip(" ,.;:")

def parse_referees(text: str) -> Tuple[List[str], List[str]]:
    """Достаём имена после тегов «Главный судья», «Линейный судья» (в любой форме)."""
    # Варианты написаний:
    # «Главные судьи: Фам Имя, Фам Имя», «Главный судья: ...»
    # «Линейные судьи: ...»
    mains: List[str] = []
    lines: List[str] = []

    # Нормализуем запятые/точки/переносы
    t = text.replace("\n", " ")
    t = re.sub(r"\s{2,}", " ", t)

    # Поищем группы с ролями:
    blocks = re.findall(r"(Главн\w+\s+суд[ьяи]:?\s*([^:]+?))(?:Линейн\w+\s+суд[ьи]:?|$)", t, flags=re.I)
    if blocks:
        for full, names in blocks:
            parts = re.split(r"[;,]\s*|\s{2,}", names)
            for p in parts:
                name = clean_role_words(p).strip()
                if len(name.split()) >= 2 and re.match(r"^[А-ЯЁA-Z]", name):
                    mains.append(name)
    # Линейные
    line_blocks = re.findall(r"(Линейн\w+\s+суд[ьи]:?\s*([^:]+))", t, flags=re.I)
    if line_blocks:
        for _, names in line_blocks:
            parts = re.split(r"[;,]\s*|\s{2,}", names)
            for p in parts:
                name = clean_role_words(p).strip()
                if len(name.split()) >= 2 and re.match(r"^[А-ЯЁA-Z]", name):
                    lines.append(name)

    # Если всё пусто — попробуем простую маску «Фам Имя» возле слова «суд»
    if not mains and not lines:
        near = re.findall(r"(?:суд[ьяи]\s*[:\-]?\s*)("+NAME_RE+r"(?:\s+"+NAME_RE+r")?)", t, flags=re.I)
        for n in near:
            nm = clean_role_words(n)
            if nm: mains.append(nm)

    # Канонизация
    mains = [canonize(n, REFS_MAP) for n in mains]
    lines = [canonize(n, REFS_MAP) for n in lines]

    # Дедуп
    def _uniq(seq): 
        out=[]; seen=set()
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return _uniq(mains), _uniq(lines)

def parse_goalies_and_lines(col_text: str) -> Roster:
    """
    Ожидаем шаблон:
    Вратари: NN Фам Имя; NN Фам Имя ...
    1 звено: Z/N + 5 записей; 2 звено: ...
    """
    rt = Roster(team="", goalies=[], lines={"1":[], "2":[], "3":[], "4":[]}, bench=[])
    t = re.sub(r"\s{2,}", " ", col_text)
    # Команда (верх страницы часто содержит «СОСТАВЫ КОМАНД <TEAM>» — берём самое длинное caps-слово)
    m = re.search(r"([А-ЯЁA-Z][А-ЯЁA-Z\s\-]{6,})", t)
    if m:
        cand = m.group(1).strip()
        # отфильтруем шум вроде «СОСТАВЫ КОМАНД»
        if "СОСТАВ" in cand:
            pass
        else:
            rt.team = cand

    # Вратари
    # Примеры: "Вратари: 31 Озолин Ярослав, 53 Долганов Филипп, 30 Любалин Семён"
    gblock = re.search(r"(?:Вратар[ьи]|Голкипер\w*):?\s+(.+?)(?:\s{0,3}1\s?[звенаип\.]|1\s?звено|1:|Первое\s?звено|$)", t, flags=re.I)
    if gblock:
        raw_g = gblock.group(1)
        # Разрезаем по ; или , или "  "
        raw_parts = re.split(r"[;,\|]\s*|\s{2,}", raw_g)
        for p in raw_parts:
            p = p.strip(" .")
            if not p: continue
            # номер + имя
            m2 = re.match(rf"({NUM_RE})\s+({NAME_RE})", p)
            if m2:
                num = m2.group(1)
                name = m2.group(2).strip()
                name = canonize(name, PLAYERS_MAP)
                rt.goalies.append({"number": num, "name": name, "gk_status": ""})

    # Звенья
    # Ищем блоки вида "1 звено: ... 2 звено: ... 3 звено: ... 4 звено: ..."
    # Допускаем варианты "1:", "1 звено", "Первое звено" и т.п.
    def grab_line(no: int) -> List[Dict[str,str]]:
        pat = rf"(?:\b{no}\s*(?:звено|звена|:)?|{['Первое','Второе','Третье','Четвёртое'][no-1]}\s*звено)\s*[:\-]?\s*(.+?)(?=\b{no+1}\s*(?:звено|:)|$)"
        m = re.search(pat, t, flags=re.I)
        out: List[Dict[str,str]] = []
        if not m:
            return out
        segment = m.group(1)
        items = re.split(r"[;,\|]\s*|\s{2,}", segment)
        for it in items:
            it = it.strip(" .")
            # позиции могут быть «З» или «Н», иногда буквы перед номером; вынем №, позицию и имя
            m3 = re.match(rf"(?:(З|Н)\s+)?({NUM_RE})\s+({NAME_RE})", it)
            if m3:
                pos = m3.group(1) or ""
                num = m3.group(2)
                name= m3.group(3)
                pos_eng = {"З":"D","Н":"F"}.get(pos, "")
                out.append({"name": canonize(name, PLAYERS_MAP), "pos": pos_eng or "F", "number": num})
        return out

    for i in (1,2,3,4):
        rt.lines[str(i)] = grab_line(i)

    return rt

# ====== основной пайплайн ======
def analyze_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    # 1) PyMuPDF слова
    words = extract_words(pdf_bytes)
    left_text, right_text = split_columns(words)

    # если совсем пусто — OCR
    ocr_used = False
    if len(left_text) + len(right_text) < 200:
        img = page_to_image(pdf_bytes, dpi=300)
        full_txt = ocr_image(img, "rus+eng")
        ocr_used = True
        # Простейший «сплит» пополам по ширине, если PyMuPDF не дал слова
        w, h = img.size
        left_img  = img.crop((0, 0, w//2, h))
        right_img = img.crop((w//2, 0, w, h))
        left_text  = ocr_image(left_img,  "rus+eng")
        right_text = ocr_image(right_img, "rus+eng")

    # 2) Судьи — берём весь «склеенный» текст, чтобы не потерять заголовки
    full_text = " ".join([w["text"] for w in words]) if words else (left_text + " " + right_text)
    mains, lines = parse_referees(full_text)

    # 3) Составы
    home = parse_goalies_and_lines(left_text)
    away = parse_goalies_and_lines(right_text)

    # 4) статусы киперов (эвристика: первый — starter, последний — reserve)
    def mark_goalies(gl: List[Dict[str,str]]):
        if not gl: return
        if len(gl) == 1:
            gl[0]["gk_status"] = "starter"
        elif len(gl) >= 2:
            gl[0]["gk_status"] = "starter"
            gl[-1]["gk_status"] = "reserve"

    mark_goalies(home.goalies)
    mark_goalies(away.goalies)

    return {
        "ok": True,
        "engine": "words" if not ocr_used else "ocr",
        "data": {
            "home": {
                "team": home.team,
                "goalies": home.goalies,
                "lines": home.lines,
                "bench": home.bench or []
            },
            "away": {
                "team": away.team,
                "goalies": away.goalies,
                "lines": away.lines,
                "bench": away.bench or []
            }
        },
        "referees": {
            "main": mains,
            "linesmen": lines
        },
        "referee_entries": (
            [{"role":"Главный судья","name": n} for n in mains] +
            [{"role":"Линейный судья","name": n} for n in lines]
        )
    }

# ====== HTTP endpoints ======
@app.get("/health")
def health():
    return jsonify({"ok": True, "engine": "ready"})

@app.get("/extract")
def extract():
    url = request.args.get("url", "").strip()
    season = request.args.get("season", "").strip()
    uid = request.args.get("uid", "").strip()
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

        pdf = fetch_pdf(pdf_url, s_int, u_int)
        data = analyze_pdf(pdf)
        data["source_url"] = pdf_url
        data["duration_s"] = round(0.0 + 0.0, 3)  # (можно засечь реально при желании)
        return jsonify(data)
    except requests.HTTPError as e:
        return jsonify({"ok": False, "error": f"http {e.response.status_code}", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": "parse-failed", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
