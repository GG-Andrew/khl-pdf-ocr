# main.py — v2.6.1-hotfix (KHL PDF OCR Server)
import os, io, re, json, sys, traceback
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from collections import defaultdict

# ---------- CONFIG ----------
APP_VERSION = "2.6.1-hotfix"
DEFAULT_SEASON = 1369

PLAYERS_CSV = os.getenv("PLAYERS_CSV", "players.csv")
REFEREES_CSV = os.getenv("REFEREES_CSV", "referees.csv")

# ---------- DICTS ----------
_dict_players: Dict[str, str] = {}
_dict_refs: Dict[str, str] = {}

def load_csv_dict(path: str) -> Dict[str, str]:
    d = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # поддержка форматов: "Фамилия Имя", или "Фамилия;Имя", или просто одна колонка
                    parts = re.split(r"[;,]", line)
                    name = " ".join([p.strip() for p in parts if p.strip()])
                    if name:
                        key = norm_key(name)
                        d[key] = name
    except Exception:
        print(f"[WARN] cannot load {path}", file=sys.stderr)
    return d

def reload_dicts():
    global _dict_players, _dict_refs
    _dict_players = load_csv_dict(PLAYERS_CSV)
    _dict_refs = load_csv_dict(REFEREES_CSV)

def norm_key(s: str) -> str:
    s = s.lower()
    s = s.replace("ё", "е")
    s = re.sub(r"[^a-zа-я0-9\s\-\.]", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- TEXT CLEAN ----------
def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\u00A0", " ")  # nbsp
    s = re.sub(r"[\u0300-\u036f\u0483-\u0489]", "", s)  # комб. акценты/ударения
    s = re.sub(r"[^\S\r\n]+", " ", s)
    return s.strip()

def normalize_tail_y(s: str) -> str:
    # убираем редкие хвостовые символы/ударения
    s = clean_text(s)
    s = s.replace("ї", "й").replace("і", "и")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def dict_fix(name: str, d: Dict[str, str]) -> str:
    key = norm_key(name)
    return d.get(key, name)

# ---------- PDF HELPERS ----------
def fetch_pdf_bytes(url: str) -> bytes:
    # простейшая загрузка через PyMuPDF (он сам не качает) — используем httpx если есть
    try:
        import httpx
        r = httpx.get(url, timeout=30)
        if r.status_code == 200:
            return r.content
    except Exception as e:
        print(f"[ERR] httpx get failed: {e}", file=sys.stderr)
    # запасной вариант через urllib
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()
    except Exception as e:
        print(f"[ERR] urllib get failed: {e}", file=sys.stderr)
    return b""

def mu_words_to_lines(words: List[Tuple]) -> List[str]:
    # words: [x0,y0,x1,y1,"text", block_no, line_no, word_no]
    # Группировка по y (с допуском)
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (int(w[1]), int(w[0])))
    lines = []
    curr_y = None
    buf = []
    for w in words_sorted:
        x0,y0,x1,y1,txt, *_ = w
        if curr_y is None:
            curr_y = y0
        if abs(y0 - curr_y) <= 2.5:  # терпимость по высоте строки
            buf.append((x0, txt))
        else:
            buf_sorted = sorted(buf, key=lambda t:t[0])
            line = " ".join([t[1] for t in buf_sorted])
            lines.append(line)
            buf = [(x0, txt)]
            curr_y = y0
    if buf:
        buf_sorted = sorted(buf, key=lambda t:t[0])
        line = " ".join([t[1] for t in buf_sorted])
        lines.append(line)
    # чистим
    lines = [clean_text(x) for x in lines if clean_text(x)]
    return lines

def split_left_right_by_median(words: List[Tuple], width: float) -> Tuple[str,str]:
    # Делим по медиане x; проще и надёжно для двух колонок
    if not words:
        return "",""
    xs = sorted([w[0] for w in words])
    mid = xs[len(xs)//2] if xs else width/2
    left_lines = []
    right_lines = []
    # группируем отдельно
    left = [w for w in words if w[0] <= mid]
    right = [w for w in words if w[0] > mid]
    left_lines = mu_words_to_lines(left)
    right_lines = mu_words_to_lines(right)
    return ("\n".join(left_lines), "\n".join(right_lines))

# ---------- REGEX ----------
HEADER_RE = re.compile(
    r"№\s*Поз\s*Фамилия,?\s*Имя(?:\s*\*?)?\s*Д\.?\s*Р\.?\s*Лет",
    re.I
)
ROW_RE = re.compile(
    r"^(?P<num>\d{1,2})\s+"
    r"(?P<pos>[ВЗН])\s+"
    r"(?P<name>[А-ЯЁA-Z][^0-9]*?)\s+"         # фамилия, имя (возможны точки/пробелы)
    r"(?:(?P<capt>[AK])\s+)?"
    r"(?P<dob>\d{2}\.\d{2}\.\d{4})"           # дата рождения
    r"(?:\s+(?P<age>\d{1,2}))?$"
)

# ---------- PARSERS ----------
def parse_lineups_block(text: str, side: str, players_dict: Dict[str,str]) -> List[Dict]:
    lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]
    items = []
    in_table = False
    for ln in lines:
        if not in_table and HEADER_RE.search(ln):
            in_table = True
            continue
        if not in_table:
            continue

        m = ROW_RE.search(ln)
        if not m:
            continue

        num = m.group("num")
        pos = m.group("pos")
        name = normalize_tail_y(m.group("name"))
        capt = m.group("capt") or ""
        dob = m.group("dob")
        age = m.group("age") or ""

        # отлепляем капитана, если прилип к имени
        if not capt:
            mcap = re.search(r"\s([AK])$", name)
            if mcap:
                capt = mcap.group(1)
                name = name[:mcap.start()].strip()

        # метки вратарей S/R в конце имени → в отдельное поле
        gk_flag = ""
        if pos == "В":
            mflag = re.search(r"\s([СР])$", name, re.I)  # русская С/Р
            if mflag:
                gk_flag = mflag.group(1).upper()
                name = name[:mflag.start()].strip()

        name = dict_fix(name, players_dict)

        gk_status = None
        if pos == "В":
            if gk_flag == "С": gk_status = "starter"
            elif gk_flag == "Р": gk_status = "reserve"

        items.append({
            "side": side, "num": num, "pos": pos, "name": name,
            "capt": capt, "dob": dob, "age": age,
            "gk_flag": gk_flag, "gk_status": gk_status
        })
    return items

def parse_goalies_from_lineups(lineups: List[Dict]) -> Dict[str, List[Dict[str,str]]]:
    out = {"home": [], "away": []}
    for it in lineups:
        if it.get("pos") == "В":
            status = it.get("gk_status") or "scratch"
            out[it["side"]].append({"name": it["name"], "status": status})
    return out

def parse_refs(left: str, right: str) -> List[Dict[str,str]]:
    blob = (left + "\n" + right).splitlines()
    blob = [clean_text(x) for x in blob if clean_text(x)]
    refs: List[Dict[str,str]] = []

    idx_main = [i for i,s in enumerate(blob) if re.search(r"Главный\s+судья", s, re.I)]
    idx_line = [i for i,s in enumerate(blob) if re.search(r"Линейный\s+судья", s, re.I)]

    def gather_after(idx_list, role):
        for i in idx_list:
            j = i + 1
            taken = 0
            while j < len(blob) and taken < 2:
                s = blob[j]
                if not re.search(r"(Главный|Линейный)\s+судья", s, re.I):
                    nm = normalize_tail_y(s)
                    nm = dict_fix(nm, _dict_refs)
                    if nm:
                        refs.append({"role": role, "name": nm})
                        taken += 1
                j += 1

    gather_after(idx_main, "Главный судья")
    gather_after(idx_line, "Линейный судья")

    # де-дуп
    seen = set()
    uniq = []
    for r in refs:
        key = (r["role"], r["name"])
        if key not in seen:
            uniq.append(r); seen.add(key)
    return uniq

# ---------- FASTAPI ----------
app = FastAPI(title="KHL PDF OCR Server", version=APP_VERSION)
reload_dicts()

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "khl-pdf-ocr",
        "version": APP_VERSION,
        "ready": True,
        "dicts": {
            "players_path": PLAYERS_CSV if os.path.exists(PLAYERS_CSV) else False,
            "referees_path": REFEREES_CSV if os.path.exists(REFEREES_CSV) else False,
            "players_loaded": len(_dict_players),
            "refs_loaded": len(_dict_refs),
        }
    }

@app.get("/ocr")
def ocr_parse(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(DEFAULT_SEASON),
    dpi: int = Query(130, ge=120, le=360),
    max_pages: int = Query(1, ge=1, le=5),
    scale: float = Query(1.0, ge=1.0, le=2.0),
    bin_thresh: int = Query(185, ge=120, le=230),
):
    # легкий/быстрый текст-слой без OCR (PyMuPDF words)
    b = fetch_pdf_bytes(pdf_url)
    if not b:
        return {"ok": False, "match_id": match_id, "season": season, "step": "GET", "status": 404}
    doc = fitz.open(stream=b, filetype="pdf")
    page = doc.load_page(0)
    words = page.get_text("words")  # list of tuples
    left, right = split_left_right_by_median(words, page.rect.width)
    text = clean_text(left + "\n" + "---RIGHT---\n" + right)
    return {
        "ok": True, "match_id": match_id, "season": season,
        "source_pdf": pdf_url, "pdf_len": len(b),
        "dpi": dpi, "pages_ocr": 1, "dur_total_s": 0.0,
        "text_len": len(text), "snippet": text[:300]
    }

@app.get("/extract")
def extract(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(DEFAULT_SEASON),
    target: str = Query("all")  # refs|goalies|lineups|all
):
    try:
        # 1) загрузка и разметка
        b = fetch_pdf_bytes(pdf_url)
        if not b:
            return {"ok": False, "match_id": match_id, "season": season, "step":"GET", "status":404}
        doc = fitz.open(stream=b, filetype="pdf")
        page = doc.load_page(0)
        words = page.get_text("words")
        left, right = split_left_right_by_median(words, page.rect.width)

        data = {}

        # 2) извлечение судей
        if target in ("refs","all"):
            data["refs"] = parse_refs(left, right)

        # 3) составы по колонкам
        if target in ("lineups","goalies","all"):
            home = parse_lineups_block(left, "home", _dict_players)
            away = parse_lineups_block(right, "away", _dict_players)
            if target in ("lineups","all"):
                data["lineups"] = {"home": home, "away": away}
            if target in ("goalies","all"):
                data["goalies"] = parse_goalies_from_lineups(home + away)

        return {
            "ok": True,
            "match_id": match_id,
            "season": season,
            "source_pdf": pdf_url,
            "pdf_len": len(b),
            "dpi": 130,
            "pages_ocr": 1,
            "dur_total_s": 0.0,
            "data": data
        }
    except Exception as e:
        err = "".join(traceback.format_exception(e))
        print("[EXTRACT ERR]\n" + err, file=sys.stderr)
        return {
            "ok": False,
            "match_id": match_id,
            "season": season,
            "error": str(e),
            "trace": err[:2000]
        }
