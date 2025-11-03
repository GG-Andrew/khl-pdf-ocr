# app.py
import io, os, re, time, json
from typing import Dict, List, Tuple, Any
from flask import Flask, request, jsonify
import requests
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# ------------ CONFIG ------------
PDF_PROXY = os.getenv("KHL_PDF_PROXY", "https://pdf2.palladiumgames2d.workers.dev")
TESS_LANG = os.getenv("TESS_LANG", "rus+eng")
TESS_CFG  = os.getenv("TESS_CFG", "--oem 1 --psm 6")
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/118.0 Safari/537.36")

SKIP = {"p.", "р.", "г.", "р", "p", "MSK", "МСК", "МСK", "№", "Матч", "Составы", "команд", "Обновлено"}
RUS_MONTHS = ("январ", "феврал", "март", "апрел", "ма", "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр")

app = Flask(__name__)

# ------------ UTILS ------------

def http_json(ok: bool, **kw):
    return jsonify(dict(ok=ok, **kw))

def load_pdf_bytes(season: str, uid: str) -> bytes:
    # Всегда через Cloudflare-proxy (ты уже поднял воркер)
    url = f"{PDF_PROXY}/khlpdf/{season}/{uid}/game-{uid}-start-ru.pdf"
    r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    if r.status_code != 200:
        raise RuntimeError(f"upstream {r.status_code}")
    return r.content

def doc_words_first_page(pdf_bytes: bytes) -> Tuple[List[Dict[str, Any]], fitz.Page]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    # words: [x0, y0, x1, y1, "text", block_no, line_no, word_no]
    words_raw = page.get_text("words")
    words = [
        {"x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "text": w[4]}
        for w in words_raw
        if str(w[4]).strip()
    ]
    return words, page

def split_columns(words: List[Dict[str, Any]], page: fitz.Page) -> Tuple[List[Dict], List[Dict]]:
    # Средина страницы: делим по X
    x_mid = (page.rect.x0 + page.rect.x1) / 2
    left  = [w for w in words if w["x1"] <= x_mid]
    right = [w for w in words if w["x0"] >  x_mid]
    # Отсортируем по y, затем x
    left.sort(key=lambda w:(round(w["y0"],1), round(w["x0"],1)))
    right.sort(key=lambda w:(round(w["y0"],1), round(w["x0"],1)))
    return left, right

def join_lines(words: List[Dict], y_tol=3.0) -> List[str]:
    """Склеиваем слова в строки, если близки по Y."""
    lines = []
    cur_y = None
    buf = []
    for w in words:
        y = round(w["y0"], 1)
        if cur_y is None or abs(y - cur_y) <= y_tol:
            buf.append(w["text"])
            cur_y = y if cur_y is None else (cur_y + y)/2
        else:
            if buf:
                lines.append(" ".join(buf).strip())
            buf = [w["text"]]
            cur_y = y
    if buf:
        lines.append(" ".join(buf).strip())
    return lines

def looks_time(s: str) -> bool:
    return bool(re.search(r"\b([01]?\d|2[0-3]):[0-5]\d\b", s))

def looks_date(s: str) -> bool:
    s2 = s.lower()
    return bool(re.search(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b", s2)) or any(m in s2 for m in RUS_MONTHS)

def clean_role_words(name: str) -> str:
    if not name: return name
    name = re.sub(r"\b(Главн\w*|Линейн\w*|Резервн\w*|Судья|судья|судьи)\b", " ", name, flags=re.I)
    name = re.sub(r"[\"'`•·\[\]\(\)\|\\/]+", " ", name)
    name = re.sub(r"\s{2,}", " ", name).strip(" ,.;:")
    return name

def only_letters(s: str) -> bool:
    return bool(re.search(r"[A-Za-zА-Яа-яЁё]", s))

def line_has_bad_tokens(s: str) -> bool:
    toks = re.split(r"[,\s]+", s)
    for tok in toks:
        t = tok.strip(" ,.;:|/\\")
        if not t: continue
        if t in SKIP: return True
        if looks_date(t): return True
        if re.fullmatch(r"\d+([/]\d+)?", t): return True
    return False

def crop_around(page: fitz.Page, center_y: float, h: float) -> Image.Image:
    r = page.rect
    box = fitz.Rect(r.x0, max(r.y0, center_y - h), r.x1, min(r.y1, center_y + h))
    pix = page.get_pixmap(matrix=fitz.Matrix(2,2), clip=box, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def ocr_text(img: Image.Image) -> List[str]:
    txt = pytesseract.image_to_string(img, lang=TESS_LANG, config=TESS_CFG)
    raw = [ln.strip() for ln in txt.splitlines()]
    return [ln for ln in raw if ln]

# ------------ PARSERS ------------

def parse_header(words: List[Dict], page: fitz.Page) -> Dict[str, Any]:
    left, right = split_columns(words, page)
    header_left  = [w for w in left if w["y0"] < 200]
    header_right = [w for w in right if w["y0"] < 200]
    linesL = join_lines(header_left)
    linesR = join_lines(header_right)

    # Команды: берём самую длинную строку с кириллицей в каждой колонке
    def best_team(lines):
        cand = [ln for ln in lines if only_letters(ln)]
        cand.sort(key=lambda s: len(s), reverse=True)
        return cand[0].strip() if cand else ""

    home = best_team(linesL)
    away = best_team(linesR)

    # Дата/время: смотрим по всей странице в верхней трети
    top_words = [w for w in words if w["y0"] < (page.rect.height/3)]
    top_lines = join_lines(sorted(top_words, key=lambda w:(w["y0"], w["x0"])))
    date, tmsk = "", ""
    for ln in top_lines:
        if not date and looks_date(ln): date = ln
        if not tmsk and looks_time(ln): tmsk = re.search(r"([01]?\d|2[0-3]):[0-5]\d", ln).group(0)
        if date and tmsk: break

    return {"teams": {"home": home, "away": away}, "date": date or "", "time_msk": tmsk or ""}

def parse_refs(words: List[Dict], page: fitz.Page, ocr_fallback=True, debug=False) -> Dict[str, List[str]]:
    # 1) попробуем по words: найти строку с "Главный судья" и дальше ограничить "Обновлено"
    lines_all = join_lines(sorted(words, key=lambda w:(w["y0"], w["x0"])))
    txt = "\n".join(lines_all)
    m1 = re.search(r"Главн\w*\s+судья.*", txt, flags=re.I)
    m2 = re.search(r"Обновлено.*", txt, flags=re.I)
    block = ""
    if m1:
        start = m1.start()
        end   = m2.start() if m2 and m2.start()>start else None
        block = txt[start:end].replace("\u00A0"," ")

    main, linesmen = [], []

    def push_names(line: str, bucket: List[str]):
        line = clean_role_words(line)
        # разложим по запятым/двойным пробелам
        parts = re.split(r"[;,]+|\s{2,}", line)
        for p in parts:
            p = p.strip(" ,.;:|/\\")
            if not p: continue
            if line_has_bad_tokens(p): continue
            # допустим «Фамилия Имя» или «Имя Фамилия», но без мусора
            if only_letters(p) and len(p.split())<=3:
                bucket.append(p)

    if block:
        # Разделить по ролям:
        # … Главный судья … Линейный судья …
        segs = re.split(r"(Главн\w*\s+судья|Линейн\w*\s+судья)", block, flags=re.I)
        role = None
        for seg in segs:
            t = seg.strip()
            if not t: continue
            if re.search(r"Главн\w*\s+судья", t, flags=re.I): role = "main"; continue
            if re.search(r"Линейн\w*\s+судья", t, flags=re.I): role = "lines"; continue
            if role == "main":
                push_names(t, main)
            elif role == "lines":
                push_names(t, linesmen)

    # OCR fallback, если плохо
    if ocr_fallback and (len(main)==0 or len(linesmen)==0):
        # найдём Y якоря "суд" в words
        y_list = [w["y0"] for w in words if re.search(r"суд", w["text"], flags=re.I)]
        center_y = (sum(y_list)/len(y_list)) if y_list else page.rect.height*0.65
        img = crop_around(page, center_y, 220)  # узкая полоска
        ocr_lines = ocr_text(img)
        block2 = "\n".join(ocr_lines)
        # ограничим «Главный судья … Обновлено»
        a = re.search(r"Главн\w*\s+судья", block2, flags=re.I)
        b = re.search(r"Обновлено", block2, flags=re.I)
        if a:
            start = a.start()
            end = b.start() if b and b.start()>start else None
            block2 = block2[start:end]
            segs = re.split(r"(Главн\w*\s+судья|Линейн\w*\s+судья)", block2, flags=re.I)
            role=None
            for seg in segs:
                t = seg.strip()
                if not t: continue
                if re.search(r"Главн\w*\s+судья", t, flags=re.I): role="main"; continue
                if re.search(r"Линейн\w*\s+судья", t, flags=re.I): role="lines"; continue
                if role=="main": push_names(t, main)
                elif role=="lines": push_names(t, linesmen)

    # финальная очистка/дедуп
    def uniq(xs): 
        out=[]
        seen=set()
        for x in xs:
            k=x.lower()
            if k in seen: continue
            seen.add(k); out.append(x)
        return out

    main = uniq(main)
    linesmen = uniq(linesmen)

    if debug:
        return {"main": main, "linesmen": linesmen, "_debug": {"raw_lines": lines_all}}
    return {"main": main, "linesmen": linesmen}

def parse_goalies(words: List[Dict], page: fitz.Page, ocr_fallback=True, debug=False) -> Dict[str, List[str]]:
    # Ищем якорь «Вратар»
    anchors = [w for w in words if re.search(r"Вратар", w["text"], flags=re.I)]
    res = {"home": [], "away": []}

    def parse_near(y_anchor: float) -> List[str]:
        # Берём 250 px ниже якоря
        win = [w for w in words if y_anchor-10 <= w["y0"] <= y_anchor+260]
        lines = join_lines(sorted(win, key=lambda w:(w["y0"], w["x0"])))
        out=[]
        for ln in lines[1:]:  # пропустим саму строку "Вратари"
            ln2 = re.sub(r"\b(Вратар\w*|Звено\s*\d+)\b.*", "", ln, flags=re.I).strip()
            ln2 = clean_role_words(ln2)
            # возможен формат: "30 Иванов Иван" или "Иванов Иван 30"
            cand = re.sub(r"\b\d+\b", "", ln2).strip(" ,.;:-|")
            if cand and only_letters(cand) and not line_has_bad_tokens(cand):
                out.append(cand)
            if len(out)>=2: break
        return out

    if anchors:
        y = min(a["y0"] for a in anchors)
        # левая/правая колонка
        left,right = split_columns(words, page)
        lh = [w for w in left  if y-10 <= w["y0"] <= y+260]
        rh = [w for w in right if y-10 <= w["y0"] <= y+260]
        res["home"] = parse_near(y) if lh else []
        res["away"] = parse_near(y) if rh else []

    # OCR fallback, если пусто
    if ocr_fallback and (len(res["home"])==0 and len(res["away"])==0):
        y = min([w["y0"] for w in anchors], default=page.rect.height*0.45)
        img = crop_around(page, y, 240)
        lines = ocr_text(img)
        # простая эвристика: имена — строки с 2 словами на кириллице
        picks = []
        for ln in lines:
            ln = clean_role_words(ln)
            ln = re.sub(r"\b\d+\b","",ln).strip(" ,.;:-")
            if not ln: continue
            if only_letters(ln) and 1 < len(ln.split()) <= 3 and not line_has_bad_tokens(ln):
                picks.append(ln)
        # первые 1–2 — домашние, след. 1–2 — гости (лучше, чем ничего)
        res["home"] = picks[:2]
        res["away"] = picks[2:4]

    if debug:
        return {"home": res["home"], "away": res["away"], "_debug": {"anchors": len(anchors)}}
    return res

# ------------ HTTP ------------

@app.route("/health")
def health():
    return jsonify({"ok": True, "engine": "ready"})

@app.route("/extract")
def extract():
    t0 = time.time()
    season = request.args.get("season", "").strip()
    uid    = request.args.get("uid", "").strip()
    mode   = (request.args.get("mode","all").strip().lower())
    debug  = request.args.get("debug","0") in ("1","true","yes")

    if not (season and uid):
        return http_json(False, error="season+uid required"), 400

    try:
        pdf = load_pdf_bytes(season, uid)
        words, page = doc_words_first_page(pdf)
    except Exception as e:
        return http_json(False, error=str(e))

    source_url = f"{PDF_PROXY}/khlpdf/{season}/{uid}/game-{uid}-start-ru.pdf"
    out = {"ok": True, "source_url": source_url}

    if mode in ("words","all"):
        header = parse_header(words, page)
        out["match"] = {"season": season, "uid": uid, **header}
        if mode == "words":
            out["engine"]="words"; out["duration_s"]=round(time.time()-t0,3); 
            return jsonify(out)

    if mode in ("refs","all"):
        refs = parse_refs(words, page, ocr_fallback=True, debug=debug)
        out["referees"] = {"main": refs.get("main",[]), "linesmen": refs.get("linesmen",[])}
        if debug and "_debug" in refs: out["_debug"]=refs["_debug"]
        if mode == "refs":
            out["engine"]="ocr-refs"; out["duration_s"]=round(time.time()-t0,3); 
            return jsonify(out)

    if mode in ("goalies","all"):
        gk = parse_goalies(words, page, ocr_fallback=True, debug=debug)
        out["goalies"] = {"home": gk.get("home",[]), "away": gk.get("away",[])}
        if debug and "_debug" in gk: out.setdefault("_debug", {})["goalies"]=gk["_debug"]
        if mode == "goalies":
            out["engine"]="gk"; out["duration_s"]=round(time.time()-t0,3); 
            return jsonify(out)

    # mode=all — возвращаем всё
    out["engine"]="all"
    out["duration_s"]=round(time.time()-t0,3)
    return jsonify(out)

# ------------ ENTRY ------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8080")))
