import io, os, re, json, time
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
HEADERS = {
    "User-Agent": UA,
    "Accept": "*/*",
    "Referer": "https://www.khl.ru/",
}

# ---- утилиты ---------------------------------------------------------------

def http_get(url, timeout=15):
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r

def warmup_preview_if_khl(pdf_url: str):
    # если это khl.ru/pdf/..., попытаться прогреть превью
    m = re.search(r"/pdf/(\d+)/(\d+)/game-\2-start-ru\.pdf", pdf_url)
    if not m:
        return
    season = m.group(1)
    uid = m.group(2)
    preview = f"https://www.khl.ru/game/{season}/{uid}/preview/"
    try:
        requests.get(preview, headers=HEADERS, timeout=8)
    except Exception:
        pass

def pdf_bytes_from_url(url: str) -> bytes:
    warmup_preview_if_khl(url)
    r = http_get(url, timeout=25)
    return r.content

# ---- базовый извлекатель: слова и быстрый OCR ------------------------------

def extract_words_first_page(pdf_bytes: bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return []
        page = doc.load_page(0)
        w = page.get_text("words")  # [x0,y0,x1,y1,"text", block_no, line_no, word_no]
        # сортируем по y, затем по x
        w.sort(key=lambda z: (round(z[1]), z[0]))
        words = [t[4] for t in w if t[4].strip()]
        return words

def ocr_first_page(pdf_bytes: bytes, dpi=300):
    # Рендер страницы → OCR по rus+eng
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return ""
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        txt = pytesseract.image_to_string(img, lang="rus+eng")
        return txt

# ---- грубый парсинг судей из текста (words → fallback OCR) -----------------

def parse_referees_from_text(text: str):
    # Ищем блок "Судьи" и ФИО
    # Примеры токенов: "Главный судья", "Линейный судья", "Резервный судья"
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    blob = " ".join(lines)
    blob = re.sub(r"[|•·∙●]", " ", blob)

    # вытащим кандидатов по словам "судья"
    chunks = re.split(r"(Главн\w*\s+судья|Линейн\w*\s+судья|Резервн\w*\s+судья)", blob, flags=re.I)
    roles, names = [], []
    for i in range(1, len(chunks), 2):
        role = chunks[i]
        tail = chunks[i+1] if i+1 < len(chunks) else ""
        # обрежем хвост до следующего маркера
        tail = re.split(r"(Главн\w*\s+судья|Линейн\w*\s+судья|Резервн\w*\s+судья)", tail, flags=re.I)[0]
        # имя = 1–3 русских слов с заглавной
        cand = re.findall(r"[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2}", tail)
        if cand:
            fio = cand[0].strip()
            roles.append(role)
            names.append(fio)

    mains = [n for r,n in zip(roles,names) if re.search(r"Глав", r, flags=re.I)]
    linesm = [n for r,n in zip(roles,names) if re.search(r"Линейн", r, flags=re.I)]
    reserves = [n for r,n in zip(roles,names) if re.search(r"Резерв", r, flags=re.I)]

    return {
        "main": list(dict.fromkeys(mains)),
        "linesmen": list(dict.fromkeys(linesm)),
        "reserve": list(dict.fromkeys(reserves))
    }

def get_referees(pdf_bytes: bytes):
    # 1) пробуем через words
    words = extract_words_first_page(pdf_bytes)
    txt_words = " ".join(words)
    refs = parse_referees_from_text(txt_words)

    # если не нашли ничего — OCR
    if not (refs["main"] or refs["linesmen"]):
        txt_ocr = ocr_first_page(pdf_bytes, dpi=300)
        refs = parse_referees_from_text(txt_ocr)

    # пост-обработка: убираем повторы
    refs["main"] = list(dict.fromkeys(refs["main"]))
    refs["linesmen"] = list(dict.fromkeys(refs["linesmen"]))
    return refs

# ---- парсинг командных составов (упрощённый words-парсер) ------------------

def parse_teams_and_lines(words):
    # Это упрощённый извлекатель, который ты потом сможешь заменить на свой прод.
    # Он ищет имена/номера в первых таблицах и собирает их "как есть".
    # У нас цель шага — поднять сервис в облаке; точность улучшим после деплоя.
    text = " ".join(words)
    # Определим команды грубо:
    home_team = re.search(r"(НЕФТЕХИМИК\s+НИЖНЕКАМСК|[А-ЯЁ][А-ЯЁ\s\-]+)", text)
    away_team = re.search(r"(САЛАВАТ\s+ЮЛАЕВ\s+УФА|[А-ЯЁ][А-ЯЁ\s\-]+)", text)
    home_name = home_team.group(1) if home_team else ""
    away_name = away_team.group(1) if away_team else ""

    # Номера и фамилии (очень грубо)
    num_name = re.findall(r"\b(\d{1,2})\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.){0,2}[а-яё\.]*)", text)
    # Разложим просто по 4 "линии" по порядку попадания:
    def pack(lines):
        out = {"1": [], "2": [], "3": [], "4": []}
        idx = 1
        for n, fio in num_name[:20]:  # не более 20 первых для примера
            out[str(idx)].append({"name": fio.strip().strip("."), "pos": "F", "number": n})
            if len(out[str(idx)]) >= 5:
                idx += 1
                if idx > 4: break
        return out

    return {
        "home": {
            "team": home_name,
            "goalies": [],  # разметка киперов потребует доп. логики — прикрутим на шаге 2
            "lines": pack(text),
            "bench": []
        },
        "away": {
            "team": away_name,
            "goalies": [],
            "lines": pack(text),
            "bench": []
        }
    }

# ---- HTTP endpoints ---------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"ok": True, "engine": "ocr", "tessdata": os.environ.get("TESSDATA_PREFIX", "")})

@app.route("/ocr")
def ocr_endpoint():
    url = request.args.get("url")
    page = int(request.args.get("page", "0"))
    if not url:
        return jsonify({"ok": False, "error": "param 'url' required"}), 400
    t0 = time.time()
    try:
        pdf = pdf_bytes_from_url(url)
        with fitz.open(stream=pdf, filetype="pdf") as doc:
            if page >= doc.page_count:
                return jsonify({"ok": False, "error": "page out of range"}), 400
            pix = doc.load_page(page).get_pixmap(dpi=300, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            txt = pytesseract.image_to_string(img, lang="rus+eng")
        return jsonify({
            "ok": True, "engine": "ocr", "page": page,
            "text_len": len(txt), "snippet": txt[:400],
            "duration_s": round(time.time()-t0, 3)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/extract")
def extract_endpoint():
    url = request.args.get("url")
    if not url:
        return jsonify({"ok": False, "error": "param 'url' required"}), 400
    t0 = time.time()
    try:
        pdf = pdf_bytes_from_url(url)
        words = extract_words_first_page(pdf)
        data_struct = parse_teams_and_lines(words)
        referees = get_referees(pdf)

        out = {
            "ok": True,
            "engine": "words",
            "data": data_struct,
            "referees": {
                "main": referees.get("main", []),
                "linesmen": referees.get("linesmen", [])
            },
            "referee_entries": (
                [{"role": "Главный судья", "name": x} for x in referees.get("main", [])] +
                [{"role": "Линейный судья", "name": x} for x in referees.get("linesmen", [])]
            ),
            "source_url": url,
            "duration_s": round(time.time()-t0, 3)
        }
        return jsonify(out)
    except requests.HTTPError as he:
        return jsonify({"ok": False, "error": f"fetch failed: {he}"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
