# app.py
import os, re, time, json, logging
from typing import List, Tuple, Dict
from flask import Flask, request, jsonify, Response
import requests, fitz, pytesseract
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)
app.json.ensure_ascii = False
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("khl-pdf-ocr")

# ---------- HTTP + Proxy ----------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.9",
    "Referer": "https://www.khl.ru/",
})
# принимает И с /khlpdf, и без него — сам подставит.
PDF_PROXY_BASE = (os.getenv("PDF_PROXY_BASE", "").rstrip("/"))
if PDF_PROXY_BASE and not PDF_PROXY_BASE.endswith("/khlpdf"):
    PDF_PROXY_BASE = PDF_PROXY_BASE + "/khlpdf"

TESS_LANG = os.getenv("TESS_LANG", "rus+eng")

def make_pdf_url(season: str, uid: str) -> str:
    path = f"{season}/{uid}/game-{uid}-start-ru.pdf"
    return f"{PDF_PROXY_BASE}/{path}" if PDF_PROXY_BASE else f"https://www.khl.ru/pdf/{path}"

def http_get(url: str, timeout=30) -> bytes:
    r = SESSION.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content

# ---------- PyMuPDF / OCR helpers ----------
def pdf_to_img(doc: fitz.Document, pno=0, dpi=300) -> Image.Image:
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = doc.load_page(pno).get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_lines(img: Image.Image) -> List[str]:
    g = img.convert("L")
    g = ImageEnhance.Contrast(g).enhance(1.4)
    g = g.filter(ImageFilter.SHARPEN)
    txt = pytesseract.image_to_string(g, lang=TESS_LANG, config="--psm 6")
    return [re.sub(r"\s+", " ", ln).strip() for ln in txt.splitlines() if ln.strip()]

def words_yx(doc: fitz.Document, pno=0) -> List[Tuple[float, float, str]]:
    page = doc.load_page(pno)
    words = page.get_text("words")
    return [(w[1], w[0], w[4]) for w in sorted(words, key=lambda w: (round(w[1],1), w[0]))]

def lines_from_words(words: List[Tuple[float, float, str]], tol=3.0) -> List[str]:
    rows: Dict[float, List[Tuple[float,str]]] = {}
    for y,x,t in words:
        key = next((ky for ky in rows if abs(ky-y)<=tol), None)
        if key is None:
            key=y; rows[key]=[]
        rows[key].append((x,t))
    out=[]
    for ky in sorted(rows):
        s=" ".join(t for _,t in sorted(rows[ky], key=lambda r:r[0]))
        s=re.sub(r"\s+"," ",s).strip()
        if s: out.append(s)
    return out

def page_spans(doc: fitz.Document, pno=0):
    page = doc.load_page(pno)
    d = page.get_text("dict")
    spans=[]
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t=(s.get("text") or "").strip()
                if not t: continue
                bbox=s.get("bbox",[0,0,0,0])
                y=float(bbox[1]); x=float(bbox[0]); size=float(s.get("size",0))
                spans.append((y,x,size,t))
    spans.sort(key=lambda r:(round(r[0],1), r[1]))
    return spans, page.rect.height

# ---------- Parsers ----------
def parse_date_time(lines: List[str]) -> Dict[str,str]:
    meta={"date":"","time_msk":""}
    for ln in lines[:160]:
        m=re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", ln)
        if m: meta["date"]=m.group(0); break
        m2=re.search(r"\b\d{1,2}\s+[А-Яа-яё]+\s+20\d{2}", ln)
        if m2: meta["date"]=m2.group(0).replace(" г.","").strip(); break
    for ln in lines[:200]:
        m=re.search(r"\b([01]\d|2[0-3]):[0-5]\d\b", ln)
        if m: meta["time_msk"]=m.group(0); break
    return meta

def find_teams_from_spans(doc: fitz.Document) -> Dict[str,str]:
    spans, h = page_spans(doc, 0)
    STOP = {"СОСТАВЫ КОМАНД","СУББОТА","ВОСКРЕСЕНЬЕ","ЛД","МАТЧ","НАЧАЛО МАТЧА",
            "ВРАТАРИ","ЗВЕНО","ТРЕНЕР","ГЛАВНЫЙ","ЛИНЕЙНЫЙ","ОБНОВЛЕНО"}
    cand=[]
    for y,x,size,t in spans:
        if y>h*0.35: break
        T=t.replace("«","").replace("»","").strip()
        if len(T)<6: continue
        letters=[ch for ch in T if "А"<=ch<="Я" or ch=="Ё" or "а"<=ch<="я" or ch=="ё"]
        if not letters: continue
        ups=sum(1 for ch in letters if "А"<=ch<="Я" or ch=="Ё")
        if ups/max(1,len(letters))<0.6: continue
        if any(w in T.upper() for w in STOP): continue
        cand.append((size,y,x,T.upper()))
    cand.sort(key=lambda r:(-r[0], r[1], r[2]))
    uniq=[]; seen=set()
    for size,y,x,T in cand:
        key=re.sub(r"\s+"," ",T)
        if key in seen: continue
        seen.add(key); uniq.append((size,y,x,key))
        if len(uniq)>=6: break
    home=uniq[0][3] if uniq else ""
    away=""
    for _,_,_,T in uniq[1:]:
        if T!=home: away=T; break
    return {"home":home,"away":away}

def extract_refs_block(lines: List[str]) -> Tuple[List[str], List[str], Dict]:
    dbg={}
    idx=-1
    for i,ln in enumerate(lines[:80]):
        if "Главный судья" in ln and "Линейный судья" in ln:
            idx=i; break
    if idx!=-1 and idx+1<len(lines):
        s = re.sub(r"Обновлено.*","", lines[idx+1]).strip()
        parts=[p for p in re.split(r"[,\|;]+|\s+", s) if p]
        names=[]; buf=[]
        for p in parts:
            if re.match(r"^[А-ЯЁ][а-яё\-]+$", p):
                buf.append(p)
                if len(buf)==2:
                    names.append(" ".join(buf)); buf=[]
            else:
                buf=[]
        if len(names)<4 and len(parts)>=4:
            alt=[]
            for j in range(len(parts)-1):
                a,b=parts[j],parts[j+1]
                if all(re.match(r"^[А-ЯЁ][а-яё\-]+$", x) for x in (a,b)):
                    alt.append(a+" "+b)
            if len(alt)>=4: names=alt[:4]
        main=names[:2]; linesmen=names[2:4]
        dbg["raw"]=s
        return main, linesmen, dbg
    return [], [], {"note":"header not found"}

def extract_goalies_from_lines(all_lines: List[str]) -> Dict[str,List[Dict]]:
    idxs=[i for i,ln in enumerate(all_lines) if ln.strip().startswith("Вратари")]
    def collect(si):
        acc=[]
        for j in range(si+1, min(si+40,len(all_lines))):
            t=all_lines[j]
            if t.startswith("Звено"): break
            m=re.search(r"^[А-ЯЁ][а-яё\-]+ [А-ЯЁ][а-яё\-]+(?: [А-ЯЁ][а-яё\-]+)?", t)
            if not m: continue
            nm=m.group(0); flag=""
            if re.search(r"\bС\b", t): flag="C"
            elif re.search(r"\bР\b", t): flag="R"
            acc.append({"name":nm,"flag":flag})
        return acc
    home=collect(idxs[0]) if idxs else []
    away=collect(idxs[1]) if len(idxs)>1 else []
    return {"home":home,"away":away}

# ---------- Extractors ----------
def ex_words(doc: fitz.Document) -> Dict:
    w=words_yx(doc,0); lines=lines_from_words(w)
    meta=parse_date_time(lines)
    meta["teams"]=find_teams_from_spans(doc)
    return {"ok":True,"engine":"words","match":meta}

def ex_refs(doc: fitz.Document, debug=False) -> Dict:
    w=words_yx(doc,0); lines=lines_from_words(w)
    main, linesmen, dbg = extract_refs_block(lines)
    if not(main and linesmen):
        # OCR верхней трети
        img=pdf_to_img(doc,0,300); crop=img.crop((0,0,img.width,int(img.height*0.33)))
        ol=ocr_lines(crop)
        i=-1
        for k,ln in enumerate(ol[:120]):
            if "Главный судья" in ln and "Линейный судья" in ln:
                i=k; break
        if i!=-1 and i+1<len(ol):
            s=ol[i+1]
            parts=[p for p in re.split(r"[,|;]|\s+", s) if p]
            cand=[]
            for j in range(len(parts)-1):
                a,b=parts[j],parts[j+1]
                if all(re.match(r"^[А-ЯЁ][а-яё\-]+$", x) for x in (a,b)):
                    cand.append(a+" "+b)
            if len(cand)>=4:
                main=cand[:2]; linesmen=cand[2:4]
                dbg["ocr"]=s
    out={"ok":True,"engine":"ocr-refs","referees":{"main":main,"linesmen":linesmen}}
    if debug: out["_debug"]=dbg
    return out

def ex_goalies(doc: fitz.Document, debug=False) -> Dict:
    w=words_yx(doc,0); lines=lines_from_words(w)
    g=extract_goalies_from_lines(lines)
    if not g["home"] and not g["away"]:
        img=pdf_to_img(doc,0,300)
        crop=img.crop((0,int(img.height*0.15), img.width, int(img.height*0.55)))
        gl=ocr_lines(crop)
        gg=extract_goalies_from_lines(gl)
        if gg["home"] or gg["away"]: g=gg
    out={"ok":True,"engine":"gk","goalies":g}
    if debug: out["_debug"]={"lines_used":len(lines)}
    return out

def ex_all(doc: fitz.Document, season: str, uid: str, debug=False) -> Dict:
    t0=time.time()
    meta=ex_words(doc)
    refs=ex_refs(doc, debug)
    gk=ex_goalies(doc, debug)
    out={
        "ok":True, "engine":"all",
        "match":{"season":season,"uid":uid, **meta.get("match",{})},
        "referees":refs.get("referees",{"main":[],"linesmen":[]}),
        "goalies":gk.get("goalies",{"home":[],"away":[]}),
        "duration_s":round(time.time()-t0,3),
    }
    if debug: out["_debug"]={"words_match":meta.get("match")}
    return out

# ---------- Routes ----------
@app.get("/health")
def health():
    return jsonify({"ok":True,"engine":"ready"})

@app.get("/extract")
def extract():
    season=(request.args.get("season") or "").strip()
    uid=(request.args.get("uid") or "").strip()
    mode=(request.args.get("mode") or "all").strip().lower()
    debug=(request.args.get("debug") in ("1","true","yes"))
    if not season or not uid:
        return jsonify({"ok":False,"error":"season or uid missing"}), 400

    url=make_pdf_url(season, uid)
    try:
        pdf=http_get(url, timeout=30)
    except requests.HTTPError as e:
        return jsonify({"ok":False,"error":f"http {e.response.status_code}","detail":str(e)}), 502
    except Exception as e:
        return jsonify({"ok":False,"error":"download_error","detail":str(e)}), 502

    try:
        doc=fitz.open(stream=pdf, filetype="pdf")
    except Exception as e:
        return jsonify({"ok":False,"error":"pdf_open_error","detail":str(e)}), 500

    try:
        if mode=="refs":   res=ex_refs(doc, debug)
        elif mode in ("gk","goalies"): res=ex_goalies(doc, debug)
        elif mode=="words": res=ex_words(doc)
        else:               res=ex_all(doc, season, uid, debug)
        res["source_url"]=url
        return Response(json.dumps(res, ensure_ascii=False), mimetype="application/json")
    except pytesseract.TesseractNotFoundError:
        return jsonify({"ok":False,"error":"tesseract_missing",
                        "detail":"Install tesseract-ocr + rus/eng langs"}), 500
    except Exception as e:
        return jsonify({"ok":False,"error":"extract_error","detail":str(e)}), 500

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")), debug=False)
