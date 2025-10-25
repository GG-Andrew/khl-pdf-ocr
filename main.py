# --- ВСТАВИТЬ/ОБНОВИТЬ ГДЕ УДОБНО В РАЗДЕЛЕ УТИЛИТ ---
import re
import unicodedata
from statistics import median

ROLE_ALIASES = {
    "Главный судья": "Главный судья",
    "Главные судьи": "Главный судья",
    "Линейный судья": "Линейный судья",
    "Линейные судьи": "Линейный судья",
}

def _norm_txt(s: str) -> str:
    # снимаем ударения/диакритики, нормализуем пробелы, приводим ё/й и т.п.
    s = s.replace('\xa0',' ').replace('  ',' ')
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('й','й').replace('Й','Й').replace('ё','е').replace('Ё','Е')
    return re.sub(r'\s+', ' ', s).strip()

def find_roster_page(doc):
    # ищем первую страницу, где встречается заголовок состава
    for i in range(len(doc)):
        t = _norm_txt(doc.load_page(i).get_text())
        if re.search(r'Составы|Составы команд', t, re.I):
            return i
    return 0  # fallback

def split_two_columns(words):
    # words: список кортежей (x0,y0,x1,y1,word,block_no,...) из get_text("words")
    # строим гистограмму "просвета" по x и ищем вертикальную щель между колонками
    xs = sorted([ (w[0]+w[2]) / 2.0 for w in words ])  # центры слов
    if not xs:
        return [], []
    # берём медиану как первичную границу, потом уточняем по плотности
    mid = median(xs)
    left = [w for w in words if ((w[0]+w[2])/2.0) <= mid]
    right = [w for w in words if ((w[0]+w[2])/2.0) > mid]
    # иногда медиана попадает криво — если один список слишком мал, просто делим по ширине страницы
    if len(left) < 10 or len(right) < 10:
        page_w = max(w[2] for w in words)
        cut = page_w * 0.5
        left  = [w for w in words if ((w[0]+w[2])/2.0) <= cut]
        right = [w for w in words if ((w[0]+w[2])/2.0) > cut]
    # сортируем по y, затем x
    left.sort(key=lambda w: (w[1], w[0]))
    right.sort(key=lambda w: (w[1], w[0]))
    return left, right

def words_to_lines(words, y_tolerance=3.0):
    # группируем слова в строки по близости Y
    lines = []
    cur = []
    last_y = None
    for w in words:
        y = w[1]
        if last_y is None or abs(y - last_y) <= y_tolerance:
            cur.append(w)
        else:
            cur.sort(key=lambda z: z[0])
            line_txt = ' '.join(_norm_txt(z[4]) for z in cur)
            if line_txt.strip():
                lines.append(line_txt)
            cur = [w]
        last_y = y
    if cur:
        cur.sort(key=lambda z: z[0])
        line_txt = ' '.join(_norm_txt(z[4]) for z in cur)
        if line_txt.strip():
            lines.append(line_txt)
    return lines

ROSTER_ROW_RE = re.compile(
    r'^(?P<num>\d{1,3})\s+(?P<pos>[ВЗН])\s+(?P<name>[A-ЯЁA-Za-z\-’\'\.]+(?:\s+[A-ЯЁA-Za-z\-’\'\.]+){0,3})\s+(?P<dob>\d{2}\.\d{2}\.\d{4})\s+(?P<age>\d{1,2})$'
)

def parse_roster_lines(lines, side):
    out = []
    for ln in lines:
        ln = _norm_txt(ln)
        m = ROSTER_ROW_RE.match(ln)
        if not m:
            continue
        d = m.groupdict()
        out.append({
            "side": side,
            "num": d["num"],
            "pos": d["pos"],
            "name": d["name"].strip().strip('*').strip(),
            "capt": "",
            "dob": d["dob"],
            "age": d["age"],
            "gk_flag": "S" if re.search(r'\sS\b', ln) else ("R" if re.search(r'\sR\b', ln) else ""),
            "gk_status": "starter" if re.search(r'\sS\b', ln) else ("reserve" if re.search(r'\sR\b', ln) else None),
        })
    return out

REFS_BLOCK_RE = re.compile(r'(Судьи|Судейская бригада)(.*?)(\n\n|$)', re.S|re.I)
REF_LINE_RE = re.compile(
    r'(?P<role>Главн(ые|ый)\s+судья|Линейн(ые|ый)\s+судья|Линейный судья|Главный судья)\s*[:\-–]?\s*'
    r'(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё\.]+){1,2})'
)

def extract_refs_from_text(full_text):
    full_text = _norm_txt(full_text)
    m = REFS_BLOCK_RE.search(full_text)
    if not m:
        return []
    block = m.group(0)
    refs = []
    for mm in REF_LINE_RE.finditer(block):
        role = ROLE_ALIASES.get(_norm_txt(mm.group('role')).replace('Линейн','Линейн'), 'Линейный судья')
        name = _norm_txt(mm.group('name'))
        if name and name not in ('Главный судья','Линейный судья'):
            refs.append({"role": role, "name": name})
    # dedupe, сохраняем порядок
    uniq = []
    seen = set()
    for r in refs:
        k = (r['role'], r['name'])
        if k not in seen:
            uniq.append(r); seen.add(k)
    return uniq
