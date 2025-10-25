from fastapi import FastAPI, Query
from pydantic import BaseModel
import requests
from pdf2image import convert_from_bytes
import pytesseract

app = FastAPI(title="KHL PDF OCR Service")

class PDFResponse(BaseModel):
    ok: bool
    match_id: int
    season: int
    source_pdf: str
    pages_ocr: int
    text_len: int
    snippet: str

@app.get("/extract", response_model=PDFResponse)
def extract_pdf(
    match_id: int = Query(...),
    pdf_url: str = Query(...),
    season: int = Query(...),
):
    try:
        # Скачиваем PDF
        r = requests.get(pdf_url, timeout=15)
        r.raise_for_status()
        pdf_bytes = r.content

        # OCR страниц PDF
        images = convert_from_bytes(pdf_bytes, dpi=130)
        full_text = ""
        for img in images:
            full_text += pytesseract.image_to_string(img, lang="rus+eng")

        snippet = full_text[:300]  # первые 300 символов для превью

        return PDFResponse(
            ok=True,
            match_id=match_id,
            season=season,
            source_pdf=pdf_url,
            pages_ocr=len(images),
            text_len=len(full_text),
            snippet=snippet
        )
    except Exception as e:
        return {
            "ok": False,
            "match_id": match_id,
            "season": season,
            "error": str(e)
        }
