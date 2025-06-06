# scripts/extract_text.py

import re
from pathlib import Path
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_pdf(pdf_path):
    try:
        # Attempt native text extraction
        reader = PdfReader(str(pdf_path))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        if text.strip():
            return text
    except Exception:
        pass

    # If native extraction fails, fallback to OCR
    print(f"⚠️ OCR fallback for: {pdf_path.name}")
    ocr_text = ""
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            ocr_text += pytesseract.image_to_string(image)
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return ""

    return ocr_text

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\-]", "", text)  # Keep only clean characters
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# def save_processed_text(input_dir, output_dir):
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     for pdf in Path(input_dir).glob("*.pdf"):
#         raw = extract_text_from_pdf(pdf)
#         cleaned = clean_text(raw)

#         # Filter garbage: require at least 10 words
#         if len(cleaned.split()) < 10:
#             print(f"⚠️ Skipping low-quality output for: {pdf.name}")
#             continue

#         out_file = Path(output_dir) / f"{pdf.stem}.txt"
#         out_file.write_text(cleaned, encoding="utf-8")
#         print(f"✅ Saved cleaned text for: {pdf.name}")

def save_processed_text(input_dir, output_dir, specific_pdf=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pdfs = [specific_pdf] if specific_pdf else Path(input_dir).glob("*.pdf")

    for pdf in pdfs:
        raw = extract_text_from_pdf(pdf)
        cleaned = clean_text(raw)

        if len(cleaned.split()) < 10:
            print(f"⚠️ Skipping low-quality output for: {pdf.name}")
            continue

        out_file = Path(output_dir) / f"{pdf.stem}.txt"
        out_file.write_text(cleaned, encoding="utf-8")
        print(f"✅ Saved cleaned text for: {pdf.name}")

