# scripts/extract_text.py

import re
from pathlib import Path
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import multiprocessing
import threading

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


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        if len(chunk) < 10:
            break
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks


def process_pdf_chunking(args):
    pdf, input_dir, output_dir, chunk_size, overlap, chunk_dir = args
    raw = extract_text_from_pdf(pdf)
    cleaned = clean_text(raw)
    if len(cleaned.split()) < 10:
        print(f"⚠️ Skipping low-quality output for: {pdf.name}")
        return []
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        print(f"⚠️ No valid chunks for: {pdf.name}")
        return []
    chunk_files = []
    for idx, chunk in enumerate(chunks, 1):
        out_file = Path(chunk_dir or output_dir) / f"{pdf.stem}_chunk{idx}.txt"
        out_file.write_text(chunk, encoding="utf-8")
        chunk_files.append(out_file)
    print(f"✅ Saved {len(chunks)} chunk(s) for: {pdf.name}")
    return chunk_files


def save_processed_text(input_dir, output_dir, specific_pdf=None, chunk_size=300, overlap=50, chunk_dir=None, use_multiprocessing=False, use_multithreading=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if chunk_dir:
        Path(chunk_dir).mkdir(parents=True, exist_ok=True)
    pdfs = [specific_pdf] if specific_pdf else list(Path(input_dir).glob("*.pdf"))

    all_chunk_files = []
    if use_multithreading and len(pdfs) > 1:
        threads = []
        results = [[] for _ in pdfs]
        def thread_target(i, args):
            results[i] = process_pdf_chunking(args)
        for i, pdf in enumerate(pdfs):
            args = (pdf, input_dir, output_dir, chunk_size, overlap, chunk_dir)
            t = threading.Thread(target=thread_target, args=(i, args))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        for chunk_list in results:
            all_chunk_files.extend(chunk_list)
    else:
        for pdf in pdfs:
            chunk_files = process_pdf_chunking((pdf, input_dir, output_dir, chunk_size, overlap, chunk_dir))
            all_chunk_files.extend(chunk_files)
    return all_chunk_files

