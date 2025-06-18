# scripts/extract_text.py

import re
from pathlib import Path
import pdfplumber
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import multiprocessing
import threading
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

def extract_text_from_pdf(pdf_path):
    # Try pdfplumber first
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if text and text.strip():
            return text
    except Exception as e:
        print(f"[pdfplumber] Extraction failed for {pdf_path.name}: {e}")
    # Fallback to PyPDF2
    try:
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


def recursive_chunk(text, chunk_size=300, overlap=50):
    # 1. Try to split by section headings (e.g., JUDGMENT, ORDER, PRAYER)
    section_pattern = re.compile(
        r"\n\s*(JUDGMENT|ORDER|PRAYER|DECREE|SUMMARY|CONCLUSION|CORAM|PARTIES|APPEAL|PETITION|RESPONDENT|APPELLANT|DATE|FACTS|ARGUMENTS|FINDINGS|REASONING|DISPOSITION|DECISION|HELD|CASE NO|CASE NUMBER|CASE STATUS|BACKGROUND|INTRODUCTION|PROCEEDINGS|SUBMISSIONS|CONTENTIONS|EVIDENCE|ANALYSIS|DISCUSSION|CONCLUSION|RESULT|OUTCOME|RELIEF|RECOMMENDATION|ANNEXURE|APPENDIX|EXHIBIT|REFERENCE|FOOTNOTE|ENDNOTE|INDEX|TABLE OF CONTENTS|LIST OF AUTHORITIES|CITATION|CITATIONS|REFERENCES|NOTES|NOTE|NOTE:|NOTE -)\b.*\n",
        re.IGNORECASE
    )
    sections = section_pattern.split(text)
    if len(sections) > 1:
        # Merge section headings with their content
        merged = []
        for i in range(0, len(sections)-1, 2):
            heading = sections[i+1].strip()
            content = sections[i+2].strip() if i+2 < len(sections) else ''
            merged.append(f"{heading}\n{content}")
        sections = merged
    else:
        sections = [text]
    chunks = []
    for section in sections:
        # 2. Split by paragraphs (double newline)
        paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
        for para in paragraphs:
            # 3. Split by sentences (spaCy if available, else regex)
            if nlp:
                doc = nlp(para)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                sentences = re.split(r'(?<=[.!?])\s+', para)
            # 4. Recursively build chunks
            current = []
            for sent in sentences:
                words = sent.split()
                if not words:
                    continue
                if len(words) > chunk_size:
                    # If sentence is too long, split by words
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk = words[i:i+chunk_size]
                        if len(chunk) >= 10:
                            chunks.append(' '.join(chunk))
                else:
                    current.extend(words)
                    if len(current) >= chunk_size:
                        chunks.append(' '.join(current[:chunk_size]))
                        # Overlap
                        current = current[chunk_size - overlap:]
            # Add any remaining words as a chunk
            if current and len(current) >= 10:
                chunks.append(' '.join(current))
    return chunks


def chunk_text(text, chunk_size=300, overlap=50):
    # Use recursive chunking
    return recursive_chunk(text, chunk_size=chunk_size, overlap=overlap)


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

