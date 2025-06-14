from scripts.extract_text import save_processed_text
from scripts.metadata import extract_metadata
from scripts.vector_store import (
    upsert_to_pinecone,
    delete_from_pinecone,
    pinecone_vector_exists,                   # 👈 NEW
)
from scripts.search_pipeline import search_query , hybrid_search
from scripts.hash_utils import compute_md5
from scripts.filter_utils import generate_filter

from pathlib import Path
import json
import os

INPUT       = "data/input_pdf_data"
TEXTS       = "data/processed_data"
OUTPUT      = "data/output_data"
HASH_STORE  = "data/pdf_hashes.json"
CHUNKS      = "data/chunks"

# ------------------------------------------------------------------ #
# 1.  Load stored MD5 hashes (or start empty)                        #
# ------------------------------------------------------------------ #
try:
    with open(HASH_STORE, "r") as fh:
        stored_hashes = json.load(fh)
except FileNotFoundError:
    stored_hashes = {}

updated_files: list[str] = []            # PDFs whose text has changed
# ------------------------------------------------------------------ #
# 2.  (Re)process PDFs when needed                                   #
# ------------------------------------------------------------------ #
if not os.path.exists(TEXTS) or not os.listdir(TEXTS):
    print("⚠️  No processed files found. Reprocessing all PDFs …")
    pdf_iter = Path(INPUT).glob("*.pdf")
else:
    pdf_iter = Path(INPUT).glob("*.pdf")

for pdf_file in pdf_iter:
    current_hash = compute_md5(pdf_file)

    needs_reprocess = (
        pdf_file.name not in stored_hashes              # new file
        or stored_hashes[pdf_file.name] != current_hash # changed file
    )

    # Check if chunk files exist for this PDF
    chunk_prefix = f"{pdf_file.stem}_chunk"
    chunk_files = list(Path(CHUNKS).glob(f"{chunk_prefix}*.txt"))
    if not chunk_files:
        needs_reprocess = True

    if needs_reprocess:
        label = "Processing" if pdf_file.name not in stored_hashes else "🔄 Updating"
        print(f"{label} {pdf_file.name} …")
        try:
            # Use multiprocessing for chunking if more than one PDF
            save_processed_text(INPUT, TEXTS, specific_pdf=pdf_file, chunk_size=300, overlap=50, chunk_dir=CHUNKS, use_multiprocessing=False)
            chunk_files = list(Path(CHUNKS).glob(f"{chunk_prefix}*.txt"))
            if chunk_files:
                stored_hashes[pdf_file.name] = current_hash
                updated_files.extend([f.stem for f in chunk_files])
                print(f"✅  Finished {pdf_file.name} ({len(chunk_files)} chunk(s))")
            else:
                print(f"❌  Failed to create text chunks for {pdf_file.name}")
        except Exception as e:
            print(f"⛔ Error while processing {pdf_file.name}: {e}")

# If you want to process all PDFs in parallel, you can call:
# save_processed_text(INPUT, TEXTS, chunk_size=300, overlap=50, chunk_dir=CHUNKS, use_multiprocessing=True)

# ------------------------------------------------------------------ #
# 3.  Persist updated hashes                                         #
# ------------------------------------------------------------------ #
with open(HASH_STORE, "w") as fh:  
    json.dump(stored_hashes, fh, indent=2)

# ------------------------------------------------------------------ #
# 4.  Build the *need_upsert* set                                    #
#     • everything just updated, plus                                #
#     • any existing TXT whose vector is absent in Pinecone          #
# ------------------------------------------------------------------ #
need_upsert: set[str] = set(updated_files)

# Use CHUNKS directory for downstream processing
for txt_file in Path(CHUNKS).glob("*.txt"):
    stem = txt_file.stem
    json_path = Path(OUTPUT) / f"{stem}.json"
    if not json_path.exists():
        text = txt_file.read_text(encoding="utf-8").strip()
        # metadata = extract_metadata(text) | {"filename": txt_file.name}
        # Add original PDF filename as document_name
        # Assumes chunk name is like 'MyPDF_chunk1.txt' and original PDF is 'MyPDF.pdf'
        pdf_stem = txt_file.stem.rsplit('_chunk', 1)[0]
        pdf_name = f"{pdf_stem}.pdf"
        metadata = extract_metadata(text, document_name=pdf_name) | {"filename": txt_file.name}
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"📝 Metadata JSON created for {stem}")
        if not pinecone_vector_exists(stem):
            need_upsert.add(stem)
    else:
        if not pinecone_vector_exists(stem):
            need_upsert.add(stem)
        # Print metadata for debugging
        with open(json_path, "r", encoding="utf-8") as fh:
            meta_dbg = json.load(fh)
        print(f"[DEBUG] Metadata for {stem}: {json.dumps(meta_dbg, ensure_ascii=False)}")

# ------------------------------------------------------------------ #
# 5.  Ensure metadata JSON exists for everything in need_upsert      #
# ------------------------------------------------------------------ #
for stem in list(need_upsert):  # copy—it may shrink
    txt_path   = Path(TEXTS)  / f"{stem}.txt"
    json_path  = Path(OUTPUT) / f"{stem}.json"

    # If metadata JSON does not exist, create it
    if not json_path.exists():
        if not txt_path.exists():
            print(f"⛔ Missing TXT for {stem}; skipping.")
            need_upsert.discard(stem)
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        pdf_stem = stem.rsplit('_chunk', 1)[0]
        pdf_name = f"{pdf_stem}.pdf"
        metadata = extract_metadata(text, document_name=pdf_name) | {"filename": txt_path.name}
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"📝 Metadata JSON created for {stem}")

# ------------------------------------------------------------------ #
# 6.  Delete old vectors for files that were *updated*               #
# ------------------------------------------------------------------ #
for stem in updated_files:
    delete_from_pinecone(stem)

# ------------------------------------------------------------------ #
# 7.  Upsert everything that’s needed                                #
# ------------------------------------------------------------------ #
if need_upsert:
    print(f"🚀 Upserting {len(need_upsert)} vector(s) …")
    upsert_to_pinecone(OUTPUT, only_ids=need_upsert)
else:
    print("✅ Nothing new to upsert—Pinecone already up-to-date.")

# ------------------------------------------------------------------ #
# 8.  Interactive query                                              #
# ------------------------------------------------------------------ #
user_question = input("❓ Enter your question: ")

# AUTOMATIC FILTER GENERATION
user_filter = generate_filter(user_question)
print(f"[INFO] Auto-generated metadata filter: {user_filter}")

search_query(user_question, top_k=1, filter=user_filter)
results = hybrid_search(user_question, top_k=1, filter=user_filter)

# for match in results:
#     print(f"Score: {match['score']}, Metadata: {match['metadata']}")
