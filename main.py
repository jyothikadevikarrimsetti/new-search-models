from scripts.extract_text import save_processed_text
from scripts.metadata import extract_metadata
from scripts.vector_store import (
    upsert_to_pinecone,
    delete_from_pinecone,
    pinecone_vector_exists,                   # 👈 NEW
    get_splade_encoder
)
from scripts.search_pipeline import search_query , hybrid_search
from scripts.hash_utils import compute_md5
from scripts.filter_utils import generate_filter

from pathlib import Path
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ------------------------------------------------------------------ #
# 2.  (Re)process PDFs when needed (MULTITHREADED)                  #
# ------------------------------------------------------------------ #
pdf_files = list(Path(INPUT).glob("*.pdf"))
if not os.path.exists(TEXTS) or not os.listdir(TEXTS):
    print("⚠️  No processed files found. Reprocessing all PDFs …")
    # Use multithreading for all PDFs
    save_processed_text(INPUT, TEXTS, chunk_size=300, overlap=50, chunk_dir=CHUNKS, use_multithreading=True)
    pdf_iter = pdf_files
else:
    pdf_iter = pdf_files

updated_files: list[str] = []

# --- NEW: Precompute document-level entities for each PDF ---
doc_entities_map = {}
for pdf_file in pdf_files:
    try:
        # Read full text for the PDF
        from scripts.extract_text import extract_text_from_pdf, clean_text
        full_text = clean_text(extract_text_from_pdf(pdf_file))
        doc_metadata = extract_metadata(full_text, document_name=pdf_file.name)
        doc_entities_map[pdf_file.stem] = set(doc_metadata.get("entities", []))
    except Exception as e:
        print(f"[WARN] Could not extract document-level entities for {pdf_file.name}: {e}")
        doc_entities_map[pdf_file.stem] = set()

# Compute hashes and check for updates in parallel
with ThreadPoolExecutor() as executor:
    future_to_pdf = {executor.submit(compute_md5, pdf_file): pdf_file for pdf_file in pdf_iter}
    hash_results = {}
    for future in as_completed(future_to_pdf):
        pdf_file = future_to_pdf[future]
        try:
            current_hash = future.result()
            hash_results[pdf_file] = current_hash
        except Exception as e:
            print(f"⛔ Error computing hash for {pdf_file.name}: {e}")

for pdf_file, current_hash in hash_results.items():
    needs_reprocess = (
        pdf_file.name not in stored_hashes              # new file
        or stored_hashes[pdf_file.name] != current_hash # changed file
    )
    chunk_prefix = f"{pdf_file.stem}_chunk"
    chunk_files = list(Path(CHUNKS).glob(f"{chunk_prefix}*.txt"))
    if not chunk_files:
        needs_reprocess = True
    if needs_reprocess:
        label = "Processing" if pdf_file.name not in stored_hashes else "🔄 Updating"
        print(f"{label} {pdf_file.name} …")
        try:
            save_processed_text(INPUT, TEXTS, specific_pdf=pdf_file, chunk_size=300, overlap=50, chunk_dir=CHUNKS, use_multithreading=True)
            chunk_files = list(Path(CHUNKS).glob(f"{chunk_prefix}*.txt"))
            if chunk_files:
                stored_hashes[pdf_file.name] = current_hash
                updated_files.extend([f.stem for f in chunk_files])
                print(f"✅  Finished {pdf_file.name} ({len(chunk_files)} chunk(s))")
            else:
                print(f"❌  Failed to create text chunks for {pdf_file.name}")
        except Exception as e:
            print(f"⛔ Error while processing {pdf_file.name}: {e}")

# ------------------------------------------------------------------ #
# 3.  Persist updated hashes                                         #
# ------------------------------------------------------------------ #
with open(HASH_STORE, "w") as fh:  
    json.dump(stored_hashes, fh, indent=2)

# ------------------------------------------------------------------ #
# 4.  Build the *need_upsert* set (METADATA EXTRACTION THREADED)    #
# ------------------------------------------------------------------ #
need_upsert: set[str] = set(updated_files)
chunk_txt_files = list(Path(CHUNKS).glob("*.txt"))

def process_metadata(txt_file):
    stem = txt_file.stem
    json_path = Path(OUTPUT) / f"{stem}.json"
    if not json_path.exists():
        text = txt_file.read_text(encoding="utf-8").strip()
        pdf_stem = txt_file.stem.rsplit('_chunk', 1)[0]
        pdf_name = f"{pdf_stem}.pdf"
        # --- Extract chunk-level metadata ---
        metadata = extract_metadata(text, document_name=pdf_name) | {"filename": txt_file.name}
        # --- Merge document-level entities ---
        doc_entities = doc_entities_map.get(pdf_stem, set())
        chunk_entities = set(metadata.get("entities", []))
        merged_entities = sorted(doc_entities | chunk_entities)
        metadata["entities"] = merged_entities
        print(f"[DEBUG] Entities for {stem}: {metadata['entities']}")  # <-- Debug print
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"📝 Metadata JSON created for {stem}")
        if not pinecone_vector_exists(stem):
            return stem
    else:
        if not pinecone_vector_exists(stem):
            return stem
        with open(json_path, "r", encoding="utf-8") as fh:
            meta_dbg = json.load(fh)
        print(f"[DEBUG] Metadata for {stem}: {json.dumps(meta_dbg, ensure_ascii=False)}")
    return None

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_metadata, txt_file) for txt_file in chunk_txt_files]
    for future in as_completed(futures):
        result = future.result()
        if result:
            need_upsert.add(result)

# ------------------------------------------------------------------ #
# 5.  Ensure metadata JSON exists for everything in need_upsert      #
#     (already handled above in parallel)                            #
# ------------------------------------------------------------------ #
for stem in list(need_upsert):
    txt_path   = Path(TEXTS)  / f"{stem}.txt"
    json_path  = Path(OUTPUT) / f"{stem}.json"
    if not json_path.exists():
        if not txt_path.exists():
            print(f"⛔ Missing TXT for {stem}; skipping.")
            need_upsert.discard(stem)
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        pdf_stem = stem.rsplit('_chunk', 1)[0]
        pdf_name = f"{pdf_stem}.pdf"
        metadata = extract_metadata(text, document_name=pdf_name) | {"filename": txt_path.name}
        # --- Merge document-level entities here as well ---
        doc_entities = doc_entities_map.get(pdf_stem, set())
        chunk_entities = set(metadata.get("entities", []))
        merged_entities = sorted(doc_entities | chunk_entities)
        metadata["entities"] = merged_entities
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"📝 Metadata JSON created for {stem}")

# ------------------------------------------------------------------ #
# 6.  Delete old vectors for files that were *updated* (THREADED)    #
# ------------------------------------------------------------------ #
def delete_vector_thread(stem):
    delete_from_pinecone(stem)

with ThreadPoolExecutor() as executor:
    list(executor.map(delete_vector_thread, updated_files))

# ------------------------------------------------------------------ #
# 7.  Upsert everything that’s needed (THREADED)                     #
# ------------------------------------------------------------------ #
if need_upsert:
    print(f"🚀 Upserting {len(need_upsert)} vector(s) …")
    for stem in need_upsert:
        upsert_to_pinecone(OUTPUT, only_ids=[stem])
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
