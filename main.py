from scripts.extract_text import save_processed_text
from scripts.metadata import extract_metadata
from scripts.vector_store import (
    upsert_to_pinecone,
    delete_from_pinecone,
    pinecone_vector_exists
)
from scripts.search_pipeline import hybrid_search  # Removed unused search_query
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
# 1. Load stored MD5 hashes                                          #
# ------------------------------------------------------------------ #
try:
    with open(HASH_STORE, "r") as fh:
        stored_hashes = json.load(fh)
except FileNotFoundError:
    stored_hashes = {}

# ------------------------------------------------------------------ #
# 2. Chunk PDFs if needed                                            #
# ------------------------------------------------------------------ #
CHUNK_FLAG = Path(CHUNKS).exists() and any(Path(CHUNKS).glob("*.txt"))
pdf_files = list(Path(INPUT).glob("*.pdf"))

if not os.path.exists(TEXTS) or not os.listdir(TEXTS):
    print("⚠️  No processed files found. Reprocessing all PDFs …")
    save_processed_text(INPUT, TEXTS, chunk_size=300, overlap=50, chunk_dir=CHUNKS, use_multithreading=True)
elif not CHUNK_FLAG:
    print("⚠️  No chunks found. Reprocessing all PDFs …")
    save_processed_text(INPUT, TEXTS, chunk_size=300, overlap=50, chunk_dir=CHUNKS, use_multithreading=True)
else:
    print("✅ Chunks already present. Skipping chunking step.")

pdf_iter = pdf_files

# ------------------------------------------------------------------ #
# 3. Document-level entity aggregation                               #
# ------------------------------------------------------------------ #
# [REMOVED] Document-level entity aggregation for performance
# print(f"[DEBUG] Aggregating document-level entities for {len(pdf_files)} PDFs…")
# doc_entities_map = {}
# for pdf_file in pdf_files:
#     try:
#         chunk_prefix = f"{pdf_file.stem}_chunk"
#         chunk_jsons = sorted(Path(OUTPUT).glob(f"{chunk_prefix}*.json"))
#         all_entities = set()
#         for chunk_json in chunk_jsons:
#             with open(chunk_json, "r", encoding="utf-8") as f:
#                 chunk_metadata = json.load(f)
#             chunk_entities = set(chunk_metadata.get("entities", []))
#             all_entities.update(chunk_entities)
#         doc_entities_map[pdf_file.stem] = all_entities
#     except Exception as e:
#         print(f"[WARN] Could not aggregate entities for {pdf_file.name}: {e}")
#         doc_entities_map[pdf_file.stem] = set()

# ------------------------------------------------------------------ #
# 4. Check for updates and reprocess if needed                       #
# ------------------------------------------------------------------ #
updated_files: list[str] = []

# Shared thread pool for all parallel operations
MAX_WORKERS = 4  # Adjust as needed for your system
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 1. Hash computation
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
            pdf_file.name not in stored_hashes or
            stored_hashes[pdf_file.name] != current_hash
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

    with open(HASH_STORE, "w") as fh:  
        json.dump(stored_hashes, fh, indent=2)

    # 2. Metadata extraction
    need_upsert: set[str] = set(updated_files)
    chunk_txt_files = list(Path(CHUNKS).glob("*.txt"))
    print(f"[DEBUG] Starting metadata extraction for {len(chunk_txt_files)} chunks...")
    def process_metadata(txt_file):
        stem = txt_file.stem
        json_path = Path(OUTPUT) / f"{stem}.json"
        if not json_path.exists():
            text = txt_file.read_text(encoding="utf-8").strip()
            pdf_stem = txt_file.stem.rsplit('_chunk', 1)[0]
            pdf_name = f"{pdf_stem}.pdf"
            metadata = extract_metadata(text, document_name=pdf_name) | {"filename": txt_file.name}
            chunk_entities = set(metadata.get("entities", []))
            metadata["entities"] = sorted(chunk_entities)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
            print(f"📝 Metadata JSON created for {stem}")
            if not pinecone_vector_exists(stem):
                return stem
        else:
            if not pinecone_vector_exists(stem):
                return stem
        return None
    futures = [executor.submit(process_metadata, txt_file) for txt_file in chunk_txt_files]
    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if result:
            need_upsert.add(result)
        if i % 10 == 0 or i == len(chunk_txt_files):
            print(f"[INFO] Processed {i}/{len(chunk_txt_files)} chunks...")

    # 3. Delete old vectors for updated files
    def delete_vector_thread(stem):
        delete_from_pinecone(stem)
    list(executor.map(delete_vector_thread, updated_files))

    # 4. Upsert vectors to Pinecone
    if need_upsert:
        print(f"🚀 Upserting {len(need_upsert)} vector(s) …")
        for stem in need_upsert:
            upsert_to_pinecone(OUTPUT, only_ids=[stem])
    else:
        print("✅ Nothing new to upsert—Pinecone already up-to-date.")

# ------------------------------------------------------------------ #
# 8. Interactive Query                                               #
# ------------------------------------------------------------------ #
user_question = input("❓ Enter your question: ")
user_filter = generate_filter(user_question)
print(f"[INFO] Auto-generated metadata filter: {user_filter}")
results = hybrid_search(user_question, top_k=1, filter=user_filter)
