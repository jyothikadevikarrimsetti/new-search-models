from scripts.extract_text import save_processed_text
from scripts.metadata import extract_metadata
from scripts.vector_store import (
    upsert_to_pinecone,
    delete_from_pinecone,
    pinecone_vector_exists,                   # 👈 NEW
)
from scripts.search_pipeline import search_query , hybrid_search
from scripts.hash_utils import compute_md5

from pathlib import Path
import json
import os

INPUT       = "data/input_pdf_data"
TEXTS       = "data/processed_data"
OUTPUT      = "data/output_data"
HASH_STORE  = "data/pdf_hashes.json"

# ------------------------------------------------------------------ #
# 1.  Load stored MD5 hashes (or start empty)                        #
# ------------------------------------------------------------------ #
try:
    with open(HASH_STORE, "r") as fh:
        stored_hashes = json.load(fh)
except FileNotFoundError:
    stored_hashes = {}

<<<<<<< HEAD
updated_files = []
delete_from_pinecone("MHC_CaseStatus_511652")
# Check if processed_data folder is empty or doesn't exist
if not os.path.exists(TEXTS) or not os.listdir(TEXTS):
    print("⚠️ No processed files found. Reprocessing all PDFs...")
    for pdf_file in Path(INPUT).glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        try:
            save_processed_text(INPUT, TEXTS, specific_pdf=pdf_file)
            txt_path = Path(TEXTS) / f"{pdf_file.stem}.txt"
            if txt_path.exists():
                stored_hashes[pdf_file.name] = compute_md5(pdf_file)
                updated_files.append(pdf_file.stem)
                print(f"✅ Successfully processed {pdf_file.name}")
            else:
                print(f"❌ Failed to create text file for {pdf_file.name}")
        except Exception as e:
            print(f"⛔ Error processing {pdf_file.name}: {str(e)}")
else:
    # Check for updated PDFs
    for pdf_file in Path(INPUT).glob("*.pdf"):
        current_hash = compute_md5(pdf_file)
        file_key = pdf_file.name
        txt_path = Path(TEXTS) / f"{pdf_file.stem}.txt"

        if stored_hashes.get(file_key) != current_hash:
            print(f"🔄 PDF updated: {pdf_file.name}")
            try:
                save_processed_text(INPUT, TEXTS, specific_pdf=pdf_file)
                if txt_path.exists():
                    stored_hashes[file_key] = current_hash
                    updated_files.append(pdf_file.stem)
                else:
                    print(f"❌ Text extraction failed for: {pdf_file.name}")
            except Exception as e:
                print(f"⛔ Error processing {pdf_file.name}: {str(e)}")

# Save updated hashes
with open(HASH_STORE, "w") as f:
    json.dump(stored_hashes, f, indent=2)

# Re-extract metadata for updated files
for pdf_stem in updated_files:
    txt_path = Path(TEXTS) / f"{pdf_stem}.txt"
    json_path = Path(OUTPUT) / f"{pdf_stem}.json"

    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        metadata = extract_metadata(text)
        metadata["filename"] = txt_path.name

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        delete_from_pinecone(pdf_stem)

# Upsert updated vectors - CORRECTED parameter name here
upsert_to_pinecone(OUTPUT, only_ids=updated_files)

# Ask a question
user_question = input("❓ Enter your question: ")
search_query(user_question, top_k=1)
=======
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
    txt_path = Path(TEXTS) / f"{pdf_file.stem}.txt"
    current_hash = compute_md5(pdf_file)

    needs_reprocess = (
        pdf_file.name not in stored_hashes              # new file
        or stored_hashes[pdf_file.name] != current_hash # changed file
        or not txt_path.exists()                        # txt missing
    )

    if needs_reprocess:
        label = "Processing" if pdf_file.name not in stored_hashes else "🔄 Updating"
        print(f"{label} {pdf_file.name} …")
        try:
            save_processed_text(INPUT, TEXTS, specific_pdf=pdf_file)
            if txt_path.exists():
                stored_hashes[pdf_file.name] = current_hash
                updated_files.append(pdf_file.stem)
                print(f"✅  Finished {pdf_file.name}")
            else:
                print(f"❌  Failed to create text for {pdf_file.name}")
        except Exception as e:
            print(f"⛔ Error while processing {pdf_file.name}: {e}")

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

for txt_file in Path(TEXTS).glob("*.txt"):
    stem = txt_file.stem
    json_path = Path(OUTPUT) / f"{stem}.json"
    if not json_path.exists():
        text = txt_file.read_text(encoding="utf-8").strip()
        metadata = extract_metadata(text) | {"filename": txt_file.name}
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"📝 Metadata JSON created for {stem}")
        # Only add to need_upsert if not already in Pinecone
        if not pinecone_vector_exists(stem):
            need_upsert.add(stem)
    else:
        # If JSON exists, check if vector exists in Pinecone
        if not pinecone_vector_exists(stem):
            need_upsert.add(stem)

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
        metadata = extract_metadata(text) | {"filename": txt_path.name}

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
search_query(user_question, top_k=1)
# results =
hybrid_search(user_question, top_k=1)
# for match in results:
#     print(f"Score: {match['score']}, Metadata: {match['metadata']}")
>>>>>>> 9914bd6788a2ae7dc73974a15c37bc071852eb84
