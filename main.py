# from scripts.extract_text import save_processed_text
# from scripts.metadata import extract_metadata
# from scripts.vector_store import upsert_to_pinecone
# from scripts.search_pipeline import search_query
# import json
# from pathlib import Path
# import time

# INPUT = "data/input_pdf_data"
# TEXTS = "data/processed_data"
# OUTPUT = "data/output_data"

# # ✅ STEP 1: Only extract and clean new PDFs
# for pdf_file in Path(INPUT).glob("*.pdf"):
#     txt_path = Path(TEXTS) / f"{pdf_file.stem}.txt"
#     if not txt_path.exists():
#         print(f"📄 Processing PDF: {pdf_file.name}")
#         save_processed_text(INPUT, TEXTS)
#         break
#     else:
#         print(f"✅ Skipped (already processed): {pdf_file.name}")

# # ✅ STEP 2: Only extract metadata for new .txt files
# for txt_file in Path(TEXTS).glob("*.txt"):
#     json_path = Path(OUTPUT) / f"{txt_file.stem}.json"
#     if not json_path.exists():
#         print(f"🔍 Extracting metadata for: {txt_file.name}")
#         text = txt_file.read_text()
#         meta = extract_metadata(text)
#         # meta = extract_metadata(text, filename=txt_file.name)

#         meta["filename"] = txt_file.name
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(meta, f, indent=2)
#     else:
#         print(f"✅ Skipped metadata (already exists): {txt_file.name}")

# # ✅ STEP 3: Push metadata to Pinecone only if vector not yet present (simple retry)
# upsert_to_pinecone(OUTPUT)

# # ✅ STEP 4: Run user query with re-ranking and show top 1 result with timing
# user_question = input("❓ Enter your question: ")
# # start_time = time.time()
# search_query(user_question, top_k=1)
# # end_time = time.time()

# # print("\n🎯 Top Answer:")
# # print(answer)
# # print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
from scripts.extract_text import save_processed_text
from scripts.metadata import extract_metadata
from scripts.vector_store import upsert_to_pinecone, delete_from_pinecone
from scripts.search_pipeline import search_query
from scripts.hash_utils import compute_md5
from pathlib import Path
import json
import os

INPUT = "data/input_pdf_data"
TEXTS = "data/processed_data"
OUTPUT = "data/output_data"
HASH_STORE = "data/pdf_hashes.json"

# Load existing hashes
try:
    with open(HASH_STORE, "r") as f:
        stored_hashes = json.load(f)
except FileNotFoundError:
    stored_hashes = {}

updated_files = []
delete_from_pinecone("MHC_CaseStatus_511661")
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
