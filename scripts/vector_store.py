"""
Utilities for upserting JSON embeddings and deleting vectors by ID
using the *latest* Pinecone Python client.
"""
from pathlib import Path
import json
from typing import Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.pinecone_utils import index
from pinecone_text.sparse import SpladeEncoder

_splade_encoder_instance = None

def get_splade_encoder():
    global _splade_encoder_instance
    if _splade_encoder_instance is None:
        _splade_encoder_instance = SpladeEncoder()
    return _splade_encoder_instance
# ------------------------------------------------------------------ #
# Upsert                                                             #
# ------------------------------------------------------------------ #

def upsert_to_pinecone(
    json_dir: str | Path,
    only_ids: Optional[Iterable[str]] = None,
    namespace: str = "__default__",
) -> None:
    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"{json_dir} is not a directory")

    splade = get_splade_encoder()

    def upsert_one(json_file):
        vector_id = json_file.stem
        if only_ids and vector_id not in only_ids:
            return None
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            dense_vec = data["embedding"]
            text_for_sparse = data.get("text", "").lower()
            sparse_vec = splade.encode_documents([text_for_sparse])[0]
            print(f"Upserting {vector_id}: SPLADE sparse_vec indices[:5]={sparse_vec.get('indices', [])[:5]}, values[:5]={sparse_vec.get('values', [])[:5]}")
            vectors = [
                {
                    "id": vector_id,
                    "values": dense_vec,
                    # "sparse_values": sparse_vec,
                    "metadata": {
                        "keywords": data.get("keywords", []),  # store as list
                        "entities": data.get("entities", []),  # store as list
                        "intent": data.get("intent", ""),
                        "summary": data.get("summary", ""),
                        "document_name": data.get("document_name", "")  # <-- Add document name from metadata JSON
                    },
                }
            ]
            index.upsert(vectors=vectors, namespace=namespace)
            print(f"✅  Upserted '{vector_id}' → namespace '{namespace}' (SPLADE)")
        except Exception as err:
            print(f"❌  Error upserting '{vector_id}': {err}")
        return None

    json_files = list(json_dir.glob("*.json"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(upsert_one, json_file) for json_file in json_files]
        for future in as_completed(futures):
            _ = future.result()

# ------------------------------------------------------------------ #
# Delete                                                             #
# ------------------------------------------------------------------ #
def delete_from_pinecone(vector_id: str, namespace: str = "__default__"):
    if not vector_id or not isinstance(vector_id, str) or not vector_id.strip():
        print(f"❌ Invalid or empty vector ID: {vector_id!r}")
        return

    # Confirm fetch
    fetch_resp = index.fetch(ids=[vector_id])
    if not fetch_resp.vectors or vector_id not in fetch_resp.vectors:
        print(f"⚠️ Vector '{vector_id}' does not exist in namespace '{namespace}'. Aborting delete.")
        return

    print(f"🧪 Deleting vector ID '{vector_id}' from namespace '{namespace}'")
    index.delete(ids=[vector_id])
    print(f"✅ Deleted vector '{vector_id}' successfully.")

# ------------------------------------------------------------------
# 🔍 Helper: does a vector already live in Pinecone?
# ------------------------------------------------------------------
def pinecone_vector_exists(vector_id: str, namespace: str = "__default__") -> bool:
    """Check if a vector exists in Pinecone using the updated client."""
    try:
        print(f"🔍 Checking if vector ID '{vector_id}' exists in namespace '{namespace}'...")
        response = index.fetch(ids=[vector_id], namespace=namespace)
        # print(f"📦 Fetch response: {response.vectors}")
        return vector_id in response.vectors
    except Exception as err:
        print(f"⚠️ Could not check vector ID '{vector_id}': {err}")
        return False


