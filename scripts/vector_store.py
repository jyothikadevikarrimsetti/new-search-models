
"""
Utilities for upserting JSON embeddings and deleting vectors by ID
using the *latest* Pinecone Python client.
"""
from pathlib import Path
import json
from typing import Iterable, Optional

from scripts.pinecone_utils import index
from pinecone_text.sparse import BM25Encoder


# Fit BM25 on your corpus (all texts)
def get_bm25_encoder(texts):
    bm25 = BM25Encoder()
    bm25.fit(texts)
    return bm25
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

    # Gather all texts for BM25 fitting
    all_texts = []
    for json_file in json_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            all_texts.append(data.get("text", ""))

    bm25 = get_bm25_encoder(all_texts)

    for json_file in json_dir.glob("*.json"):
        vector_id = json_file.stem
        if only_ids and vector_id not in only_ids:
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            dense_vec = data["embedding"]
            sparse_vec = bm25.encode_documents([data.get("text", "")])[0]

            vectors = [
                {
                    "id": vector_id,
                    "values": dense_vec,
                    "sparse_values": sparse_vec,
                    "metadata": {
                        "keywords": ", ".join(data.get("keywords", [])),
                        "entities": ", ".join(data.get("entities", [])),
                        "intent": data.get("intent", ""),
                        "summary": data.get("summary", ""),
                        "text": data.get("text", ""),
                    },
                }
            ]

            index.upsert(vectors=vectors, namespace=namespace)
            print(f"✅  Upserted '{vector_id}' → namespace '{namespace}'")

        except Exception as err:
            print(f"❌  Error upserting '{vector_id}': {err}")

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


