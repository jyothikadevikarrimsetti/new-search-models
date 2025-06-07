# import json
# from scripts.pinecone_utils import index
# from pathlib import Path

# def upsert_to_pinecone(json_dir):
#     for json_file in Path(json_dir).glob("*.json"):
#         with open(json_file, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         metadata = {
#             "keywords": ", ".join(data.get("keywords", [])),
#             "entities": ", ".join(data.get("entities", [])),
#             "intent": data.get("intent", ""),
#             "summary": data.get("summary", "")
#         }

#         index.upsert([(json_file.stem, data["embedding"], metadata)])


# scripts/vector_store.py
"""
Utilities for upserting JSON embeddings and deleting vectors by ID
using the *latest* Pinecone Python client.
"""
from pathlib import Path
import json
from typing import Iterable, Optional

from scripts.pinecone_utils import index


# ------------------------------------------------------------------ #
# Upsert                                                             #
# ------------------------------------------------------------------ #
def upsert_to_pinecone(
    json_dir: str | Path,
    only_ids: Optional[Iterable[str]] = None,
    namespace: str = "__default__",
) -> None:
    """
    Read every *.json file in `json_dir`, extract the embedding and metadata,
    and upsert it to Pinecone.

    Each JSON file must have:
        {
            "embedding": [...],
            "keywords": [...],
            "entities": [...],
            "intent": "...",
            "summary": "..."
        }

    The file name (without .json) is used as the vector ID.
    """
    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"{json_dir} is not a directory")

    for json_file in json_dir.glob("*.json"):
        vector_id = json_file.stem
        if only_ids and vector_id not in only_ids:
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            vectors = [
                (
                    vector_id,
                    data["embedding"],
                    {
                        "keywords": ", ".join(data.get("keywords", [])),
                        "entities": ", ".join(data.get("entities", [])),
                        "intent": data.get("intent", ""),
                        "summary": data.get("summary", ""),
                        "text": data.get("text", ""),  # <-- Add this line if you want to store the original text

                    },
                )
            ]

            index.upsert(vectors=vectors)
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


