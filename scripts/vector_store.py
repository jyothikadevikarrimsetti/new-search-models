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

from scripts.pinecone_utils import index
from pathlib import Path
import json

def upsert_to_pinecone(json_dir, only_ids=None, namespace="default"):
    """Upsert vectors to a specific Pinecone namespace"""
    for json_file in Path(json_dir).glob("*.json"):
        if only_ids and json_file.stem not in only_ids:
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = {
            "keywords": ", ".join(data.get("keywords", [])),
            "entities": ", ".join(data.get("entities", [])),
            "intent": data.get("intent", ""),
            "summary": data.get("summary", "")
        }

        index.upsert(
            vectors=[(json_file.stem, data["embedding"], metadata)],
            namespace=namespace
        )
        print(f"✅ Upserted vector {json_file.stem} to namespace '{namespace}'")

def delete_from_pinecone(vector_id, namespace="default"):
    """Delete vector from a specific Pinecone namespace"""
    try:
        index.delete(
            ids=[vector_id], 
            namespace=namespace
        )
        print(f"🗑️ Deleted vector {vector_id} from namespace '{namespace}'")
    except Exception as e:
        print(f"❌ Error deleting {vector_id}: {str(e)}")
