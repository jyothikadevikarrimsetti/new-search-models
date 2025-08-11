import os
import json
from typing import List
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from models.datamodel_pdantic import VectorDocument, MetadataModel


from scripts.mongo_utils import upsert_vector_document
from models.datamodel_pdantic import VectorDocument, MetadataModel
import json
import os

def build_vector_document_from_json(json_path):
    """Builds a VectorDocument from a JSON metadata file."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Compose metadata
    metadata = {
        "Key1": raw.get("intent", "unknown"),
        "Key2": "N/A",
        "Key3": "N/A"
    }
    for entity in raw.get("entity_details", []):
        if metadata["Key2"] == "N/A" and entity["type"] == "PERSON":
            metadata["Key2"] = entity["text"].strip()
        elif metadata["Key3"] == "N/A" and entity["type"] in ["PRODUCT", "ORG"]:
            metadata["Key3"] = entity["text"].strip()
        if metadata["Key2"] != "N/A" and metadata["Key3"] != "N/A":
            break

    file_id = os.path.basename(json_path).replace(".json", "")
    doc = VectorDocument(
        _id=file_id,
        path=f"output_data/{file_id}.pdf",
        href=f"output_data/{file_id}.pdf",
        title=file_id.replace("_", " ").title(),
        summary=raw.get("summary", ""),
        text=raw.get("text", ""),
        embedding=raw.get("embedding", []),
        metadata=MetadataModel(**metadata)
    )
    return doc

# ...existing code...

