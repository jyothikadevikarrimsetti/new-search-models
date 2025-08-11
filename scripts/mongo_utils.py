import os
import json
from typing import List
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from models.datamodel_pdantic import VectorDocument, MetadataModel

# -------------------------------
# MongoDB Setup
# -------------------------------
load_dotenv("config/.env")
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client[os.getenv("MONGO_DB_NAME", "rag_db")]
collection = db[os.getenv("MONGO_COLLECTION_NAME", "dmodel")]


def upsert_vector_document(doc: VectorDocument):
    """Upsert a VectorDocument into MongoDB Atlas."""
    try:
        result = collection.update_one(
            {"_id": doc.id},
            {"$set": doc.model_dump(by_alias=True)},
            upsert=True
        )
        print(f"✅ Upserted: {doc.id}")
        return result
    except Exception as e:
        print(f"❌ Upsert failed for {doc.id}: {e}")
        return None
    