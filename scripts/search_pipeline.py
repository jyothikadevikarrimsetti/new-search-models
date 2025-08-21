
from scripts.filter_utils import generate_filter, combine_filters
from dotenv import load_dotenv
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from pinecone_text.sparse import SpladeEncoder
from openai import AzureOpenAI
import logging
import concurrent.futures
from scripts.filter_utils import extract_query_metadata, is_entity_lookup_query
import tiktoken
import json

import os

intent_examples_path = os.path.join(
    os.environ.get('PROJECT_ROOT', ''), 'data', 'intent_categories', 'intent_examples.json'
)
with open(intent_examples_path, 'r', encoding='utf-8') as f:
    intent_examples = json.load(f)
# Load environment variables
load_dotenv("config/.env")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Helper functions for embeddings and similarity
def get_openai_embedding(text, timeout=15):
    """Get embeddings using Azure OpenAI's text-embedding model with context window truncation and timeout."""
    # Truncate text to fit within model context window (e.g., 8000 tokens for text-embedding-3-small)
    max_tokens = 8000
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    def call():
        return client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logging.error(f"OpenAI embedding call timed out for text: {text[:50]}")
            raise TimeoutError("OpenAI embedding call timed out.")

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def get_openai_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding



# --- SPLADEEncoder cache for efficiency ---
_splade_encoder_cache = None

def get_splade_encoder_for_query():
    global _splade_encoder_cache
    if _splade_encoder_cache is not None:
        return _splade_encoder_cache
    splade = SpladeEncoder()
    _splade_encoder_cache = splade
    return splade


# Helper to fetch chunk text from local storage using chunk_id (Pinecone vector id).
def get_chunk_text_by_id(chunk_id, chunk_dir="data/chunks"):
    """Retrieve chunk text from local storage using chunk_id."""
    import os
    chunk_path = os.path.join(chunk_dir, f"{chunk_id}.txt")
    if os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def mongodb_vector_search(query_text: str, top_k: int = 3) -> dict:
    """Search MongoDB Atlas for top-k documents using vector similarity."""
    import os
    from dotenv import load_dotenv
    from pymongo import MongoClient
    import logging

    # Load environment variables
    load_dotenv("config/.env")
    MONGO_URI = os.getenv("MONGO_URI")
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["rag_db"]
    collection = db["dmodel"]


    query_vector = get_openai_embedding(query_text)
    
    query_metadata= extract_query_metadata(query_text)

    # pipeline = [
    #     {
    #         "$vectorSearch": {
    #             "queryVector": query_vector,
    #             "path": "embedding",
    #             "numCandidates": 100,
    #             "index": "vector_index",  # Make sure this matches your Atlas Search index name
    #             "limit": top_k
    #         }
    #     }
    # ]
    
    # Build $or filter only for non-empty values
    or_filters = []
    if query_metadata.get("intent"):
        or_filters.append({"metadata.intent": query_metadata["intent"]})
    if query_metadata.get("keywords"):
        or_filters.append({"metadata.Keywords": {"$in": query_metadata["keywords"]}})
    if query_metadata.get("entities"):
        or_filters.append({"metadata.entities": {"$in": query_metadata["entities"]}})

    # If no filters, remove the filter key entirely
    vector_search = {
        "index": "vector_index1",
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": 200,
        "limit": top_k,
    }
    if or_filters:
        vector_search["filter"] = {"$or": or_filters}

    pipeline = [
        {"$vectorSearch": vector_search}
    ]


    results = list(collection.aggregate(pipeline))
    docs = []
    for doc in results:
        docs.append({
            "_id": doc.get("_id", None),
            "href": doc.get("href", ""),
            "path": doc.get("path", ""),
            "title": doc.get("title",None),
            "summary": doc.get("summary", ""),
            "metadata": doc.get("metadata", {}),
            # "score": doc.get("score", None),
            # "intent": doc.get("intent", None),
            # "entities": doc.get("entities", []),
            # "keywords": doc.get("keywords", [])
        })
        # Prepare context for LLM answer
    if docs:
        context = "\n\n".join([doc["summary"] for doc in docs[:min(3, len(docs))]])
        prompt = f"You are an expert assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer in detail:"
        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            answer = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=256,
                timeout=20
            ).choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI completion error: {e}")
            answer = "LLM completion error."
    else:
        answer = "No relevant document found."
    return {
   
    "answer": answer,
    "query_filter": query_metadata,
    "results": docs,
    "count": len(docs)
}
    