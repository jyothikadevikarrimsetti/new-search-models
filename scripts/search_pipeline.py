from scripts.pinecone_utils import index
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import time
from typing import List
from pinecone_text.sparse import BM25Encoder


# Load environment variables
load_dotenv("config/.env")

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")




def search_query(query_text, top_k=1):
    print("[Dense Search Pipeline]")

    start_time = time.time()
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    pinecone_result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    summaries = [match.metadata.get("summary", "") for match in pinecone_result.matches]
    filenames = [match.metadata.get("filename", "") for match in pinecone_result.matches]
    relevance_scores = [match.score for match in pinecone_result.matches]

    summary_embeddings = model.encode(summaries, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, summary_embeddings)[0]

    top_idx = cosine_scores.argmax().item() if summaries else None
    top_summary = summaries[top_idx] if summaries else ""
    elapsed = time.time() - start_time

    print(f"\n✅ Top Answer: {top_summary}")
    print(f"⏱️  Search Time: {elapsed:.2f} seconds")

    for i, match in enumerate(pinecone_result.matches):
        print(f"\nPDF: {filenames[i]}")
        print(f"Relevance Score (Pinecone): {relevance_scores[i]:.6f}")
        print(f"Reranking Score (Cosine): {float(cosine_scores[i]):.6f}")
        print(f"Summary: {summaries[i][:300]}{'...' if len(summaries[i]) > 300 else ''}")


# Fit BM25 on your corpus (do this once, or load from disk)
def get_bm25_encoder_for_query():
    # Load all texts from your output JSONs
    from pathlib import Path
    import json
    OUTPUT = "data/output_data"
    all_texts = []
    for json_file in Path(OUTPUT).glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            all_texts.append(data.get("text", ""))
    bm25 = BM25Encoder()
    bm25.fit(all_texts)
    return bm25

def hybrid_search(query: str, top_k: int = 1, namespace: str = "__default__"):
    print("[Hybrid Search Pipeline]")

    start_time = time.time()
    query_embedding = model.encode(query, convert_to_tensor=True)

    bm25 = get_bm25_encoder_for_query()
    sparse_query_vector = bm25.encode_queries([query])[0]

    pinecone_result = index.query(
        vector=query_embedding.tolist(),
        sparse_vector=sparse_query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=None,
    )

    summaries = [match.metadata.get("summary", "") for match in pinecone_result.matches]
    filenames = [match.metadata.get("filename", "") for match in pinecone_result.matches]
    relevance_scores = [match.score for match in pinecone_result.matches]

    if summaries:
        summary_embeddings = model.encode(summaries, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, summary_embeddings)[0]
        top_idx = cosine_scores.argmax().item()
        top_summary = summaries[top_idx]
    else:
        top_summary = ""
        cosine_scores = []

    elapsed = time.time() - start_time
    print(f"\n✅ Top Answer: {top_summary}")
    print(f"⏱️  Search Time: {elapsed:.2f} seconds")

    for i, match in enumerate(pinecone_result.matches):
        print(f"\nPDF: {filenames[i]}")
        print(f"Relevance Score (Pinecone): {relevance_scores[i]:.6f}")
        print(f"Reranking Score (Cosine): {float(cosine_scores[i]):.6f}")
        print(f"Summary: {summaries[i][:300]}{'...' if len(summaries[i]) > 300 else ''}")
    return None