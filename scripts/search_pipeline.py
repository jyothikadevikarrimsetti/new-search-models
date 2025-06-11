from scripts.pinecone_utils import index
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import time
from typing import List
from pinecone_text.sparse import SpladeEncoder
from openai import AzureOpenAI
import json


# Load environment variables
load_dotenv("config/.env")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")



def search_query(query_text, top_k=3):
    print("[Dense Search Pipeline]")
    start_time = time.time()
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    pinecone_result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    summaries = [match.metadata.get("summary", "") for match in pinecone_result.matches]
    filenames = [match.id for match in pinecone_result.matches]
    relevance_scores = [match.score for match in pinecone_result.matches]
    # Use both summary and text for reranking
    rerank_texts = [
        (match.metadata.get("summary", "") + "\n" + match.metadata.get("text", ""))
        for match in pinecone_result.matches
    ]
    summary_embeddings = model.encode(rerank_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, summary_embeddings)[0] if rerank_texts else []
    # Sort by rerank score
    reranked = sorted(
        zip(filenames, summaries, relevance_scores, cosine_scores.tolist()),
        key=lambda x: x[3], reverse=True
    )
    elapsed = time.time() - start_time
    # RAG-style LLM answer using top reranked chunk(s)
    if reranked:
        context = "\n\n".join([r[1] for r in reranked[:min(3, len(reranked))]])
        prompt = f"You are an expert assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer in detail:"
        answer = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256
        ).choices[0].message.content.strip()
    else:
        answer = "No relevant document found."
    # Prepare results for UI
    results = []
    for i, (fname, summary, rel_score, rerank_score) in enumerate(reranked):
        results.append({
            "document_name": fname,
            "summary": summary,
            "pinecone_score": rel_score,
            "rerank_score": rerank_score
        })
    return {"answer": answer, "results": results, "search_time": elapsed}


# --- SPLADEEncoder cache for efficiency ---
_splade_encoder_cache = None

def get_splade_encoder_for_query():
    global _splade_encoder_cache
    if _splade_encoder_cache is not None:
        return _splade_encoder_cache
    splade = SpladeEncoder()
    _splade_encoder_cache = splade
    return splade


# Hybrid search function combining dense and sparse retrieval
def hybrid_search(query: str, top_k: int = 3, namespace: str = "__default__", alpha: float = 0.5, filter: dict = None):
    print("[Local Hybrid Search Pipeline - SPLADE rerank]")
    start_time = time.time()
    query_lower = query.lower()
    query_embedding = model.encode(query_lower, convert_to_tensor=True)

    # 1. Dense search with Pinecone
    pinecone_result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k*3,  # get more for reranking
        namespace=namespace,
        include_metadata=True,
        filter=filter
    )

    # 2. Locally rerank with SPLADE sparse vectors
    splade = get_splade_encoder_for_query()
    sparse_query_vector = splade.encode_queries([query_lower])[0]

    doc_texts = [match.metadata.get("text", "") for match in pinecone_result.matches]
    doc_sparse_vecs = splade.encode_documents([t.lower() for t in doc_texts])

    # Compute sparse (SPLADE) score: dot product between query and doc sparse vectors
    def sparse_score(q, d):
        q_indices, q_values = q.get("indices", []), q.get("values", [])
        d_indices, d_values = d.get("indices", []), d.get("values", [])
        q_map = dict(zip(q_indices, q_values))
        d_map = dict(zip(d_indices, d_values))
        # Intersection only
        return sum(q_map[i] * d_map[i] for i in set(q_map) & set(d_map))

    sparse_scores = [sparse_score(sparse_query_vector, dvec) for dvec in doc_sparse_vecs]

    # 3. Prepare results and rerank by hybrid score (alpha * dense + (1-alpha) * sparse)
    summaries = [match.metadata.get("summary", "") for match in pinecone_result.matches]
    filenames = [match.id for match in pinecone_result.matches]
    relevance_scores = [match.score for match in pinecone_result.matches]
    rerank_texts = [
        (match.metadata.get("summary", "") + "\n" + match.metadata.get("text", ""))
        for match in pinecone_result.matches
    ]
    if rerank_texts:
        summary_embeddings = model.encode(rerank_texts, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, summary_embeddings)[0].tolist()
    else:
        cosine_scores = []

    # Hybrid score
    hybrid_scores = [alpha * dense + (1-alpha) * sparse for dense, sparse in zip(cosine_scores, sparse_scores)]
    # Sort by hybrid score
    reranked = sorted(
        zip(filenames, summaries, relevance_scores, cosine_scores, sparse_scores, hybrid_scores),
        key=lambda x: x[5], reverse=True
    )[:top_k]

    elapsed = time.time() - start_time
    if reranked:
        context = "\n\n".join([r[1] for r in reranked[:min(3, len(reranked))]])
        prompt = f"You are an expert assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer in detail:"
        answer = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256
        ).choices[0].message.content.strip()
    else:
        answer = "No relevant document found."

    # Prepare top-k results for UI
    results = []
    for i, (fname, summary, rel_score, dense_score, sparse_score_val, hybrid_score) in enumerate(reranked):
        results.append({
            "document_name": fname,
            "summary": summary,
            "reranking_score": hybrid_score,
            "dense_score": dense_score,
            "sparse_score": sparse_score_val,
        })
    # Return top result's answer and all top-k results for UI
    return {
        "document_name": reranked[0][0] if reranked else None,
        "answer": answer,
        "time_complexity": f"{elapsed:.2f} seconds",
        "reranking_score": reranked[0][5] if reranked else None,
        "summary": reranked[0][1] if reranked else None,
        "results": results
    }
