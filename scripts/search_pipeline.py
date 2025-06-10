from scripts.pinecone_utils import index
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import time
from typing import List
from rank_bm25 import BM25Okapi
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


# Fit BM25 on your corpus (do this once, or load from disk)
def get_bm25_encoder_for_query():
    from pathlib import Path
    OUTPUT = "data/output_data"
    all_texts = []
    for json_file in Path(OUTPUT).glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            all_texts.append(data.get("text", ""))
    # Tokenize for BM25Okapi
    tokenized_corpus = [text.split() for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, all_texts


# Hybrid search function combining dense and sparse retrieval
def hybrid_search(query: str, top_k: int = 3, namespace: str = "__default__"):
    print("[Hybrid Search Pipeline]")
    start_time = time.time()
    query_embedding = model.encode(query, convert_to_tensor=True)
    bm25, all_texts = get_bm25_encoder_for_query()
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    # Get a large pool for fusion (increase recall)
    bm25_top_n = 100  # Even larger pool for best recall
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_top_n]
    from pathlib import Path
    OUTPUT = "data/output_data"
    json_files = list(Path(OUTPUT).glob("*.json"))
    summaries = []
    filenames = []
    bm25_selected_scores = []
    dense_selected_scores = []
    for idx in top_indices:
        with open(json_files[idx], "r", encoding="utf-8") as fh:
            data = json.load(fh)
            # Use both summary and text for dense scoring
            summary = data.get("summary", "")
            text = data.get("text", "")
            combined = summary + "\n" + text
            summaries.append(combined)
            filenames.append(json_files[idx].stem)
            bm25_selected_scores.append(bm25_scores[idx])
    # Rerank with dense embeddings
    if summaries:
        summary_embeddings = model.encode(summaries, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, summary_embeddings)[0].tolist()
        dense_selected_scores = cosine_scores
        # Only use dense scores for final ranking (true semantic hybrid)
        reranked = sorted(
            zip(filenames, summaries, bm25_selected_scores, dense_selected_scores),
            key=lambda x: x[3], reverse=True
        )
        reranked = reranked[:top_k]
        top_summaries = [r[1] for r in reranked]
        context = "\n\n".join(top_summaries)
    else:
        reranked = []
        context = ""
    elapsed = time.time() - start_time
    # RAG-style LLM answer
    if context:
        prompt = f"You are an expert assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer in detail:"
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
    for i, (fname, combined, bm25_score, dense_score) in enumerate(reranked):
        results.append({
            "document_name": fname,
            "summary": combined[:1000],
            "bm25_score": bm25_score,
            "cosine_score": dense_score
        })
    return {"answer": answer, "results": results, "search_time": elapsed}
