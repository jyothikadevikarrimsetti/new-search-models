from scripts.pinecone_utils import index
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


def safe_pinecone_query(**kwargs):
    try:
        logging.info(f"Pinecone query kwargs: {kwargs}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(index.query, **kwargs)
            return future.result(timeout=20)
    except concurrent.futures.TimeoutError:
        logging.error("Pinecone query timed out.")
        raise TimeoutError("Pinecone query timed out.")
    except Exception as e:
        logging.error(f"Pinecone query failed: {e}")
        raise

def search_query(query_text, top_k=10, metadata_filter=None, return_count_for_intent=False):
    print("[Dense Search Pipeline]")
    logging.info(f"[Dense Search Pipeline] Query: {query_text}")
    
    query_metadata = extract_query_metadata(query_text)
    query_filter = generate_filter(query_text) if not metadata_filter else None
    if metadata_filter and query_filter:
        combined_filter = combine_filters([metadata_filter, query_filter])
    else:
        combined_filter = metadata_filter or query_filter
    print(f"[Dense Search Pipeline] Using filter: {combined_filter}")
    logging.info(f"[Dense Search Pipeline] Using filter: {combined_filter}")
    start_time = time.time()
    try:
        query_embedding = get_openai_embedding(query_text)
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return {"answer": "Embedding error.", "results": []}
    pinecone_query_kwargs = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True
    }
    if combined_filter:
        pinecone_query_kwargs["filter"] = combined_filter
    try:
        pinecone_result = safe_pinecone_query(**pinecone_query_kwargs)
    except Exception as e:
        logging.error(f"Pinecone error: {e}")
        return {"answer": "Pinecone error.", "results": []}
    summaries = [match.metadata.get("summary", "") for match in pinecone_result.matches]
    filenames = [match.id for match in pinecone_result.matches]
    relevance_scores = [match.score for match in pinecone_result.matches]
    rerank_texts = [
        (match.metadata.get("summary", "") + "\n" + match.metadata.get("text", ""))
        for match in pinecone_result.matches
    ]
    if rerank_texts:
        try:
            rerank_embeddings = [get_openai_embedding(text) for text in rerank_texts]
            cosine_scores = [cosine_sim(query_embedding, emb) for emb in rerank_embeddings]
        except Exception as e:
            logging.error(f"Rerank embedding error: {e}")
            cosine_scores = [0.0 for _ in rerank_texts]
    else:
        cosine_scores = []
    reranked = sorted(
        zip(filenames, summaries, relevance_scores, cosine_scores),
        key=lambda x: x[3], reverse=True
    )
    elapsed = time.time() - start_time
    print("[DEBUG] All Pinecone matches:")
    for fname, summary, rel_score, rerank_score in reranked:
        print(f"  - {fname} | rerank_score: {rerank_score}")
    # --- ENTITY LOOKUP LOGIC ---
    if is_entity_lookup_query(query_metadata):
        return entity_lookup_output(reranked, query_metadata, start_time, pinecone_matches=pinecone_result.matches)
    # --- INTENT COUNT LOGIC ---
    if return_count_for_intent:
        count = len(reranked)
        search_time = time.time() - start_time
        return {
            "answer": f"{count} document(s) reference this query.",
            "count": count,
            "search_time": search_time,
            "results": [
                {"document_name": fname, "summary": summary, "reranking_score": rerank_score}
                for fname, summary, rel_score, rerank_score in reranked
            ]
        }
    # RAG-style LLM answer using top reranked chunk(s)
    if reranked:
        context = "\n\n".join([r[1] for r in reranked[:min(3, len(reranked))]])
        prompt = f"You are an expert assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer in detail:"
        try:
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
    # Prepare results for UI
    results = []
    for i, (fname, summary, rel_score, rerank_score, match) in enumerate(
            zip(filenames, summaries, relevance_scores, cosine_scores, pinecone_result.matches)):
        results.append({
            "document_name": fname,
            "summary": summary,
            "reranking_score": rerank_score,
            # "intent": match.metadata.get("intent", "[not present]"),
            # "entities": match.metadata.get("entities", "[not present]"),
            # "keywords": match.metadata.get("keywords", "[not present]")
        })
    top_doc = results[0] if results else {}
    search_time = time.time() - start_time
    return {
        "document_names": top_doc.get("document_name"),
        "answer": answer,
        "search_time": search_time,
        "reranking_score": top_doc.get("reranking_score"),
        "results": results
    }


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


# Hybrid search function combining dense and sparse retrieval
def hybrid_search(query: str, top_k: int = 3, namespace: str = "__default__", alpha: float = 0.5, metadata_filter: dict = None):
    print("[Local Hybrid Search Pipeline - SPLADE rerank]")
    start_time = time.time()
    query_lower = query.lower()
    query_embedding = get_openai_embedding(query_lower)    # Generate and combine filters
    query_filter = generate_filter(query) if not metadata_filter else None
    combined_filter = None
    if metadata_filter and query_filter:
        combined_filter = combine_filters([metadata_filter, query_filter])
    else:
        combined_filter = metadata_filter or query_filter

    pinecone_query_kwargs = {
        "vector": query_embedding,
        "top_k": top_k*3,  # get more for reranking
        "namespace": namespace,
        "include_metadata": True
    }
    if combined_filter:
        pinecone_query_kwargs["filter"] = combined_filter
    pinecone_result = index.query(**pinecone_query_kwargs)

    # Print all matching chunk names (IDs)
    chunk_names = [match.id for match in pinecone_result.matches]
    print(f"[HybridSearch] All matching chunk names: {chunk_names}")

    # 2. Locally rerank with SPLADE sparse vectors
    splade = get_splade_encoder_for_query()
    sparse_query_vector = splade.encode_queries([query_lower])[0]

    filenames = chunk_names
    # Retrieve chunk text locally using chunk_id (filename)
    doc_texts = [get_chunk_text_by_id(chunk_id) for chunk_id in filenames]
    if not doc_texts or all(not t.strip() for t in doc_texts):
        return {
            "document_name": None,
            "answer": "No relevant document found.",
            "time_complexity": f"{time.time() - start_time:.2f} seconds",
            "reranking_score": None,
            "summary": None,
            "results": []
        }
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
        (match.metadata.get("summary", "") + "\n" + doc_texts[i])
        for i, match in enumerate(pinecone_result.matches)
    ]
    if rerank_texts:
        rerank_embeddings = [get_openai_embedding(text) for text in rerank_texts]
        import numpy as np
        def cosine_sim(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cosine_scores = [cosine_sim(query_embedding, emb) for emb in rerank_embeddings]
    else:
        cosine_scores = []

    # Hybrid score
    hybrid_scores = [alpha * dense + (1-alpha) * sparse for dense, sparse in zip(cosine_scores, sparse_scores)]

    # --- SOFT INTENT BOOST LOGIC ---
    from scripts.filter_utils import extract_query_metadata
    query_metadata = extract_query_metadata(query)
    query_intent = query_metadata.get("intent")
    intent_boost = 0.2  # Tune as needed
    reranked_with_boost = []
    for i, (fname, summary, rel_score, dense_score, sparse_score_val, hybrid_score) in enumerate(
            zip(filenames, summaries, relevance_scores, cosine_scores, sparse_scores, hybrid_scores)):
        chunk_intent = pinecone_result.matches[i].metadata.get("intent")
        score = hybrid_score
        if chunk_intent == query_intent and query_intent is not None:
            score += intent_boost
        reranked_with_boost.append((fname, summary, rel_score, dense_score, sparse_score_val, score))
    # Now sort by the new score
    reranked = sorted(reranked_with_boost, key=lambda x: x[5], reverse=True)[:top_k]

    elapsed = time.time() - start_time
    if reranked:
        context = "\n\n".join([r[1] for r in reranked[:min(3, len(reranked))]])
        system_msg = "You are a direct and concise assistant. Do not use phrases like 'Based on the context' or 'According to the information'. Just answer the question directly."
        user_msg = f"Information:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        answer = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
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
            # "intent": match.metadata.get("intent", "[not present]"),
            # "entities": match.metadata.get("entities", "[not present]"),
            # "keywords": match.metadata.get("keywords", "[not present]")
        })
    # --- ENTITY LOOKUP LOGIC ---
    query_metadata = extract_query_metadata(query)
    if is_entity_lookup_query(query_metadata):
        # Use document_name from metadata, not chunk name
        doc_names = []
        for match in pinecone_result.matches:
            doc_name = match.metadata.get("document_name")
            if doc_name and doc_name not in doc_names:
                doc_names.append(doc_name)
        count = len(doc_names)
        entity = (query_metadata["entities"] or query_metadata["keywords"])[0]
        doc_list_str = "\n".join(f"- {name}" for name in doc_names)
        # Always use Azure OpenAI to generate the answer
        if count > 0:
            prompt = (
                f"You are an expert assistant. The user searched for the entity or keyword '{entity}'. "
                f"Here is a list of all document names that reference this entity or keyword:\n{doc_list_str}\n\n"
                f"For each document, explain in 1-2 sentences why it is relevant to the entity or keyword '{entity}', using the document's summary if available. "
                f"List all document names and provide a brief explanation for each."
            )
            try:
                ai_answer = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=256,
                    timeout=20
                ).choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"OpenAI completion error (entity lookup): {e}")
                ai_answer = f"{count} document(s) reference '{entity}':\n" + doc_list_str
        else:
            ai_answer = f"No documents reference '{entity}'."
        search_time = time.time() - start_time
        # Build results with summary and reranking_score for each unique document_name
        hybrid_results = []
        for doc_name in doc_names:
            for match in pinecone_result.matches:
                if match.metadata.get("document_name") == doc_name:
                    hybrid_results.append({
                        "document_name": doc_name,
                        "summary": match.metadata.get("summary", ""),
                        "reranking_score": match.score
                    })
                    break
        return {
            "answer": ai_answer,
            "count": count,
            "document_names": doc_names,
            "search_time": search_time,
            "reranking_score": hybrid_results[0]["reranking_score"] if hybrid_results else None,
            "results": hybrid_results
        }
    # Return only the top result's doc name, answer, time, and reranking score
    top_doc = results[0] if results else {}
    return {
        "document_names": top_doc.get("document_name"),
        "answer": answer,
        "search_time": elapsed,
        "reranking_score": top_doc.get("reranking_score"),
        "results": results
    }


def entity_lookup_output(reranked, query_metadata: Dict[str, Any], start_time: float, pinecone_matches=None) -> Dict[str, Any]:
    """Handle entity lookup output for the search pipeline."""
    count = len(reranked)
    entity = (query_metadata["entities"] or query_metadata["keywords"])[0]
    # Use document_name from metadata, not chunk name
    doc_names = []
    if pinecone_matches is not None:
        for match in pinecone_matches:
            doc_name = match.metadata.get("document_name")
            if doc_name and doc_name not in doc_names:
                doc_names.append(doc_name)
    else:
        # fallback: try to get from reranked if pinecone_matches not passed
        for row in reranked:
            if isinstance(row, dict) and "document_name" in row:
                doc_name = row["document_name"]
                if doc_name and doc_name not in doc_names:
                    doc_names.append(doc_name)
            elif isinstance(row, (list, tuple)) and len(row) > 0:
                doc_name = row[0]
                if doc_name and doc_name not in doc_names:
                    doc_names.append(doc_name)
    doc_list_str = "\n".join(f"- {name}" for name in doc_names)
    # Always use Azure OpenAI to generate the answer
    if count > 0:
        # Build a detailed answer that always includes document names and their summaries
        doc_summaries = []
        for doc_name in doc_names:
            summary = None
            for match in pinecone_matches:
                if match.metadata.get("document_name") == doc_name:
                    summary = match.metadata.get("summary", "")
                    break
            if summary:
                doc_summaries.append(f"- {doc_name}: {summary}")
            else:
                doc_summaries.append(f"- {doc_name}")
        doc_list_str = "\n".join(doc_summaries)
        prompt = (
            f"The following documents reference the entity or keyword '{entity}':\n{doc_list_str}\n\n"
            f"List all document names and provide a brief explanation for each, using the summary if available."
        )
        try:
            ai_answer = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=256,
                timeout=20
            ).choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI completion error (entity lookup): {e}")
            ai_answer = f"{count} document(s) reference '{entity}':\n" + doc_list_str
    else:
        ai_answer = f"No documents reference '{entity}'."
    search_time = time.time() - start_time
    # Build results with summary and reranking_score for each unique document_name
    hybrid_results = []
    pinecone_matches = pinecone_matches if pinecone_matches is not None else []
    for doc_name in doc_names:
        for match in pinecone_matches:
            if match.metadata.get("document_name") == doc_name:
                hybrid_results.append({
                    "document_name": doc_name,
                    "summary": match.metadata.get("summary", ""),
                    "reranking_score": match.score
                })
                break
    return {
        "answer": ai_answer,
        "count": count,
        "document_names": doc_names,
        "search_time": search_time,
        "reranking_score": hybrid_results[0]["reranking_score"] if hybrid_results else None,
        "results": hybrid_results
    }
