# Minimal FastAPI server to expose dense and hybrid search endpoints
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append('./scripts')
from search_pipeline import search_query, hybrid_search
from filter_utils import generate_filter
from scripts.extract_text import save_processed_text
from scripts.metadata import extract_metadata
from scripts.vector_store import upsert_to_pinecone, delete_from_pinecone, pinecone_vector_exists
import shutil
import os
from pathlib import Path
from scripts.pinecone_utils import index
from scripts.filter_utils import extract_query_metadata, is_entity_lookup_query
from fastapi import Query

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running. Use /search/dense or /search/hybrid."}

class DenseSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class HybridSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    namespace: Optional[str] = "__default__"
    alpha: Optional[float] = 0.5

class IntentCountRequest(BaseModel):
    query: str
    top_k: Optional[int] = 50

class Result():
    DocumentName: str
    answer: str
    search_time: float
    
    reranking_score: Optional[float] = None
    # results: list[dict]
    

    def __init__(self,DocumentName : str, answer: str, search_time: float, reranking_score: Optional[float] = None):
        self.document_name = DocumentName
        self.answer = answer
        self.search_time = search_time
        self.reranking_score = reranking_score

MIN_RERANK_SCORE = 0.15  # Lowered threshold for more results
@app.post("/search/dense", tags=["search"])
def dense_search(request: DenseSearchRequest):
    try:
        # Always generate filter automatically
        filter_dict = generate_filter(request.query)
        print(f"[API] Dense Search - Generated metadata filter: {filter_dict}")
        results = search_query(
            query_text=request.query,
            top_k=request.top_k,
            metadata_filter=filter_dict
        )
        # If no results, or low reranking score, show 'Document not found'
        no_results = not results.get('results')
        query_metadata = extract_query_metadata(request.query)
        if is_entity_lookup_query(query_metadata):
            min_score = 0.05  # Lower threshold for entity lookup
        else:
            min_score = MIN_RERANK_SCORE
        low_score = (
            results.get('reranking_score') is not None and
            results.get('reranking_score') < min_score
        )
        # --- Always return top result if Pinecone returned any matches, even if low_score or no_results ---
        if no_results or low_score:
            # Try to get top result from Pinecone matches if available
            if results.get('results') and len(results['results']) > 0:
                return {
                    "document_names": results.get('document_names', []),
                    "answer": results.get('answer', 'No answer found'),
                    "search_time": results.get('search_time', 0.0),
                    "reranking_score": results.get('reranking_score', None),
                    "results": results.get('results', [])
                    
                }
            else:
                return {"answer": "Document not found.", "results": []}
        # Pass through all fields from backend, including LLM answer and all document names/results
        return {
            "document_names": results.get('document_names', []),
            "answer": results.get('answer', 'No answer found'),     
            "search_time": results.get('search_time', 0.0),
            "reranking_score": results.get('reranking_score', None),       
            "results": results.get('results', []),
           
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/hybrid", tags=["search"])
def hybrid_search_endpoint(request: HybridSearchRequest):
    try:
        # Always generate filter automatically
        filter_dict = generate_filter(request.query)
        print(f"[API] Hybrid Search - Generated metadata filter: {filter_dict}")
        results = hybrid_search(
            query=request.query,
            top_k=request.top_k,
            namespace=request.namespace,
            alpha=request.alpha,
            metadata_filter=filter_dict
        )
        no_results = not results.get('results')
        query_metadata = extract_query_metadata(request.query)
        if is_entity_lookup_query(query_metadata):
            min_score = 0.05  # Lower threshold for entity lookup
        else:
            min_score = MIN_RERANK_SCORE
        low_score = (
            results.get('reranking_score') is not None and
            results.get('reranking_score') < min_score
        )
        # --- Always return top result if Pinecone returned any matches, even if low_score or no_results ---
        if no_results or low_score:
            # Try to get top result from Pinecone matches if available
            if results.get('results') and len(results['results']) > 0:
                return {
                    "document_names": results.get('document_names', []),
                    "answer": results.get('answer', 'No answer found'),
                    "search_time": results.get('search_time', 0.0),
                    "reranking_score": results.get('reranking_score', None),
                    "results": results.get('results', []),

                }
            else:
                return {"answer": "Document not found.", "results": []}
        # Pass through all fields from backend, including LLM answer and all document names/results
        return {
            "document_names": results.get('document_names', []),
            "answer": results.get('answer', 'No answer found'),
            "search_time": results.get('search_time', 0.0),
            "reranking_score": results.get('reranking_score', None),
            "results": results.get('results', []),
            
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/intent_count", tags=["search"])
def intent_count_search(request: IntentCountRequest):
    try:
        filter_dict = generate_filter(request.query)
        print(f"[API] Intent Count - Generated metadata filter: {filter_dict}")
        from search_pipeline import search_query
        results = search_query(
            query_text=request.query,
            top_k=request.top_k,
            metadata_filter=filter_dict,
            return_count_for_intent=True
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/pdf", tags=["ingest"])
def ingest_pdf(pdf: UploadFile = File(...), chunk_size: int = Form(300), overlap: int = Form(50), chunk_dir: str = Form("data/chunks"), output_dir: str = Form("data/output_data")):
    """Upload/process a PDF, chunk, extract metadata, upsert to Pinecone."""
    try:
        # Save uploaded PDF to disk
        input_dir = "data/input_pdf_data"
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        pdf_path = Path(input_dir) / pdf.filename
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf.file, buffer)
        # Process and chunk PDF
        chunk_files = save_processed_text(input_dir, output_dir, specific_pdf=pdf_path, chunk_size=chunk_size, overlap=overlap, chunk_dir=chunk_dir)
        # Extract metadata and upsert each chunk
        for chunk_file in chunk_files:
            text = chunk_file.read_text(encoding="utf-8")
            metadata = extract_metadata(text)
            # Save metadata JSON for upsert
            json_path = Path(output_dir) / f"{chunk_file.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                import json
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        # Upsert all new metadata JSONs
        upsert_to_pinecone(output_dir)
        return {"message": f"PDF '{pdf.filename}' processed, chunked, and upserted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class VectorDeleteRequest(BaseModel):
    vector_id: str
    namespace: Optional[str] = "__default__"

@app.post("/vector/delete", tags=["vector"])
def delete_vector(request: VectorDeleteRequest):
    try:
        delete_from_pinecone(request.vector_id, namespace=request.namespace)
        return {"message": f"Vector '{request.vector_id}' deleted from namespace '{request.namespace}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector/exists", tags=["vector"])
def vector_exists(vector_id: str, namespace: str = "__default__"):
    try:
        exists = pinecone_vector_exists(vector_id, namespace=namespace)
        return {"vector_id": vector_id, "namespace": namespace, "exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class NamespaceDeleteRequest(BaseModel):
    namespace: str

@app.post("/namespace/delete", tags=["vector"])
def delete_namespace(request: NamespaceDeleteRequest):
    try:
        # Pinecone does not have a direct delete_namespace, so we delete all vectors in the namespace
        if not request.namespace or not isinstance(request.namespace, str):
            raise ValueError("A valid namespace string must be provided.")
        # Remove all vectors in the namespace
        index.delete(delete_all=True, namespace=request.namespace)
        return {"message": f"Namespace '{request.namespace}' deleted (all vectors removed)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/hybrid/match_details", tags=["search"])
def hybrid_search_match_details(request: HybridSearchRequest):
    """Return all FilterUtils details and hybrid search chunk names as structured fields, without dense/sparse scores."""
    try:
        query_metadata = extract_query_metadata(request.query)
        filter_dict = generate_filter(request.query)
        results = hybrid_search(
            query=request.query,
            top_k=request.top_k,
            namespace=request.namespace,
            alpha=request.alpha,
            metadata_filter=filter_dict
        )
        # Get all matching chunk names (if available)
        matching_chunk_names = [r.get('document_name') for r in results.get('results', []) if r.get('document_name')]
        match_details = []
        for r in results.get('results', []):
            explanation = []
            hybrid_score = r.get('hybrid_score', r.get('reranking_score', 0))
            if hybrid_score > 0.5:
                explanation.append("High hybrid score: strong semantic and keyword/entity match.")
            elif r.get('dense_score', 0) > 0.5:
                explanation.append("High dense (semantic) similarity.")
            elif r.get('sparse_score', 0) > 0.5:
                explanation.append("High sparse (keyword/entity) match.")
            else:
                explanation.append("Low score: weak match.")
            match_details.append({
                "chunk_id": r.get("document_name"),
                "document_name": r.get("document_name"),
                "summary": r.get("summary"),
                "hybrid_score": hybrid_score,
                "explanation": " ".join(explanation)
            })
        return {
            "query": request.query,
            "entities": query_metadata.get("entities"),
            "intent": query_metadata.get("intent"),
            "keywords": query_metadata.get("keywords"),
            "generated_filter": filter_dict,
            "matching_chunk_names": matching_chunk_names,
            "matches": match_details,
            "top_k": request.top_k,
            "namespace": request.namespace,
            "alpha": request.alpha
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
