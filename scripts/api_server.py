# Minimal FastAPI server to expose dense and hybrid search endpoints
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append('./scripts')
from search_pipeline import search_query, hybrid_search
from filter_utils import generate_filter

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
        low_score = (
            results.get('reranking_score') is not None and
            results.get('reranking_score') < MIN_RERANK_SCORE
        )
        if no_results or low_score:
            return {"answer": "Document not found.", "results": []}
        result = Result(
            DocumentName=results.get('document_name', 'Unknown'),
            answer=results.get('answer', 'No answer found'),
            search_time=results.get('time_taken', 0.0),
            reranking_score=results.get('reranking_score', None)
        )
        return result
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
        low_score = (
            results.get('reranking_score') is not None and
            results.get('reranking_score') < MIN_RERANK_SCORE
        )
        if no_results or low_score:
            return {"answer": "Document not found.", "results": []}
        result = Result(
            DocumentName=results.get('document_name', 'Unknown'),
            answer=results.get('answer', 'No answer found'),
            search_time=results.get('time_taken', 0.0),
            reranking_score=results.get('reranking_score', None)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
