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
        return results
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
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
