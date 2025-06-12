# Minimal FastAPI server to expose dense and hybrid search endpoints
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append('./scripts')
from search_pipeline import search_query, hybrid_search

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running. Use /search/dense or /search/hybrid."}

class DenseSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    metadata_filter: Optional[dict] = None

class HybridSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    namespace: Optional[str] = "__default__"
    alpha: Optional[float] = 0.5
    metadata_filter: Optional[dict] = None

@app.post("/search/dense", tags=["search"])
def dense_search(request: DenseSearchRequest):
    try:
        results = search_query(
            query_text=request.query,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/hybrid", tags=["search"])
def hybrid_search_endpoint(request: HybridSearchRequest):
    try:
        results = hybrid_search(
            query=request.query,
            top_k=request.top_k,
            namespace=request.namespace,
            alpha=request.alpha,
            metadata_filter=request.metadata_filter
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
