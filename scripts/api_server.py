# Minimal FastAPI server to expose dense and hybrid search endpoints
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append('./scripts')
from search_pipeline import mongodb_vector_search


app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running. Use /search/dense or /search/hybrid."}

class DenseSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class   HybridSearchRequest(BaseModel):
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



from search_pipeline import mongodb_vector_search

class MongoVectorSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

@app.post("/search/mongo_vector", tags=["search"])
def mongo_vector_search_endpoint(request: MongoVectorSearchRequest):
    """
    Search MongoDB Atlas for top-k documents using vector similarity.
    """
    try:
        results = mongodb_vector_search(request.query, top_k=request.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 