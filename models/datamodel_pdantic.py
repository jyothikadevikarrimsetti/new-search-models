from pydantic import BaseModel, Field
from typing import List

class MetadataModel(BaseModel):
    Keywords: List[str]
    intent: str
    entities: List[str]

class VectorDocument(BaseModel):
    id: str = Field(..., alias="_id")
    path: str
    href: str
    title: str
    summary: str
    text: str
    embedding: List[float]
    metadata: MetadataModel
