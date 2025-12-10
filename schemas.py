#--- Shared request/response schemas
from typing import List, Optional

try:
    from pydantic import BaseModel, Field, ConfigDict
    BaseConfigDict = ConfigDict
except ImportError:  # Pydantic v1 fallback
    from pydantic import BaseModel, Field
    BaseConfigDict = None


class CamelModel(BaseModel):
    if BaseConfigDict:
        model_config = BaseConfigDict(populate_by_name=True)
    else:
        class Config:
            allow_population_by_field_name = True
            allow_population_by_alias = True


class SearchRequest(CamelModel):
    query: str
    top_k: int = Field(3, alias="topK")


class SearchResultItem(BaseModel):
    rank: int
    score: float
    chunk_id: Optional[int]
    text: str


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


class AskRequest(CamelModel):
    query: str
    top_k: int = Field(3, alias="topK")


class AskResponse(BaseModel):
    answer: str
    used_chunks: List[SearchResultItem]
