from fastapi import FastAPI

from qa import generate_answer
from schemas import AskRequest, AskResponse, SearchRequest, SearchResponse
from search_service import search_faiss
from vector_store import bootstrap_vector_store
from services.logger import Logger

#--- App setup
app = FastAPI(title="Memory Assistant API", version="0.1.0")
logger = Logger()

#--- Dependencies
embeddings, meta, model, index = bootstrap_vector_store()


#--- Routes
@app.get("/health")
def health():
    return {"status": "ok", "embeddings": embeddings.shape[0], "dim": embeddings.shape[1]}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    results = search_faiss(
        query=request.query,
        model=model,
        index=index,
        meta=meta,
        top_k=request.top_k,
    )
    return SearchResponse(results=results)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    results = search_faiss(
        query=request.query,
        model=model,
        index=index,
        meta=meta,
        top_k=request.top_k,
    )
    answer = generate_answer(request.query, results)
    logger.log_interaction(request.query, answer, results)
    return AskResponse(
        answer=answer,
        used_chunks=results,
    )
