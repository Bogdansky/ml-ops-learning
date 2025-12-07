from pathlib import Path
from typing import List, Optional

import json

import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# ---------- Конфиг ----------

EMBEDDINGS_PATH = Path("embeddings.npy")
META_PATH = Path("chunks_meta.json")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------- Модели запрос/ответ ----------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchResultItem(BaseModel):
    rank: int
    score: float
    chunk_id: Optional[int]
    text: str


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


# ---------- Инициализация всего добра ----------

print("▶ Загружаю эмбеддинги...")
if not EMBEDDINGS_PATH.exists():
    raise FileNotFoundError(f"Не найден {EMBEDDINGS_PATH}")
embeddings = np.load(EMBEDDINGS_PATH)

if embeddings.ndim != 2:
    raise ValueError(f"Ожидалась матрица вида (N, D), а не {embeddings.shape}")

# FAISS любит float32
if embeddings.dtype != np.float32:
    embeddings = embeddings.astype(np.float32)

print("Форма эмбеддингов:", embeddings.shape)

print("▶ Загружаю метаданные чанков...")
if not META_PATH.exists():
    raise FileNotFoundError(f"Не найден {META_PATH}")
meta = json.loads(META_PATH.read_text(encoding="utf-8"))

if not isinstance(meta, list):
    raise ValueError("В chunks_meta.json ожидается список объектов")

if embeddings.shape[0] != len(meta):
    raise ValueError(
        f"Количество эмбеддингов ({embeddings.shape[0]}) "
        f"не равно количеству чанков в метаданных ({len(meta)})"
    )

# Нормализуем эмбеддинги, на всякий случай
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
mask = norms.squeeze() > 0
embeddings[mask] = embeddings[mask] / norms[mask]

print("▶ Загружаю модель эмбеддингов:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

print("▶ Строю FAISS IndexFlatIP...")
num_vecs, dim = embeddings.shape
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print("В индекс добавлено векторов:", index.ntotal)


# ---------- Логика поиска ----------

def search_faiss(query: str, top_k: int = 3) -> List[SearchResultItem]:
    # эмбеддинг запроса
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    query_vec = np.expand_dims(query_vec, axis=0)

    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_vec, k)

    idxs = indices[0]
    scores = distances[0]

    results: List[SearchResultItem] = []
    for rank, (idx, score) in enumerate(zip(idxs, scores), start=1):
        idx = int(idx)
        if idx < 0:
            continue
        chunk_info = meta[idx]
        results.append(
            SearchResultItem(
                rank=rank,
                score=float(score),
                chunk_id=chunk_info.get("chunk_id"),
                text=chunk_info.get("text", ""),
            )
        )
    return results


# ---------- FastAPI-приложение ----------

app = FastAPI(title="Memory Assistant API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "embeddings": embeddings.shape[0], "dim": embeddings.shape[1]}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    results = search_faiss(request.query, request.top_k)
    return SearchResponse(results=results)
