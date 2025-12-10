#--- Semantic search helpers
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from schemas import SearchResultItem


def search_faiss(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    meta: List[dict],
    top_k: int,
) -> List[SearchResultItem]:
    # E5 любит префикс "query: "
    e5_query = f"query: {query}"
    query_vec = model.encode(
        [e5_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    query_vec = np.expand_dims(query_vec, axis=0)

    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_vec, k)

    results: List[SearchResultItem] = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        idx = int(idx)
        if idx < 0:
            continue

        chunk = meta[idx]
        results.append(
            SearchResultItem(
                rank=rank,
                score=float(score),
                chunk_id=chunk.get("chunk_id"),
                text=chunk.get("text", ""),
            )
        )

    return results
