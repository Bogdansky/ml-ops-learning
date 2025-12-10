#--- Loading embeddings and model
import json
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDINGS_PATH, META_PATH, MODEL_NAME


def _load_embeddings(path=EMBEDDINGS_PATH) -> np.ndarray:
    print("▶ Загружаю эмбеддинги...")
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}")

    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"Ожидалась матрица вида (N, D), а не {embeddings.shape}")

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings


def _load_meta(path=META_PATH) -> List[dict]:
    print("▶ Загружаю метаданные чанков...")
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}")

    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, list):
        raise ValueError("В chunks_meta.json ожидается список объектов")
    return meta


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    mask = norms.squeeze() > 0
    embeddings[mask] = embeddings[mask] / norms[mask]
    return embeddings


def _build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    num_vecs, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("В индекс добавлено векторов:", index.ntotal)
    return index


def bootstrap_vector_store() -> Tuple[np.ndarray, List[dict], SentenceTransformer, faiss.Index]:
    embeddings = _load_embeddings()
    meta = _load_meta()

    if embeddings.shape[0] != len(meta):
        raise ValueError(
            f"Количество эмбеддингов ({embeddings.shape[0]}) "
            f"не равно количеству чанков в метаданных ({len(meta)})"
        )

    print("Форма эмбеддингов:", embeddings.shape)
    embeddings = _normalize_embeddings(embeddings)

    print("▶ Загружаю модель эмбеддингов:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("▶ Строю FAISS IndexFlatIP...")
    index = _build_index(embeddings)

    return embeddings, meta, model, index
