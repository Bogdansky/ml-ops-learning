from pathlib import Path
import json
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path("embeddings.npy")
META_PATH = Path("chunks_meta.json")
EVAL_DATASET_PATH = Path("./evaluation/eval_dataset.json")
MODEL_NAME = "intfloat/multilingual-e5-base"  # или твоя текущая модель


def load_embeddings_and_meta():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Не найден {EMBEDDINGS_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Не найден {META_PATH}")

    embeddings = np.load(EMBEDDINGS_PATH)
    if embeddings.ndim != 2:
        raise ValueError(f"Ожидалась матрица (N, D), а не {embeddings.shape}")

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if not isinstance(meta, list):
        raise ValueError("В chunks_meta.json ожидается список объектов")

    if embeddings.shape[0] != len(meta):
        raise ValueError(
            f"Количество эмбеддингов ({embeddings.shape[0]}) "
            f"не равно количеству чанков в метаданных ({len(meta)})"
        )

    # Нормализуем эмбеддинги
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    mask = norms.squeeze() > 0
    embeddings[mask] = embeddings[mask] / norms[mask]

    return embeddings, meta


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    num_vecs, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def load_eval_dataset() -> List[dict]:
    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Не найден {EVAL_DATASET_PATH}")
    data = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("В eval_dataset.json ожидается список объектов")
    # ожидаем поля: query, expected_chunk_id
    for ex in data:
        if "query" not in ex or "expected_chunk_id" not in ex:
            raise ValueError(
                "Каждый пример в eval_dataset.json должен содержать "
                '"query" и "expected_chunk_id"'
            )
    return data


def search_faiss(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    meta: List[dict],
    top_k: int,
) -> List[dict]:
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
    idxs = indices[0]
    scores = distances[0]

    results: List[dict] = []
    for rank, (idx, score) in enumerate(zip(idxs, scores), start=1):
        idx = int(idx)
        if idx < 0:
            continue
        chunk_info = meta[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk_info.get("chunk_id"),
                "text": chunk_info.get("text", ""),
            }
        )
    return results


def evaluate_hit_at_k(
    eval_data: List[dict],
    model: SentenceTransformer,
    index: faiss.Index,
    meta: List[dict],
    k_values=(1, 3),
    max_k_for_search: int = 10,
):
    stats = {k: 0 for k in k_values}
    total = len(eval_data)

    print(f"▶ Начинаю оценку на {total} примерах...\n")

    for i, example in enumerate(eval_data, start=1):
        query = example["query"]
        expected_id = example["expected_chunk_id"]

        results = search_faiss(
            query=query,
            model=model,
            index=index,
            meta=meta,
            top_k=max_k_for_search,
        )

        # ищем ранг ожидаемого чанка
        found_rank = -1
        for r in results:
            if r["chunk_id"] == expected_id:
                found_rank = r["rank"]
                break

        # обновляем hit@k
        for k in k_values:
            if found_rank != -1 and found_rank <= k:
                stats[k] += 1

        print(f'Пример {i}: "{query}"')
        print(f"  expected_chunk_id = {expected_id}, found_rank = {found_rank}")
        print("  Top результаты:")
        for r in results[:3]:
            preview = r["text"].replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120].rstrip() + "..."
            print(f"    - (rank={r['rank']}, score={r['score']:.4f}, chunk_id={r['chunk_id']}) {preview}")
        print()

    print("▶ Результаты:")
    for k in k_values:
        hit = stats[k]
        hit_at_k = hit / total if total > 0 else 0.0
        print(f"  hit@{k}: {hit} / {total} = {hit_at_k:.3f}")


def main():
    print("▶ Загружаю эмбеддинги и метаданные...")
    embeddings, meta = load_embeddings_and_meta()

    print("▶ Строю FAISS-индекс...")
    index = build_faiss_index(embeddings)

    print("▶ Загружаю eval_dataset.json...")
    eval_data = load_eval_dataset()

    print("▶ Загружаю модель:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    evaluate_hit_at_k(
        eval_data,
        model,
        index,
        meta,
        k_values=(1, 3),
        max_k_for_search=10,
    )


if __name__ == "__main__":
    main()
