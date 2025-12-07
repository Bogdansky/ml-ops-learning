from pathlib import Path
import json
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embeddings(emb_path: Path) -> np.ndarray:
    if not emb_path.exists():
        raise FileNotFoundError(f"Файл с эмбеддингами не найден: {emb_path}")
    embeddings = np.load(emb_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Ожидалась матрица 2D, а не форма {embeddings.shape}")
    return embeddings


def load_meta(meta_path: Path) -> list[dict]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Файл с метаданными не найден: {meta_path}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Ожидался список в chunks_meta.json")
    return data


def build_model() -> SentenceTransformer:
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"▶ Загружаю модель эмбеддингов: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    meta: list[dict],
    top_k: int = 3,
):
    # Считаем эмбеддинг запроса
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,  # так же, как в embed_chunks.py
    )[0]

    # Косинусное сходство для нормированных векторов = скалярное произведение
    scores = embeddings @ query_vec  # shape: (num_chunks,)

    # Берём top_k индексов по убыванию
    top_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score = float(scores[idx])
        chunk_info = meta[idx]
        results.append(
            {
                "rank": rank,
                "score": score,
                "chunk_id": chunk_info.get("chunk_id"),
                "text": chunk_info.get("text", ""),
            }
        )
    return results


def pretty_print_results(results: list[dict], max_chars: int = 400):
    if not results:
        print("Ничего не найдено.")
        return

    print("\n========= РЕЗУЛЬТАТЫ ПОИСКА =========")
    for r in results:
        text = r["text"].replace("\n", " ")
        if len(text) > max_chars:
            text_preview = text[:max_chars].rstrip() + "..."
        else:
            text_preview = text

        print(f"\n#{r['rank']}  (score={r['score']:.4f}, chunk_id={r['chunk_id']})")
        print(text_preview)
    print("=====================================\n")


def main():
    parser = argparse.ArgumentParser(description="Поиск по чанкам с помощью эмбеддингов")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="embeddings.npy",
        help="Путь к файлу с эмбеддингами (.npy)",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="chunks_meta.json",
        help="Путь к файлу с метаданными по чанкам (.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Сколько ближайших чанков показывать",
    )

    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    meta_path = Path(args.meta)

    print("▶ Загружаю эмбеддинги...")
    embeddings = load_embeddings(emb_path)
    print(f"Форма матрицы эмбеддингов: {embeddings.shape}")

    print("▶ Загружаю метаданные по чанкам...")
    meta = load_meta(meta_path)
    print(f"Чанков в метаданных: {len(meta)}")

    if embeddings.shape[0] != len(meta):
        raise ValueError(
            f"Количество эмбеддингов ({embeddings.shape[0]}) "
            f"не совпадает с количеством чанков в метаданных ({len(meta)})"
        )

    model = build_model()

    print("\nГотово. Можно искать.")
    print('Введи запрос и нажми Enter. Для выхода введи пустую строку или "q".\n')

    while True:
        try:
            query = input("🔎 Запрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not query or query.lower() in {"q", "quit", "exit"}:
            print("Выход.")
            break

        results = search(
            query=query,
            model=model,
            embeddings=embeddings,
            meta=meta,
            top_k=args.top_k,
        )
        pretty_print_results(results)


if __name__ == "__main__":
    main()
