from pathlib import Path
import re
import json

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNK_HEADER_PATTERN = re.compile(r"^===== CHUNK (\d+) =====\s*$")


def load_chunks(chunks_path: Path) -> list[dict]:
    """
    Парсим chunks.txt вида:

    ===== CHUNK 1 =====
    текст...

    ===== CHUNK 2 =====
    текст...    
    """
    text = chunks_path.read_text(encoding="utf-8")

    chunks: list[dict] = []
    current_id: int | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        header_match = CHUNK_HEADER_PATTERN.match(line)
        if header_match:
            # Сохраняем предыдущий чанк
            if current_id is not None:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append(
                        {"id": current_id, "text": chunk_text}
                    )
            # Начинаем новый
            current_id = int(header_match.group(1))
            current_lines = []
        else:
            # Накопление текста чанка
            if current_id is not None:
                current_lines.append(line)

    # Последний чанк
    if current_id is not None:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append({"id": current_id, "text": chunk_text})

    return chunks


def main():
    chunks_path = Path("chunks.txt")
    if not chunks_path.exists():
        raise FileNotFoundError("Файл chunks.txt не найден. Сначала запусти chunker.py")

    print("▶ Загружаю чанки...")
    chunks = load_chunks(chunks_path)
    print(f"Найдено чанков: {len(chunks)}")

    if not chunks:
        raise ValueError("Файл chunks.txt не содержит чанков")

    # Берём модель для многоязычных предложений (включая русский)
    model_name = "intfloat/multilingual-e5-base"
    print(f"▶ Загружаю модель эмбеддингов: {model_name} ...")
    model = SentenceTransformer(model_name)

    # E5 ожидает префикс "passage: " для документов
    texts = [f"passage: {c['text']}" for c in chunks]

    print("▶ Считаю эмбеддинги...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # удобно для cosine / IP
    )

    print("▶ Сохраняю эмбеддинги и метаданные...")

    # Сохраняем эмбеддинги в .npy
    embeddings_path = Path("embeddings.npy")
    np.save(embeddings_path, embeddings)

    # Сохраняем метаданные по чанкам
    meta = [
        {
            "chunk_id": c["id"],
            "text": c["text"],
        }
        for c in chunks
    ]
    meta_path = Path("chunks_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Готово.")
    print(f"   Эмбеддинги: {embeddings_path}")
    print(f"   Метаданные: {meta_path}")
    print(f"   Форма матрицы эмбеддингов: {embeddings.shape}")


if __name__ == "__main__":
    main()
