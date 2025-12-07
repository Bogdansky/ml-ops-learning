import argparse
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    # Приводим переносы строк к единому виду
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Убираем служебные маркеры страниц вида "===== PAGE xxx ====="
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # если строка выглядит как наш заголовок страницы — пропускаем
        if stripped.startswith("===== PAGE") and stripped.endswith("====="):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Заменяем множественные пустые строки на максимум одну
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Убираем лишние пробелы внутри строк
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def split_into_chunks(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[str]:
    """
    Режем текст на чанки по символам с перекрытием.
    При этом стараемся не разрывать предложения посередине.
    """

    # Грубо делим на предложения
    sentences = re.split(r"(?<=[.!?…])\s+", text)

    chunks: list[str] = []
    current = ""

    for sent in sentences:
        # Если предложение пустое — пропускаем
        if not sent.strip():
            continue

        # Если текущее предложение само по себе длиннее max_chars,
        # просто режем его по кускам.
        if len(sent) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sent), max_chars):
                chunk_part = sent[i : i + max_chars]
                chunks.append(chunk_part.strip())
            continue

        # Если добавление предложения переполнит чанк — сохраняем текущий
        if len(current) + len(sent) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            current = sent
        else:
            if current:
                current += " " + sent
            else:
                current = sent

    # Добавляем последний кусок
    if current:
        chunks.append(current.strip())

    # Делаем перекрытие по символам
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped_chunks: list[str] = []
        prev_chunk = ""

        for chunk in chunks:
            if not prev_chunk:
                overlapped_chunks.append(chunk)
            else:
                overlap = prev_chunk[-overlap_chars:]
                merged = (overlap + " " + chunk).strip()
                overlapped_chunks.append(merged)
            prev_chunk = chunk

        chunks = overlapped_chunks

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Simple text chunker")
    parser.add_argument(
        "--input",
        type=str,
        default="input.txt",
        help="Путь к входному текстовому файлу",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chunks.txt",
        help="Путь к файлу, куда сохранить чанки",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Максимальное число символов в чанке",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=200,
        help="Перекрытие чанков в символах",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Файл {input_path} не найден")

    text = input_path.read_text(encoding="utf-8")
    normalized = normalize_text(text)
    chunks = split_into_chunks(
        normalized,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )

    # Сохраняем чанки в файл, отделяя их разделителем
    with output_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"===== CHUNK {i} =====\n")
            f.write(chunk)
            f.write("\n\n")

    print(f"Готово. Сохранено {len(chunks)} чанков в {output_path}")


if __name__ == "__main__":
    main()
