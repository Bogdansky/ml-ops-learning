from pathlib import Path

import pytesseract
from PIL import Image


def ocr_image(image_path: Path, lang: str = "rus") -> str:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text


def main():
    images_dir = Path("ocr_data")
    output_path = Path("input.txt")

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Папка {images_dir} не найдена. Создай её и положи туда картинки."
        )

    all_text_parts: list[str] = []

    # Берём только самые распространённые форматы
    image_paths = sorted(
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
    )

    if not image_paths:
        raise FileNotFoundError(
            f"В папке {images_dir} не найдено ни одной .png/.jpg/.jpeg картинки."
        )

    for path in image_paths:
        print(f"Обрабатываю {path.name} ...")
        text = ocr_image(path, lang="rus+eng")
        # Добавим небольшой разделитель между страницами
        all_text_parts.append(f"===== PAGE {path.name} =====\n{text.strip()}\n")

    full_text = "\n\n".join(all_text_parts).strip()
    output_path.write_text(full_text, encoding="utf-8")

    print(f"\nГотово. Текст сохранён в {output_path}")


if __name__ == "__main__":
    main()