# python chunker.py --input input.txt --output chunks.txt --max-chars 1200 --overlap-chars 200
source .venv/bin/activate
python ocr_to_text.py
python chunker.py