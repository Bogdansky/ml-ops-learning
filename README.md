# Memory Assistant

FastAPI service and CLI tools for turning raw images into searchable text. The pipeline runs OCR, cleans and chunks text, embeds it with `intfloat/multilingual-e5-base`, builds a FAISS index, and serves retrieval/QA endpoints.

## Repository layout
- `ocr_to_text.py` — OCR images in `ocr_data/` into `input.txt` (pytesseract + Pillow).
- `chunker.py` — normalize and split text into overlapping chunks, write `chunks.txt`.
- `embed_chunks.py` — embed chunks, save `embeddings.npy` and `chunks_meta.json`.
- `search.py` / `search_faiss.py` — interactive local search (numpy or FAISS).
- `api.py` — FastAPI app (`/health`, `/search`, `/ask`) using `vector_store.py`, `search_service.py`, `qa.py`, `schemas.py`, `config.py`, `llm.py`.
- `evaluation/eval_retrieval.py` — hit@k evaluation using `evaluation/eval_dataset.json`.
- Shell helpers: `install_dependencies.sh`, `init.sh` (Tesseract via brew), `run_ocr.sh`, `get_chunks.sh`, `get_embenddings.sh`, `run_app.sh`.

## Quick start
1) System deps: Python 3.x and Tesseract. On macOS: `brew install tesseract` (see `init.sh`).
2) Python env: `bash install_dependencies.sh` (creates `.venv`, installs pytesseract, pillow, sentence-transformers, numpy, faiss-cpu). Activate with `source .venv/bin/activate`.
3) Prepare text  
   - Images → text: put `.png/.jpg/.jpeg` pages into `ocr_data/`, run `python ocr_to_text.py` (or `bash run_ocr.sh`). Output: `input.txt`.  
   - Or supply your own `input.txt` directly.
4) Chunk text: `python chunker.py --input input.txt --output chunks.txt --max-chars 1200 --overlap-chars 200` (defaults shown; see `get_chunks.sh`).
5) Embed chunks: `python embed_chunks.py` (or `bash get_embenddings.sh`). Outputs `embeddings.npy`, `chunks_meta.json`.

## Running the API
- Start: `uvicorn api:app --reload` (or `bash run_app.sh`).
- `/health` — basic status + embedding dims.
- `/search` — body `{ "query": "...", "topK": 3 }`, returns ranked chunks.
- `/ask` — same body, returns `answer` plus `used_chunks`. The answer comes from `llm.call_llm`, currently a stub in `llm.py`; plug in your provider there.
- Config knobs in `config.py`: embedding/model paths and `MAX_CONTEXT_CHARS`.

## Local search (CLI)
- Cosine via numpy: `python search.py --top-k 5`.
- FAISS: `python search_faiss.py --top-k 5`.
Type queries interactively; results include rank, score, chunk_id, and a preview.

## Evaluating retrieval
`python evaluation/eval_retrieval.py` builds a FAISS index from current embeddings and reports hit@1 and hit@3 for the examples in `evaluation/eval_dataset.json` (`query`, `expected_chunk_id`, `expected_substring` for reference).

## Notes
- All generated artifacts stay in the repo root by default (`input.txt`, `chunks.txt`, `embeddings.npy`, `chunks_meta.json`).
- For QA quality, ensure `call_llm` normalizes the prompt in `qa.py` and respects the context limit. Update `MODEL_NAME`/paths in `config.py` if you change the encoder.

## LLM usage
Project uses Ollama - open source localy deployed Large Language Model. Before exploring application, **make sure Ollama is launched**.