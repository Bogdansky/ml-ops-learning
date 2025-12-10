#--- Paths and model setup
from pathlib import Path

EMBEDDINGS_PATH = Path("embeddings.npy")
META_PATH = Path("chunks_meta.json")
MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_CONTEXT_CHARS = 2000
