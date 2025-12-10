from datetime import datetime
import json
from pathlib import Path
from typing import List
from schemas import SearchResultItem

class Logger:
    __logPath: str

    def __init__(self, log_path: str = "logs/ask_log.jsonl"):
        self.__logPath = Path(log_path)
        self.__logPath.parent.mkdir(exist_ok=True)

    def log_interaction(self, query: str, answer: str, chunks: List[SearchResultItem]) -> None:
        """
        Логируем запрос, ответ и использованные чанки в JSONL-файл.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "answer": answer,
            "chunks": [
                {
                    "rank": c.rank,
                    "score": c.score,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                }
                for c in chunks
            ],
        }

        with self.__logPath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
