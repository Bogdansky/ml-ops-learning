import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

#--- LLM client placeholder
def call_llm(prompt: str) -> str:
    """
    Вызывает локальную LLM через Ollama.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,  # для простоты берём нестриминговый режим
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        # на первый MVP просто вернём текст ошибки
        return f"Ошибка при обращении к LLM: {e}"

    data = resp.json()
    # в Ollama ответ обычно лежит в поле "response"
    return data.get("response", "").strip() or "LLM не вернула текст ответа."