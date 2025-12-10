#--- Answer generation helpers
from typing import List

from config import MAX_CONTEXT_CHARS
from llm import call_llm
from schemas import SearchResultItem


def _build_prompt(query: str, chunks: List[SearchResultItem]) -> str:
    context_parts = [c.text.strip() for c in chunks if c.text.strip()]
    full_context = "\n\n".join(context_parts)

    if len(full_context) > MAX_CONTEXT_CHARS:
        full_context = full_context[:MAX_CONTEXT_CHARS].rsplit(" ", 1)[0] + "..."

    prompt = f"""
Ты — помощник, который отвечает на вопросы на основе заданного текста.
Отвечай кратко и по делу, только используя информацию из контекста.
Если ответа нет в тексте, честно скажи, что в записях этого нет.

Контекст:
{full_context}

Вопрос:
{query}

Ответ:
""".strip()
    return prompt


def generate_answer(query: str, chunks: List[SearchResultItem]) -> str:
    if not chunks:
        return "Я не нашёл подходящую информацию в архиве."

    context_parts = []
    for c in chunks:
        header_id = c.chunk_id if c.chunk_id is not None else c.rank
        header = f"[Фрагмент {header_id}]"
        text = c.text.strip()
        if not text:
            continue
        context_parts.append(f"{header}\n{text}")

    full_context = "\n\n".join(context_parts).strip()

    max_context_chars = 3000
    if len(full_context) > max_context_chars:
        full_context = full_context[:max_context_chars].rsplit(" ", 1)[0] + "..."

    prompt = f"""
Ты — внимательный и честный помощник, который отвечает на вопросы по семейным записям.

Тебе даны несколько фрагментов текста (помечены как [Фрагмент N]).

Правила:
1. Опирайся на информацию из фрагментов и делай НЕТРУДНЫЕ логические выводы.
   Например, если написано, что "Саша проявлял большие способности к изобретениям",
   и тебя спрашивают "Кто проявлял большие способности к изобретениям?", —
   ответ "Саша" считается корректным.
2. Не придумывай факты, которых явно нет или которые противоречат тексту.
3. Если в тексте совсем нет информации, которая помогает ответить на вопрос,
   честно напиши: "В этих записях это не указано."
4. Отвечай по-русски, в 2–4 предложениях, понятно и по-человечески.
5. Если фрагменты расходятся между собой, опиши это и не выдумывай однозначный ответ.

КОНТЕКСТ:
{full_context}

ВОПРОС:
{query}

Ответ:
""".strip()

    answer_text = call_llm(prompt)
    return answer_text



