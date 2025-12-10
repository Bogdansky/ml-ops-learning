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

    # Собираем контекст с небольшими заголовками для фрагментов
    context_parts = []
    for c in chunks:
        header_id = c.chunk_id if c.chunk_id is not None else c.rank
        header = f"[Фрагмент {header_id}]"
        text = c.text.strip()
        if not text:
            continue
        context_parts.append(f"{header}\n{text}")

    full_context = "\n\n".join(context_parts).strip()

    # Ограничим размер контекста, чтобы не ломать LLM
    max_context_chars = 3000
    if len(full_context) > max_context_chars:
        full_context = full_context[:max_context_chars].rsplit(" ", 1)[0] + "..."

    prompt = f"""
Ты — аккуратный и честный ассистент, который отвечает на вопросы по семейным записям.

Тебе даны несколько фрагментов текста (они помечены как [Фрагмент N]).
Следуй этим правилам:

1. Отвечай ТОЛЬКО на основе информации из этих фрагментов.
2. Не придумывай новые факты, события, даты или имена.
3. Если в тексте явно нет ответа, честно напиши: "В этих записях это не указано."
4. Можно кратко цитировать фрагменты, если это помогает, но не переписывай текст полностью.
5. Отвечай по-русски, тёплым тоном, в 2–4 предложениях.
6. Если фрагменты противоречат друг другу, опиши это и не выдумывай, "как было на самом деле".

КОНТЕКСТ:
{full_context}

ВОПРОС:
{query}

Ответ:
""".strip()

    answer_text = call_llm(prompt)
    return answer_text


