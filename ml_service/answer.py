"""Слой генерации ответа поверх найденных сообщений.

LLM опциональна: если Ollama не запущена или модель не скачана,
is_available() вернёт False, а интерфейс просто покажет найденные
сообщения как обычно.
"""

import logging
import os
import re

import httpx

logger = logging.getLogger("web-service")

# Обращение к модели через OpenAI-совместимый эндпоинт /v1 - движок
# задаётся переменной LLM_URL и меняется без правки кода.
# OLLAMA_URL оставлен как запасной для обратной совместимости.
LLM_URL = os.getenv("LLM_URL", os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")

CONTEXT_MESSAGES = int(os.getenv("LLM_CONTEXT_MESSAGES", "8"))
MAX_MESSAGE_CHARS = 700

SYSTEM_PROMPT = """Ты — ассистент, который отвечает на вопросы по истории рабочего чата.

Правила:
1. Отвечай ТОЛЬКО на основе приведённых сообщений. Не додумывай факты.
2. После каждого утверждения ставь ссылку на источник в квадратных скобках: [1], [2].
3. Если есть хоть одно сообщение по теме вопроса — отвечай по нему, даже если
   вопрос сформулирован коротко или неточно, а ответ получается неполным.
4. Фразу "В найденных сообщениях ответа нет" используй только тогда, когда
   НИ ОДНО сообщение не относится к теме вопроса.
5. Вопрос может утверждать то, чего в сообщениях нет. Если названного в
   вопросе человека, продукта или события в сообщениях не встречается —
   так и напиши, не подтверждай предпосылку вопроса. Никогда не переноси
   имя из вопроса в ответ, если оно не встречается в самих сообщениях.
6. Сообщения найдены автоматически, среди них есть посторонние из других
   обсуждений — их просто не упоминай.
7. Отвечай кратко, 1-3 предложения. По-русски.
8. Не начинай ответ с приветствий и не копируй текст сообщений дословно —
   формулируй своими словами."""


NAME_RE = re.compile(r"\b[А-ЯЁA-Z][а-яёa-zА-ЯЁA-Z]{2,}\b")
WORD_RE = re.compile(r"\w+", re.UNICODE)
NO_ANSWER = "В найденных сообщениях ответа нет."
STEM_MIN_LEN = 4


def _same_stem(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    n = min(len(a), len(b))
    if n < STEM_MIN_LEN:
        return a == b
    return a[:n] == b[:n]


def _unsupported_names(question: str, answer: str, sources: list[dict]) -> set[str]:
    corpus_words = {
        w
        for src in sources
        for field in ("text", "sender")
        for w in WORD_RE.findall(str(src.get(field, "")).lower())
    }
    from_question = NAME_RE.findall(question)
    unsupported = set()
    for name in NAME_RE.findall(answer):
        if not any(_same_stem(name, q) for q in from_question):
            continue
        if not any(_same_stem(name, w) for w in corpus_words):
            unsupported.add(name)
    return unsupported


class LLMUnavailable(RuntimeError):
    """Движок не отвечает или нужная модель не загружена."""


def _llm_text(message: dict) -> str:
    return message.get("text_llm") or message.get("text") or ""


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{LLM_URL}/v1/models", timeout=3)
            response.raise_for_status()
            models = response.json().get("data") or []
    except Exception:
        return False

    if not models:
        return False

    names = {m.get("id", "") for m in models}
    base = LLM_MODEL.split(":")[0]
    if any(name == LLM_MODEL or name.startswith(base) for name in names):
        return True
    return len(models) == 1


def build_prompt(question: str, messages: list[dict]) -> str:
    lines = []
    for i, message in enumerate(messages[:CONTEXT_MESSAGES], start=1):
        text = _llm_text(message).strip()[:MAX_MESSAGE_CHARS]
        if not text:
            continue
        lines.append(f"[{i}] ({message.get('date', '?')}) {text}")

    context = "\n\n".join(lines)
    return f"Сообщения из чата:\n\n{context}\n\nВопрос: {question}\n\nОтвет:"


async def generate(question: str, messages: list[dict]) -> tuple[str, list[dict]]:
    used = [m for m in messages[:CONTEXT_MESSAGES] if _llm_text(m).strip()]
    if not used:
        raise LLMUnavailable("нет текстов для генерации")

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, messages)},
        ],
        "temperature": 0.2,
        "max_tokens": 320,
        "stop": ["\nВопрос:", "\n\nВопрос:", "\nСообщения из чата:"],
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LLM_URL}/v1/chat/completions", json=payload, timeout=180
            )
            response.raise_for_status()
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"LLM недоступна: {e}") from e

    choices = response.json().get("choices") or []
    answer = (choices[0]["message"]["content"].strip() if choices else "")
    if not answer:
        raise LLMUnavailable("модель вернула пустой ответ")

    sources = [
        {
            "n": i,
            "id": m["id"],
            "sender": m["sender"],
            "date": m["date"],
            "text": _llm_text(m)[:MAX_MESSAGE_CHARS],
        }
        for i, m in enumerate(used, start=1)
    ]

    unsupported = _unsupported_names(question, answer, sources)
    if unsupported:
        logger.info("Ответ отклонён: имена не подтверждены источниками: %s",
                    ", ".join(sorted(unsupported)))
        answer = NO_ANSWER

    return answer, sources
