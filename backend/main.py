"""API поверх поискового пайплайна - общий для любых клиентов.

Сервис search возвращает только message_ids. Здесь к ним подставляются
тексты сообщений, добавляется генерация ответа и управление чатами.


"""

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from corpus import DATA_DIR, PROTECTED, Corpus, available_chats
from models import (
    AnswerRequest,
    AnswerResponse,
    InfoResponse,
    SearchRequest,
    SearchResponse,
    ServiceState,
    SwitchChatRequest,
    SwitchChatResponse,
    UploadResponse,
)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8002")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8004")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8003")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("web-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.corpus = Corpus()
    corpus = app.state.corpus
    logger.info(
        "Загружен чат %s: %d сообщений, %d размеченных вопросов",
        corpus.chat["name"], len(corpus.messages), len(corpus.questions),
    )
    await _sync_index(app.state.http, corpus)
    try:
        yield
    finally:
        await app.state.http.aclose()


async def _sync_index(client: httpx.AsyncClient, corpus: Corpus) -> None:
    for _ in range(30):
        try:
            response = await client.get(f"{ORCHESTRATOR_URL}/indexed", timeout=5)
            response.raise_for_status()
            indexed = response.json()["chat_name"]
            break
        except httpx.HTTPError:
            await asyncio.sleep(2)
    else:
        logger.warning("orchestrator недоступен, индекс не синхронизирован")
        return

    if indexed == corpus.chat["name"]:
        logger.info("Индекс соответствует чату %s", indexed)
        return

    logger.info("В индексе %s, нужен %s — переиндексирую",
                indexed or "пусто", corpus.chat["name"])
    try:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/index", json={"file": corpus.file_name}, timeout=1800
        )
        response.raise_for_status()
        logger.info("Проиндексировано чанков: %d", response.json()["chunks"])
    except httpx.HTTPError as e:
        logger.warning("Не удалось переиндексировать: %s", e)


app = FastAPI(
    title="Chat Search API",
    description="Поиск по истории чатов с подстановкой текстов и генерацией ответа",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    corpus: Corpus = app.state.corpus
    return InfoResponse(
        chat_name=corpus.chat["name"],
        message_count=len(corpus.messages),
        questions=corpus.question_texts(),
        chat_file=corpus.file_name,
        available_chats=available_chats(),
    )


@app.get("/api/status", response_model=dict[str, ServiceState])
async def status() -> dict[str, ServiceState]:
    client: httpx.AsyncClient = app.state.http
    checks = {
        "search": f"{SEARCH_URL}/health",
        "orchestrator": f"{ORCHESTRATOR_URL}/health",
        "ml-service": f"{ML_SERVICE_URL}/health",
        "qdrant": f"{QDRANT_URL}/collections/{QDRANT_COLLECTION_NAME}",
    }

    result: dict[str, ServiceState] = {}
    for name, url in checks.items():
        try:
            response = await client.get(url, timeout=3)
            result[name] = ServiceState(ok=response.status_code == 200)
            if name == "qdrant" and response.status_code == 200:
                result[name].points = response.json()["result"]["points_count"]
        except Exception as e:
            result[name] = ServiceState(ok=False, error=str(e))
    return result


@app.post("/api/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    client: httpx.AsyncClient = app.state.http
    corpus: Corpus = app.state.corpus
    question = payload.text.strip()

    started = time.time()
    try:
        response = await client.post(
            f"{SEARCH_URL}/search", json={"question": {"text": question}}, timeout=120
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"search недоступен: {e}") from e
    elapsed_ms = int((time.time() - started) * 1000)

    results = response.json().get("results") or []
    message_ids = results[0]["message_ids"] if results else []

    gold_ids = corpus.gold_ids_for(question)
    return SearchResponse(
        messages=corpus.resolve(message_ids, gold_ids),
        gold_ids=gold_ids,
        elapsed_ms=elapsed_ms,
        llm_available=await _llm_available(client),
    )


async def _llm_available(client: httpx.AsyncClient) -> bool:
    try:
        response = await client.get(f"{ML_SERVICE_URL}/llm/status", timeout=5)
        response.raise_for_status()
        return bool(response.json().get("available"))
    except httpx.HTTPError:
        return False


@app.post("/api/answer", response_model=AnswerResponse)
async def answer(payload: AnswerRequest) -> AnswerResponse:
    client: httpx.AsyncClient = app.state.http
    corpus: Corpus = app.state.corpus

    messages = [corpus.messages[mid] for mid in payload.message_ids if mid in corpus.messages]
    if not messages:
        raise HTTPException(status_code=400, detail="сообщения не найдены в корпусе")

    started = time.time()
    try:
        response = await client.post(
            f"{ML_SERVICE_URL}/generate",
            json={"question": payload.text.strip(), "messages": messages},
            timeout=180,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e)) if e.response.content else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"ml-service недоступен: {e}") from e

    body = response.json()
    return AnswerResponse(
        answer=body["answer"],
        sources=body["sources"],
        model=body["model"],
        elapsed_ms=int((time.time() - started) * 1000),
    )


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
UPLOAD_MESSAGE_LIMIT = int(os.getenv("UPLOAD_MESSAGE_LIMIT", "3000"))


@app.post("/api/upload", response_model=UploadResponse)
async def upload_chat(file: UploadFile = File(...)) -> UploadResponse:
    client: httpx.AsyncClient = app.state.http
    content = await file.read()
    try:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/upload",
            files={"file": (file.filename, content, "application/json")},
            timeout=600,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e)) if e.response.content else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"orchestrator недоступен: {e}") from e

    body = response.json()
    return UploadResponse(
        chat_name=body["chat_name"],
        chat_file=body["chat_file"],
        message_count=body["message_count"],
        available_chats=available_chats(),
    )


@app.post("/api/chat", response_model=SwitchChatResponse)
async def switch_chat(payload: SwitchChatRequest) -> SwitchChatResponse:
    if payload.file not in available_chats():
        raise HTTPException(status_code=404, detail=f"чат не найден: {payload.file}")

    client: httpx.AsyncClient = app.state.http
    try:
        corpus = Corpus(chat_file=payload.file)
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"не удалось прочитать чат: {e}") from e

    logger.info("Переключаюсь на %s, переиндексирую...", payload.file)
    try:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/index", json={"file": payload.file}, timeout=1800
        )
        response.raise_for_status()
        points = response.json()["chunks"]
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e)) if e.response.content else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"orchestrator недоступен: {e}") from e

    app.state.corpus = corpus
    logger.info("Активен чат %s: %d сообщений, %d чанков",
                corpus.chat["name"], len(corpus.messages), points)
    return SwitchChatResponse(
        chat_name=corpus.chat["name"],
        chat_file=corpus.file_name,
        message_count=len(corpus.messages),
        chunks=points,
    )


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
