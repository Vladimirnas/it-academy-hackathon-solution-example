"""Сервис индексации: собирает чанки и складывает их в Qdrant.

На хакатоне эту роль выполняла закрытая система проверки. Здесь тот же
конвейер, но своим сервисом: index строит чанки, ml-service считает
плотные векторы, разреженные приходят из index/sparse_embedding.
Коллекция в Qdrant одна, поэтому переключение чата её пересоздаёт.
"""

import json
import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import telegram

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
PROTECTED = {"Go Nova.json", "demo_chat.json"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
UPLOAD_MESSAGE_LIMIT = int(os.getenv("UPLOAD_MESSAGE_LIMIT", "3000"))

INDEX_SERVICE_URL = os.getenv("INDEX_SERVICE_URL", "http://index:8000")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
DENSE_MODEL = os.getenv(
    "EMBEDDINGS_DENSE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DENSE_DIM = int(os.getenv("DENSE_DIM", "384"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("orchestrator")

app = FastAPI(
    title="Orchestrator Service",
    description="Построение чанков и запись их в Qdrant",
    version="0.1.0",
)


class IndexRequest(BaseModel):
    file: str = Field(min_length=1, description="Имя файла чата в каталоге данных")


class IndexResponse(BaseModel):
    chat_name: str
    chat_file: str
    message_count: int
    chunks: int


class UploadResponse(BaseModel):
    chat_name: str
    chat_file: str
    message_count: int


class IndexedChat(BaseModel):
    chat_name: str | None


class PipelineError(RuntimeError):
    """Не удалось построить или записать индекс."""


async def recreate_collection(client: httpx.AsyncClient) -> None:
    await client.delete(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=30)
    response = await client.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        json={
            "vectors": {"dense": {"size": DENSE_DIM, "distance": "Cosine"}},
            "sparse_vectors": {"sparse": {"modifier": "idf"}},
        },
        timeout=30,
    )
    response.raise_for_status()


async def reindex(chat: dict, messages: list[dict]) -> int:
    async with httpx.AsyncClient() as client:
      return await _reindex(client, chat, messages)


async def _reindex(client: httpx.AsyncClient, chat: dict, messages: list[dict]) -> int:
    response = await client.post(
        f"{INDEX_SERVICE_URL}/index",
        json={"data": {"chat": chat, "overlap_messages": [], "new_messages": messages}},
        timeout=300,
    )
    response.raise_for_status()
    chunks = response.json()["results"]
    if not chunks:
        return 0

    dense_response = await client.post(
        f"{ML_SERVICE_URL}/embeddings",
        json={"model": DENSE_MODEL, "input": [c["dense_content"] for c in chunks]},
        timeout=600,
    )
    dense_response.raise_for_status()
    by_index = {item["index"]: item["embedding"] for item in dense_response.json()["data"]}
    dense_vectors = [by_index[i] for i in range(len(chunks))]

    sparse_response = await client.post(
        f"{INDEX_SERVICE_URL}/sparse_embedding",
        json={"texts": [c["sparse_content"] for c in chunks]},
        timeout=300,
    )
    sparse_response.raise_for_status()
    sparse_vectors = sparse_response.json()["vectors"]

    messages_by_id = {m["id"]: m for m in messages}
    points = []
    for i, chunk in enumerate(chunks):
        chunk_messages = [
            messages_by_id[mid] for mid in chunk["message_ids"] if mid in messages_by_id
        ]
        mentions: set[str] = set()
        for m in chunk_messages:
            mentions.update(m.get("mentions") or [])

        points.append({
            "id": i,
            "vector": {
                "dense": dense_vectors[i],
                "sparse": {
                    "indices": sparse_vectors[i]["indices"],
                    "values": sparse_vectors[i]["values"],
                },
            },
            "payload": {
                "page_content": chunk["page_content"],
                "metadata": {
                    "chat_name": chat["name"],
                    "chat_type": chat["type"],
                    "chat_id": chat["id"],
                    "chat_sn": chat["sn"],
                    "thread_sn": None,
                    "message_ids": chunk["message_ids"],
                    "start": "",
                    "end": "",
                    "participants": sorted(
                        {m["sender_id"] for m in chunk_messages if m.get("sender_id")}
                    ),
                    "mentions": sorted(mentions),
                    "contains_forward": any(m.get("is_forward") for m in chunk_messages),
                    "contains_quote": any(m.get("is_quote") for m in chunk_messages),
                },
            },
        })

    await recreate_collection(client)
    upsert = await client.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points", json={"points": points}, timeout=300
    )
    upsert.raise_for_status()
    return len(points)


async def indexed_chat_name() -> str | None:
    async with httpx.AsyncClient() as client:
        return await _indexed_chat_name(client)


async def _indexed_chat_name(client: httpx.AsyncClient) -> str | None:
    try:
        response = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json={"limit": 1, "with_payload": True},
            timeout=10,
        )
        response.raise_for_status()
        points = response.json()["result"]["points"]
    except (httpx.HTTPError, KeyError):
        return None
    if not points:
        return None
    return points[0]["payload"].get("metadata", {}).get("chat_name")


def load_chat(name: str) -> dict:
    path = DATA_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"чат не найден: {name}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/indexed", response_model=IndexedChat)
async def indexed() -> IndexedChat:
    return IndexedChat(chat_name=await indexed_chat_name())


@app.post("/index", response_model=IndexResponse)
async def index_chat(payload: IndexRequest) -> IndexResponse:
    raw = load_chat(payload.file)
    chat, messages = raw["chat"], raw["messages"]

    logger.info("Индексирую %s (%d сообщений)", chat["name"], len(messages))
    try:
        chunks = await reindex(chat, messages)
    except PipelineError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    logger.info("Готово: %d чанков", chunks)
    return IndexResponse(
        chat_name=chat["name"],
        chat_file=payload.file,
        message_count=len(messages),
        chunks=chunks,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"файл больше {MAX_UPLOAD_BYTES // 1024 // 1024} МБ")
    try:
        raw = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"не похоже на JSON: {e}") from e

    if not isinstance(raw, dict) or "messages" not in raw:
        raise HTTPException(status_code=400,
                            detail="не похоже на экспорт Telegram: нет поля messages")

    converted = telegram.convert(raw, limit=UPLOAD_MESSAGE_LIMIT, anonymize=False)
    if not converted["messages"]:
        raise HTTPException(status_code=400, detail="в экспорте нет сообщений с текстом")

    target = DATA_DIR / f"{telegram.slugify(converted['chat']['name'])}.json"
    if target.name in PROTECTED:
        raise HTTPException(status_code=409,
                            detail=f"имя {target.name} занято штатным корпусом")
    target.write_text(json.dumps(converted, ensure_ascii=False), encoding="utf-8")
    logger.info("Загружен чат %s: %d сообщений",
                converted["chat"]["name"], len(converted["messages"]))

    return UploadResponse(
        chat_name=converted["chat"]["name"],
        chat_file=target.name,
        message_count=len(converted["messages"]),
    )


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
