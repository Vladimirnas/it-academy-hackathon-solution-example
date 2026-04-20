import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
import asyncio

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ваш сервис должен считывать эти переменные из окружения (env), так как проверяющая система управляет ими
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8004"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")


# Модель данных, которую мы предоставляем и рассчитываем получать от вас
class Chat(BaseModel):
    id: str
    name: str
    sn: str
    type: str  # group, channel, private
    is_public: bool | None = None
    members_count: int | None = None
    members: list[dict[str, Any]] | None = None


class Message(BaseModel):
    id: str
    thread_sn: str | None = None
    time: int
    text: str
    sender_id: str
    file_snippets: str
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None
    member_event: dict[str, Any] | None = None
    is_system: bool
    is_hidden: bool
    is_forward: bool
    is_quote: bool


class ChatData(BaseModel):
    chat: Chat
    overlap_messages: list[Message]
    new_messages: list[Message]


class IndexAPIRequest(BaseModel):
    data: ChatData



class IndexAPIItem(BaseModel):
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]


class IndexAPIResponse(BaseModel):
    results: list[IndexAPIItem]


class SparseEmbeddingRequest(BaseModel):
    texts: list[str]


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]


@asynccontextmanager
async def lifespan(app: FastAPI):


    await asyncio.to_thread(get_sparse_model)
    yield

app = FastAPI(title="Index Service", version="0.1.0", lifespan=lifespan)



MESSAGES_PER_CHUNK = 5   
MESSAGES_OVERLAP = 3      
CHUNK_STEP = 5            
MAX_CHUNK_CHARS = 6000    
SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"



UVICORN_WORKERS = 8

MONTHS_RU = [
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
]


def render_message(message: Message) -> str:
    parts: list[str] = []

    if message.text:
        parts.append(message.text)

    if message.parts:
        for part in message.parts:
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text:
                parts.append(part_text)

    return "\n".join(parts)


def render_message_rich(message: Message) -> str:


    parts: list[str] = []

    if message.text:
        parts.append(message.text)

    if message.parts:
        for part in message.parts:
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text:
                parts.append(part_text)

    if message.file_snippets:
        parts.append(message.file_snippets)

    return "\n".join(parts)


def build_chunks(
    chat: Chat,
    overlap_messages: list[Message],
    new_messages: list[Message],
) -> list[IndexAPIItem]:


    def has_text(m: Message) -> bool:
        if m.text:
            return True
        if m.parts:
            return any(isinstance(p.get("text"), str) and p.get("text") for p in m.parts)
        return False

    overlap = [m for m in overlap_messages if has_text(m)]
    new = [m for m in new_messages if has_text(m)]

    if not new:
        return []

  
  
    all_messages = overlap[-MESSAGES_OVERLAP:] + new

    result: list[IndexAPIItem] = []
    step = CHUNK_STEP

    for start in range(0, len(all_messages), step):
        window = all_messages[start : start + MESSAGES_PER_CHUNK + MESSAGES_OVERLAP]
        if not window:
            break



        window_new = [m for m in window if m in new]
        if not window_new:
            continue



        tagged_lines: list[str] = []
        for m in window:
            text = render_message(m)
            if not text:
                continue


            if m.file_snippets:
                text += " " + m.file_snippets[:300]
            if m in new:
                tagged_lines.append(f"[{m.id}|{m.sender_id}] {text}")
            else:
                tagged_lines.append(f"{m.sender_id}: {text}")
        page_content = "\n".join(tagged_lines)[:MAX_CHUNK_CHARS]

        
        dense_lines = [render_message(m) for m in window if render_message(m)]
        dense_text = "\n".join(dense_lines)
        dense_content = f"[{chat.name}] {dense_text}"[:MAX_CHUNK_CHARS]


        sparse_lines = [render_message_rich(m) for m in window if render_message_rich(m)]
        sparse_extras: list[str] = []
        sparse_extras.append(chat.name)


        participants = set(m.sender_id for m in window if m.sender_id)
        if participants:
            sparse_extras.append(" ".join(participants))
        all_mentions: set[str] = set()
        for m in window:
            if m.mentions:
                all_mentions.update(m.mentions)
        if all_mentions:
            sparse_extras.append(" ".join(all_mentions))


        if window:
            start_ts = min(m.time for m in window)
            end_ts = max(m.time for m in window)
            start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
            date_tokens: set[str] = set()
            date_tokens.add(f"{MONTHS_RU[start_dt.month - 1]} {start_dt.year}")
            date_tokens.add(f"{MONTHS_RU[end_dt.month - 1]} {end_dt.year}")
            sparse_extras.extend(date_tokens)
        sparse_content = ("\n".join(sparse_lines) + "\n" + " ".join(sparse_extras))[:MAX_CHUNK_CHARS]

        result.append(
            IndexAPIItem(
                page_content=page_content,
                dense_content=dense_content,
                sparse_content=sparse_content,
                message_ids=[m.id for m in window_new],
            )
        )

    return result



@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    return IndexAPIResponse(
        results=build_chunks(
            payload.data.chat,
            payload.data.overlap_messages,
            payload.data.new_messages,
        )
    )


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding




    logger.info(
        "Loading sparse model %s from cache %s",
        SPARSE_MODEL_NAME,
        FASTEMBED_CACHE_PATH,
    )
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    vectors: list[dict[str, list[int] | list[float]]] = []

    for item in model.embed(texts):
        vectors.append(
            {
                "indices": item.indices.tolist(),
                "values": item.values.tolist(),
            }
        )

    return vectors


@app.post("/sparse_embedding")
async def sparse_embedding(payload: SparseEmbeddingRequest) -> dict[str, Any]:


    vectors = await asyncio.to_thread(embed_sparse_texts, payload.texts)
    return {"vectors": vectors}

# красивая обработка ошибок
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=UVICORN_WORKERS,
    )


if __name__ == "__main__":
    main()
