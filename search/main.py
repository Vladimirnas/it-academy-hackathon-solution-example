import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = os.getenv("RERANKER_URL")
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
QDRANT_FUSION = os.getenv("QDRANT_FUSION", "DBSF").upper()
REQUIRED_ENV_VARS = [
    "EMBEDDINGS_DENSE_URL",
    "RERANKER_URL",
    "QDRANT_URL",
]
 
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")


def validate_required_env() -> None:
    if bool(OPEN_API_LOGIN) != bool(OPEN_API_PASSWORD):
        raise RuntimeError("OPEN_API_LOGIN and OPEN_API_PASSWORD must be set together")

    if not API_KEY and not (OPEN_API_LOGIN and OPEN_API_PASSWORD):
        raise RuntimeError("Either API_KEY or OPEN_API_LOGIN and OPEN_API_PASSWORD must be set")

    missing_env_vars = [
        name for name in REQUIRED_ENV_VARS if os.getenv(name) is None or os.getenv(name) == ""
    ]
    if not missing_env_vars:
        return

    logger.error("Empty required env vars: %s", ", ".join(missing_env_vars))
    raise RuntimeError(f"Empty required env vars: {', '.join(missing_env_vars)}")


validate_required_env()


def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}

    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
        return kwargs

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    return kwargs


class DateRange(BaseModel):
    from_: str = Field(alias="from")
    to: str


class Entities(BaseModel):
    people: list[str] | None = None
    emails: list[str] | None = None
    documents: list[str] | None = None
    names: list[str] | None = None
    links: list[str] | None = None


class Question(BaseModel):
    text: str
    asker: str = ""
    asked_on: str = ""
    variants: list[str] | None = None
    hyde: list[str] | None = None
    keywords: list[str] | None = None
    entities: Entities | None = None
    date_mentions: list[str] | None = None
    date_range: DateRange | None = None
    search_text: str = ""


class SearchAPIRequest(BaseModel):
    question: Question


class SearchAPIItem(BaseModel):
    message_ids: list[str]


class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]


class DenseEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class DenseEmbeddingResponse(BaseModel):
    data: list[DenseEmbeddingItem]


class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]


class ChunkMetadata(BaseModel):
    chat_name: str
    chat_type: str
    chat_id: str
    chat_sn: str
    thread_sn: str | None = None
    message_ids: list[str]
    start: str
    end: str
    participants: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    contains_forward: bool = False
    contains_quote: bool = False


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=API_KEY,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.1.0", lifespan=lifespan)


DENSE_PREFETCH_K = 150
SPARSE_PREFETCH_K = 200
VARIANT_SPARSE_K = 100
RETRIEVE_K = 100
RERANK_CANDIDATES = 20
MESSAGE_RERANK_CANDIDATES = 60
MAX_RERANK_TEXT_CHARS = 1200
MAX_MESSAGE_IDS = 50
MAX_VARIANTS = 3
MAX_DENSE_TEXTS = 3
TIER1_LIMIT = 35           


async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": [text],
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload.data:
        raise ValueError("Dense embedding response is empty")
    return payload.data[0].embedding


async def embed_dense_batch(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": texts,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload.data:
        raise ValueError("Dense embedding response is empty")
    sorted_data = sorted(payload.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


async def embed_sparse(text: str) -> SparseVector:
    vectors = list(get_sparse_model().embed([text]))
    if not vectors:
        raise ValueError("Sparse embedding response is empty")
    item = vectors[0]
    return SparseVector(
        indices=[int(i) for i in item.indices.tolist()],
        values=[float(v) for v in item.values.tolist()],
    )


async def embed_sparse_multi(texts: list[str]) -> list[SparseVector]:
    def _embed() -> list[SparseVector]:
        model = get_sparse_model()
        results: list[SparseVector] = []
        for item in model.embed(texts):
            results.append(SparseVector(
                indices=[int(i) for i in item.indices.tolist()],
                values=[float(v) for v in item.values.tolist()],
            ))
        return results
    return await asyncio.to_thread(_embed)


async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
    entity_values: list[str] | None = None,
) -> list[Any]:
    prefetch: list[models.Prefetch] = []


    for dv in dense_vectors:
        prefetch.append(
            models.Prefetch(
                query=dv,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=DENSE_PREFETCH_K,
            )
        )


    for i, sv in enumerate(sparse_vectors):
        limit = SPARSE_PREFETCH_K if i == 0 else VARIANT_SPARSE_K
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(
                    indices=sv.indices,
                    values=sv.values,
                ),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=limit,
            )
        )


    if entity_values and dense_vectors:
        try:
            prefetch.append(
                models.Prefetch(
                    query=dense_vectors[0],
                    using=QDRANT_DENSE_VECTOR_NAME,
                    limit=50,
                    filter=models.Filter(
                        should=[
                            models.FieldCondition(
                                key="metadata.mentions",
                                match=models.MatchAny(any=entity_values),
                            ),
                            models.FieldCondition(
                                key="metadata.participants",
                                match=models.MatchAny(any=entity_values),
                            ),
                        ]
                    ),
                )
            )
        except Exception:
            pass

    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF if QDRANT_FUSION == "RRF" else models.Fusion.DBSF
        ),
        limit=RETRIEVE_K,
        with_payload=True,
    )
    return response.points or []


def extract_message_ids(point: Any) -> list[str]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    return [str(mid) for mid in (metadata.get("message_ids") or [])]


async def get_rerank_scores(
    client: httpx.AsyncClient,
    query: str,
    targets: list[str],
) -> list[float]:
    if not targets:
        return []
    response = await client.post(
        RERANKER_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": RERANKER_MODEL,
            "encoding_format": "float",
            "text_1": query,
            "text_2": targets,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json().get("data") or []
    return [float(s["score"]) for s in data]


async def rerank_points(
    client: httpx.AsyncClient,
    query: str,
    points: list[Any],
) -> list[Any]:
    if not points:
        return points
    candidates = points[:RERANK_CANDIDATES]
    try:
        targets = [p.payload.get("page_content", "") for p in candidates]
        scores = await get_rerank_scores(client, query, targets)
    except Exception as e:
        logger.warning("Reranker failed, skipping: %s", e)
        return points
    if not scores:
        return points
    reranked = [
        point
        for _, point in sorted(
            zip(scores, candidates, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )
    ]
    return reranked + points[RERANK_CANDIDATES:]


async def rerank_messages(
    client: httpx.AsyncClient,
    query: str,
    points: list[Any],
) -> list[str]:
    texts: dict[str, str] = {}
    order: list[str] = []

    for point in points:
        tagged = parse_tagged_messages(point.payload.get("page_content", ""))
        for mid in extract_message_ids(point):
            if mid in texts:
                continue
            text = tagged.get(mid, "").strip()
            if not text:
                continue
            texts[mid] = text[:MAX_RERANK_TEXT_CHARS]
            order.append(mid)
            if len(order) >= MESSAGE_RERANK_CANDIDATES:
                break
        if len(order) >= MESSAGE_RERANK_CANDIDATES:
            break

    if not order:
        return []

    try:
        scores = await get_rerank_scores(client, query, [texts[m] for m in order])
    except Exception as e:
        logger.warning("Message reranker failed, keeping retrieval order: %s", e)
        return order
    if len(scores) != len(order):
        logger.warning("Reranker returned %d scores for %d texts, keeping order",
                       len(scores), len(order))
        return order

    return [
        mid
        for _, mid in sorted(zip(scores, order, strict=True), key=lambda x: x[0], reverse=True)
    ]


_TAG_RE = re.compile(r'\[([^\]|]+)\|([^\]]+)\]\s*')


def parse_tagged_messages(page_content: str) -> dict[str, str]:


    result: dict[str, str] = {}
    matches = list(_TAG_RE.finditer(page_content))
    for i, match in enumerate(matches):
        msg_id = match.group(1)
        sender = match.group(2)
        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(page_content)
        text = page_content[text_start:text_end].strip()
        result[msg_id] = f"{sender} {text}"
    return result


STOP_WORDS = {
    "кто", "что", "как", "где", "когда", "какой", "какая", "какие", "какое",
    "чем", "чём", "кого", "кому", "чей", "зачем", "почему", "куда", "откуда",
    "был", "была", "было", "были", "есть", "это", "этот", "эта", "эти",
    "для", "или", "если", "так", "там", "тут", "она", "они", "оно", "все",
    "всех", "нужно", "можно", "какому", "каком", "каких",
}


def _clean_term(word: str) -> str:
    return word.strip(".,!?;:()[]«»\"'…")


def build_query_terms(question: Question) -> set[str]:

    terms: set[str] = set()
    for word in question.text.lower().split():
        word = _clean_term(word)
        if len(word) >= 3 and word not in STOP_WORDS:
            terms.add(word)
    if question.keywords:
        for kw in question.keywords:
            for word in kw.lower().split():
                word = _clean_term(word)
                if len(word) >= 3 and word not in STOP_WORDS:
                    terms.add(word)
    if question.entities:
        for field in (question.entities.people, question.entities.names,
                      question.entities.emails, question.entities.documents):
            if field:
                for entity in field:
                    for word in entity.lower().split():
                        if len(word) >= 2:
                            terms.add(word)
    if question.date_mentions:
        for dm in question.date_mentions:
            for word in dm.lower().split():
                if len(word) >= 3:
                    terms.add(word)
    return terms


STEM_MIN_LEN = 4


def _same_stem(term: str, word: str) -> bool:
    n = min(len(term), len(word))
    if n < STEM_MIN_LEN:
        return term == word
    return term[:n] == word[:n]


def keyword_score(terms: set[str], text: str) -> float:

    if not terms or not text:
        return 0.0
    words = set(re.findall(r"\w+", text.lower()))
    matches = sum(1 for t in terms if any(_same_stem(t, w) for w in words))
    return matches / len(terms) if terms else 0.0


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    question = payload.question
    query = question.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="question.text is required")

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

   
    dense_texts = [query]
    if question.hyde and question.hyde[0].strip():
        dense_texts.append(question.hyde[0].strip())
    if question.search_text and question.search_text.strip() and question.search_text.strip() != query:
        if len(dense_texts) < MAX_DENSE_TEXTS:
            dense_texts.append(question.search_text.strip())


    sparse_query = query
    if question.keywords:
        sparse_query = query + " " + " ".join(question.keywords)
    entity_filter_values: list[str] = []
    if question.entities:
        entity_terms: list[str] = []
        for field in (question.entities.people, question.entities.names,
                      question.entities.emails, question.entities.documents,
                      question.entities.links):
            if field:
                entity_terms.extend(field)
        if entity_terms:
            sparse_query = sparse_query + " " + " ".join(entity_terms)


        for field in (question.entities.people, question.entities.names):
            if field:
                entity_filter_values.extend(field)
    if question.date_mentions:
        sparse_query = sparse_query + " " + " ".join(question.date_mentions)

    sparse_texts = [sparse_query]
    if question.search_text and question.search_text.strip() and question.search_text.strip() != query:
        sparse_texts.append(question.search_text.strip())
    if question.variants:
        for v in question.variants[:MAX_VARIANTS]:
            v_text = v.strip()
            if v_text:
                sparse_texts.append(v_text)

    logger.info("Dense texts (%d, hyde=%s): %s",
                len(dense_texts), bool(question.hyde), dense_texts[0][:80])
    logger.info("Sparse texts (%d): %s",
                len(sparse_texts), sparse_texts[0][:80])


    dense_vectors, sparse_vectors = await asyncio.gather(
        embed_dense_batch(client, dense_texts),
        embed_sparse_multi(sparse_texts),
    )
    points = await qdrant_search(
        qdrant, dense_vectors, sparse_vectors,
        entity_values=entity_filter_values or None,
    )

    if not points:
        return SearchAPIResponse(results=[])


    query_terms = build_query_terms(question)

    tier1 = (await rerank_messages(client, query, points))[:TIER1_LIMIT]
    seen: set[str] = set(tier1)


    tier2_scores: dict[str, float] = {}
    for point in points:
        page_content = point.payload.get("page_content", "")
        mids = extract_message_ids(point)
        tagged = parse_tagged_messages(page_content)
        for mid in mids:
            if mid not in seen and mid not in tier2_scores:
                text = tagged.get(mid, "")
                tier2_scores[mid] = keyword_score(query_terms, text)

    tier2 = sorted(tier2_scores.keys(), key=lambda m: tier2_scores[m], reverse=True)

    message_ids = (tier1 + tier2)[:MAX_MESSAGE_IDS]

    return SearchAPIResponse(
        results=[SearchAPIItem(message_ids=message_ids)]
    )


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    detail = str(exc) or repr(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return JSONResponse(status_code=500, content={"detail": detail})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
