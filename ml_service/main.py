import logging
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import answer as llm


DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
RERANK_MODEL_NAME_LOCAL = "jinaai/jina-reranker-v2-base-multilingual-quantized"
FASTEMBED_CACHE_PATH = os.getenv("FASTEMBED_CACHE_PATH", "/models/fastembed")
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ml-service")

app = FastAPI(title="Mock ML Service", version="0.1.0")


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: list[str]


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    data: list[EmbeddingItem]


class ScoreRequest(BaseModel):
    model: str | None = None
    text_1: str
    text_2: list[str]


class ScoreItem(BaseModel):
    score: float


class ScoreResponse(BaseModel):
    data: list[ScoreItem]


class GenerateRequest(BaseModel):
    question: str
    messages: list[dict]


class Source(BaseModel):
    n: int
    id: str
    sender: str
    date: str
    text: str


class GenerateResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str


class LLMStatus(BaseModel):
    available: bool
    model: str


@lru_cache(maxsize=1)
def get_dense_model():
    from fastembed import TextEmbedding

    logger.info("Loading local dense model %s", DENSE_MODEL_NAME)
    return TextEmbedding(model_name=DENSE_MODEL_NAME, cache_dir=FASTEMBED_CACHE_PATH)


@lru_cache(maxsize=1)
def get_rerank_model():
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    TextCrossEncoder.add_custom_model(
        model=RERANK_MODEL_NAME_LOCAL,
        sources=ModelSource(hf="jinaai/jina-reranker-v2-base-multilingual"),
        model_file="onnx/model_quantized.onnx",
        size_in_gb=0.27,
        license="cc-by-nc-4.0",
    )
    logger.info("Loading local rerank model %s", RERANK_MODEL_NAME_LOCAL)
    return TextCrossEncoder(
        model_name=RERANK_MODEL_NAME_LOCAL, cache_dir=FASTEMBED_CACHE_PATH
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/llm/status", response_model=LLMStatus)
async def llm_status() -> LLMStatus:
    return LLMStatus(available=await llm.is_available(), model=llm.LLM_MODEL)


@app.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    try:
        text, sources = await llm.generate(payload.question, payload.messages)
    except llm.LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return GenerateResponse(answer=text, sources=sources, model=llm.LLM_MODEL)


@app.post("/embeddings", response_model=EmbeddingsResponse)
async def embeddings(payload: EmbeddingsRequest) -> EmbeddingsResponse:
    model = get_dense_model()
    data: list[EmbeddingItem] = []
    for start in range(0, len(payload.input), EMBED_BATCH_SIZE):
        batch = payload.input[start : start + EMBED_BATCH_SIZE]
        for offset, vector in enumerate(model.embed(batch, batch_size=EMBED_BATCH_SIZE)):
            data.append(EmbeddingItem(index=start + offset, embedding=vector.tolist()))
    return EmbeddingsResponse(data=data)


@app.post("/score", response_model=ScoreResponse)
async def score(payload: ScoreRequest) -> ScoreResponse:
    if not payload.text_2:
        return ScoreResponse(data=[])

    model = get_rerank_model()
    scores: list[float] = []
    for start in range(0, len(payload.text_2), RERANK_BATCH_SIZE):
        batch = payload.text_2[start : start + RERANK_BATCH_SIZE]
        scores.extend(
            float(s) for s in model.rerank(payload.text_1, batch, batch_size=RERANK_BATCH_SIZE)
        )
    return ScoreResponse(data=[ScoreItem(score=s) for s in scores])


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, workers=1)


if __name__ == "__main__":
    main()
