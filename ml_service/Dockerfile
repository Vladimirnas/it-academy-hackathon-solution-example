FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

ENV HOST=0.0.0.0
ENV PORT=8000
ENV FASTEMBED_CACHE_PATH=/models/fastembed


COPY .model_cache /models/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/models/fastembed')"
RUN python -c "from fastembed.common.model_description import ModelSource; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder.add_custom_model(model='jinaai/jina-reranker-v2-base-multilingual-quantized', sources=ModelSource(hf='jinaai/jina-reranker-v2-base-multilingual'), model_file='onnx/model_quantized.onnx', size_in_gb=0.27, license='cc-by-nc-4.0'); TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual-quantized', cache_dir='/models/fastembed')"

EXPOSE 8000

CMD ["python", "main.py"]
