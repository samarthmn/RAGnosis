from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from tqdm import tqdm

from app.common.models import EmbeddingModelConfig, embedding_config

OPENAI_MAX_INPUTS_PER_BATCH = 2048
LOCAL_MAX_INPUTS_PER_BATCH = 1
MAX_TOKENS_PER_BATCH = 290_000


def encoding_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


class OpenAICompatibleEmbeddings(Embeddings):
    def __init__(self, config: EmbeddingModelConfig) -> None:
        if not config["api_key"]:
            raise RuntimeError(f"No API key configured for {config['model']!r}")
        self.client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        self.model = config["model"]
        self.is_paid = config["is_paid_model"]
        self.usd_per_1m_tokens = config.get("usd_per_1m_tokens")
        self.encoding = encoding_for(self.model)
        self.max_inputs = (
            OPENAI_MAX_INPUTS_PER_BATCH
            if config["is_openai_model"] and config["is_paid_model"]
            else LOCAL_MAX_INPUTS_PER_BATCH
        )

    def batched(self, texts: list[str], token_counts: list[int]) -> Iterator[list[str]]:
        batch: list[str] = []
        batch_tokens = 0
        for text, token_count in zip(texts, token_counts):
            if batch and (
                len(batch) >= self.max_inputs
                or batch_tokens + token_count > MAX_TOKENS_PER_BATCH
            ):
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += token_count
        if batch:
            yield batch

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        token_counts = [len(self.encoding.encode(text)) for text in texts]
        if self.is_paid:
            tokens = sum(token_counts)
            message = (
                f"Embedding {len(texts)} chunks (~{tokens:,} tokens) with {self.model}"
            )
            if self.usd_per_1m_tokens is not None:
                cost = tokens / 1_000_000 * self.usd_per_1m_tokens
                message += f"; estimated cost ${cost:.4f}"
            print(message)

        vectors: list[list[float]] = []
        batches = list(self.batched(texts, token_counts))
        with tqdm(
            total=len(texts), desc=f"Embedding ({self.model})", unit="chunk"
        ) as progress:
            for batch in batches:
                response = self.client.embeddings.create(model=self.model, input=batch)
                vectors.extend(item.embedding for item in response.data)
                progress.update(len(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


def get_embeddings(model: str | None = None) -> Embeddings:
    return OpenAICompatibleEmbeddings(embedding_config(model))


def _reset_chroma_cache() -> None:
    """Clear Chroma's in-process client/collection cache.

    Chroma caches clients (and their collections) per persist path within a process.
    Deleting the directory on disk does not evict that cache, so when several runs
    share one process (e.g. the multirun runner) a rebuilt store can reuse a stale
    collection — fatal when the new embedding model has a different dimension
    (``Collection expecting embedding with dimension of 384, got 4096``). Clearing the
    cache forces a fresh client to read the now-empty directory.
    """
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def create_embeddings(
    chunks: list[Document],
    *,
    persist_directory: Path,
    embedding_model: str | None = None,
) -> Chroma:
    db_path = Path(persist_directory).resolve()
    if db_path.exists():
        print(f"Deleting existing vector database at {db_path}")
        shutil.rmtree(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _reset_chroma_cache()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(embedding_model),
        persist_directory=str(db_path),
    )
    collection = vectorstore._collection
    count = collection.count()
    sample = collection.get(limit=1, include=["embeddings"])["embeddings"]
    dimensions = len(sample[0]) if len(sample) else 0
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore
