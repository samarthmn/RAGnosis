"""Local cross-encoder reranking with ``BAAI/bge-reranker-v2-m3``.

Unlike the LLM re-ranker (which asks a chat model to order chunk ids), this runs a
cross-encoder locally via FlagEmbedding. The model scores each (query, chunk) pair
directly, which is what rerankers are trained for, so it is both faster and more
accurate than prompting a general chat model to rank.

The model (~600MB) is downloaded from the Hugging Face Hub on first use and cached
under ``~/.cache/huggingface``. Loading is lazy and memoized, so the weights are read
once per process (and reused across runs in a multirun batch).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from app.common.models import RERANK_MODEL

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker


@lru_cache(maxsize=None)
def load_reranker(model_name: str = RERANK_MODEL) -> "FlagReranker":
    """Load (and cache) the FlagEmbedding reranker for ``model_name``.

    ``use_fp16`` is only enabled on CUDA — half precision is unsupported / unstable on
    CPU and Apple MPS, where it would raise or silently degrade results.
    """
    from FlagEmbedding import FlagReranker
    import torch

    return FlagReranker(model_name, use_fp16=torch.cuda.is_available())


def rerank_local(
    question: str,
    chunks: list[Document],
    *,
    model_name: str = RERANK_MODEL,
) -> list[Document]:
    """Return ``chunks`` reordered most- to least-relevant for ``question``.

    Every chunk is kept (this only reorders); the caller truncates to top-k. Errors
    propagate deliberately: a reranker that cannot load is a hard configuration
    problem, and silently returning the original order would corrupt eval numbers
    while looking fine.
    """
    if len(chunks) <= 1:
        return chunks
    reranker = load_reranker(model_name)
    pairs = [[question, chunk.page_content] for chunk in chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):  # single-pair calls return a bare float
        scores = [scores]
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked]
