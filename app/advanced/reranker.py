"""Local reranking, supporting two model families selected by ``rerank=<repo-id>``.

The family is detected from the model's own ``config.json`` (only that tiny file is
read for the decision), so routing is not fragile name-matching:

- **Cross-encoders** (default ``BAAI/bge-reranker-v2-m3``): an XLM-RoBERTa
  sequence-classification head scores each (query, chunk) pair directly, via
  FlagEmbedding. Fast, permissively licensed (Apache-2.0).
- **Jina listwise** (``jinaai/jina-reranker-v3``): a Qwen3-based ``JinaForRanking``
  model that ranks all chunks together in one pass using late-interaction
  embeddings, via its ``trust_remote_code`` ``.rerank()`` API on transformers.
  Stronger relevance, but the weights are CC-BY-NC-4.0 (non-commercial) and loading
  executes model code fetched from the Hub. Runs on GPU/MPS/CPU.

Both paths return chunks reordered most- to least-relevant; the caller truncates to
top-k. Weights are downloaded from the Hugging Face Hub on first use and cached under
``~/.cache/huggingface``. Loading is lazy and memoized, so each model is read once per
process (and reused across runs in a multirun batch).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from app.common.models import RERANK_MODEL

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker


@lru_cache(maxsize=None)
def _is_jina_reranker(model_name: str) -> bool:
    """True for a Jina listwise reranker (``JinaForRanking``), False for a cross-encoder.

    Reads only the tiny ``config.json`` (cached after first fetch), so the routing
    works without a hardcoded model list.
    """
    from transformers import AutoConfig

    architectures = getattr(
        AutoConfig.from_pretrained(model_name, trust_remote_code=False),
        "architectures",
        None,
    )
    return any("JinaForRanking" in str(arch) for arch in (architectures or []))


# --- Cross-encoder path (FlagEmbedding) -----------------------------------------------


@lru_cache(maxsize=None)
def load_reranker(model_name: str = RERANK_MODEL) -> "FlagReranker":
    """Load (and cache) the FlagEmbedding cross-encoder for ``model_name``.

    ``use_fp16`` is only enabled on CUDA — half precision is unsupported / unstable on
    CPU and Apple MPS, where it would raise or silently degrade results.
    """
    from FlagEmbedding import FlagReranker
    import torch

    return FlagReranker(model_name, use_fp16=torch.cuda.is_available())


def _rerank_cross_encoder(
    question: str, chunks: list[Document], model_name: str
) -> list[Document]:
    reranker = load_reranker(model_name)
    pairs = [[question, chunk.page_content] for chunk in chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):  # single-pair calls return a bare float
        scores = [scores]
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked]


# --- Jina listwise path (transformers, trust_remote_code) -----------------------------


@lru_cache(maxsize=None)
def _load_jina_reranker(model_name: str):
    """Load (and cache) the Jina listwise reranker via its Hub-hosted model code.

    Loading runs ``trust_remote_code`` code fetched from the model repo — only used
    for the explicitly-selected Jina reranker, never for the default cross-encoder.
    The model is placed on CUDA if available, else Apple MPS, else CPU.
    """
    import torch
    from transformers import AutoModel

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = AutoModel.from_pretrained(model_name, dtype="auto", trust_remote_code=True)
    return model.to(device).eval()


def _rerank_jina(
    question: str, chunks: list[Document], model_name: str
) -> list[Document]:
    model = _load_jina_reranker(model_name)
    documents = [chunk.page_content for chunk in chunks]
    # .rerank returns dicts sorted best-first, each carrying the input `index`.
    results = model.rerank(question, documents)
    return [chunks[result["index"]] for result in results]


def rerank_local(
    question: str,
    chunks: list[Document],
    *,
    model_name: str = RERANK_MODEL,
) -> list[Document]:
    """Return ``chunks`` reordered most- to least-relevant for ``question``.

    The scorer is picked automatically from the model's architecture (cross-encoder
    vs Jina listwise). Every chunk is kept (this only reorders); the caller truncates
    to top-k. Errors propagate deliberately: a reranker that cannot load is a hard
    configuration problem, and silently returning the original order would corrupt
    eval numbers while looking fine.
    """
    if len(chunks) <= 1:
        return chunks
    if _is_jina_reranker(model_name):
        return _rerank_jina(question, chunks, model_name)
    return _rerank_cross_encoder(question, chunks, model_name)
