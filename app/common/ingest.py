"""Embed a pipeline's stored chunks into a Chroma vector DB (shared by both pipelines).

Reads ``app/<pipeline>/chunks.jsonl`` and rebuilds ``app/vector_db/<pipeline>``. Run it
for a pipeline with::

    uv run python -m app.common.ingest basic
    uv run python -m app.common.ingest advanced  embedding="all-minilm:l6-v2"
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma

from app.common.chunking import load_chunks
from app.common.embeddings import create_embeddings
from app.common.models import (
    parse_model_overrides,
    selected_embedding_model,
    set_model_overrides,
)
from app.common.paths import APP_ROOT

PIPELINES = ("basic", "advanced")


def db_path_for(pipeline: str) -> Path:
    """Where a pipeline's Chroma store lives: ``app/vector_db/<pipeline>``."""
    return APP_ROOT / "vector_db" / pipeline


def ingest(
    pipeline: str,
    *,
    chunks_path: Path | None = None,
    embedding_model: str | None = None,
) -> Chroma:
    chunks = load_chunks(pipeline, chunks_path)
    return create_embeddings(
        chunks,
        persist_directory=db_path_for(pipeline),
        embedding_model=embedding_model,
    )


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in PIPELINES:
        raise SystemExit(
            f"usage: python -m app.common.ingest <{'|'.join(PIPELINES)}> [key=value ...]"
        )
    pipeline, overrides = args[0], args[1:]
    set_model_overrides(pipeline, parse_model_overrides(overrides))
    ingest(pipeline, embedding_model=selected_embedding_model(pipeline))
    print(f"{pipeline.title()} ingestion complete")


if __name__ == "__main__":
    main()
