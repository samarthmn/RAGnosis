from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.common.chunking import parents_path_for
from app.common.chunks import load_parents
from app.common.embeddings import get_embeddings
from app.common.models import selected_embedding_model

SYSTEM_PROMPT = """
You answer questions about a synthetic Synthea healthcare dataset loaded from CSV tables.
Use only the retrieved context.
If the context does not contain the answer, say that the dataset context does not show it.
Do not provide medical advice. Treat all people as synthetic patients.

Retrieved context:
{context}
""".strip()


def load_vectorstore(
    persist_directory: str | Path,
    *,
    pipeline_name: str,
    embedding_model: str | None = None,
) -> Chroma:
    db_path = Path(persist_directory)
    if not db_path.exists():
        raise FileNotFoundError(
            f"{pipeline_name.title()} vector database not found at {db_path}. "
            f"Run `uv run python -m app.{pipeline_name}.ingest` first."
        )
    return Chroma(
        persist_directory=str(db_path),
        embedding_function=get_embeddings(
            embedding_model or selected_embedding_model(pipeline_name)
        ),
    )


# Cache the parent index per pipeline, keyed by the sidecar's mtime so a
# re-chunk (e.g. between multirun configs) transparently invalidates it.
_PARENTS_CACHE: dict[str, tuple[tuple[str, int], dict[str, Document]]] = {}


def _parent_id_of(doc: Document) -> str:
    return doc.metadata.get("parent_id") or doc.metadata.get("record_id", "")


def parents_by_id(pipeline: str) -> dict[str, Document]:
    """Load ``parents.jsonl`` as ``{parent_id: Document}`` (mtime-cached)."""
    path = parents_path_for(pipeline)
    key = (str(path), path.stat().st_mtime_ns if path.exists() else 0)
    cached = _PARENTS_CACHE.get(pipeline)
    if cached and cached[0] == key:
        return cached[1]
    index = {_parent_id_of(parent): parent for parent in load_parents(path)}
    _PARENTS_CACHE[pipeline] = (key, index)
    return index


def expand_to_parents(
    children: list[Document], pipeline: str, *, k: int
) -> list[Document]:
    """Map ranked child chunks up to their parent documents.

    Preserves child ranking, deduplicates by ``parent_id``, and returns up to
    ``k`` parents. Falls back to the child (or to plain truncation when no
    parents sidecar exists) so behaviour degrades gracefully before a re-chunk.
    """
    index = parents_by_id(pipeline)
    if not index:
        return children[:k]
    out: list[Document] = []
    seen: set[str] = set()
    for child in children:
        parent_id = _parent_id_of(child)
        if parent_id in seen:
            continue
        seen.add(parent_id)
        out.append(index.get(parent_id, child))
        if len(out) >= k:
            break
    return out


def format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        source = metadata.get("source_csv_names", metadata.get("source_tables", "unknown"))
        patient = (
            metadata.get("patient_name")
            or metadata.get("patient_id")
            or metadata.get("record_id", "")
        )
        blocks.append(f"[Context {index}] source={source} patient={patient}\n{doc.page_content}")
    return "\n\n".join(blocks)


def history_to_messages(history: list[dict] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": str(content)})
    return messages
