from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

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
