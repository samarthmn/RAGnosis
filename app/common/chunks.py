from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    documents: list[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents with a recursive character splitter (shared by both pipelines)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk_item in enumerate(chunks):
        chunk_item.metadata["chunk_id"] = index
        chunk_item.metadata["chunk_size"] = len(chunk_item.page_content)
        chunk_item.metadata["chunking_strategy"] = "recursive_character"
    print(f"Divided into {len(chunks)} chunks")
    return chunks


def save_chunks(chunks: list[Document], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    {
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Saved {len(chunks)} chunks to {path}")
    return path


def load_chunks(path: Path, *, command_hint: str) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(
            f"Stored chunks not found at {path}. Run `{command_hint}` first."
        )
    chunks: list[Document] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(
                Document(
                    page_content=str(record["page_content"]),
                    metadata=dict(record.get("metadata", {})),
                )
            )
    return chunks
