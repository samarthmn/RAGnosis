from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from tqdm import tqdm

from app.common.chat import chat_structured
from app.common.chunking import build_documents
from app.common.chunks import load_chunks as load_stored_documents
from app.common.chunks import save_chunks as save_documents
from app.common.models import (
    PREPROCESS_MODEL,
    parse_model_overrides,
    set_model_overrides,
)
from app.common.paths import DATASET_DIR

# Enriched documents live next to this module (same folder), ready for chunking.
ENRICHED_DOCS_PATH = Path(__file__).resolve().parent / "enriched_documents.jsonl"


def document_enrichment_prompt(document: Document) -> str:
    meta = document.metadata
    text = document.page_content

    return f"""You write retrieval-friendly metadata for documents in a synthetic clinic knowledge base (patients, doctors, departments, appointments, medical records, prescriptions and billing).

Your job is ONLY to produce a title and summary.

Document metadata:
- Document type: {meta.get("doc_type", "unknown")}
- Source file: {meta.get("source_csv_names", "unknown")}
- Patient ID: {meta.get("patient_id") or "n/a"}
- Patient name: {meta.get("patient_name") or "n/a"}
- Record ID: {meta.get("record_id") or "n/a"}
- Row count: {meta.get("row_count", "unknown")}

Document text:
{text}

Instructions:
1. Title: one line that helps search. Include the person or entity name when present, the document type (patient, doctor, department), and the main topic.
2. Summary: a few sentences describing what this document contains and what questions it can answer.


3. Ground every claim in the document. If truncated, describe overall scope, not individual rows you cannot see.
4. Use plain English. Prefer names over IDs.
""".strip()


class DocumentEnrichment(BaseModel):
    title: str = Field(
        description="One line that helps search. Include patient name when present, table type, and the main topic."
    )
    summary: str = Field(
        description="A few sentences describing what this document contains and what questions it can answer."
    )


def llm_document_enrichment(document: Document) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": document_enrichment_prompt(document),
        },
        {
            "role": "user",
            "content": f"Document metadata: {document.metadata} \n\n Document: {document.page_content} \n\n Please enrich the document with a title and summary.",
        },
    ]
    result = chat_structured(
        messages, DocumentEnrichment, model=PREPROCESS_MODEL
    )
    return {
        "title": result.title,
        "summary": result.summary,
    }


def enrich_document(document: Document) -> Document:
    """Prepend an LLM title + summary to a document so retrieval has more to match on."""
    enrichment = llm_document_enrichment(document)
    title = enrichment["title"].strip()
    summary = enrichment["summary"].strip()
    enriched_text = (
        f"Title: {title}\nSummary: {summary}\n\nPage content: {document.page_content}"
    )
    metadata = {
        **document.metadata,
        "title": title,
        "summary": summary,
        "enriched": True,
    }
    return Document(page_content=enriched_text, metadata=metadata)


def load_enriched_documents(path: Path = ENRICHED_DOCS_PATH) -> list[Document]:
    return load_stored_documents(
        path,
        command_hint="uv run python -m app.advanced.preprocess",
    )


def preprocess(
    *,
    csv_dir: Path = DATASET_DIR,
    limit: int | None = None,
    output_path: Path = ENRICHED_DOCS_PATH,
) -> list[Document]:
    documents = build_documents(csv_dir)
    documents_to_process = documents if limit is None else documents[:limit]
    enriched = [
        enrich_document(document)
        for document in tqdm(documents_to_process, desc="Enriching documents")
    ]
    save_documents(enriched, output_path)
    print(f"Saved {len(enriched)} enriched documents to {output_path}")
    return enriched


if __name__ == "__main__":
    set_model_overrides("advanced", parse_model_overrides(sys.argv[1:]))
    preprocess()
    print("LLM Preprocessing complete")
