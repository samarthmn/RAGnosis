from __future__ import annotations

import json
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
from app.common.models import (
    PREPROCESS_MODEL,
    parse_model_overrides,
    set_model_overrides,
)
from app.common.paths import DATASET_DIR

# Enriched documents live next to this module (same folder), ready for chunking.
ENRICHED_DOCS_PATH = Path(__file__).resolve().parent / "enriched_documents1.jsonl"


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


def fallback_enrichment(document: Document) -> dict[str, str]:
    """Deterministic title/summary from metadata, used when the LLM call fails.

    Keeps a long enrichment run alive: a document the model can't enrich (e.g.
    it returns an empty completion) still gets usable, grounded retrieval text
    instead of crashing the whole pass.
    """
    meta = document.metadata
    doc_type = str(meta.get("doc_type", "document"))
    name = meta.get("patient_name") or meta.get("record_id") or "unknown"
    record_id = meta.get("record_id") or "n/a"
    title = f"{doc_type.capitalize()} {name} ({record_id})"
    summary = (
        f"{doc_type.capitalize()} record for {name} (ID {record_id}), "
        f"source {meta.get('source_csv_names', 'unknown')}. "
        "Contains the fields shown in the document body."
    )
    return {"title": title, "summary": summary}


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
    try:
        result = chat_structured(messages, DocumentEnrichment, model=PREPROCESS_MODEL)
    except Exception as exc:
        record_id = document.metadata.get("record_id", "?")
        print(f"  ! enrichment failed for {record_id}, using fallback: {exc}")
        return fallback_enrichment(document)
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


def _document_key(document: Document) -> tuple[str, str]:
    """Stable identity for a built document, used to resume an interrupted run."""
    meta = document.metadata
    return (str(meta.get("doc_type", "")), str(meta.get("record_id", "")))


def _load_checkpoint(path: Path) -> dict[tuple[str, str], Document]:
    """Load already-enriched documents from a previous (possibly interrupted) run."""
    if not path.exists():
        return {}
    done: dict[tuple[str, str], Document] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            document = Document(
                page_content=str(record["page_content"]),
                metadata=dict(record.get("metadata", {})),
            )
            done[_document_key(document)] = document
    return done


def _append_document(document: Document, path: Path) -> None:
    """Append one enriched document to the checkpoint file as it completes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {"page_content": document.page_content, "metadata": document.metadata},
                ensure_ascii=False,
            )
            + "\n"
        )


def preprocess(
    *,
    csv_dir: Path = DATASET_DIR,
    limit: int | None = None,
    output_path: Path = ENRICHED_DOCS_PATH,
) -> list[Document]:
    documents = build_documents(csv_dir)
    documents_to_process = documents if limit is None else documents[:limit]

    # Resume: reuse documents enriched by a previous run, enriching only the rest.
    # Each newly enriched document is appended immediately, so an interruption
    # costs at most the in-flight document.
    done = _load_checkpoint(output_path)
    if done:
        print(f"Resuming: {len(done)} documents already enriched in {output_path}")

    enriched: list[Document] = []
    for document in tqdm(documents_to_process, desc="Enriching documents"):
        cached = done.get(_document_key(document))
        if cached is not None:
            enriched.append(cached)
            continue
        enriched_document = enrich_document(document)
        _append_document(enriched_document, output_path)
        enriched.append(enriched_document)

    print(f"Saved {len(enriched)} enriched documents to {output_path}")
    return enriched


if __name__ == "__main__":
    set_model_overrides("advanced", parse_model_overrides(sys.argv[1:]))
    preprocess()
    print("LLM Preprocessing complete")
