from __future__ import annotations

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.common.chat import chat, chat_structured
from app.common.ingest import db_path_for
from app.common.models import RERANK_MODEL, REWRITE_MODEL, selected_chat_model
from app.common.rag import SYSTEM_PROMPT, format_context, history_to_messages
from app.common.rag import load_vectorstore as load_common_vectorstore

RETRIEVAL_K = 20
FINAL_K = 10


def chat_model() -> str:
    return selected_chat_model("advanced")


class RankOrder(BaseModel):
    order: list[int] = Field(description="Chunk ids from most to least relevant")


def load_vectorstore():
    return load_common_vectorstore(db_path_for("advanced"), pipeline_name="advanced")


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    prompt = f"""
You are about to search a healthcare knowledge base to answer the user's question.

Conversation so far:
{history or []}

User's current question:
{question}

Respond with only a short, specific search query most likely to surface relevant content.
""".strip()
    try:
        rewritten = chat(
            [{"role": "system", "content": prompt}], model=REWRITE_MODEL
        ).strip()
        return rewritten or question
    except Exception:
        return question


def merge_chunks(primary: list[Document], secondary: list[Document]) -> list[Document]:
    merged = list(primary)
    seen = {doc.page_content for doc in primary}
    for doc in secondary:
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    return merged


def rerank(question: str, chunks: list[Document]) -> list[Document]:
    if len(chunks) <= 1:
        return chunks
    prompt_parts = [
        f"The user asked:\n\n{question}\n\nRank the chunks by relevance, most relevant first. Include every chunk id exactly once.\n\nChunks:\n"
    ]
    for index, chunk in enumerate(chunks, start=1):
        prompt_parts.append(f"# CHUNK ID {index}:\n{chunk.page_content}\n")
    messages = [
        {
            "role": "system",
            "content": "You are a document re-ranker. Reply only with the reranked list of chunk ids.",
        },
        {"role": "user", "content": "\n".join(prompt_parts)},
    ]
    try:
        order = chat_structured(messages, RankOrder, model=RERANK_MODEL).order
        reranked = [chunks[index - 1] for index in order if 1 <= index <= len(chunks)]
        missing = [
            chunk
            for index, chunk in enumerate(chunks, start=1)
            if index not in set(order)
        ]
        return reranked + missing if reranked else chunks
    except Exception:
        return chunks


def fetch_context(question: str, k: int = FINAL_K) -> list[Document]:
    vectorstore = load_vectorstore()
    rewritten = rewrite_query(question)
    primary = vectorstore.similarity_search(question, k=RETRIEVAL_K)
    expanded = vectorstore.similarity_search(rewritten, k=RETRIEVAL_K)
    return rerank(question, merge_chunks(primary, expanded))[:k]


def answer_question(
    question: str,
    history: list[dict] | None = None,
) -> tuple[str, list[Document]]:
    docs = fetch_context(question, FINAL_K)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=format_context(docs)),
        }
    ]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": question})
    model = chat_model()
    try:
        return chat(messages, model=model), docs
    except Exception as exc:
        return (
            f"Chat model {model!r} is unavailable ({exc}). Retrieved context is still available.",
            docs,
        )
