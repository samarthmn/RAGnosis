from __future__ import annotations

from langchain_core.documents import Document

from app.common.chat import chat
from app.common.ingest import db_path_for
from app.common.models import selected_chat_model
from app.common.rag import SYSTEM_PROMPT, format_context, history_to_messages
from app.common.rag import load_vectorstore as load_common_vectorstore

RETRIEVAL_K = 8


def load_vectorstore():
    return load_common_vectorstore(db_path_for("basic"), pipeline_name="basic")


def combined_question(question: str, history: list[dict] | None = None) -> str:
    prior = "\n".join(
        str(message.get("content", ""))
        for message in history or []
        if message.get("role") == "user" and message.get("content")
    )
    return f"{prior}\n{question}".strip() if prior else question


def fetch_context(question: str, k: int = RETRIEVAL_K) -> list[Document]:
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search(question, k=k)


def answer_question(
    question: str,
    history: list[dict] | None = None,
) -> tuple[str, list[Document]]:
    docs = fetch_context(combined_question(question, history), RETRIEVAL_K)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=format_context(docs))}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": question})
    model = selected_chat_model("basic")
    try:
        return chat(messages, model=model), docs
    except Exception as exc:
        return (
            f"Chat model {model!r} is unavailable ({exc}). Retrieved context is still available.",
            docs,
        )
