from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast


def _load_app_env() -> None:
    """Load ``app/.env`` (KEY=VALUE lines) so config travels with the app.

    Keeps the app self-contained and independent of the parent repo. Real
    environment variables take precedence (``setdefault``), so a value set in
    the shell always wins over the bundled default.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_app_env()


class EmbeddingModelConfig(TypedDict):
    model: str
    api_key: str | None
    base_url: str | None
    is_paid_model: bool
    is_openai_model: bool
    usd_per_1m_tokens: NotRequired[float | None]


class LlmModelConfig(TypedDict):
    model: str
    api_key: str | None
    base_url: str | None
    is_paid_model: bool
    is_openai_model: bool
    supports_temperature: NotRequired[bool]


class SelectedPipelineConfig(TypedDict):
    # Only these two vary per pipeline; the rest are shared constants (see below).
    embedding_model: EmbeddingModelConfig
    chat_model: LlmModelConfig


EmbeddingModelName = Literal[
    "all-minilm:l6-v2",
    "qwen3-embedding:latest",
    "bge-large:latest",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

PipelineModelType = Literal[
    "embedding_model",
    "chat_model",
    "preprocess_model",
    "rewrite_model",
    "rerank_model",
    "judge_model",
]

DEFAULT_EMBEDDING_MODEL: EmbeddingModelName = "all-minilm:l6-v2"


def ollama_base_url() -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if host.endswith("/v1"):
        return host
    return f"{host}/v1"


EMBEDDING_MODELS: dict[EmbeddingModelName, EmbeddingModelConfig] = {
    "all-minilm:l6-v2": {
        "model": "all-minilm:l6-v2",
        "api_key": "ollama",
        "base_url": ollama_base_url(),
        "is_paid_model": False,
        "is_openai_model": False,
    },
    "qwen3-embedding:latest": {
        "model": "qwen3-embedding:latest",
        "api_key": "ollama",
        "base_url": ollama_base_url(),
        "is_paid_model": False,
        "is_openai_model": False,
    },
    "bge-large:latest": {
        "model": "bge-large:latest",
        "api_key": "ollama",
        "base_url": ollama_base_url(),
        "is_paid_model": False,
        "is_openai_model": False,
    },
    "text-embedding-3-small": {
        "model": "text-embedding-3-small",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": "https://api.openai.com/v1",
        "is_paid_model": True,
        "is_openai_model": True,
        "usd_per_1m_tokens": 0.02,
    },
    "text-embedding-3-large": {
        "model": "text-embedding-3-large",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": "https://api.openai.com/v1",
        "is_paid_model": True,
        "is_openai_model": True,
        "usd_per_1m_tokens": 0.13,
    },
}

LLM_MODELS: dict[str, LlmModelConfig] = {
    "deepseek-r1:1.5b": {
        "model": "deepseek-r1:1.5b",
        "api_key": "ollama",
        "base_url": ollama_base_url(),
        "is_paid_model": False,
        "is_openai_model": False,
    },
    "gpt-oss:20b": {
        "model": "gpt-oss:20b",
        "api_key": "ollama",
        "base_url": ollama_base_url(),
        "is_paid_model": False,
        "is_openai_model": False,
    },
    "gemma4:e4b": {
        "model": "gemma4:e4b",
        "api_key": "ollama",
        "base_url": ollama_base_url(),
        "is_paid_model": False,
        "is_openai_model": False,
    },
}

# Shared across every pipeline — these never vary between basic and advanced.
# Only embedding_model and chat_model differ per pipeline (see SELECTED_MODELS).
PREPROCESS_MODEL = LLM_MODELS["gpt-oss:20b"]["model"]
REWRITE_MODEL = LLM_MODELS["gemma4:e4b"]["model"]
JUDGE_MODEL = LLM_MODELS["gemma4:e4b"]["model"]

# The advanced pipeline's re-ranker: a local cross-encoder, run in-process via
# FlagEmbedding (downloaded from the Hugging Face Hub on first use; see
# app/advanced/reranker.py). Override per run with rerank=<repo-id>.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


SELECTED_MODELS: dict[str, SelectedPipelineConfig] = {
    "basic": {
        "embedding_model": EMBEDDING_MODELS["bge-large:latest"],
        "chat_model": LLM_MODELS["gpt-oss:20b"],
    },
    "advanced": {
        "embedding_model": EMBEDDING_MODELS["bge-large:latest"],
        "chat_model": LLM_MODELS["gpt-oss:20b"],
    },
}


# --------------------------------------------------------------------------------------
# Runtime model overrides (CLI params take priority over SELECTED_MODELS)
# --------------------------------------------------------------------------------------

# pipeline -> {model_type: model_name}, populated from CLI `key=value` args.
_MODEL_OVERRIDES: dict[str, dict[str, str]] = {}

# Accepted CLI keys (and friendly aliases) mapped to their PipelineModelType.
# Only embedding_model and chat_model are overridable; the rest are constants.
CLI_MODEL_KEYS: dict[str, PipelineModelType] = {
    "embedding": "embedding_model",
    "embedding_model": "embedding_model",
    "chat": "chat_model",
    "chat_model": "chat_model",
    # Advanced-only: the re-ranker (the bge cross-encoder id, or an LLM_MODELS tag).
    "rerank": "rerank_model",
    "rerank_model": "rerank_model",
}


def set_model_overrides(pipeline: str, overrides: dict[str, str | None]) -> None:
    """Register CLI model overrides for ``pipeline`` (empty values are ignored)."""
    cleaned = {key: value for key, value in overrides.items() if value}
    if cleaned:
        _MODEL_OVERRIDES.setdefault(pipeline, {}).update(cleaned)


def model_override(pipeline: str, model_type: PipelineModelType) -> str | None:
    return _MODEL_OVERRIDES.get(pipeline, {}).get(model_type)


def clear_model_overrides(pipeline: str | None = None) -> None:
    """Drop registered overrides (for one pipeline, or all).

    Needed when several runs share a process (e.g. the multirun runner) so one
    run's overrides don't leak into the next.
    """
    if pipeline is None:
        _MODEL_OVERRIDES.clear()
    else:
        _MODEL_OVERRIDES.pop(pipeline, None)


def _override_token(arg: str) -> tuple[PipelineModelType, str] | None:
    """Parse one ``key=value`` token into ``(model_type, value)`` or ``None``.

    Accepts both ``embedding_model=...`` and ``--embedding_model=...`` styles.
    """
    key, separator, value = arg.partition("=")
    if not separator:
        return None
    model_type = CLI_MODEL_KEYS.get(key.strip().lstrip("-").lower())
    if model_type is None:
        return None
    value = value.strip().strip('"').strip("'")
    return (model_type, value) if value else None


def parse_model_overrides(args: Iterable[str]) -> dict[str, str]:
    """Parse ``key=value`` model overrides, erroring on unknown keys.

    Use for entrypoints (ingest/chunking) whose only CLI args are model overrides.
    """
    overrides: dict[str, str] = {}
    for arg in args:
        parsed = _override_token(arg)
        if parsed is None:
            raise ValueError(
                f"Unrecognized argument {arg!r}. Expected key=value with key in "
                f"{sorted(set(CLI_MODEL_KEYS))}."
            )
        overrides[parsed[0]] = parsed[1]
    return overrides


def split_model_overrides(args: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    """Split argv into (model overrides, remaining args) for mixed CLIs (evaluator)."""
    overrides: dict[str, str] = {}
    leftover: list[str] = []
    for arg in args:
        parsed = _override_token(arg)
        if parsed is None:
            leftover.append(arg)
        else:
            overrides[parsed[0]] = parsed[1]
    return overrides, leftover


def selected_config(pipeline: str) -> SelectedPipelineConfig:
    try:
        return SELECTED_MODELS[pipeline]
    except KeyError as exc:
        raise ValueError(
            f"Unknown pipeline {pipeline!r}. Choose from {sorted(SELECTED_MODELS)}."
        ) from exc


def selected_rerank_model(pipeline: str) -> str:
    """Resolve the re-ranker for ``pipeline`` (CLI override → ``RERANK_MODEL``)."""
    return model_override(pipeline, "rerank_model") or RERANK_MODEL


def selected_pipeline_models(pipeline: str) -> dict[PipelineModelType, str | None]:
    return {
        "embedding_model": selected_embedding_model(pipeline),
        "chat_model": selected_chat_model(pipeline),
        "preprocess_model": PREPROCESS_MODEL,
        "rewrite_model": REWRITE_MODEL,
        "rerank_model": selected_rerank_model(pipeline),
        "judge_model": JUDGE_MODEL,
    }


def selected_embedding_model(pipeline: str) -> EmbeddingModelName:
    name = model_override(pipeline, "embedding_model") or (
        selected_config(pipeline)["embedding_model"]["model"]
    )
    return cast(EmbeddingModelName, name)


def selected_chat_model(pipeline: str) -> str:
    return model_override(pipeline, "chat_model") or (
        selected_config(pipeline)["chat_model"]["model"]
    )


def embedding_config(
    model: str | None = None, *, pipeline: str = "basic"
) -> EmbeddingModelConfig:
    model = model or selected_embedding_model(pipeline)
    config = EMBEDDING_MODELS.get(cast(EmbeddingModelName, model))
    if config is not None:
        if config["is_openai_model"]:
            return {
                **config,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": "https://api.openai.com/v1",
            }
        return config
    return {
        "model": model,
        "api_key": "ollama",
        "base_url": ollama_base_url(),
        "is_paid_model": False,
        "is_openai_model": False,
    }


def chat_model_config(model: str) -> LlmModelConfig:
    for alias, config in LLM_MODELS.items():
        if model in {alias, config["model"]}:
            if config["is_openai_model"]:
                return {
                    **config,
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": "https://api.openai.com/v1",
                }
            return config
    if model.startswith("gpt-") or model.startswith("o"):
        return {
            "model": model,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
            "is_paid_model": True,
            "is_openai_model": True,
            "supports_temperature": not model.startswith("gpt-5"),
        }
    return {
        "model": model,
        "api_key": "ollama",
        "base_url": ollama_base_url(),
        "is_paid_model": False,
        "is_openai_model": False,
    }
