from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.common.models import chat_model_config

TModel = TypeVar("TModel", bound=BaseModel)


def json_instruction(schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True)
    return (
        "Respond with only one valid JSON object. Do not include markdown, prose, "
        f"or code fences. The JSON object must match this schema: {schema_json}"
    )


def messages_with_json_instruction(
    messages: list[dict[str, str]],
    schema: type[BaseModel],
) -> list[dict[str, str]]:
    return [*messages, {"role": "system", "content": json_instruction(schema)}]


def temperature_kwargs(model: str, temperature: float) -> dict[str, float]:
    if not chat_model_config(model).get("supports_temperature", True):
        return {}
    return {"temperature": temperature}


def chat_client(model: str) -> OpenAI:
    config = chat_model_config(model)
    if not config["api_key"]:
        raise RuntimeError(f"No API key configured for {config['model']!r}")
    return OpenAI(api_key=config["api_key"], base_url=config["base_url"])


def chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
) -> str:
    response = chat_client(model).chat.completions.create(
        model=model,
        messages=messages,
        **temperature_kwargs(model, temperature),
    )
    return response.choices[0].message.content or ""


def strip_reasoning(text: str) -> str:
    """Drop ``<think>...</think>`` blocks emitted by reasoning models (e.g. deepseek-r1).

    These blocks often contain stray braces that derail naive JSON extraction.
    """
    return re.sub(
        r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    )


def _balanced_json_objects(text: str):
    """Yield candidate ``{...}`` substrings with balanced, string-aware braces."""
    depth = in_string = escape = 0
    start = -1
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = 0
            elif char == "\\":
                escape = 1
            elif char == '"':
                in_string = 0
            continue
        if char == '"':
            in_string = 1
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start : index + 1]


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = strip_reasoning(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for candidate in _balanced_json_objects(text):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def chat_structured(
    messages: list[dict[str, str]],
    schema: type[TModel],
    *,
    model: str,
    temperature: float = 0.0,
) -> TModel:
    client = chat_client(model)
    json_messages = messages_with_json_instruction(messages, schema)
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=json_messages,
            response_format=schema,
            **temperature_kwargs(model, temperature),
        )
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed
    except Exception:
        pass

    errors: list[str] = []
    for kwargs in (
        {"response_format": {"type": "json_object"}},
        {},
    ):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=json_messages,
                **kwargs,
                **temperature_kwargs(model, temperature),
            )
            content = response.choices[0].message.content or ""
            data = extract_json_object(content)
            if data is None:
                raise ValueError(
                    f"Could not parse structured response as JSON: {content[:200]!r}"
                )
            return schema.model_validate(data)
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError("Structured response failed: " + " | ".join(errors))
