import os
from typing import Any

from llm_axe.models import OllamaChat
from llm_axe.va4_product_discoverer import DEFAULT_MODEL_QA, ask_program_question


def make_ollama(model: str, timeout_seconds: float | None = None) -> OllamaChat:
    return OllamaChat(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model=model,
        timeout=timeout_seconds,
    )


def answer_program_question(raw: dict[str, Any], message: str, model: str | None = None) -> tuple[str, bool, str | None]:
    model_name = (model or os.getenv("OLLAMA_QA_MODEL") or os.getenv("OLLAMA_USER_MODEL") or DEFAULT_MODEL_QA).strip()
    classification = raw.get("classification")
    if not isinstance(classification, dict):
        classification = {
            "primary_category": "other",
            "confidence": 0.0,
            "is_relevant": False,
            "secondary_categories": [],
            "key_features": [],
        }

    try:
        answer = ask_program_question(
            make_ollama(model_name),
            raw,
            classification,
            message,
            temperature=0.3,
            num_ctx=8192,
        )
        return answer.strip(), True, model_name
    except Exception as exc:
        return f"Δεν μπόρεσα να πάρω απάντηση από το QA model ({model_name}): {exc}", False, model_name


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if not value:
        return default
    if value.strip().casefold() in {"none", "null", "off", "false", "0"}:
        return None
    try:
        return float(value)
    except ValueError:
        return default
