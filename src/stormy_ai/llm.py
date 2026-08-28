"""Construct the chat model from Stormy AI settings."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from stormy_ai.config import LLMConfig, get_settings


def huggingface_model_id(model: str, inference_provider: str) -> str:
    """Build the HF router model ID with an Inference Provider suffix."""

    # Model IDs are org/name; a trailing :provider routes via the HF router.
    leaf = model.rsplit("/", 1)[-1]
    base = model.rsplit(":", 1)[0] if ":" in leaf else model
    return f"{base}:{inference_provider}"


def create_chat_model(llm: LLMConfig | None = None) -> BaseChatModel:
    """Build a LangChain chat model for the configured LLM provider."""

    config = llm or get_settings().llm

    if config.provider == "ollama":
        return ChatOllama(
            model=config.model,
            base_url=config.ollama.base_url,
            temperature=config.temperature,
        )

    if config.provider == "huggingface":
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise ValueError(
                "HF_TOKEN is required when llm.provider is 'huggingface'. "
                "Add it to .env or your environment."
            )

        model_id = huggingface_model_id(
            config.model,
            config.huggingface.inference_provider,
        )
        return ChatOpenAI(
            model=model_id,
            base_url=config.huggingface.base_url,
            api_key=api_key,
            temperature=config.temperature,
        )

    raise ValueError(f"Unsupported llm.provider: {config.provider!r}")
