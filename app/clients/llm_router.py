"""Central LLM client.

Every other module in `app/` must obtain its LLM through `route_chat()` /
`route_embedding()` so that:

- Approved provider/model pairs are enforced in a single place.
- The Xygeni AI Inventory detectors can pick up framework + model evidence
  from one well-known module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from anthropic import Anthropic
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


Provider = Literal["openai", "anthropic", "azure_openai"]


@dataclass(frozen=True)
class ModelChoice:
    provider: Provider
    model: str
    temperature: float = 0.2


# --- approved chat models ----------------------------------------------------
PRIMARY_CHAT = ModelChoice(provider="openai", model="gpt-4o-mini")
FALLBACK_CHAT = ModelChoice(provider="anthropic", model="claude-3-5-sonnet-20241022")
SENSITIVE_CHAT = ModelChoice(provider="azure_openai", model="gpt-4o", temperature=0.0)

# --- approved embedding model ------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"


def _openai_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    )


def _anthropic_client() -> Anthropic:
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def route_chat(choice: ModelChoice = PRIMARY_CHAT):
    """Return a configured chat client for the given approved model."""
    if choice.provider == "openai":
        return ChatOpenAI(model=choice.model, temperature=choice.temperature)
    if choice.provider == "anthropic":
        return ChatAnthropic(model=choice.model, temperature=choice.temperature)
    if choice.provider == "azure_openai":
        # Azure OpenAI uses deployment names instead of model names; both are
        # surfaced so the inventory detector picks up `deployment` evidence.
        return ChatOpenAI(
            model=choice.model,
            deployment_name="support-prod-gpt-4o",
            temperature=choice.temperature,
        )
    raise ValueError(f"Unsupported provider: {choice.provider}")


def route_embedding() -> tuple[OpenAI, str]:
    """Return a raw OpenAI client and the approved embedding model id."""
    return _openai_client(), EMBEDDING_MODEL


def transcribe(audio_path: str) -> str:
    """Transcribe a customer voicemail using Whisper."""
    client = _openai_client()
    model = "whisper-1"
    with open(audio_path, "rb") as fh:
        return client.audio.transcriptions.create(model=model, file=fh).text
