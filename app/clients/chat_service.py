"""Customer-facing chat service.

Uses the next-generation OpenAI flagship model for the public chat endpoint.
The model is pinned via an explicit `model="gpt-5.4"` argument so that the
Xygeni AI Inventory detector picks it up as an AIModel asset.
"""

from __future__ import annotations

import os

from openai import OpenAI

CHAT_MODEL = "gpt-5.4"


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def reply(user_msg: str) -> str:
    """Return the assistant reply for the given user message."""
    client = _client()
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": "You are a helpful support assistant."},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content