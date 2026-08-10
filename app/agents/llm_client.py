"""Claude API client for SQL generation.

A small Protocol, not the Anthropic SDK, is what text_to_sql_agent.py
depends on -- that keeps the orchestration layer testable with a fake
client and swappable to another provider later.
"""
from __future__ import annotations

import os
from typing import Protocol

DEFAULT_MODEL = "claude-sonnet-5"


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class ClaudeLLMClient:
    """Generates text via the Anthropic Messages API.

    Requires ANTHROPIC_API_KEY (from the environment or a .env file at the
    project root). The key is only read on first actual call, not at
    construction or import time, so building an agent pipeline never
    requires a key until you run it live.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        from dotenv import load_dotenv

        load_dotenv()

        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to a .env file at the project "
                "root (see .env.example) or export it before running the agent."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
