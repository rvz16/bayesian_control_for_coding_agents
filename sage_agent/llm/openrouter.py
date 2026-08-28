"""OpenRouter LLM client for SAGE-Agent.

This module provides an LLMClient implementation using OpenRouter's
OpenAI-compatible API endpoint.

Example:
    ```python
    from sage_agent.llm import OpenRouterClient

    client = OpenRouterClient(
        model="openai/gpt-4-turbo",
        api_key="your-key",  # or set OPENROUTER_API_KEY env var
    )

    response = client.complete("What is 2 + 2?")
    ```
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from .llm import LLMClient


@dataclass
class OpenRouterClient:
    """LLM client using OpenRouter API.

    OpenRouter provides access to multiple LLM providers through a
    unified OpenAI-compatible API.

    Attributes:
        model: Model identifier (e.g., "openai/gpt-4-turbo", "anthropic/claude-3-opus")
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        base_url: OpenRouter API base URL
        verbose: Whether to print prompts and responses

    Environment Variables:
        SAGE_OPENROUTER_API_KEY: Primary API key lookup
        OPENROUTER_API_KEY: Fallback API key lookup
    """
    model: str
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    verbose: bool = False

    def __post_init__(self) -> None:
        """Initialize the OpenAI client with OpenRouter configuration."""
        api_key = (
            self.api_key
            or os.getenv("SAGE_OPENROUTER_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "OpenRouter API key not set. "
                "Set SAGE_OPENROUTER_API_KEY or OPENROUTER_API_KEY environment variable, "
                "or pass api_key to constructor."
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url.rstrip("/"),
        )

    def complete(self, prompt: str) -> str:
        """Complete a prompt using OpenRouter.

        Args:
            prompt: The prompt to complete

        Returns:
            The model's response text
        """
        if self.verbose:
            print(f"\n--- OpenRouter prompt ({self.model}) ---\n{prompt}\n")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""

        if self.verbose:
            print(f"--- OpenRouter response ({self.model}) ---\n{content}\n")

        return content


# Ensure it implements the protocol
_: LLMClient = OpenRouterClient(model="test", api_key="test")  # type: ignore
