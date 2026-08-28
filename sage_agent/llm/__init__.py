"""LLM backends for SAGE-Agent.

This module provides various LLM client implementations that satisfy
the LLMClient protocol for use with SAGE-Agent.

Available clients:
    - OpenRouterClient: OpenRouter API (multi-provider)
    - TTSServiceClient: Test-Time Scaling service with uncertainty

Example:
    ```python
    from sage_agent.llm import OpenRouterClient, LLMClient

    # Using OpenRouter
    client = OpenRouterClient(model="openai/gpt-4-turbo")

    # Use with SAGE generators
    from sage_agent import LLMBackedCandidateGenerator
    generator = LLMBackedCandidateGenerator(client)
    ```
"""

from .llm import (
    LLMClient,
    PromptCandidateGenerator,
    PromptQuestionGenerator,
)

# Optional imports - only available if dependencies are installed
try:
    from .openrouter import OpenRouterClient
except ImportError:
    OpenRouterClient = None  # type: ignore

try:
    from .tts_service import TTSServiceClient
except ImportError:
    TTSServiceClient = None  # type: ignore

__all__ = [
    "LLMClient",
    "PromptCandidateGenerator",
    "PromptQuestionGenerator",
    "OpenRouterClient",
    "TTSServiceClient",
]
