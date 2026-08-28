"""Test-Time Scaling (TTS) LLM client for SAGE-Agent.

This module provides an LLMClient implementation that integrates with
llm-tts service for uncertainty-aware inference using test-time scaling
strategies like self-consistency.

The client exposes uncertainty and confidence scores from the TTS service,
which can be used by SAGE-Agent for enhanced uncertainty quantification.

Example:
    ```python
    from sage_agent.llm import TTSServiceClient

    client = TTSServiceClient(
        base_url="http://localhost:8000",
        model="gpt-4",
        tts_strategy="self_consistency",
        tts_budget=8,
    )

    response = client.complete("What is the capital of France?")
    print(f"Response: {response}")
    print(f"Uncertainty: {client.last_uncertainty}")
    print(f"Confidence: {client.last_confidence}")
    ```

Note:
    Requires the llm-tts package: pip install llm-tts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .llm import LLMClient


@dataclass
class TTSServiceClient:
    """LLM client with Test-Time Scaling uncertainty quantification.

    Integrates with llm-tts service to provide uncertainty scores
    alongside LLM responses. These scores can be used by SAGE-Agent
    to weight candidate probabilities and make better decisions.

    Attributes:
        base_url: URL of the llm-tts service
        model: Model identifier to use
        tts_strategy: TTS strategy (e.g., "self_consistency", "best_of_n")
        tts_budget: Number of samples for TTS (default: 8)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: 4096)
        timeout: Request timeout in seconds (default: 120.0)
        system_prompt: System prompt for the model
        verbose: Whether to print debug information

    Properties:
        last_uncertainty: Uncertainty score from last response [0, 1]
        last_confidence: Confidence/consensus score from last response [0, 1]
    """
    base_url: str
    model: str
    tts_strategy: str = "self_consistency"
    tts_budget: int = 8
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0
    system_prompt: str = (
        "You are a precise assistant. "
        "Respond with one line only in this exact format: Answer: <final answer>."
    )
    verbose: bool = False

    # Internal state
    _llm: Any = field(default=None, repr=False)
    _last_metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Initialize the TTS LLM client."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from llm_tts.integration import ChatTTS
        except ImportError as e:
            raise ImportError(
                "llm-tts and langchain-core are required for TTSServiceClient. "
                "Install with: pip install llm-tts langchain-core"
            ) from e

        self._llm = ChatTTS(
            base_url=self.base_url,
            model=self.model,
            tts_strategy=self.tts_strategy,
            tts_budget=self.tts_budget,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        self._last_metadata = {}

    def complete(self, prompt: str) -> str:
        """Complete a prompt using TTS service.

        Args:
            prompt: The prompt to complete

        Returns:
            The model's response text

        Side Effects:
            Updates last_uncertainty and last_confidence properties
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]

        if self.verbose:
            print(f"\n--- TTS prompt ({self.model}) ---\n{prompt}\n")

        response = self._llm.invoke(messages)
        self._last_metadata = response.response_metadata.get("tts_metadata", {})

        content = response.content
        if self.verbose:
            print(f"--- TTS response ({self.model}) ---\n{content}\n")
            if self._last_metadata:
                print(f"--- TTS metadata ---\n{self._last_metadata}\n")

        return content

    @property
    def last_uncertainty(self) -> Optional[float]:
        """Get uncertainty score from last response.

        Returns:
            Uncertainty score in [0, 1], or None if not available.
            Higher values indicate more uncertainty.
        """
        if not self._last_metadata:
            return None
        return self._last_metadata.get("uncertainty_score")

    @property
    def last_confidence(self) -> Optional[float]:
        """Get confidence/consensus score from last response.

        Returns:
            Confidence score in [0, 1], or None if not available.
            Higher values indicate more agreement across samples.
        """
        if not self._last_metadata:
            return None
        return self._last_metadata.get("consensus_score")

    @property
    def last_metadata(self) -> Dict[str, Any]:
        """Get full metadata from last response.

        Returns:
            Dictionary with all TTS metadata (uncertainty, consensus, etc.)
        """
        return dict(self._last_metadata)
