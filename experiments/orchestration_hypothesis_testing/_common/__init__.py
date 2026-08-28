"""Shared helpers used by every calibration / iter / analysis pipeline.

This package is the single source of truth for:
  - Generator registry + client factory  (`generators` submodule)
  - Python critics L0/L1/L3              (`critics` submodule)
  - Code extraction from LLM output      (`extract` submodule)
  - Cost accounting + CostTracker        (`cost` submodule)

Convenience re-exports below let callers do `from _common import X` for
the most commonly used symbols, or `from _common.<sub> import ...` for
the full surface of any submodule.
"""
from __future__ import annotations

from .generators import (
    GENERATORS,
    canonical_generator_key,
    _make_client,
    _load_openrouter_key,
    OPENROUTER_KEY_NAMES,
)
from .critics import critic_L0_syntax, critic_L1_lint, critic_L3_review
from .extract import extract_code
from .cost import (
    cost_for_call,
    CostTracker,
    extract_usage,
    project_cost,
)

__all__ = [
    # generators
    "GENERATORS", "canonical_generator_key", "_make_client",
    "_load_openrouter_key", "OPENROUTER_KEY_NAMES",
    # critics
    "critic_L0_syntax", "critic_L1_lint", "critic_L3_review",
    # extract
    "extract_code",
    # cost
    "cost_for_call", "CostTracker", "extract_usage", "project_cost",
]
