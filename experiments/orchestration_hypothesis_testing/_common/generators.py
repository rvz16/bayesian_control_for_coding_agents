"""Generator registry + OpenAI-compatible client factory.

Single source of truth for every generator we calibrate against. Each
generator entry is a 3-tuple (model_id, human_label, base_url):
  - model_id   passed to chat.completions.create()
  - human_label friendly description for logs
  - base_url   None -> OpenRouter; non-None -> local vLLM endpoint

Originally lived inside lcb_calibrate.py; extracted here so every
calibration/iter/analysis module imports from one place.
"""
from __future__ import annotations

import os
from typing import Optional


GENERATORS: dict[str, tuple[str, str, Optional[str]]] = {
    "gpt5_mini":   ("openai/gpt-5-mini",                "OpenAI gpt-5-mini",                          None),
    "qwen3_coder": ("qwen/qwen3-coder",                 "Qwen3 Coder",                                None),
    "haiku45":     ("anthropic/claude-haiku-4.5",       "Claude Haiku 4.5",                           None),
    "sonnet45":    ("anthropic/claude-sonnet-4.5",      "Claude Sonnet 4.5",                          None),
    "qwen25_7b":   ("Qwen/Qwen2.5-Coder-7B-Instruct",   "Qwen2.5-Coder-7B (open-weight, local vLLM)", "http://127.0.0.1:8001/v1"),
    "qwen25_32b":  ("Qwen/Qwen2.5-Coder-32B-Instruct",  "Qwen2.5-Coder-32B (open-weight, local vLLM)", "http://127.0.0.1:8003/v1"),
    "gpt_oss_20b": ("openai/gpt-oss-20b:free",          "gpt-oss-20b",                                None),
}


def canonical_generator_key(raw: str) -> str:
    """Map CLI aliases (e.g. hyphens) to keys in GENERATORS."""
    g = raw.strip()
    if not g:
        raise ValueError("empty generator key")
    if g in GENERATORS:
        return g
    alt = g.replace("-", "_")
    if alt in GENERATORS:
        return alt
    raise SystemExit(
        f"unknown generator {raw!r}; known: {', '.join(sorted(GENERATORS))}"
    )


OPENROUTER_KEY_NAMES = ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY", "OPEN_ROUTER")


def _load_openrouter_key() -> str:
    for key_name in OPENROUTER_KEY_NAMES:
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    raise SystemExit(
        "OpenRouter API key not set. Expected one of: "
        "OPENROUTER_API_KEY, OPEN_ROUTER_API_KEY, OPEN_ROUTER."
    )


def _make_client(generator_key: str | None = None):
    """Build an OpenAI-compatible client.

    For qwen25_* (and any other entry with a non-None base_url slot):
    local vLLM at base_url stored in GENERATORS tuple slot 2.
    Otherwise: OpenRouter.
    """
    from openai import OpenAI
    base_url = None
    if generator_key and generator_key in GENERATORS and len(GENERATORS[generator_key]) >= 3:
        base_url = GENERATORS[generator_key][2]
    if base_url:
        return OpenAI(api_key="EMPTY", base_url=base_url)
    return OpenAI(api_key=_load_openrouter_key(), base_url="https://openrouter.ai/api/v1")
