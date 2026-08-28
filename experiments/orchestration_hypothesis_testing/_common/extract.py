"""Extract Python code from a freeform LLM response.

LLMs typically wrap code in ```python ... ``` fences. This helper tries
the fenced form first, falling back to the raw response if no fence is
present.
"""
from __future__ import annotations

import re


def extract_code(response: str) -> str:
    """Pull the first Python code block out of a model response.

    Tries fenced ```python ... ``` (or unlabeled ``` ... ```) blocks
    first; falls back to the stripped raw response if no fence is found.
    """
    if not response:
        return ""
    m = re.search(r"```(?:python)?\s*\n([\s\S]+?)```", response)
    if m:
        return m.group(1).strip()
    return response.strip()
