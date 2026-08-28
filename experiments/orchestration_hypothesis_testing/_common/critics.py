"""Language-agnostic Python critics (L0 syntax, L1 lint, L3 LLM review).

These critics take a Python `code` string and return a bool (or for L3,
a (bool, cost_usd) tuple). They are independent of benchmark format -- a
critic doesn't know if it's reviewing an LCB solution or a MBPP function.

Originally lived inside lcb_calibrate.py; extracted here so every
calibration/iter pipeline shares one implementation.
"""
from __future__ import annotations

import ast
import logging
import os
import subprocess
import tempfile

log = logging.getLogger(__name__)


def critic_L0_syntax(code: str) -> bool:
    """L0: parses cleanly under ast.parse(). Essentially free."""
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def critic_L1_lint(code: str) -> bool:
    """L1: ruff with conservative ruleset (F821, F811, E999).

    Same ruleset as the SWE-bench Lite production critic. If ruff isn't
    installed, this returns True (don't fail-closed on missing tooling).
    """
    if not code.strip():
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        proc = subprocess.run(
            ["ruff", "check", "--quiet", "--no-cache", "--select", "F821,F811,E999", tmp],
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True
    finally:
        os.unlink(tmp)


def critic_L3_review(problem: str, code: str, client) -> tuple[bool, float]:
    """L3: small-model PASS/FAIL judgment on (problem, code).

    Default reviewer model is claude-haiku-4.5. Caller passes an
    OpenAI-compatible client (typically the OpenRouter client). Returns
    (passed, cost_usd); on any API exception returns (False, 0.0) and
    logs a warning.
    """
    prompt = (
        "You are a senior software engineer reviewing a code submission.\n\n"
        f"## Problem\n{problem[:3000]}\n\n"
        f"## Submitted code\n```python\n{code[:6000]}\n```\n\n"
        "Does this code correctly solve the problem? Respond with exactly one word: "
        "PASS or FAIL. No explanation."
    )
    try:
        resp = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10,
        )
        text = resp.choices[0].message.content.strip().upper()
        usage = resp.usage
        cost = (usage.prompt_tokens / 1_000_000) * 1.0 + (usage.completion_tokens / 1_000_000) * 5.0
        return ("PASS" in text and "FAIL" not in text), cost
    except Exception as e:
        log.warning("L3 failed: %s", e)
        return False, 0.0
