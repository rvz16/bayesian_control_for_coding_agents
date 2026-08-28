"""Shared types and helpers for live fitted-controller runs.

The adapters in this package deliberately reuse the existing calibration
scripts' loaders, prompts, critics, and verifiers.  Their job is only to
present a common interface to the live controller loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


CRITIC_FIELDS = {
    "L0": "L0_syntax",
    "L2": "L2_public_tests",
    "L3": "L3_llm_review",
}


@dataclass
class Candidate:
    """A generated candidate solution or patch."""

    payload: str
    raw_text: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticResult:
    passed: bool | None
    detail: str = ""
    api_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class VerifyResult:
    passed: bool
    detail: str = ""


class BenchmarkAdapter(Protocol):
    """Protocol implemented by benchmark-specific live adapters."""

    benchmark: str

    def load_instances(self) -> list[dict]:
        ...

    def instance_id(self, instance: dict) -> str:
        ...

    def build_prompt(
        self,
        instance: dict,
        previous: Candidate | None,
        action_log: list[dict[str, Any]],
    ) -> str:
        ...

    def extract_candidate(self, instance: dict, response_text: str) -> Candidate:
        ...

    def run_critic(
        self,
        critic: str,
        instance: dict,
        candidate: Candidate,
        reviewer_client,
    ) -> CriticResult:
        ...

    def verify(
        self,
        instance: dict,
        candidate: Candidate,
        run_id: str,
    ) -> VerifyResult:
        ...


def safe_stem(value: str, max_len: int = 120) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    if not stem:
        stem = "item"
    return stem[:max_len]


def feedback_block(previous: Candidate | None, action_log: list[dict[str, Any]]) -> str:
    """Compact feedback appended to regeneration prompts."""
    if previous is None:
        return ""
    recent = action_log[-8:]
    lines = []
    for rec in recent:
        action = rec.get("action")
        if action in {"L0", "L2", "L3"}:
            status = "PASS" if rec.get("passed") else "FAIL"
            detail = rec.get("detail") or ""
            lines.append(f"- {action}: {status} {str(detail)[:300]}".rstrip())
        elif action == "verify":
            status = "PASS" if rec.get("passed") else "FAIL"
            lines.append(f"- verifier: {status} {str(rec.get('detail') or '')[:300]}".rstrip())
        elif action == "generate":
            lines.append("- generated a replacement candidate")
    body = "\n".join(lines) if lines else "- no diagnostics recorded"
    if previous.kind == "diff":
        prev = f"```diff\n{previous.payload[:8000]}\n```"
    else:
        prev = f"```python\n{previous.payload[:8000]}\n```"
    return (
        "\n\n## Feedback from the previous attempt\n"
        f"{body}\n\n"
        "Previous candidate:\n"
        f"{prev}\n\n"
        "Generate a new candidate that addresses the failures above.\n"
    )


def load_jsonl_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    import json

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        inst = rec.get("instance_id")
        pol = rec.get("policy")
        if inst and pol:
            keys.add((str(inst), str(pol)))
    return keys

