"""Cost accounting: per-call price model + thread-safe running tracker.

Two pieces:
  - cost_for_call(model_id, prompt_tokens, completion_tokens) -> USD
    Returns the approximate cost in USD given the model identifier and
    token counts. Open-weight local models return 0.

  - CostTracker (dataclass)
    Per-generator running cost ledger with a hard cap. Thread-safe;
    workers call .record() after every paid call; tracker flips .capped
    when total_usd >= cap_usd.

  - extract_usage(resp) + project_cost(probe, n, probe_n)
    Helpers used by spot-check scripts.

Originally lived in scripts/lcb_calibrate.py (cost_for_call) and
scripts/cost_tracker.py (CostTracker + extract_usage + project_cost);
consolidated here so callers only need one import.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------- Per-call price model ----------

def cost_for_call(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Approximate per-call cost (USD).

    Per-model pricing is hardcoded from OpenRouter rates as of Q2 2025.
    Open-weight local models (Qwen2.5-Coder, gpt-oss-20b) return 0.
    Unknown model_id falls back to the Claude Haiku rate (1/5 USD per 1M
    prompt/completion tokens) as a conservative default.
    """
    if "gpt-5-mini" in model_id:
        return (prompt_tokens / 1_000_000) * 0.5 + (completion_tokens / 1_000_000) * 4.0
    if "qwen3-coder" in model_id:
        return (prompt_tokens / 1_000_000) * 0.4 + (completion_tokens / 1_000_000) * 1.6
    if "Qwen/Qwen2.5-Coder-7B-Instruct" in model_id:
        return 0.0
    if "Qwen/Qwen2.5-Coder-32B-Instruct" in model_id:
        return 0.0
    if "claude-haiku" in model_id:
        return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 5.0
    if "claude-sonnet" in model_id:
        return (prompt_tokens / 1_000_000) * 3.0 + (completion_tokens / 1_000_000) * 15.0
    if "gpt-oss-20b" in model_id:
        return 0.0
    return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 5.0


# ---------- Running tracker ----------

@dataclass
class CostTracker:
    """Per-generator running cost ledger with a hard cap.

    Each generator gets one CostTracker. Workers call `record(cost_usd)`
    after every API response; if `total_usd` crosses `cap_usd` the
    tracker flips `capped=True`, which the caller checks at the top of
    each task to skip remaining work cleanly without cancellation games.

    A separate JSONL audit log captures one line per call so we have a
    ground-truth ledger if something goes wrong with the in-memory totals.
    """

    name: str
    cap_usd: float
    log_path: Path | None = None
    total_usd: float = 0.0
    n_calls: int = 0
    n_skipped: int = 0  # tasks not even attempted because cap was already hit
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def capped(self) -> bool:
        with self._lock:
            return self.total_usd >= self.cap_usd

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.cap_usd - self.total_usd)

    def can_proceed(self) -> bool:
        """True if we should still issue more API calls."""
        return not self.capped

    def record(
        self,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
        instance_id: str = "",
        patch_id: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> tuple[float, bool]:
        """Add a paid call; returns (new_total, still_under_cap)."""
        with self._lock:
            self.total_usd += float(cost_usd)
            self.n_calls += 1
            new_total = self.total_usd
            still_ok = new_total < self.cap_usd
        if self.log_path is not None:
            line = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "model": self.name,
                "instance_id": instance_id,
                "patch_id": patch_id,
                "cost_usd": float(cost_usd),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "cumulative_usd": new_total,
            }
            if extra:
                line.update(extra)
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(line) + "\n")
            except OSError as exc:
                log.warning("cost log write failed: %s", exc)
        return new_total, still_ok

    def note_skipped(self, n: int = 1) -> None:
        with self._lock:
            self.n_skipped += n

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model": self.name,
                "cap_usd": self.cap_usd,
                "total_usd": self.total_usd,
                "n_calls": self.n_calls,
                "n_skipped": self.n_skipped,
                "remaining_usd": max(0.0, self.cap_usd - self.total_usd),
                "capped": self.total_usd >= self.cap_usd,
            }


# ---------- Spot-check helpers ----------

def extract_usage(resp: Any) -> tuple[float, int, int]:
    """Pull (cost_usd, prompt_tokens, completion_tokens) from a chat response.

    OpenRouter sends `usage.cost` as a custom field. The OpenAI SDK exposes
    it via `model_dump()` (custom usage fields ride along in `model_extra`).
    For local vLLM endpoints there is no cost field; we return 0.0.
    """
    try:
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    except Exception:
        data = {}
    usage = data.get("usage") or {}
    cost = float(usage.get("cost") or 0.0)
    prompt_tok = int(usage.get("prompt_tokens") or 0)
    completion_tok = int(usage.get("completion_tokens") or 0)
    return cost, prompt_tok, completion_tok


def project_cost(probe_cost_usd: float, n_total_calls: int, probe_calls: int = 1) -> float:
    """Linear projection: probe cost per call * total planned calls."""
    if probe_calls <= 0:
        return 0.0
    return (probe_cost_usd / probe_calls) * n_total_calls
