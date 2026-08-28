"""Shared per-action telemetry — thread-safe JSONL writer.

Every entry-point script (calibration/*, iter/*, scripts/*) emits one row
per atomic action (generate / critic_L0 / critic_L1 / critic_L2 / critic_L3
/ verify / refine / reflect / ...) into an append-only JSONL file. The
schema is the superset of:

  - calibration/*.py's pre-existing `_ActionTelemetry` (dataset, model_name,
    instance_id, patch_id, action_type, runtime_seconds, passed,
    api_cost_usd) — feeds tab:action_latency in the paper
  - abbo's `TelemetryLogger` / `write_action` (adds run_id, belief_before,
    step) — feeds latency × policy decision joins

Promoting it to `_common/` means:

  1. Calibration callers stop importing `_ActionTelemetry` from `calibration.lcb`
     (a cross-module shim from before this module existed).
  2. iter/refine.py + iter/refine_swe.py can now record per-action timing,
     closing the latency-analysis gap (previously latency was only known
     for calibration). Without this you can't compare SR/Rfx steps' wall
     time to single-shot calibration.
  3. Future live agents that need belief_before in the row schema (the
     Bayesian DP path in `scripts/run_synthesis_live.py` and any port of
     abbo's run_codecontests_full.py) use the same writer.

JSONL row schema — required fields always present, optional ones written
only when the caller passes them, so the file stays compact for the
calibration case:

  {
    "ts": float,                  # unix epoch seconds, always
    "dataset": str,               # e.g. "lcb_calibration_hard", "mbpp"
    "model_name": str,            # e.g. "haiku45", "qwen25_32b"
    "instance_id": str,           # benchmark instance ID
    "action_type": str,           # see ACTION_TYPES below
    "runtime_seconds": float,     # wall-clock, rounded to 4 dp
    "passed": bool | None,        # critics / verify only
    "api_cost_usd": float,        # 0.0 for non-LLM actions

    # Optional, only present when applicable:
    "patch_id": int | None,       # calibration only (per-patch enumeration)
    "step": int | None,           # iter only (refinement step index)
    "belief_before": float | None,# BDP-aware agents only
    "run_id": str | None,         # multi-run grouping (W&B run id, etc.)
    "extra": dict | None,         # caller-supplied free-form metadata
  }

Canonical action types — keep this list in sync if you add new ones:

  generate     LLM patch / refinement generation
  refine       Iter refinement step (alias of generate when in iter loop)
  reflect      Reflexion verbal-reflection step
  critic_L0    Syntax check (no LLM)
  critic_L1    Lint (no LLM)
  critic_L2    Public tests (no LLM)
  critic_L3    LLM judge
  verify       Oracle test (private tests / SWE harness)
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


# Canonical action types accepted by .record(). Not strictly enforced — the
# writer accepts any string — but downstream analyzers expect this set.
ACTION_TYPES = frozenset({
    "generate", "refine", "reflect",
    "critic_L0", "critic_L1", "critic_L2", "critic_L3",
    "verify",
})


class TelemetryLogger:
    """Append-only JSONL writer for per-action telemetry. Thread-safe.

    Constructor positional signature is back-compatible with the pre-existing
    `calibration/lcb.py:_ActionTelemetry`: TelemetryLogger(path, dataset,
    model_name). Optional run_id is keyword-only.

    Usage:
        tele = TelemetryLogger(gen_dir / "action_telemetry.jsonl",
                               dataset="mbpp", model_name="haiku45")
        t0 = time.perf_counter()
        resp = client.chat.completions.create(...)
        tele.record(action_type="generate", runtime_s=time.perf_counter()-t0,
                    instance_id="Mbpp/61", patch_id=0, api_cost_usd=0.0012)
        ...
        tele.close()
    """
    def __init__(
        self,
        path: Path,
        dataset: str,
        model_name: str,
        *,
        run_id: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._dataset = dataset
        self._model_name = model_name
        self._run_id = run_id
        # Line-buffered append so killed runs preserve partial telemetry.
        self._fh = open(self._path, "a", buffering=1)
        self._lock = threading.Lock()
        self._closed = False

    def record(
        self,
        *,
        action_type: str,
        runtime_s: float,
        instance_id: str,
        patch_id: int | None = None,
        step: int | None = None,
        passed: bool | None = None,
        api_cost_usd: float = 0.0,
        belief_before: float | None = None,
        extra: dict | None = None,
    ) -> None:
        """Record one action. All args keyword-only to avoid positional
        accidents in 17+ callsites; only `extra` accepts free-form metadata."""
        row: dict[str, Any] = {
            "ts": time.time(),
            "dataset": self._dataset,
            "model_name": self._model_name,
            "instance_id": str(instance_id),
            "action_type": action_type,
            "runtime_seconds": round(float(runtime_s), 4),
            "passed": passed,
            "api_cost_usd": float(api_cost_usd),
        }
        # Only include optional fields when supplied — keeps the JSONL slim
        # for the typical calibration case.
        if patch_id is not None:
            row["patch_id"] = int(patch_id)
        if step is not None:
            row["step"] = int(step)
        if belief_before is not None:
            row["belief_before"] = float(belief_before)
        if self._run_id is not None:
            row["run_id"] = self._run_id
        if extra is not None:
            # `is not None` (not truthiness) so an explicit empty dict is
            # written rather than silently dropped — matches the pattern
            # used for the other optional fields above.
            row["extra"] = extra
        line = json.dumps(row)
        with self._lock:
            if self._closed:
                raise RuntimeError("TelemetryLogger is closed")
            self._fh.write(line + "\n")

    def close(self) -> None:
        """Idempotent close — calling twice is safe."""
        with self._lock:
            if self._closed:
                return
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_action(
    logger: TelemetryLogger,
    *,
    action_type: str,
    runtime_s: float,
    **kwargs: Any,
) -> None:
    """Convenience wrapper matching abbo's module-level call shape.

    Equivalent to `logger.record(action_type=..., runtime_s=..., **kwargs)`.
    Exists so that code ported from abbo's `from abbo.realworld.telemetry
    import write_action, TelemetryLogger` works unchanged after switching
    the import path to `_common.telemetry`.
    """
    logger.record(action_type=action_type, runtime_s=runtime_s, **kwargs)


# Backward-compat alias for the pre-existing `_ActionTelemetry` symbol that
# was inlined in calibration/lcb.py and re-imported from there by mbpp.py
# and humaneval.py. New code should import `TelemetryLogger` directly.
_ActionTelemetry = TelemetryLogger


__all__ = [
    "TelemetryLogger",
    "write_action",
    "ACTION_TYPES",
    "_ActionTelemetry",
]
