"""Shared transition-kernel utilities — post-hoc estimation + online updates.

The transition kernel describes Markovian dynamics of the per-instance
correctness signal Y across refinement steps:
  P(fix|broken)   = P(Y_{t+1}=1 | Y_t=0)
  P(break|correct) = P(Y_{t+1}=0 | Y_t=1)

This module exposes:

1) `compute_transition_kernel_from_pairs(pairs)` — Beta(α,β)-smoothed point
   estimate from a list of (Y_t, Y_{t+1}) pairs. Used by every "compute
   kernel after the run completes" script in the pipeline. Replaces the
   inline duplicates that previously lived in iter/refine.py, iter/kernel.py,
   analysis/compute_transition_kernel.py, and scripts/synthesis_transition_kernel.py.

2) `OnlineKernelCalibration` — Beta-Binomial running estimator queryable
   mid-loop. Thread-safe. Used by live agents that re-solve the Bayesian DP
   planner after each verify with an updated posterior. Small-N caveat: with
   fewer than ~10 transitions observed in a regime, the posterior is wide;
   the planner sees only the posterior mean. Inspect `.summary()` for the
   underlying counts when interpreting decisions.

3) `resolve_kernel(gen_dir, mode)` — three-way switch that callers use to pick
   between {measured-from-file, online-Beta-Binomial, hardcoded-default}.

`kernel_update(belief, kernel)` is the one-step belief propagation under the
kernel — same semantics whether the kernel is static or a snapshot from
`OnlineKernelCalibration.get()`.

Note: iter/swe_kernel.py keeps its own private compute_kernel because its
JSON output uses a different (flat, unsmoothed) schema consumed by the SWE
sections of analysis.ipynb. Migrating it would be a downstream-visible
change and is deliberately scoped out of this module.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


# Default values used by abbo + the synthesis-live agent for the initial
# uninformative kernel. Tunable per-pipeline by passing init_kernel= to
# OnlineKernelCalibration.
DEFAULT_KERNEL: dict = {"p_fix_broken": 0.50, "p_break_correct": 0.05}


def kernel_update(belief: float, kernel: Mapping[str, float]) -> float:
    """One-step belief propagation under a transition kernel.

    b' = b · (1 − p_break) + (1 − b) · p_fix

    Accepts either the lowercase schema {"p_fix_broken", "p_break_correct"}
    used by abbo / synthesis-live agents, or the uppercase schema
    {"P_fix_given_broken", "P_break_given_correct"} produced by
    `compute_transition_kernel_from_pairs` (typically under a `kernel_all`
    wrapper in JSON files).
    """
    p_fix = kernel.get("p_fix_broken", kernel.get("P_fix_given_broken"))
    p_break = kernel.get("p_break_correct", kernel.get("P_break_given_correct"))
    if p_fix is None or p_break is None:
        raise ValueError(
            "kernel missing p_fix_broken / p_break_correct "
            f"(or uppercase variants): {dict(kernel)!r}"
        )
    return belief * (1.0 - p_break) + (1.0 - belief) * p_fix


# ----------------------------------------------------------------------------
# Post-hoc kernel: Beta-smoothed point estimate from a fixed set of pairs
# ----------------------------------------------------------------------------

def _beta_smooth(success: int, total: int, alpha: float, beta: float) -> float:
    return (success + alpha) / (total + alpha + beta)


def compute_transition_kernel_from_pairs(
    pairs: Sequence[tuple[int, int]],
    *, alpha: float = 1.0, beta: float = 1.0,
) -> dict:
    """Beta(α, β)-smoothed transition kernel from (Y_t, Y_{t+1}) pairs.

    Y values must be 0 or 1; non-binary pairs are silently dropped to match
    legacy behavior. The Beta prior defaults to Laplace (alpha=beta=1).

    Schema (superset of the four variants this replaces):
        {
          "P_fix_given_broken":    p_fix,
          "P_stay_broken":         1 − p_fix     # smoothed independently
          "P_break_given_correct": p_break,
          "P_stay_correct":        1 − p_break   # smoothed independently
          "raw_counts": {"0->0": int, "0->1": int, "1->0": int, "1->1": int},
          "n_pairs": int,
          "n_broken_observed": int,    # pairs with Y_t == 0
          "n_correct_observed": int,   # pairs with Y_t == 1
          "smoothing": "Beta(alpha,beta)",
        }

    Note: when alpha == beta (e.g. Laplace), independent smoothing of stay-
    vs change-transitions exactly sums to 1 — both schemas are equivalent
    in that case. With asymmetric priors they can drift, which is by design
    (each Beta is its own posterior).
    """
    counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
    for y0, y1 in pairs:
        if y0 not in (0, 1) or y1 not in (0, 1):
            continue
        counts[f"{int(y0)}->{int(y1)}"] += 1
    n_broken = counts["0->0"] + counts["0->1"]
    n_correct = counts["1->0"] + counts["1->1"]
    p_fix = _beta_smooth(counts["0->1"], n_broken, alpha, beta)
    p_stay_broken = _beta_smooth(counts["0->0"], n_broken, alpha, beta)
    p_break = _beta_smooth(counts["1->0"], n_correct, alpha, beta)
    p_stay_correct = _beta_smooth(counts["1->1"], n_correct, alpha, beta)
    return {
        "P_fix_given_broken": p_fix,
        "P_stay_broken": p_stay_broken,
        "P_break_given_correct": p_break,
        "P_stay_correct": p_stay_correct,
        "raw_counts": counts,
        "n_pairs": n_broken + n_correct,
        "n_broken_observed": n_broken,
        "n_correct_observed": n_correct,
        "smoothing": f"Beta({alpha},{beta})",
    }


def pairs_from_trajectories(
    trajectories: Iterable[Iterable[Mapping]],
    *, y_key: str = "Y",
) -> list[tuple[int, int]]:
    """Build (Y_t, Y_{t+1}) pairs from per-instance step trajectories.

    Each trajectory is an iterable of step records, already sorted in step
    order. Pairs with missing or non-binary Y are dropped. This is the one
    helper the four post-hoc compute_kernel callers share — they each handle
    the upstream group-by-instance + sort themselves (the keys differ across
    scripts: "step", "patch_id", custom).
    """
    pairs: list[tuple[int, int]] = []
    for traj in trajectories:
        traj = list(traj)
        for i in range(len(traj) - 1):
            y0 = traj[i].get(y_key)
            y1 = traj[i + 1].get(y_key)
            if y0 is None or y1 is None:
                continue
            pairs.append((int(y0), int(y1)))
    return pairs


# ----------------------------------------------------------------------------
# Online kernel: Beta-Binomial running estimator
# ----------------------------------------------------------------------------

@dataclass
class _KernelCounts:
    n_broken: int = 0    # Y_before == 0
    k_fix: int = 0       # transitions 0 -> 1
    n_correct: int = 0   # Y_before == 1
    k_break: int = 0     # transitions 1 -> 0


class OnlineKernelCalibration:
    """Beta(α, β)-smoothed running estimator of P(fix|broken) + P(break|correct).

    Designed to be updated after each verify in a live evaluation loop.
    Falls back to `init_kernel` (or DEFAULT_KERNEL) when no observations
    have been recorded in a regime yet — i.e. the planner gets *some*
    kernel even on the very first instance.

    Thread-safe: .update() and .get() acquire an internal lock so multiple
    worker threads can share one estimator. The lock is uncontested in the
    typical sequential-loop case and adds negligible overhead.

    Usage:
        ok = OnlineKernelCalibration(init_kernel=measured_kernel)
        for step in trajectory:
            y_before = current_correctness_estimate
            run_action()
            y_after = verify()
            ok.update(y_before, y_after)
            current = ok.get()  # safe to feed back into planner re-solve
    """
    def __init__(
        self,
        init_kernel: Mapping[str, float] | None = None,
        *, alpha: float = 1.0, beta: float = 1.0,
    ) -> None:
        self.counts = _KernelCounts()
        self.alpha = float(alpha)
        self.beta = float(beta)
        # Normalize init_kernel to lowercase-key schema so .get() can serve it
        # back uniformly to callers that expect kernel_update(belief, kernel).
        raw = dict(init_kernel) if init_kernel else dict(DEFAULT_KERNEL)
        if "P_fix_given_broken" in raw and "p_fix_broken" not in raw:
            raw["p_fix_broken"] = raw["P_fix_given_broken"]
        if "P_break_given_correct" in raw and "p_break_correct" not in raw:
            raw["p_break_correct"] = raw["P_break_given_correct"]
        # Validate at construction time so a malformed init_kernel raises
        # here, not later inside .get() on an unrelated call site (KeyError
        # in the middle of a hot loop is much harder to debug).
        missing = {"p_fix_broken", "p_break_correct"} - raw.keys()
        if missing:
            raise ValueError(
                "OnlineKernelCalibration init_kernel missing required keys "
                f"{sorted(missing)} (accepted: lowercase {{p_fix_broken, "
                f"p_break_correct}} or uppercase {{P_fix_given_broken, "
                f"P_break_given_correct}}); got: {dict(raw)!r}"
            )
        self._init = raw
        # RLock so summary() can call get() while holding the lock without
        # deadlocking. Re-entrant acquire is cheap in the uncontended case.
        self._lock = threading.RLock()

    def update(self, y_before: int, y_after: int) -> None:
        """Record one observed (Y_t, Y_{t+1}) transition.

        Raises ValueError if either value does not equal 0 or 1 — silently
        accepting non-binary input (e.g. None) was a bug surface because
        anything `!= 0` previously landed in the "correct" regime, so
        `update(None, 1)` would silently log a (correct → correct)
        transition. Callers must filter their own data; both production
        call sites already do.

        Note on accepted types: the check uses `==` equality, so bool
        (True/False) and float 0.0/1.0 are accepted in addition to int
        0/1 — they all round-trip to the same counts. Anything not equal
        to 0 or 1 (None, 2, strings, etc.) raises.
        """
        if y_before not in (0, 1) or y_after not in (0, 1):
            raise ValueError(
                f"OnlineKernelCalibration.update requires y_before and "
                f"y_after to equal 0 or 1; got y_before={y_before!r}, "
                f"y_after={y_after!r}"
            )
        with self._lock:
            if y_before == 0:
                self.counts.n_broken += 1
                self.counts.k_fix += int(y_after == 1)
            else:
                self.counts.n_correct += 1
                self.counts.k_break += int(y_after == 0)

    def get(self) -> dict:
        """Current posterior estimate (mean only).

        Shape matches kernel_update()'s expected lowercase-key schema:
          {"p_fix_broken": float, "p_break_correct": float}
        """
        with self._lock:
            c = self.counts
            if c.n_broken > 0:
                p_fix = (c.k_fix + self.alpha) / (c.n_broken + self.alpha + self.beta)
            else:
                p_fix = self._init["p_fix_broken"]
            if c.n_correct > 0:
                p_break = (c.k_break + self.alpha) / (c.n_correct + self.alpha + self.beta)
            else:
                p_break = self._init["p_break_correct"]
        return {"p_fix_broken": p_fix, "p_break_correct": p_break}

    def summary(self) -> dict:
        """Diagnostic snapshot of internal counts + current estimate.

        Useful for the small-N caveat: if n_broken or n_correct is below
        ~10, the posterior mean carries little signal. Persist this at
        end-of-run so analyses can decide whether the online posterior is
        worth trusting.
        """
        with self._lock:
            c = self.counts
            current = self.get()  # OK to call: get() re-acquires the same RLock
            return {
                "n_broken_observed": c.n_broken,
                "n_correct_observed": c.n_correct,
                "k_fix": c.k_fix,
                "k_break": c.k_break,
                "alpha": self.alpha,
                "beta": self.beta,
                "init_kernel": dict(self._init),
                "current_estimate": current,
            }


# ----------------------------------------------------------------------------
# File-level resolver used by live agents and the iter/refine.py CLI
# ----------------------------------------------------------------------------

def resolve_kernel(
    gen_dir: Path,
    mode: str = "measured",
    *,
    kernel_path: Path | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[dict, str, OnlineKernelCalibration | None]:
    """Return (kernel_dict, source_label, online_or_None) for the given mode.

    `mode` is one of:
      'measured'  — load <gen_dir>/transition_kernel.json (or `kernel_path`
                    if provided), fall back to DEFAULT_KERNEL if missing.
                    Third return value is None.
      'online'    — same starting point, plus an OnlineKernelCalibration
                    initialized from it; the caller should
                    .update(y_before, y_after) after each verify.
      'hardcoded' — always return DEFAULT_KERNEL, even if a measured file
                    exists. Third return value is None.

    Source label is one of {"measured", "default", "hardcoded"} for logging.
    """
    if mode not in {"measured", "online", "hardcoded"}:
        raise ValueError(f"unknown kernel mode: {mode!r}")

    if mode == "hardcoded":
        return dict(DEFAULT_KERNEL), "hardcoded", None

    path = kernel_path or (gen_dir / "transition_kernel.json")
    if path.exists():
        kj = json.loads(path.read_text())
        k_all = kj.get("kernel_all", kj)
        kernel = {
            "p_fix_broken": k_all["P_fix_given_broken"],
            "p_break_correct": k_all["P_break_given_correct"],
        }
        source = "measured"
    else:
        kernel = dict(DEFAULT_KERNEL)
        source = "default"

    if mode == "online":
        ok = OnlineKernelCalibration(init_kernel=kernel, alpha=alpha, beta=beta)
        return kernel, source, ok
    return kernel, source, None


__all__ = [
    "DEFAULT_KERNEL",
    "kernel_update",
    "compute_transition_kernel_from_pairs",
    "pairs_from_trajectories",
    "OnlineKernelCalibration",
    "resolve_kernel",
]
