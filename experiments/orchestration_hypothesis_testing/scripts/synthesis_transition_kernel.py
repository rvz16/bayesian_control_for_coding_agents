#!/usr/bin/env python
"""Compute per-cell transition kernel for synthesis benchmarks from cached
patches in critic_results.jsonl.

For each (instance, patch_id ordered) sequence, count transitions
(Y[i] -> Y[i+1]) where Y is the verified outcome. Apply Beta(1,1) smoothing
and save as transition_kernel.json in the same format expected by abbo's
DPPlanner (`transition_kernel=` argument).

Caveat: patches in calibration data were generated INDEPENDENTLY (n_patches
sampling, no inter-patch feedback). So observed (Y[i], Y[i+1]) pairs are
not true Markov transitions of an iterative-refinement chain — they're more
like "what happens when we resample from the same model". For honest
transitions you'd want true multi-step refinement (iter_refine_bugfix.py
analog for synthesis). This approximation is the pragmatic middle ground
between hardcoded constants and full re-calibration.

Usage:
  python scripts/synthesis_transition_kernel.py \\
      --src-dir data/mbpp_full \\
      --generators haiku45,sonnet45,qwen3_coder,gpt5_mini
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.kernel import (  # noqa: E402
    DEFAULT_KERNEL,
    compute_transition_kernel_from_pairs,
    pairs_from_trajectories,
)

log = logging.getLogger("synth_kernel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


def instance_id_of(rec: dict) -> str:
    for k in ("instance_id", "question_id"):
        if k in rec:
            return str(rec[k])
    raise KeyError(f"no instance/question id in record keys: {list(rec.keys())[:5]}")


def compute_transition_kernel(records: list[dict], gen: str, benchmark: str,
                              alpha: float = 1.0, beta: float = 1.0) -> dict:
    """Group records by instance_id, sort by patch_id, count Y[i]->Y[i+1].

    Output schema preserves the source / benchmark / generator metadata that
    downstream synthesis consumers expect alongside the canonical kernel_all
    blob. Empty-regime fallback differs from the shared default: uses the
    literature prior (0.5 / 0.05) rather than Laplace-uniform.
    """
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_inst[instance_id_of(r)].append(r)
    for iid in by_inst:
        by_inst[iid].sort(key=lambda r: int(r.get("patch_id", 0)))

    pairs = pairs_from_trajectories(by_inst.values())
    k = compute_transition_kernel_from_pairs(pairs, alpha=alpha, beta=beta)

    # Empty-regime overrides: this script uses the literature prior, not the
    # Laplace-uniform that the shared helper returns when n=0.
    if k["n_broken_observed"] == 0:
        k["P_fix_given_broken"] = DEFAULT_KERNEL["p_fix_broken"]
    if k["n_correct_observed"] == 0:
        k["P_break_given_correct"] = DEFAULT_KERNEL["p_break_correct"]

    return {
        "generator": gen,
        "benchmark": benchmark,
        "source": "synthesis_transition_kernel (post-hoc from critic_results.jsonl)",
        "kernel_all": {
            "P_fix_given_broken": k["P_fix_given_broken"],
            "P_break_given_correct": k["P_break_given_correct"],
            "raw_counts": k["raw_counts"],
            "n_pairs": k["n_pairs"],
            "n_broken_observed": k["n_broken_observed"],
            "n_correct_observed": k["n_correct_observed"],
            "smoothing": f"Beta({alpha},{beta})",
        },
    }


def process_one_generator(src_dir: Path, gen: str, benchmark: str,
                          alpha: float, beta: float) -> None:
    gen_dir = src_dir / gen
    cr_path = gen_dir / "critic_results.jsonl"
    if not cr_path.exists():
        log.warning("[%s] no critic_results.jsonl at %s — skipping", gen, cr_path)
        return

    records = []
    for line in cr_path.read_text().splitlines():
        if not line.strip(): continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not records:
        log.warning("[%s] empty critic_results — skipping", gen)
        return

    # Check that we have multiple patches per instance (otherwise no transitions)
    by_inst = defaultdict(int)
    for r in records:
        by_inst[instance_id_of(r)] += 1
    n_instances_with_multiple = sum(1 for n in by_inst.values() if n > 1)
    if n_instances_with_multiple == 0:
        log.warning("[%s] only 1 patch per instance (n_patches=1?); "
                    "no transitions to count — skipping. Re-run calibration "
                    "with --n-patches >= 3 to enable kernel measurement.", gen)
        return

    kernel = compute_transition_kernel(records, gen, benchmark, alpha, beta)
    out_path = gen_dir / "transition_kernel.json"
    out_path.write_text(json.dumps(kernel, indent=2))

    k = kernel["kernel_all"]
    log.info("[%s] %d pairs (%d broken→ + %d correct→) — "
             "p_fix=%.3f, p_break=%.3f → %s",
             gen, k["n_pairs"], k["n_broken_observed"], k["n_correct_observed"],
             k["P_fix_given_broken"], k["P_break_given_correct"], out_path.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-dir", required=True, type=Path,
                   help="Directory with <gen>/critic_results.jsonl "
                        "(e.g. data/mbpp_full or data/lcb_medium_full)")
    p.add_argument("--generators", required=True,
                   help="Comma-separated generator slugs (e.g. haiku45,sonnet45)")
    p.add_argument("--benchmark", default="",
                   help="Optional benchmark label for the kernel metadata "
                        "(defaults to basename of --src-dir)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Beta prior alpha (default 1.0 = Laplace)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Beta prior beta (default 1.0)")
    args = p.parse_args()

    src_dir = args.src_dir.resolve()
    if not src_dir.exists():
        raise SystemExit(f"src-dir does not exist: {src_dir}")
    bench = args.benchmark or src_dir.name

    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    for gen in generators:
        process_one_generator(src_dir, gen, bench, args.alpha, args.beta)


if __name__ == "__main__":
    main()
