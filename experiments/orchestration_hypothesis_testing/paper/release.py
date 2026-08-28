"""Build the PAPER_RELEASE/ artifact directory for paper supplementary material.

Includes the small/medium aggregated artifacts (likelihood tables, kernels,
policy comparisons, figures, methodology outputs) and EXCLUDES bulky raw
artifacts (raw_responses/, predictions.jsonl, generation_records.jsonl,
calibration logs, eval/logs/, harness logs, swebench_lite/source/). Total
target size: ~5-15 MB.

Output structure:
  data/PAPER_RELEASE/
    README.md
    PAPER_TABLE.{json,csv}
    paper_figs/                              (8 figures)
    per_cell/
      <benchmark>/<generator>/
        likelihood_tables.json
        policy_comparison.json
        policy_comparison_kernel_*.json      (variants with measured kernel)
        policy_comparison_l3_*.json          (L3 reviewer variants)
        policy_comparison_loo.json           (leave-one-out)
        policy_comparison_iter_replay_baselines.json
        transition_kernel_iid_baseline.json
        sensitivity.{json,csv}               (where it exists)
        L3_sweep_likelihoods.json
    iter_kernels/
      <benchmark>_iter/<generator>/
        transition_kernel.json
        HARNESS_ERRORS_NOTE.json (sonnet45 SWE-Lite only)
    methodology/
      critic_independence/
      mde_power/
      fdr_correction/
      prior_ci/
      cluster_bootstrap/
    docs/
      PRE_REGISTRATION.md (linked from project root)
      EXPERIMENTAL_LOG.md (linked from project root)

Usage:
  python3 scripts/build_paper_release.py \\
      --data-root data \\
      --release-dir data/PAPER_RELEASE
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BENCHMARKS = [
    "lcb_calibration_v2",
    "lcb_calibration_medium",
    "lcb_calibration_easy",
    "mbpp_calibration",
    "humaneval_calibration",
    "swebench_lite",
    "swebench_verified",
]
ITER_BENCHMARKS = [
    "lcb_calibration_v2_iter",
    "lcb_calibration_medium_iter",
    "lcb_calibration_easy_iter",
    "swebench_verified_iter",
    "swebench_lite_iter",
]
GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45"]

# File-name patterns to include from per-cell directories
PER_CELL_KEEP = [
    "likelihood_tables.json",
    "policy_comparison.json",
    "transition_kernel_iid_baseline.json",
    "sensitivity.json", "sensitivity.csv",
    "L3_sweep_likelihoods.json",
]
PER_CELL_KEEP_GLOBS = [
    "policy_comparison_kernel_*.json",
    "policy_comparison_l3_*.json",
    "policy_comparison_loo.json",
    "policy_comparison_cver*.json",
    "policy_comparison_iter_replay_baselines.json",
]

ITER_CELL_KEEP = [
    "transition_kernel.json",
    "HARNESS_ERRORS_NOTE.json",  # sonnet45 SWE-Lite only
]

METHODOLOGY_DIRS = [
    "critic_independence",
    "mde_power",
    "fdr_correction",
    "prior_ci",
    "cluster_bootstrap",
]

PAPER_FIGS_KEEP = [
    "fig1_headline.png", "fig1_headline.pdf",
    "fig2_l3_heatmap.png", "fig2_l3_heatmap.pdf",
    "fig3_invariance.png", "fig3_invariance.pdf",
    "fig4_regime_map.png", "fig4_regime_map.pdf",
    "fig_framework_vs_baselines.png", "fig_framework_vs_baselines.pdf",
    "fig_framework_vs_baselines.csv",
    "fig_lcb_difficulty_gradient.png", "fig_lcb_difficulty_gradient.pdf",
    "fig_lcb_difficulty_gradient.csv",
]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    args = parser.parse_args()

    release = args.release_dir
    if release.exists():
        print(f"WARNING: {release} already exists; merging.")
    release.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_skipped = 0

    # Top-level table
    for f in ["PAPER_TABLE.json", "PAPER_TABLE.csv"]:
        if copy_if_exists(args.data_root / f, release / f):
            n_files += 1

    # Paper figures
    figs_dir = release / "paper_figs"
    figs_dir.mkdir(exist_ok=True)
    for fig_name in PAPER_FIGS_KEEP:
        if copy_if_exists(args.data_root / "paper_figs" / fig_name, figs_dir / fig_name):
            n_files += 1
        else:
            n_skipped += 1

    # Per-cell aggregate artifacts
    for bench in BENCHMARKS:
        for gen in GENERATORS:
            src_gen = args.data_root / bench / gen
            if not src_gen.is_dir():
                continue
            dst_gen = release / "per_cell" / bench / gen
            for fname in PER_CELL_KEEP:
                if copy_if_exists(src_gen / fname, dst_gen / fname):
                    n_files += 1
            for pat in PER_CELL_KEEP_GLOBS:
                for f in src_gen.glob(pat):
                    if copy_if_exists(f, dst_gen / f.name):
                        n_files += 1

    # Iter kernels
    for bench in ITER_BENCHMARKS:
        for gen in GENERATORS:
            src_gen = args.data_root / bench / gen
            if not src_gen.is_dir():
                # SWE-Lite May5 iter is under swebench_lite/source/<gen>/ — handle separately
                continue
            dst_gen = release / "iter_kernels" / bench / gen
            for fname in ITER_CELL_KEEP:
                if copy_if_exists(src_gen / fname, dst_gen / fname):
                    n_files += 1

    # SWE-Lite May 5 iter (gpt5+qwen3) lives under swebench_lite/source/<gen>/
    for gen in GENERATORS:
        src = args.data_root / "swebench_lite" / "source" / gen / "transition_kernel.json"
        dst = release / "iter_kernels" / "swebench_lite_may5" / gen / "transition_kernel.json"
        if copy_if_exists(src, dst):
            n_files += 1

    # Methodology output dirs
    for mname in METHODOLOGY_DIRS:
        src_dir = args.data_root / mname
        if not src_dir.is_dir():
            continue
        dst_dir = release / "methodology" / mname
        for f in src_dir.iterdir():
            if f.is_file() and (f.suffix in {".json", ".csv"} or f.suffix == ""):
                if copy_if_exists(f, dst_dir / f.name):
                    n_files += 1

    # Docs
    docs_dir = release / "docs"
    docs_dir.mkdir(exist_ok=True)
    for f in ["PRE_REGISTRATION.md", "EXPERIMENTAL_LOG.md"]:
        # These live one level up at experiments/orchestration_hypothesis_testing/
        candidate = args.data_root.parent / f
        if copy_if_exists(candidate, docs_dir / f):
            n_files += 1

    # README.md
    readme = release / "README.md"
    readme.write_text("""# PAPER_RELEASE — supplementary artifacts

Aggregated artifacts for the paper "Agentic AI Orchestration as Sequential
Hypothesis Testing for Code Generation". Excludes bulky raw artifacts
(predictions, raw_responses, calibration logs, harness eval logs).

## Structure

- `PAPER_TABLE.{json,csv}` — main results table, 28 cells × kernel/reviewer variants
- `paper_figs/` — figures referenced in the paper (PNG + PDF, plus CSV companions)
- `per_cell/<benchmark>/<generator>/` — per-cell aggregates:
  - `likelihood_tables.json` — Beta(1,1)-smoothed P(z|Y) per critic
  - `policy_comparison.json` — 8-policy utility comparison under default IID kernel
  - `policy_comparison_kernel_iterative.json` — under measured iter kernel (where available)
  - `policy_comparison_l3_<reviewer>.json` — L3 reviewer-swap variants
  - `policy_comparison_loo.json` — leave-one-out cross-validation
  - `policy_comparison_iter_replay_baselines.json` — Self-Refine + Reflexion replays
  - `transition_kernel_iid_baseline.json` — IID baseline kernel
  - `sensitivity.{json,csv}` — Tier-D θ-perturbation results (LCB-hard only)
- `iter_kernels/<benchmark>_iter/<generator>/transition_kernel.json` — measured iterative kernels
- `methodology/` — methodology rigor outputs:
  - `critic_independence/` — chi-squared + G-squared independence test (28 cells)
  - `mde_power/` — minimum detectable effect at 80% power (28 cells)
  - `fdr_correction/` — Benjamini-Hochberg FDR adjustment of policy p-values
  - `prior_ci/` — Wilson 95% CI on prior_Y1 (28 cells)
  - `cluster_bootstrap/` — within-repo cluster bootstrap on SWE-bench
- `docs/` — pre-registration and experimental log (referenced in paper)

## Cube coverage

- 4 generators: gpt-5-mini, qwen3-coder, claude-haiku-4.5, claude-sonnet-4.5
- 7 benchmarks: LiveCodeBench (hard, medium, easy), MBPP+, HumanEval+, SWE-bench (Lite, Verified)
- 3 L3 reviewers: claude-haiku-4.5, gpt-4o-mini, claude-sonnet-4.5

## Headline result

bayesian_greedy controller beats always_verify by Δ utility +5.5 to +20.3 on
LCB cells (n=29-90 instances), confirmed via paired-bootstrap CI (B=1000),
leave-one-out CV, BH-FDR correction, ±20% θ-perturbation, c_ver sensitivity
sweep, and within-repo cluster bootstrap on SWE-bench cells.

## How to reproduce

See `docs/EXPERIMENTAL_LOG.md` for full reproduction instructions and
`docs/PRE_REGISTRATION.md` for the pre-experiment methodology commitment.
""")
    n_files += 1

    print(f"Built {release}")
    print(f"  Total files: {n_files}")
    print(f"  Skipped (not present): {n_skipped}")
    print(f"  Estimated total size: ~{sum(f.stat().st_size for f in release.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
