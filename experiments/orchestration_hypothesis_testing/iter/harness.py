"""Sequentially run SWE-bench Verified harness eval on all iterative
refinement step predictions, using scg.run_swebench_eval() which
auto-loads the podman compat shim.

Usage:
  python3 run_iter_harness.py \\
    --iter-dir data/swebench_verified_iter \\
    --work-dir data/swebench_verified_iter/eval \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
    --steps 1,2,3,4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# File moved to iter/ during the scripts refactor; parents[1] is the
# package root (experiments/orchestration_hypothesis_testing/).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # for `from analysis.X import Y` etc.
sys.path.insert(0, str(ROOT / "scripts")) # for legacy `import spot_check_generators`

import spot_check_generators as scg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("iter_harness")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter-dir", required=True, type=Path,
                        help="dir containing <gen>/predictions_iter_step{X}.jsonl")
    parser.add_argument("--work-dir", required=True, type=Path,
                        help="harness work dir (reports + .podman_compat live here)")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--generators", required=True)
    parser.add_argument("--steps", default="1,2,3,4")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="default 1 — multi-worker eval races on "
                             "containerd image-pull metadata, producing "
                             "silent 'Docker API timeout' failures. Bump "
                             "explicitly only when host has been hardened.")
    args = parser.parse_args()

    iter_dir = args.iter_dir.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]

    for gen in gens:
        for step in steps:
            pred = iter_dir / gen / f"predictions_iter_step{step}.jsonl"
            if not pred.exists():
                log.warning("[%s/step%d] no predictions file, skipping", gen, step)
                continue
            run_id = f"{gen}_iter_step{step}"
            # Skip if report already exists
            existing = list(work_dir.glob(f"*{run_id}*.json"))
            if existing:
                log.info("[%s/step%d] report exists (%s), skipping",
                         gen, step, existing[0].name)
                continue
            log.info("=== %s step %d ===", gen, step)
            try:
                scg.run_swebench_eval(
                    predictions_path=pred,
                    run_id=run_id,
                    max_workers=args.max_workers,
                    work_dir=work_dir,
                    dataset_name=args.dataset,
                )
                log.info("[%s/step%d] DONE", gen, step)
            except Exception as e:
                log.error("[%s/step%d] failed: %s", gen, step, e)


if __name__ == "__main__":
    main()
