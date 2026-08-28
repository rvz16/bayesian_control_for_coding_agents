"""Run SWE-bench harness on the iterative refinement predictions.

Wraps spot_check_generators.run_swebench_eval (which includes the podman
compat shim) and runs it for steps 1..N-1 of one generator under a chosen
predictions directory.

Originally hard-coded to a single Linux/podman setup and to the
`data/spot_check_n50/` tree; now parameterised so the same script works on:

  * Mac / Docker Desktop (no podman) — for local SWE-Bench reruns
  * Linux cluster with podman 3.4.4 — opt in via --podman or env

Output naming convention (unchanged): per-step report.json under
<work-dir>/<run-id>/, where run-id = "<gen>_iter_step{k}".

Usage:
  # Mac, Docker, lite split, Self-Refine outputs
  python iter/eval_steps.py \\
      --gen gpt5_mini \\
      --data-dir data/swebench_lite_realbaselines_selfrefine_rerun \\
      --dataset princeton-nlp/SWE-bench_Lite

  # Verified split, Reflexion outputs
  python iter/eval_steps.py \\
      --gen haiku45 \\
      --data-dir data/swebench_verified_realbaselines_reflexion_rerun \\
      --dataset princeton-nlp/SWE-bench_Verified

  # Linux cluster with podman
  python iter/eval_steps.py --gen qwen3_coder \\
      --data-dir data/swebench_lite_realbaselines_selfrefine \\
      --podman

Resume-safe: existing report.json files for a (gen, step) pair are
overwritten on rerun, but already-passed instances are skipped by the
harness itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# This file lives at iter/eval_steps.py. parents[1] = orchestration_hypothesis_testing/
ROOT = Path(__file__).resolve().parents[1]
# Both paths needed: ROOT for `_common.*` imports, ROOT/scripts for
# `spot_check_generators` (mirrors refine_swe.py / harness.py setup).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _maybe_setup_podman_shim() -> None:
    """Set up Podman compat for Linux clusters. No-op on Mac (Docker Desktop)."""
    # Only attempt the podman shim when:
    #   * --podman CLI flag was given (sets SWEBENCH_PODMAN_COMPAT below), OR
    #   * the calling shell already exported SWEBENCH_PODMAN_COMPAT=1, OR
    #   * DOCKER_HOST is already pointing at a podman socket
    docker_host = os.environ.get("DOCKER_HOST", "")
    podman_flag = os.environ.get("SWEBENCH_PODMAN_COMPAT", "")
    if not (podman_flag or "podman" in docker_host.lower()):
        return
    # Heuristic for the standard rootless podman socket path. Skip the
    # setdefault if the user already gave us an explicit DOCKER_HOST.
    if not docker_host:
        os.environ["DOCKER_HOST"] = (
            f"unix:///run/user/{os.geteuid()}/podman/podman.sock"
        )
    os.environ.setdefault("SWEBENCH_PODMAN_COMPAT", "1")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen", required=True,
                   help="Generator slug, e.g. gpt5_mini / haiku45 / sonnet45 / "
                        "qwen3_coder / qwen25_coder_32b / gpt_oss_20b. Must "
                        "match the subdirectory name under --data-dir.")
    p.add_argument("--method", default=None,
                   help="Refinement method, e.g. selfrefine / reflexion. When "
                        "set, predictions are read from "
                        "<data-dir>/<gen>/<method>/predictions_iter_step{N}.jsonl "
                        "(refine_swe.py layout), run_id becomes "
                        "<gen>_<method>_iter_step{N}, and the default work-dir "
                        "becomes <data-dir>/<gen>/<method>/eval so that "
                        "swe_backfill_y can find the reports under the cell "
                        "directory. Leave unset for legacy flat layouts.")
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Directory containing <gen>/predictions_iter_step{1..N}"
                        ".jsonl (or <gen>/<method>/... when --method is set). "
                        "Typically the --output-dir from a refine_swe.py run, "
                        "e.g. data/swebench_lite_realbaselines_selfrefine_full.")
    p.add_argument("--n-steps", type=int, default=5,
                   help="Iterate step in 1..N-1 (matches refine_swe.py default).")
    p.add_argument("--start-step", type=int, default=1,
                   help="First step to evaluate. Useful for resume after an "
                        "accepted persistent infra error in an earlier step.")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                   choices=["princeton-nlp/SWE-bench_Lite",
                            "princeton-nlp/SWE-bench_Verified"],
                   help="HuggingFace dataset id forwarded to the harness. Must "
                        "match the dataset used by the refine run.")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Where the harness writes its report files. "
                        "Default: <data-dir>/eval")
    p.add_argument("--max-workers", type=int,
                   default=int(os.environ.get("SWE_EVAL_MAX_WORKERS", "1")),
                   help="Concurrent harness workers. Default 1 to avoid "
                        "containerd metadata race (see iter/harness.py). "
                        "Override via SWE_EVAL_MAX_WORKERS env or flag.")
    p.add_argument("--max-attempts", type=int, default=1,
                   help="Maximum harness attempts per step. Existing per-instance "
                        "report.json files are resume-safe and are skipped.")
    p.add_argument("--retry-delay-seconds", type=int, default=30,
                   help="Delay between attempts when infra error_ids remain.")
    p.add_argument("--require-no-errors", action="store_true",
                   help="Retry a harness aggregate that contains error_ids.")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Stop before later steps if this step exhausts retries.")
    p.add_argument("--continue-after-errors", action="store_true",
                   help="After retries are exhausted, accept the aggregate with "
                        "remaining error_ids and continue to later steps.")
    p.add_argument("--podman", action="store_true",
                   help="Use podman-compat shim instead of regular Docker. "
                        "Required on Linux clusters where DOCKER_HOST points at "
                        "a podman.sock. No-op on Mac / Docker Desktop.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.retry_delay_seconds < 0:
        raise SystemExit("--retry-delay-seconds must be >= 0")
    if not 1 <= args.start_step < args.n_steps:
        raise SystemExit("--start-step must be in 1..n-steps-1")
    if args.podman:
        os.environ["SWEBENCH_PODMAN_COMPAT"] = "1"
    _maybe_setup_podman_shim()

    # Late import — has to come AFTER any podman env wiring because
    # spot_check_generators imports docker SDK at module level.
    import spot_check_generators as scg  # noqa: E402

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"ERROR: --data-dir does not exist: {data_dir}", file=sys.stderr)
        return 1

    if args.method:
        cell_dir = data_dir / args.gen / args.method
        default_work = cell_dir / "eval"
        run_id_tag = f"{args.gen}_{args.method}"
    else:
        cell_dir = data_dir / args.gen
        default_work = data_dir / "eval"
        run_id_tag = args.gen
    work_dir = (args.work_dir or default_work).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"eval config: gen={args.gen}  method={args.method or '<none>'}  "
          f"data-dir={data_dir}  cell-dir={cell_dir}  n-steps={args.n_steps}  "
          f"max-workers={args.max_workers}  dataset={args.dataset}  "
          f"work-dir={work_dir}")

    n_ok = n_skip = n_err = 0
    for step in range(args.start_step, args.n_steps):
        pred_path = cell_dir / f"predictions_iter_step{step}.jsonl"
        if not pred_path.exists() or pred_path.stat().st_size == 0:
            reason = "not found" if not pred_path.exists() else "empty"
            print(f"skip step {step}: {pred_path} {reason}")
            n_skip += 1
            continue
        run_id = f"{run_id_tag}_iter_step{step}"
        print(f"\n==== eval {run_id_tag} step {step} (run_id={run_id}) ====")
        step_ok = False
        for attempt in range(1, args.max_attempts + 1):
            print(f"  attempt {attempt}/{args.max_attempts}")
            try:
                report_path = scg.run_swebench_eval(
                    predictions_path=pred_path,
                    run_id=run_id,
                    max_workers=args.max_workers,
                    work_dir=work_dir,
                    dataset_name=args.dataset,
                )
                report = json.loads(report_path.read_text())
                error_ids = report.get("error_ids", [])
                print(f"  -> {report_path.name}  infra_errors={len(error_ids)}")
                if args.require_no_errors and error_ids:
                    if attempt < args.max_attempts:
                        print(
                            f"  retrying {len(error_ids)} infra errors after "
                            f"{args.retry_delay_seconds}s; completed reports "
                            "will be skipped"
                        )
                        time.sleep(args.retry_delay_seconds)
                        continue
                    print(
                        f"  ERROR: {len(error_ids)} infra errors remain after "
                        f"{args.max_attempts} attempts"
                    )
                    if args.continue_after_errors:
                        print("  accepting persistent infra errors and continuing")
                        step_ok = True
                    break
                step_ok = True
                break
            except Exception as e:
                print(f"  ERROR attempt {attempt}/{args.max_attempts}: {e}")
                if attempt < args.max_attempts:
                    time.sleep(args.retry_delay_seconds)

        if step_ok:
            n_ok += 1
        else:
            n_err += 1
            if args.stop_on_error:
                print("  stopping before later steps because this step failed")
                break

    print(f"\nDone: ok={n_ok}  skipped={n_skip}  errored={n_err}")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
