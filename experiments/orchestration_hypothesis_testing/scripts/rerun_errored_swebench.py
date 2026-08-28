#!/usr/bin/env python3
"""Re-run SWE-Bench harness on just the instances that errored in a previous
calibration run.

Use case
========

A run of `spot_check_generators.py` over SWE-Bench Lite / Verified writes one
report per (generator, patch_id) under
    <calibration-dir>/eval/<gen>__p<pid>.<gen>_p<pid>.json
Some instances land in `error_ids` because their per-instance Docker / podman
build failed for an environment reason (tar uid mapping, missing pip flag, …)
rather than because the model patch was wrong. Once the underlying build
issue is fixed (see `patch_swebench_harness.py`), those same instances will
build cleanly — but the original report is frozen with `error` rows.

This script re-runs the harness on a *filtered* `predictions_p<pid>.jsonl`
that contains only the errored instance IDs, using `scg.run_swebench_eval`
(the same wrapper the calibration uses, so paths and report locations match
exactly). It then merges the rerun outcomes back into the original report so
downstream consumers (`from_spotcheck.py`, `refine_swe.py`, the notebook's
harness-error filter) see the new resolved/unresolved verdicts without
having to know a rerun happened.

The original report is preserved as `<report>.pre_rerun.bak` on first merge.
Idempotent — running twice does nothing extra.

Usage
-----

    python rerun_errored_swebench.py \
        --calibration-dir data/swebench_lite_calibration_full \
        --gen qwen3_coder \
        --dataset princeton-nlp/SWE-bench_Lite \
        --n-patches 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _maybe_setup_podman_shim() -> None:
    """Mirrors iter/eval_steps._maybe_setup_podman_shim — set DOCKER_HOST /
    SWEBENCH_PODMAN_COMPAT if we're on a podman host and the calling shell
    hasn't already wired up the env.
    """
    docker_host = os.environ.get("DOCKER_HOST", "")
    podman_flag = os.environ.get("SWEBENCH_PODMAN_COMPAT", "")
    if not (podman_flag or "podman" in docker_host.lower()):
        return
    if not docker_host:
        os.environ["DOCKER_HOST"] = (
            f"unix:///run/user/{os.geteuid()}/podman/podman.sock"
        )
    os.environ.setdefault("SWEBENCH_PODMAN_COMPAT", "1")


def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def filter_predictions(src: Path, dst: Path, keep_ids: set[str]) -> int:
    n = 0
    with open(src) as inf, open(dst, "w") as outf:
        for line in inf:
            row = json.loads(line)
            if row.get("instance_id") in keep_ids:
                outf.write(line)
                n += 1
    return n


def merge_rerun_into_original(orig_path: Path, rerun_path: Path) -> dict:
    """Move instances from orig.error_ids into orig.resolved_ids /
    orig.unresolved_ids based on the rerun outcome. Returns a stats dict.
    """
    orig = load_report(orig_path)
    rerun = load_report(rerun_path)
    if orig is None:
        return {"err": "orig missing"}
    if rerun is None:
        return {"err": "rerun missing"}

    bak = orig_path.with_suffix(".json.pre_rerun.bak")
    if not bak.exists():
        bak.write_text(json.dumps(orig, indent=2))

    orig_err = set(orig.get("error_ids", []))
    orig_resolved = set(orig.get("resolved_ids", []))
    orig_unresolved = set(orig.get("unresolved_ids", []))
    rerun_resolved = set(rerun.get("resolved_ids", []))
    rerun_unresolved = set(rerun.get("unresolved_ids", []))
    rerun_err = set(rerun.get("error_ids", []))

    promoted_resolved = orig_err & rerun_resolved
    promoted_unresolved = orig_err & rerun_unresolved
    still_errored = orig_err & rerun_err

    new_err = orig_err - promoted_resolved - promoted_unresolved
    new_resolved = orig_resolved | promoted_resolved
    new_unresolved = orig_unresolved | promoted_unresolved

    orig["resolved_ids"] = sorted(new_resolved)
    orig["unresolved_ids"] = sorted(new_unresolved)
    orig["error_ids"] = sorted(new_err)
    # Keep the canonical ID partition internally consistent. Older versions
    # updated completed_instances but left completed_ids frozen at the
    # pre-rerun value, which made otherwise valid merged reports fail audits.
    orig["completed_ids"] = sorted(new_resolved | new_unresolved)
    orig["resolved_instances"] = len(new_resolved)
    orig["unresolved_instances"] = len(new_unresolved)
    orig["error_instances"] = len(new_err)
    orig["completed_instances"] = len(new_resolved) + len(new_unresolved)
    orig["_rerun_meta"] = {
        "rerun_report": rerun_path.name,
        "promoted_resolved": sorted(promoted_resolved),
        "promoted_unresolved": sorted(promoted_unresolved),
        "still_errored": sorted(still_errored),
    }

    orig_path.write_text(json.dumps(orig, indent=2))
    return {
        "errored_before": len(orig_err),
        "promoted_resolved": len(promoted_resolved),
        "promoted_unresolved": len(promoted_unresolved),
        "still_errored": len(still_errored),
        "errored_after": len(new_err),
    }


def rerun_patch(args, pid: int) -> dict:
    cal = args.calibration_dir
    eval_dir = cal / "eval"
    report_path = eval_dir / f"{args.gen}__p{pid}.{args.gen}_p{pid}.json"
    src_predictions = cal / args.gen / f"predictions_p{pid}.jsonl"

    report = load_report(report_path)
    if report is None:
        print(f"  p{pid}: report missing at {report_path}; skipping")
        return {"skip": True}
    err_ids = set(report.get("error_ids", []))
    if not err_ids:
        print(f"  p{pid}: 0 errors in original report; nothing to re-run")
        return {"errored_before": 0}
    if not src_predictions.exists():
        print(f"  p{pid}: predictions file missing at {src_predictions}")
        return {"err": "predictions missing"}

    rerun_pred = cal / args.gen / f"predictions_p{pid}_rerun.jsonl"
    n_kept = filter_predictions(src_predictions, rerun_pred, err_ids)
    print(f"  p{pid}: filtered {len(err_ids)} error_ids -> {n_kept} rows "
          f"in {rerun_pred.name}")

    rerun_run_id = f"{args.gen}_p{pid}_rerun"
    # Harness writes its report as
    #     <work_dir>/<model_name_or_path>.<run_id>.json
    # where model_name_or_path comes from the JSONL row (still "<gen>__p<pid>"
    # because we filtered the same file in-place). Compute the expected
    # location ahead of time so the merge step doesn't have to glob.
    rerun_report = eval_dir / f"{args.gen}__p{pid}.{rerun_run_id}.json"

    if rerun_report.exists() and not args.force:
        print(f"  p{pid}: rerun report already present at {rerun_report.name}; "
              "merging without re-invoking the harness (pass --force to re-eval)")
    else:
        if rerun_report.exists() and args.force:
            # Drop the stale report so a partial harness run doesn't make us
            # silently merge old results.
            rerun_report.unlink()
            print(f"  p{pid}: --force: removed stale {rerun_report.name}")
        # Reuse the calibration wrapper so DOCKER_HOST / SWEBENCH_PODMAN_COMPAT
        # / --namespace none / --cache_level instance defaults match exactly.
        import spot_check_generators as scg  # noqa: E402 (late import; needs podman env wired up)
        out_path = scg.run_swebench_eval(
            predictions_path=rerun_pred,
            run_id=rerun_run_id,
            max_workers=args.max_workers,
            work_dir=eval_dir,
            dataset_name=args.dataset,
        )
        print(f"  p{pid}: harness wrote {Path(out_path).name}")
        if not rerun_report.exists():
            print(f"  p{pid}: expected report at {rerun_report} not found; "
                  f"harness returned {out_path}")
            return {"err": "no rerun report"}

    stats = merge_rerun_into_original(report_path, rerun_report)
    print(f"  p{pid}: {stats}")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration-dir", required=True, type=Path,
                   help="The same dir passed as --output-dir to "
                        "spot_check_generators.py.")
    p.add_argument("--gen", required=True,
                   help="Generator slug whose error_ids should be re-run.")
    p.add_argument("--dataset", required=True,
                   choices=["princeton-nlp/SWE-bench_Lite",
                            "princeton-nlp/SWE-bench_Verified"])
    p.add_argument("--n-patches", type=int, default=3,
                   help="Re-run patches p0..p<n-1>.")
    p.add_argument("--patches", type=int, nargs="*", default=None,
                   help="Override --n-patches with an explicit list (e.g. --patches 0).")
    p.add_argument("--max-workers", type=int,
                   default=int(os.environ.get("SWE_EVAL_MAX_WORKERS", "4")))
    p.add_argument("--force", action="store_true",
                   help="Re-invoke the harness even if a prior rerun report "
                        "is already present.")
    p.add_argument("--podman", action="store_true",
                   help="Use podman-compat shim instead of regular Docker. "
                        "Required on Linux clusters where DOCKER_HOST points "
                        "at a podman.sock and the caller has not already "
                        "exported SWEBENCH_PODMAN_COMPAT. No-op otherwise.")
    p.add_argument("--namespace", type=str, default="none",
                   help="SWE-Bench harness --namespace. Default 'none' (build "
                        "instance images locally so the patcher's fixes "
                        "apply), since this script's purpose is to repair "
                        "build-errored instances. Set to e.g. 'swebench' to "
                        "fall back to docker.io/swebench/* pulls instead. "
                        "Set to '' to use the harness's own default.")
    args = p.parse_args()
    if args.podman:
        os.environ["SWEBENCH_PODMAN_COMPAT"] = "1"
    # Honour the wrapper's namespace contract: SWEBENCH_NAMESPACE env var
    # controls whether scg.run_swebench_eval passes --namespace through.
    # Set unconditionally so --namespace '' from the caller actually clears
    # any value inherited from the shell (otherwise the wrapper keeps
    # using whatever was already exported, defeating the CLI override).
    os.environ["SWEBENCH_NAMESPACE"] = args.namespace
    _maybe_setup_podman_shim()

    args.calibration_dir = args.calibration_dir.resolve()
    if not args.calibration_dir.exists():
        print(f"ERROR: calibration-dir missing: {args.calibration_dir}",
              file=sys.stderr)
        return 1

    # spot_check_generators lives in the scripts/ dir alongside us.
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    patches = args.patches if args.patches is not None else list(range(args.n_patches))
    print(f"rerun config: gen={args.gen}  dataset={args.dataset}  "
          f"patches={patches}  max-workers={args.max_workers}  "
          f"calibration-dir={args.calibration_dir}")

    total_promoted = 0
    total_still = 0
    for pid in patches:
        print(f"\n==== p{pid} ====")
        stats = rerun_patch(args, pid)
        total_promoted += stats.get("promoted_resolved", 0) + stats.get("promoted_unresolved", 0)
        total_still += stats.get("still_errored", 0)

    print(f"\nDone. Promoted out of error_ids: {total_promoted}.  "
          f"Still errored: {total_still}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
