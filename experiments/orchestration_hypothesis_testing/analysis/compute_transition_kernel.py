"""Compute transition kernel P(Y_{t+1} | Y_t) from iter_records.jsonl.

Step 0 = original spot-check patch (Y from harness).
Steps 1..N = refinement attempts (Y reported here is from a separate
Phase-2 harness eval of the refinement predictions; if not yet run, Y
is None and the trajectory is dropped from kernel computation).

Adds an `identical_to_prev` flag per step (sha256 of diff/code text) so we can
either include or exclude trivial "stay" transitions caused by the model
emitting byte-identical patches across steps.

Output: <generator>/transition_kernel.json with both:
  - kernel_all: includes identical-patch transitions (worst case for
    P(fix|broken), since identical = always-stay)
  - kernel_changed_only: only counts transitions where the patch
    actually changed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.kernel import compute_transition_kernel_from_pairs  # noqa: E402


def _artifact_hash(record: dict) -> str:
    text = record.get("diff")
    if not isinstance(text, str) or not text:
        text = record.get("code")
    return hashlib.sha256((text or "").encode()).hexdigest()


def _annotate_identical(records: list[dict]) -> list[dict]:
    """Group by instance, sort by step, set identical_to_prev flag."""
    by_inst: dict[str, list[dict]] = {}
    for r in records:
        by_inst.setdefault(r["instance_id"], []).append(r)
    annotated = []
    for inst, rs in by_inst.items():
        rs = sorted(rs, key=lambda r: r["step"])
        prev_hash = None
        for r in rs:
            r["diff_hash"] = _artifact_hash(r)
            r["identical_to_prev"] = (prev_hash is not None and r["diff_hash"] == prev_hash)
            prev_hash = r["diff_hash"]
            annotated.append(r)
    return annotated


def _kernel_from_pairs(pairs: list[tuple[int, int]]) -> dict:
    """pairs: list of (Y_t, Y_{t+1}) with Y in {0, 1}.

    Output schema preserved for downstream consumers: includes the legacy
    `n_broken_transitions` / `n_correct_transitions` aliases alongside the
    canonical `n_broken_observed` / `n_correct_observed`.
    """
    k = compute_transition_kernel_from_pairs(pairs)
    # Legacy alias names — keep both so old jq queries against existing
    # transition_kernel.json files still work.
    k["n_broken_transitions"] = k["n_broken_observed"]
    k["n_correct_transitions"] = k["n_correct_observed"]
    # The pre-shared-module impl reported "Beta(1,1)" verbatim regardless of
    # the actual prior; preserve that exact string for back-compat.
    k["smoothing"] = "Beta(1,1)"
    return k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        records_path = out_dir / gen / "iter_records.jsonl"
        if not records_path.exists():
            print(f"[{gen}] no iter_records.jsonl, skipping")
            continue
        with open(records_path) as f:
            records = [json.loads(line) for line in f]
        records = _annotate_identical(records)
        # rewrite with annotations
        records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        # Build pairs by instance
        by_inst: dict[str, list[dict]] = {}
        for r in records:
            by_inst.setdefault(r["instance_id"], []).append(r)

        pairs_all: list[tuple[int, int]] = []
        pairs_changed: list[tuple[int, int]] = []
        n_identical = 0
        n_y_missing = 0
        for inst, rs in by_inst.items():
            rs = sorted(rs, key=lambda r: r["step"])
            for i in range(len(rs) - 1):
                yt = rs[i].get("Y")
                yt1 = rs[i + 1].get("Y")
                if yt is None or yt1 is None:
                    n_y_missing += 1
                    continue
                pairs_all.append((int(yt), int(yt1)))
                if rs[i + 1].get("identical_to_prev"):
                    n_identical += 1
                else:
                    pairs_changed.append((int(yt), int(yt1)))

        kernel = {
            "generator": gen,
            "n_records": len(records),
            "n_instances": len(by_inst),
            "n_y_missing": n_y_missing,
            "n_identical_transitions": n_identical,
            "kernel_all": _kernel_from_pairs(pairs_all),
            "kernel_changed_only": _kernel_from_pairs(pairs_changed),
        }
        out_path = out_dir / gen / "transition_kernel.json"
        out_path.write_text(json.dumps(kernel, indent=2))
        print(f"\n=== {gen} transition kernel ===")
        print(f"  n_pairs (all):           {kernel['kernel_all']['n_pairs']}")
        print(f"  n_pairs (changed only):  {kernel['kernel_changed_only']['n_pairs']}")
        print(f"  n_identical_transitions: {n_identical}")
        print(f"  n_y_missing:             {n_y_missing}")
        print(f"  P(fix|broken)   all={kernel['kernel_all']['P_fix_given_broken']:.3f}  "
              f"changed={kernel['kernel_changed_only']['P_fix_given_broken']:.3f}")
        print(f"  P(break|correct) all={kernel['kernel_all']['P_break_given_correct']:.3f}  "
              f"changed={kernel['kernel_changed_only']['P_break_given_correct']:.3f}")


if __name__ == "__main__":
    main()
