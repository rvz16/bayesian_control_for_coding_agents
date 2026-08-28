"""Split iter_records.jsonl into per-step predictions files for the harness.

Reads <gen>/iter_records.jsonl, writes <gen>/predictions_iter_step{0..N}.jsonl
with one record per (instance, step) in the format the SWE-bench harness
expects:

  {"instance_id": ..., "model_name_or_path": "<gen>__iter_step{S}", "model_patch": <diff>}

Use these files as inputs to `python -m swebench.harness.run_evaluation`.

Usage:
  python3 split_iter_predictions.py --output-dir data/spot_check_n50 \\
      --generators gpt5_mini,qwen3_coder
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        rec_path = out_dir / gen / "iter_records.jsonl"
        if not rec_path.exists():
            print(f"[{gen}] no iter_records.jsonl, skipping")
            continue
        by_step: dict[int, list[dict]] = defaultdict(list)
        with open(rec_path) as f:
            for line in f:
                r = json.loads(line)
                row = {
                    "instance_id": r["instance_id"],
                    "model_name_or_path": f"{gen}__iter_step{r['step']}",
                    "model_patch": r.get("diff", "") or "",
                }
                by_step[r["step"]].append(row)
        for step, rows in sorted(by_step.items()):
            out_path = out_dir / gen / f"predictions_iter_step{step}.jsonl"
            out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            print(f"  {out_path.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
