"""Re-evaluate L2_public_tests on existing MBPP+ data using the lenient runner.

Reads <gen>/critic_results.jsonl + <gen>/raw_responses/<inst>_p<pid>.txt and
recomputes only L2_public_tests using the new (lenient) run_assertions from
mbpp_calibrate.py. Y / L0 / L1 / L3 are preserved as-is.

Backups the old jsonl to <gen>/critic_results.jsonl.bak_strict before
overwriting. Idempotent — running twice yields the same result.

Usage:
  python3 mbpp_re_eval_l2.py \\
    --output-dir data/mbpp_calibration \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibration.mbpp import run_assertions, load_mbpp_plus  # noqa: E402
from calibration.lcb import extract_code  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mbpp_reeval")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    # Pre-load problem map (instance_id -> {test_list})
    log.info("loading MBPP+...")
    problems = load_mbpp_plus(0)  # n=0 → no sampling, all 378
    by_id = {str(p["task_id"]): p for p in problems}
    log.info("indexed %d problems", len(by_id))

    out_dir = args.output_dir.resolve()
    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        gen_dir = out_dir / gen
        rec_path = gen_dir / "critic_results.jsonl"
        raw_dir = gen_dir / "raw_responses"
        if not rec_path.exists() or not raw_dir.exists():
            log.warning("[%s] missing data, skipping", gen)
            continue

        records = [json.loads(l) for l in open(rec_path) if l.strip()]
        log.info("[%s] loaded %d records", gen, len(records))

        # Backup
        bak = rec_path.with_suffix(".jsonl.bak_strict")
        if not bak.exists():
            shutil.copy2(rec_path, bak)
            log.info("[%s] backed up old strict L2 to %s", gen, bak.name)

        def reeval_one(idx_rec):
            idx, rec = idx_rec
            inst_id = str(rec["instance_id"])
            pid = int(rec["patch_id"])
            problem = by_id.get(inst_id)
            if problem is None:
                return idx, rec.get("L2_public_tests"), None
            test_list = problem.get("test_list") or []
            raw_path = raw_dir / f"{inst_id}_p{pid}.txt"
            if not raw_path.exists():
                return idx, rec.get("L2_public_tests"), None
            code = extract_code(raw_path.read_text())
            l2_pass, l2_total = run_assertions(code, test_list)
            new_l2 = (l2_pass == l2_total) and l2_total > 0
            return idx, rec.get("L2_public_tests"), new_l2

        n_changed = 0
        n_total = 0
        n_pos_to_neg = 0  # was True, now False
        n_neg_to_pos = 0  # was False, now True
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(reeval_one, (i, r)) for i, r in enumerate(records)]
            for fut in as_completed(futures):
                idx, old, new = fut.result()
                if new is None:
                    continue
                n_total += 1
                if bool(old) != bool(new):
                    n_changed += 1
                    if old and not new: n_pos_to_neg += 1
                    if (not old) and new: n_neg_to_pos += 1
                records[idx]["L2_public_tests"] = bool(new)

        log.info("[%s] re-evaluated %d records: %d changed (%d True→False, %d False→True)",
                 gen, n_total, n_changed, n_pos_to_neg, n_neg_to_pos)

        # Write back
        with open(rec_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        log.info("[%s] wrote %s", gen, rec_path.name)

        # Print before/after critic stats
        y1 = [r for r in records if r["Y"] == 1]
        y0 = [r for r in records if r["Y"] == 0]
        def rate(rs, k): return sum(bool(r.get(k)) for r in rs) / max(len(rs), 1)
        l2_y1 = rate(y1, "L2_public_tests"); l2_y0 = rate(y0, "L2_public_tests")
        print(f"  [{gen}] new L2: P(z|Y=1)={l2_y1:.3f}  P(z|Y=0)={l2_y0:.3f}  gap={l2_y1-l2_y0:+.3f}")


if __name__ == "__main__":
    main()
