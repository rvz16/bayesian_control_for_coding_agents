#!/usr/bin/env python
"""Apply train/test split to existing synthesis-benchmark calibration data.

Reads `<src-dir>/<gen>/critic_results.jsonl` from a previous calibration run
(produced by lcb_calibrate.py / mbpp_calibrate.py / humaneval_calibrate.py),
deterministically splits instance_ids into train/test using SPLIT_SEED,
refits `likelihood_tables.json` from the train slice only, and writes a
`split.json` recording the partition.

This is the analog of N_TRAIN/SPLIT_SEED in the abbo bug-fix scripts
(test_codecontests_calibration.py, run_codecontests_full.py): the
likelihoods that will feed into GrFt/DPFt are computed on the train slice,
and the live agent runs only on test_ids.

After running this script, run_synthesis_endtoend.py picks up the same
SPLIT_SEED to reproduce the partition and runs Bayesian agents on test.

Usage:
  python scripts/synthesis_train_test_split.py \\
      --src-dir data/lcb_calibration_medium \\
      --generators haiku45,sonnet45,qwen3_coder,gpt5_mini \\
      --n-train 20 \\
      --split-seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from collections import defaultdict

log = logging.getLogger("synth_split")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# Default critic key names; we tolerate either spelling
CRITIC_KEYS = ["L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review"]


def load_records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            log.warning("skipping malformed line: %s", e)
    return rows


def instance_id_of(rec: dict) -> str:
    """Return the canonical instance id from a record (LCB uses question_id,
    MBPP/HumanEval+ use instance_id)."""
    for k in ("instance_id", "question_id"):
        if k in rec:
            return str(rec[k])
    raise KeyError(f"no instance/question id in record: {list(rec.keys())[:5]}")


def fit_likelihoods(records: list[dict], alpha: float = 1.0, beta: float = 1.0) -> dict:
    """Beta(alpha,beta)-smoothed estimate of P(z_k = pass | Y) per critic.

    Returns a dict shaped like likelihood_tables.json:
      {"prior_Y1": float,
       "critic_likelihoods": {
            "L0_syntax": {"P_pass_given_Y1": ..., "P_pass_given_Y0": ...,
                         "gap": ..., "TP": ..., "FP": ..., "TN": ..., "FN": ...},
            ...}}
    """
    n = len(records)
    n_y1 = sum(int(r.get("Y", 0)) for r in records)
    prior_y1 = n_y1 / n if n else 0.0

    out_critics: dict[str, dict] = {}
    for critic in CRITIC_KEYS:
        TP = FP = TN = FN = 0
        for r in records:
            z = r.get(critic)
            if z is None:
                continue
            y = int(r.get("Y", 0))
            passed = bool(z)
            if y == 1 and passed: TP += 1
            elif y == 0 and passed: FP += 1
            elif y == 0 and not passed: TN += 1
            elif y == 1 and not passed: FN += 1
        # Beta(α,β) smoothing
        p1 = (TP + alpha) / (TP + FN + alpha + beta) if (TP + FN) or True else 0.5
        p0 = (FP + alpha) / (FP + TN + alpha + beta) if (FP + TN) or True else 0.5
        out_critics[critic] = {
            "P_pass_given_Y1": p1,
            "P_pass_given_Y0": p0,
            "gap": p1 - p0,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "alpha": alpha, "beta": beta,
        }
    return {"prior_Y1": prior_y1, "critic_likelihoods": out_critics}


def split_one_generator(
    src_dir: Path, gen: str, n_train: int, split_seed: int, alpha: float, beta: float
) -> None:
    gen_dir = src_dir / gen
    cr_path = gen_dir / "critic_results.jsonl"
    if not cr_path.exists():
        log.warning("no critic_results.jsonl at %s — skipping", cr_path)
        return
    records = load_records(cr_path)
    if not records:
        log.warning("empty critic_results.jsonl at %s — skipping", cr_path)
        return

    # Group by instance_id
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_inst[instance_id_of(r)].append(r)
    all_ids = sorted(by_inst.keys())

    if n_train >= len(all_ids):
        raise SystemExit(
            f"[{gen}] n_train={n_train} >= n_instances={len(all_ids)}; "
            f"need at least one held-out test instance."
        )

    # Deterministic shuffle with split_seed
    rng = random.Random(split_seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    train_ids = shuffled[:n_train]
    test_ids = shuffled[n_train:]
    log.info("[%s] %d train, %d test (split_seed=%d)",
             gen, len(train_ids), len(test_ids), split_seed)

    # Fit likelihoods from train slice
    train_records = [r for r in records if instance_id_of(r) in set(train_ids)]
    fitted = fit_likelihoods(train_records, alpha=alpha, beta=beta)
    fitted["source"] = "synthesis_train_test_split"
    fitted["n_train_instances"] = len(train_ids)
    fitted["n_train_records"] = len(train_records)
    fitted["split_seed"] = split_seed
    fitted["generator"] = gen

    # Save
    lk_path = gen_dir / "likelihood_tables.json"
    if lk_path.exists():
        # Backup the existing one (which was fit in-sample on all instances)
        backup = gen_dir / "likelihood_tables.insample_backup.json"
        if not backup.exists():
            backup.write_text(lk_path.read_text())
            log.info("[%s] backed up insample likelihoods → %s", gen, backup.name)
    lk_path.write_text(json.dumps(fitted, indent=2))
    log.info("[%s] wrote train-fit likelihoods → %s", gen, lk_path.name)

    # Save split manifest
    split_path = gen_dir / "split.json"
    split_path.write_text(json.dumps({
        "split_seed": split_seed,
        "n_instances_total": len(all_ids),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_ids": sorted(train_ids),
        "test_ids": sorted(test_ids),
    }, indent=2))
    log.info("[%s] wrote split → %s", gen, split_path.name)

    # Print summary
    print(f"\n=== {gen} ===")
    print(f"  prior_Y1 = {fitted['prior_Y1']:.3f}  (train n={len(train_ids)} instances)")
    for name, lk in fitted["critic_likelihoods"].items():
        print(f"  {name:<20} P(pass|Y=1)={lk['P_pass_given_Y1']:.3f} "
              f"P(pass|Y=0)={lk['P_pass_given_Y0']:.3f} gap={lk['gap']:+.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-dir", required=True, type=Path,
                   help="Directory with <gen>/critic_results.jsonl (output of "
                        "lcb_calibrate.py / mbpp_calibrate.py / humaneval_calibrate.py)")
    p.add_argument("--generators", required=True,
                   help="Comma-separated generator slugs (e.g. haiku45,sonnet45)")
    p.add_argument("--n-train", type=int, required=True,
                   help="Number of instances used for fitting likelihoods (train slice)")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Random seed for the train/test partition (default 42)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Beta prior alpha for likelihood fitting (default 1.0)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Beta prior beta for likelihood fitting (default 1.0)")
    args = p.parse_args()

    src_dir = args.src_dir.resolve()
    if not src_dir.exists():
        raise SystemExit(f"src-dir does not exist: {src_dir}")

    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    for gen in generators:
        split_one_generator(src_dir, gen, args.n_train, args.split_seed,
                            args.alpha, args.beta)


if __name__ == "__main__":
    main()
