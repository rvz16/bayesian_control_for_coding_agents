"""Analyze the L3 reviewer sweep produced by lcb_l3_sweep.py.

Computes, per (output_dir, generator):
  - Per-reviewer P(pass|Y=1), P(pass|Y=0), gap (with Beta(1,1) smoothing)
  - Pairwise reviewer agreement (Cohen-style fraction-agree by Y stratum)
  - The 2D (generator × reviewer) gap heatmap as JSON

Output: <gen>/L3_sweep_likelihoods.json
        <output-dir>/L3_sweep_summary.json   (rolled up across generators)
        <output-dir>/L3_sweep_summary.csv    (paper-friendly)

Usage:
  python lcb_l3_analyze.py \\
    --output-dir data/lcb_calibration_v2 \\
    --generators gpt5_mini,qwen3_coder,haiku45
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def per_reviewer_likes(rows: list[dict], reviewers: list[str]) -> dict[str, dict]:
    out = {}
    y1 = [r for r in rows if r.get("Y") == 1]
    y0 = [r for r in rows if r.get("Y") == 0]
    for lbl in reviewers:
        key = f"L3_{lbl}"
        # Filter rows where this reviewer was actually evaluated
        y1_v = [r for r in y1 if r.get(key) is not None]
        y0_v = [r for r in y0 if r.get(key) is not None]
        TP = sum(1 for r in y1_v if r[key])
        FN = sum(1 for r in y1_v if not r[key])
        FP = sum(1 for r in y0_v if r[key])
        TN = sum(1 for r in y0_v if not r[key])
        # Beta(1,1) smoothing
        p1 = (TP + 1) / (TP + FN + 2)
        p0 = (FP + 1) / (FP + TN + 2)
        out[lbl] = {
            "n_y1": len(y1_v), "n_y0": len(y0_v),
            "TP": TP, "FN": FN, "FP": FP, "TN": TN,
            "P_pass_given_Y1": p1, "P_pass_given_Y0": p0,
            "gap": p1 - p0,
        }
    return out


def pairwise_agreement(rows: list[dict], reviewers: list[str]) -> dict:
    """For each pair (a, b), compute fraction-agree on Y=0 records and Y=1 records."""
    out = {}
    for i, a in enumerate(reviewers):
        for b in reviewers[i+1:]:
            ka, kb = f"L3_{a}", f"L3_{b}"
            for stratum, label in ((1, "y1"), (0, "y0")):
                both = [r for r in rows if r.get("Y") == stratum
                        and r.get(ka) is not None and r.get(kb) is not None]
                if not both:
                    out[f"{a}|{b}|{label}"] = {"n": 0, "agree": None}
                    continue
                agree = sum(1 for r in both if r[ka] == r[kb])
                out[f"{a}|{b}|{label}"] = {"n": len(both),
                                            "agree_rate": agree / len(both)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    summary = {"generators": {}, "reviewers_seen": []}
    csv_lines = ["generator,reviewer,n_y1,n_y0,P_pass_Y1,P_pass_Y0,gap"]

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        gen_dir = out_dir / gen
        sweep_path = gen_dir / "L3_sweep.jsonl"
        if not sweep_path.exists():
            print(f"[{gen}] no L3_sweep.jsonl, skipping")
            continue
        # Dedup: keep the latest row per (instance_id, patch_id)
        # (append-mode writes can produce multiple rows per key across reruns;
        # later rows take precedence because they include more reviewer columns)
        latest: dict[tuple, dict] = {}
        for line in open(sweep_path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (str(r.get("instance_id", "")), int(r.get("patch_id", -1)))
            prev = latest.get(key, {})
            # Merge: later writes overwrite, but None values don't clobber non-None
            merged = dict(prev)
            for k, v in r.items():
                if v is None and k in merged and merged[k] is not None:
                    continue
                merged[k] = v
            latest[key] = merged
        rows = list(latest.values())
        # Discover reviewer columns from the data
        reviewers = sorted({k[3:] for r in rows for k in r if k.startswith("L3_")})
        if not reviewers:
            print(f"[{gen}] no L3_* columns")
            continue
        likes = per_reviewer_likes(rows, reviewers)
        agree = pairwise_agreement(rows, reviewers)
        out = {
            "generator": gen, "n_records": len(rows),
            "reviewers": reviewers,
            "per_reviewer": likes,
            "pairwise_agreement": agree,
        }
        (gen_dir / "L3_sweep_likelihoods.json").write_text(json.dumps(out, indent=2))
        summary["generators"][gen] = {"n_records": len(rows), "per_reviewer": likes}
        summary["reviewers_seen"] = sorted(set(summary["reviewers_seen"]) | set(reviewers))

        print(f"\n=== {gen} (n={len(rows)}) ===")
        print(f"{'reviewer':<14} {'n_y1':>5} {'n_y0':>5} {'P(pass|Y=1)':>12} {'P(pass|Y=0)':>12} {'gap':>8}")
        for lbl in reviewers:
            l = likes[lbl]
            print(f"  {lbl:<12} {l['n_y1']:>5} {l['n_y0']:>5} "
                  f"{l['P_pass_given_Y1']:>12.3f} {l['P_pass_given_Y0']:>12.3f} {l['gap']:>+8.3f}")
            csv_lines.append(f"{gen},{lbl},{l['n_y1']},{l['n_y0']},"
                             f"{l['P_pass_given_Y1']:.4f},{l['P_pass_given_Y0']:.4f},{l['gap']:+.4f}")

    (out_dir / "L3_sweep_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "L3_sweep_summary.csv").write_text("\n".join(csv_lines) + "\n")

    # 2D heatmap as JSON: rows=generators, cols=reviewers, vals=gap
    heatmap = {"generators": list(summary["generators"].keys()),
               "reviewers": summary["reviewers_seen"], "gap": []}
    for gen in heatmap["generators"]:
        per_r = summary["generators"][gen]["per_reviewer"]
        heatmap["gap"].append([per_r[lbl]["gap"] if lbl in per_r else None
                                for lbl in heatmap["reviewers"]])
    (out_dir / "L3_sweep_heatmap.json").write_text(json.dumps(heatmap, indent=2))
    print(f"\nwrote summary: {out_dir / 'L3_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
