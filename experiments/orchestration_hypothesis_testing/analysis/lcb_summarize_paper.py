"""Consolidate all calibration results into a single paper-ready table.

Walks any number of (label, dir) pairs, collects per-generator
policy_comparison*.json and likelihood_tables.json, emits a flat CSV +
a structured JSON.

Output:
  <root>/PAPER_TABLE.csv   — flat: cell, generator, l3_reviewer, policy,
                              utility, pass_rate, diff, ci_lo, ci_hi
  <root>/PAPER_TABLE.json  — keyed by (cell, generator, l3_reviewer)

Usage (legacy — still supported):
  python lcb_summarize_paper.py \\
    --hard-dir data/lcb_calibration_v2 \\
    --medium-dir data/lcb_calibration_medium \\
    --output-root data

Usage (generic — preferred for ≥3 benchmarks):
  python lcb_summarize_paper.py \\
    --cells lcb_hard=data/lcb_calibration_v2,lcb_medium=data/lcb_calibration_medium,\\
            lcb_easy=data/lcb_calibration_easy,mbpp=data/mbpp_calibration,\\
            humaneval=data/humaneval_calibration \\
    --output-root data
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Allow optional kernel-source suffix: policy_comparison_l3_<rev>.json,
# policy_comparison_kernel_measured.json, policy_comparison.json
COMPARISON_RE = re.compile(r"^policy_comparison(?:_l3_(\w+)|_(?:kernel_)?(\w+))?\.json$")


def collect_one_generator(gen_dir: Path, cell: str) -> list[dict]:
    """Read all policy_comparison*.json + likelihood_tables.json from a
    generator dir, return flat row list."""
    out = []
    likes_path = gen_dir / "likelihood_tables.json"
    likes = {}
    if likes_path.exists():
        likes = json.loads(likes_path.read_text())
    prior = likes.get("prior_Y1")
    cl = likes.get("critic_likelihoods", {})

    for f in gen_dir.glob("policy_comparison*.json"):
        m = COMPARISON_RE.match(f.name)
        if not m:
            continue
        # Three filename forms:
        #   policy_comparison.json                      — base
        #   policy_comparison_l3_<reviewer>.json        — reviewer-swap (group 1)
        #   policy_comparison_kernel_measured.json etc. — kernel ablation (group 2)
        l3_label = m.group(1) or "haiku45_default"
        kernel_tag = m.group(2)
        if kernel_tag:
            l3_label = f"haiku45_default__{kernel_tag}"
        data = json.loads(f.read_text())

        # File schemas differ:
        #   policy_comparison.json — flat dict {policy: {...}}
        #   policy_comparison_l3_*.json — wraps under "policies"
        #   policy_comparison_kernel_*.json — wraps under "policies" + adds kernel_source
        if "policies" in data:
            policies = data["policies"]
            l3_gap_used = data.get("L3_gap_with_reviewer")
        else:
            policies = data
            l3_gap_used = cl.get("L3_llm_review", {}).get("gap")

        for name, r in policies.items():
            out.append({
                "cell": cell,
                "generator": gen_dir.name,
                "l3_reviewer": l3_label,
                "prior_Y1": prior,
                "L0_gap": cl.get("L0_syntax", {}).get("gap"),
                "L2_gap": cl.get("L2_public_tests", {}).get("gap"),
                "L3_gap_used": l3_gap_used,
                "policy": name,
                "mean_utility": r.get("mean_utility"),
                "pass_rate": r.get("pass_rate"),
                "diff_vs_always_verify": r.get("diff_vs_always_verify"),
                "ci95_lo": r.get("ci95_lo"),
                "ci95_hi": r.get("ci95_hi"),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-dir", type=Path, default=None)
    parser.add_argument("--medium-dir", type=Path, default=None)
    parser.add_argument("--cells", type=str, default=None,
                        help="Comma-separated label=path pairs covering any number of cells. "
                             "If supplied, supersedes --hard-dir/--medium-dir.")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    cells: list[tuple[str, Path]] = []
    if args.cells:
        for pair in args.cells.split(","):
            pair = pair.strip()
            if not pair: continue
            label, p = pair.split("=", 1)
            cells.append((label.strip(), Path(p.strip())))
    else:
        if args.hard_dir is not None:
            cells.append(("hard", args.hard_dir))
        if args.medium_dir is not None:
            cells.append(("medium", args.medium_dir))
    if not cells:
        raise SystemExit("must pass either --cells or --hard-dir/--medium-dir")

    rows: list[dict] = []
    for cell_label, dir_ in cells:
        if not dir_.exists():
            print(f"skip {cell_label}: {dir_} missing")
            continue
        for gen_dir in sorted(p for p in dir_.iterdir() if p.is_dir()):
            if gen_dir.name.startswith("_") or gen_dir.name.startswith("."):
                continue
            new_rows = collect_one_generator(gen_dir, cell_label)
            if new_rows:
                print(f"  {cell_label}/{gen_dir.name}: {len(new_rows)} rows")
                rows.extend(new_rows)

    # CSV
    cols = ["cell", "generator", "l3_reviewer", "prior_Y1",
            "L0_gap", "L2_gap", "L3_gap_used",
            "policy", "mean_utility", "pass_rate",
            "diff_vs_always_verify", "ci95_lo", "ci95_hi"]
    csv_path = args.output_root / "PAPER_TABLE.csv"
    def fmt(v):
        if v is None: return ""
        if isinstance(v, float): return f"{v:.4f}"
        return str(v)
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(fmt(r.get(c)) for c in cols) + "\n")

    # JSON keyed by (difficulty, generator, l3_reviewer, policy)
    structured: dict = {}
    for r in rows:
        d = structured.setdefault(r["cell"], {})
        g = d.setdefault(r["generator"], {})
        rev = g.setdefault(r["l3_reviewer"], {"prior_Y1": r["prior_Y1"],
                                                "L0_gap": r["L0_gap"],
                                                "L2_gap": r["L2_gap"],
                                                "L3_gap_used": r["L3_gap_used"],
                                                "policies": {}})
        rev["policies"][r["policy"]] = {
            "mean_utility": r["mean_utility"],
            "pass_rate": r["pass_rate"],
            "diff_vs_always_verify": r["diff_vs_always_verify"],
            "ci95_lo": r["ci95_lo"],
            "ci95_hi": r["ci95_hi"],
        }
    json_path = args.output_root / "PAPER_TABLE.json"
    json_path.write_text(json.dumps(structured, indent=2))

    # Print headline: for each (cell, generator), the bayesian_greedy diff
    # under the default L3 reviewer (haiku45_default).
    print("\n=== HEADLINE: bayesian_greedy Δ vs always_verify ===")
    print(f"{'cell':<14} {'generator':<14} {'l3_reviewer':<18} {'L2_gap':>8} {'Δ utility':>10} {'95% CI':>22}")
    for diff in sorted(structured.keys()):
        for gen, revs in structured[diff].items():
            for rev_name, rev in revs.items():
                p = rev["policies"].get("bayesian_greedy")
                if not p or rev_name != "haiku45_default":
                    continue
                ci = f"[{p['ci95_lo']:>+6.2f},{p['ci95_hi']:>+6.2f}]"
                l2 = rev.get("L2_gap")
                l2_s = f"{l2:.3f}" if l2 is not None else "n/a"
                print(f"{diff:<14} {gen:<14} {rev_name:<18} {l2_s:>8} "
                      f"{p['diff_vs_always_verify']:>+10.2f} {ci:>22}")

    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
