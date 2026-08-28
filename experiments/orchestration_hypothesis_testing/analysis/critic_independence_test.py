"""Tests conditional independence of critic likelihoods given Y.

For each (benchmark, generator) cell:
- Load critic_results.jsonl
- Filter to records with Y in {0,1} and all 4 critic columns present
- For each Y stratum (Y=0, Y=1):
  - Observed joint distribution over 2^4 = 16 cells of (L0, L1, L2, L3)
  - Expected joint under independence = product of marginals
  - Pearson chi^2 statistic on the 16 cells
  - G^2 (likelihood ratio) statistic as cross-check
- Pool strata: total chi^2 = chi^2(Y=0) + chi^2(Y=1), df = 11 + 11 = 22
- Pairwise chi^2 tests for each (critic_i, critic_j) pair per stratum (6 pairs * 2 strata)
- Effect size: Cramer's V on each pairwise table

Reports:
- per-cell joint independence test result (chi^2, df, p-value, min expected count)
- per-cell pairwise tests (which pairs of critics are most correlated within Y stratum)
- aggregate across the 28-cell cube

Outputs:
- data/critic_independence/per_cell.json     (full results)
- data/critic_independence/per_cell.csv      (flat summary for paper appendix)
- data/critic_independence/pairwise.csv      (pairwise table for localization)

Usage:
  python3 scripts/critic_independence_test.py \\
      --data-root data \\
      --output-root data/critic_independence
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations, product
from pathlib import Path

from scipy.stats import chi2

CRITICS = ["L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review"]

# (benchmark_label, dir_name) — order matters for the output table.
BENCHMARKS = [
    ("LCB-hard", "lcb_calibration_v2"),
    ("LCB-medium", "lcb_calibration_medium"),
    ("LCB-easy", "lcb_calibration_easy"),
    ("MBPP+", "mbpp_calibration"),
    ("HumanEval+", "humaneval_calibration"),
    ("SWE-Lite", "swebench_lite"),
    ("SWE-Verified", "swebench_verified"),
]

GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45"]


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def filter_complete(records: list[dict]) -> list[dict]:
    """Keep records with Y in {0,1} and all 4 critic columns as int 0/1."""
    out = []
    for r in records:
        y = r.get("Y")
        if y not in (0, 1):
            continue
        ok = True
        for c in CRITICS:
            v = r.get(c)
            if v not in (0, 1, True, False):
                ok = False
                break
        if not ok:
            continue
        out.append(r)
    return out


def joint_counts(records: list[dict]) -> dict[tuple[int, int, int, int], int]:
    """Count 2^4 = 16 cells of (L0, L1, L2, L3) outcomes."""
    counts = {tup: 0 for tup in product([0, 1], repeat=4)}
    for r in records:
        key = tuple(int(bool(r[c])) for c in CRITICS)
        counts[key] += 1
    return counts


def marginal_probs(counts: dict[tuple[int, ...], int], n: int) -> list[float]:
    """For each critic index, return P(L_k = 1)."""
    probs = []
    for k in range(len(CRITICS)):
        p1 = sum(c for tup, c in counts.items() if tup[k] == 1) / max(n, 1)
        probs.append(p1)
    return probs


def expected_joint(probs: list[float], n: int) -> dict[tuple[int, ...], float]:
    """E[count(tup)] = n * prod_k P(L_k = tup[k])."""
    exp = {}
    for tup in product([0, 1], repeat=len(CRITICS)):
        p = 1.0
        for k, b in enumerate(tup):
            p *= probs[k] if b == 1 else (1 - probs[k])
        exp[tup] = n * p
    return exp


def chi2_stat(observed: dict, expected: dict, eps: float = 1e-12) -> float:
    return sum(
        ((observed[k] - expected[k]) ** 2) / max(expected[k], eps) for k in observed
    )


def g2_stat(observed: dict, expected: dict, eps: float = 1e-12) -> float:
    return 2 * sum(
        observed[k] * math.log(observed[k] / max(expected[k], eps))
        for k in observed
        if observed[k] > 0
    )


def independence_test_one_stratum(records: list[dict]) -> dict:
    """4-way joint independence test (Pearson chi^2 + G^2).

    Marks the stratum as degenerate when any critic's marginal probability is in
    {0, 1} (exact). Under such a marginal, the joint distribution collapses
    along that axis and the independence test is vacuous (chi^2 = 0).
    """
    n = len(records)
    if n == 0:
        return {"n": 0, "skipped": "empty stratum"}
    obs = joint_counts(records)
    probs = marginal_probs(obs, n)
    degenerate = any(p == 0.0 or p == 1.0 for p in probs)
    exp = expected_joint(probs, n)
    df = (2**4 - 1) - 4  # 16 - 1 - 4 = 11
    x2 = chi2_stat(obs, exp)
    g2 = g2_stat(obs, exp)
    p_x2 = 1 - chi2.cdf(x2, df)
    p_g2 = 1 - chi2.cdf(g2, df)
    min_exp = min(exp.values())
    n_low = sum(1 for v in exp.values() if v < 5)
    return {
        "n": n,
        "marginal_probs": dict(zip(CRITICS, probs)),
        "chi2": x2,
        "G2": g2,
        "df": df,
        "p_chi2": p_x2,
        "p_G2": p_g2,
        "min_expected": min_exp,
        "n_cells_low_expected": n_low,
        "warning_chi2_invalid": min_exp < 5,
        "degenerate": degenerate,
    }


def pairwise_tests(records: list[dict]) -> dict[str, dict]:
    """For each pair of critics, 2x2 contingency chi^2 + Cramer's V.

    Within a Y stratum, P(L_i, L_j) should equal P(L_i) * P(L_j) under
    pairwise independence. Returns per-pair statistics.
    """
    n = len(records)
    out = {}
    for i, j in combinations(range(len(CRITICS)), 2):
        # 2x2 table: rows = L_i, cols = L_j
        tab = [[0, 0], [0, 0]]
        for r in records:
            a = int(bool(r[CRITICS[i]]))
            b = int(bool(r[CRITICS[j]]))
            tab[a][b] += 1
        # marginals
        row_tot = [tab[0][0] + tab[0][1], tab[1][0] + tab[1][1]]
        col_tot = [tab[0][0] + tab[1][0], tab[0][1] + tab[1][1]]
        if n == 0 or min(row_tot) == 0 or min(col_tot) == 0:
            out[f"{CRITICS[i]}__{CRITICS[j]}"] = {
                "n": n,
                "skipped": "degenerate table",
                "table": tab,
            }
            continue
        # expected
        exp = [[(row_tot[r] * col_tot[c]) / n for c in (0, 1)] for r in (0, 1)]
        x2 = sum(((tab[r][c] - exp[r][c]) ** 2) / max(exp[r][c], 1e-12) for r in (0, 1) for c in (0, 1))
        df = 1
        p = 1 - chi2.cdf(x2, df)
        # Cramer's V for 2x2 = sqrt(chi2 / n)
        v = math.sqrt(x2 / n) if n > 0 else 0.0
        out[f"{CRITICS[i]}__{CRITICS[j]}"] = {
            "n": n,
            "table": tab,
            "expected": exp,
            "chi2": x2,
            "df": df,
            "p_value": p,
            "cramers_v": v,
        }
    return out


def analyze_cell(records: list[dict]) -> dict:
    """Run the full battery on one (benchmark, generator) cell."""
    by_y = {0: [r for r in records if r["Y"] == 0],
            1: [r for r in records if r["Y"] == 1]}
    out = {
        "n_total": len(records),
        "n_Y0": len(by_y[0]),
        "n_Y1": len(by_y[1]),
    }
    for y in (0, 1):
        out[f"joint_Y{y}"] = independence_test_one_stratum(by_y[y])
        out[f"pairwise_Y{y}"] = pairwise_tests(by_y[y])
    # Pooled chi^2 across strata: sum of stratum stats, df sum
    pooled_chi2 = 0.0
    pooled_g2 = 0.0
    pooled_df = 0
    for y in (0, 1):
        s = out[f"joint_Y{y}"]
        if "chi2" in s:
            pooled_chi2 += s["chi2"]
            pooled_g2 += s["G2"]
            pooled_df += s["df"]
    if pooled_df > 0:
        out["pooled"] = {
            "chi2": pooled_chi2,
            "G2": pooled_g2,
            "df": pooled_df,
            "p_chi2": 1 - chi2.cdf(pooled_chi2, pooled_df),
            "p_G2": 1 - chi2.cdf(pooled_g2, pooled_df),
        }
    else:
        out["pooled"] = {"skipped": "no usable stratum"}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="root containing <benchmark_dir>/<gen>/critic_results.jsonl")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="dir for per_cell.json + per_cell.csv + pairwise.csv")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_per_cell: dict[str, dict] = {}

    for bench_label, bench_dir in BENCHMARKS:
        for gen in GENERATORS:
            path = args.data_root / bench_dir / gen / "critic_results.jsonl"
            recs = filter_complete(load_records(path))
            cell_id = f"{bench_label}__{gen}"
            if not recs:
                out_per_cell[cell_id] = {"skipped": f"no records at {path}"}
                continue
            out_per_cell[cell_id] = analyze_cell(recs)

    # Write JSON
    json_path = args.output_root / "per_cell.json"
    json_path.write_text(json.dumps(out_per_cell, indent=2))

    # Flat CSV summary
    csv_path = args.output_root / "per_cell.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "generator", "n_total", "n_Y0", "n_Y1",
            "Y0_chi2", "Y0_p", "Y0_min_exp", "Y0_warn",
            "Y1_chi2", "Y1_p", "Y1_min_exp", "Y1_warn",
            "pooled_chi2", "pooled_df", "pooled_p", "pooled_p_G2",
        ])
        for bench_label, _ in BENCHMARKS:
            for gen in GENERATORS:
                cell = out_per_cell.get(f"{bench_label}__{gen}", {})
                if "skipped" in cell:
                    w.writerow([bench_label, gen, "skipped"] + [""] * 14)
                    continue
                y0 = cell.get("joint_Y0", {})
                y1 = cell.get("joint_Y1", {})
                pl = cell.get("pooled", {})
                w.writerow([
                    bench_label, gen,
                    cell.get("n_total"), cell.get("n_Y0"), cell.get("n_Y1"),
                    f"{y0.get('chi2', float('nan')):.3f}" if "chi2" in y0 else "",
                    f"{y0.get('p_chi2', float('nan')):.4f}" if "p_chi2" in y0 else "",
                    f"{y0.get('min_expected', float('nan')):.2f}" if "min_expected" in y0 else "",
                    "yes" if y0.get("warning_chi2_invalid") else "no",
                    f"{y1.get('chi2', float('nan')):.3f}" if "chi2" in y1 else "",
                    f"{y1.get('p_chi2', float('nan')):.4f}" if "p_chi2" in y1 else "",
                    f"{y1.get('min_expected', float('nan')):.2f}" if "min_expected" in y1 else "",
                    "yes" if y1.get("warning_chi2_invalid") else "no",
                    f"{pl.get('chi2', float('nan')):.3f}" if "chi2" in pl else "",
                    pl.get("df", ""),
                    f"{pl.get('p_chi2', float('nan')):.4f}" if "p_chi2" in pl else "",
                    f"{pl.get('p_G2', float('nan')):.4f}" if "p_G2" in pl else "",
                ])

    # Pairwise CSV
    pairwise_path = args.output_root / "pairwise.csv"
    with open(pairwise_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "generator", "Y", "critic_i", "critic_j",
                    "chi2", "p", "cramers_v", "n"])
        for bench_label, _ in BENCHMARKS:
            for gen in GENERATORS:
                cell = out_per_cell.get(f"{bench_label}__{gen}", {})
                if "skipped" in cell:
                    continue
                for y in (0, 1):
                    pw = cell.get(f"pairwise_Y{y}", {})
                    for pair_key, stat in pw.items():
                        if "skipped" in stat:
                            continue
                        ci, cj = pair_key.split("__")
                        w.writerow([
                            bench_label, gen, y, ci, cj,
                            f"{stat['chi2']:.3f}",
                            f"{stat['p_value']:.4f}",
                            f"{stat['cramers_v']:.3f}",
                            stat["n"],
                        ])

    # Aggregate summary printed to stdout
    n_cells = sum(1 for v in out_per_cell.values() if "skipped" not in v)
    print(f"Analyzed {n_cells} cells.")
    print(f"Per-cell JSON: {json_path}")
    print(f"Per-cell CSV:  {csv_path}")
    print(f"Pairwise CSV:  {pairwise_path}")
    print()

    # Headline: focus on Y=0 stratum (Y=1 is often degenerate when critics
    # deterministically PASS on correct solutions). Use G^2 (more robust to
    # sparse cells than chi^2).
    print("=== Headline: Y=0 joint independence test (G^2) ===")
    print(f"{'cell':40} {'n_Y0':>5}  {'G2':>8}  {'p_G2':>8}  {'degen':>6}")
    by_gen: dict[str, list[float]] = {g: [] for g in GENERATORS}
    n_reject_y0 = 0
    n_nondegen_y0 = 0
    for bench_label, _ in BENCHMARKS:
        for gen in GENERATORS:
            cell = out_per_cell.get(f"{bench_label}__{gen}", {})
            if "skipped" in cell:
                continue
            y0 = cell.get("joint_Y0", {})
            if "G2" not in y0:
                continue
            label = f"{bench_label}/{gen}"[:40]
            degen = "yes" if y0.get("degenerate") else "no"
            print(f"{label:40} {y0['n']:>5}  {y0['G2']:>8.2f}  {y0['p_G2']:>8.4f}  {degen:>6}")
            if not y0.get("degenerate"):
                n_nondegen_y0 += 1
                by_gen[gen].append(y0["p_G2"])
                if y0["p_G2"] < 0.05:
                    n_reject_y0 += 1
    print()
    print(f"Y=0 strata where independence rejected (G^2 p<0.05): {n_reject_y0}/{n_nondegen_y0} non-degenerate cells")
    print()
    print("=== Per-generator pattern (Y=0 G^2 rejection rate) ===")
    for gen in GENERATORS:
        ps = by_gen[gen]
        rejects = sum(1 for p in ps if p < 0.05)
        print(f"  {gen:12} reject={rejects}/{len(ps)} cells")
    print()

    # Pairwise summary: which (critic_i, critic_j) pairs are problematic?
    print("=== Pairwise correlation summary (Y=0 stratum, max Cramer's V across cells) ===")
    pair_max: dict[tuple[str, str], dict[str, float]] = {}
    for bench_label, _ in BENCHMARKS:
        for gen in GENERATORS:
            cell = out_per_cell.get(f"{bench_label}__{gen}", {})
            if "skipped" in cell:
                continue
            for pair_key, stat in cell.get("pairwise_Y0", {}).items():
                if "skipped" in stat:
                    continue
                pair_max.setdefault(pair_key, {}).setdefault(gen, 0.0)
                if stat["cramers_v"] > pair_max[pair_key][gen]:
                    pair_max[pair_key][gen] = stat["cramers_v"]
    print(f"{'pair':50} " + "  ".join(f"{g:>11}" for g in GENERATORS))
    for pair_key in pair_max:
        line = f"{pair_key:50} "
        for g in GENERATORS:
            v = pair_max[pair_key].get(g, float("nan"))
            line += f"  {v:>11.3f}"
        print(line)


if __name__ == "__main__":
    main()
