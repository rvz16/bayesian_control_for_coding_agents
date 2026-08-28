"""Within-repo cluster bootstrap for SWE-bench cells.

The default paired-bootstrap CIs in policy_comparison.json assume IID
instances. SWE-bench Lite/Verified have within-repo correlation: instances
from the same repo (django, sympy, etc.) share library/style structure that
makes them more similar than random. This violates the IID assumption and
underestimates CI width.

Cluster bootstrap: resample REPOS (with replacement), then take all the
instances belonging to each resampled repo. This gives an honest CI under
within-repo dependence.

Reads `critic_results.jsonl` for SWE cells; uses repo prefix from
instance_id (e.g., "astropy__astropy-7746" -> "astropy") as cluster.

Output:
  data/cluster_bootstrap/per_cell.csv
  data/cluster_bootstrap/per_cell.json

Usage:
  python3 scripts/cluster_bootstrap_swe.py \\
      --data-root data \\
      --output-root data/cluster_bootstrap
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# Cost model for re-computing utility from critic_results
COSTS = {"c_gen": 5, "c_L0": 1, "c_L2": 2, "c_L3": 5, "c_ver": 30, "reward": 100}

SWE_CELLS = [
    ("SWE-Lite", "swebench_lite"),
    ("SWE-Verified", "swebench_verified"),
]
GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45"]


def repo_from_instance(inst_id: str) -> str:
    """Extract repo prefix as cluster id."""
    if "__" in inst_id:
        return inst_id.split("__")[0]
    return inst_id.split("-")[0] if "-" in inst_id else inst_id


def load_records(path: Path) -> dict[str, list[dict]]:
    """Group critic_results.jsonl by instance_id."""
    if not path.exists():
        return {}
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "instance_id" not in r:
            continue
        by_inst[r["instance_id"]].append(r)
    for rs in by_inst.values():
        rs.sort(key=lambda r: r.get("patch_id", 0))
    return dict(by_inst)


def utility_always_verify(traj: list[dict]) -> float:
    if not traj:
        return 0.0
    y0 = int(bool(traj[0].get("Y") or 0))
    return COSTS["reward"] * y0 - COSTS["c_ver"]


def utility_threshold_L2(traj: list[dict]) -> float:
    """threshold_L2: run L2 cheaply, verify only if L2 PASS, else regenerate
    next patch (cost c_gen) and try L2 again. Up to 3 patches."""
    cost = 0.0
    reward = 0.0
    for i, rec in enumerate(traj[:3]):
        cost += COSTS["c_L2"]
        if rec.get("L2_public_tests"):
            cost += COSTS["c_ver"]
            reward = COSTS["reward"] * (rec.get("Y") or 0)
            return reward - cost
        if i < len(traj) - 1:
            cost += COSTS["c_gen"]
    return reward - cost  # gave up


def cluster_bootstrap_ci(by_inst: dict[str, list[dict]], policy_fn,
                          baseline_fn, n_boot: int = 1000,
                          seed: int = 42) -> tuple[float, float, float]:
    """Cluster bootstrap by repo. Returns (mean_diff, ci_lo, ci_hi)."""
    insts = list(by_inst.keys())
    by_repo: dict[str, list[str]] = defaultdict(list)
    for inst in insts:
        by_repo[repo_from_instance(inst)].append(inst)
    repos = list(by_repo.keys())

    # Compute observed mean diff
    diffs_per_inst = {}
    for inst, traj in by_inst.items():
        diffs_per_inst[inst] = policy_fn(traj) - baseline_fn(traj)
    observed = sum(diffs_per_inst.values()) / len(diffs_per_inst) if diffs_per_inst else 0.0

    # Cluster bootstrap: resample repos
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sampled_repos = [rng.choice(repos) for _ in range(len(repos))]
        sample_diffs = []
        for repo in sampled_repos:
            for inst in by_repo[repo]:
                sample_diffs.append(diffs_per_inst[inst])
        if not sample_diffs:
            continue
        boot_means.append(sum(sample_diffs) / len(sample_diffs))
    boot_means.sort()
    n = len(boot_means)
    if n < 100:
        return (observed, observed, observed)
    lo = boot_means[int(0.025 * n)]
    hi = boot_means[int(0.975 * n)]
    return (observed, lo, hi)


def iid_bootstrap_ci(by_inst: dict[str, list[dict]], policy_fn,
                      baseline_fn, n_boot: int = 1000,
                      seed: int = 42) -> tuple[float, float, float]:
    """Standard paired-bootstrap CI assuming IID instances. For comparison."""
    insts = list(by_inst.keys())
    diffs = {inst: policy_fn(by_inst[inst]) - baseline_fn(by_inst[inst]) for inst in insts}
    observed = sum(diffs.values()) / len(diffs) if diffs else 0.0
    rng = random.Random(seed)
    boot_means = []
    inst_list = list(insts)
    for _ in range(n_boot):
        sampled = [rng.choice(inst_list) for _ in range(len(inst_list))]
        sample_diffs = [diffs[inst] for inst in sampled]
        boot_means.append(sum(sample_diffs) / len(sample_diffs))
    boot_means.sort()
    n = len(boot_means)
    return (observed, boot_means[int(0.025 * n)], boot_means[int(0.975 * n)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell_label, dir_name in SWE_CELLS:
        for gen in GENERATORS:
            path = args.data_root / dir_name / gen / "critic_results.jsonl"
            by_inst = load_records(path)
            if not by_inst:
                continue
            n_inst = len(by_inst)
            n_repos = len(set(repo_from_instance(inst) for inst in by_inst))
            # threshold_L2 vs always_verify
            iid_d, iid_lo, iid_hi = iid_bootstrap_ci(by_inst, utility_threshold_L2,
                                                     utility_always_verify, args.n_boot)
            cl_d, cl_lo, cl_hi = cluster_bootstrap_ci(by_inst, utility_threshold_L2,
                                                       utility_always_verify, args.n_boot)
            iid_width = iid_hi - iid_lo
            cl_width = cl_hi - cl_lo
            rows.append({
                "cell": cell_label, "generator": gen, "policy": "threshold_L2",
                "n_instances": n_inst, "n_repos": n_repos,
                "iid_delta": iid_d, "iid_ci_lo": iid_lo, "iid_ci_hi": iid_hi, "iid_width": iid_width,
                "cluster_delta": cl_d, "cluster_ci_lo": cl_lo, "cluster_ci_hi": cl_hi, "cluster_width": cl_width,
                "ci_width_inflation": cl_width / iid_width if iid_width > 0 else float("inf"),
            })

    csv_path = args.output_root / "per_cell.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "generator", "policy", "n_instances", "n_repos",
                    "iid_delta", "iid_ci_lo", "iid_ci_hi", "iid_width",
                    "cluster_delta", "cluster_ci_lo", "cluster_ci_hi", "cluster_width",
                    "ci_width_inflation"])
        for r in rows:
            w.writerow([r["cell"], r["generator"], r["policy"], r["n_instances"], r["n_repos"],
                        f"{r['iid_delta']:+.3f}", f"{r['iid_ci_lo']:+.3f}", f"{r['iid_ci_hi']:+.3f}", f"{r['iid_width']:.3f}",
                        f"{r['cluster_delta']:+.3f}", f"{r['cluster_ci_lo']:+.3f}", f"{r['cluster_ci_hi']:+.3f}", f"{r['cluster_width']:.3f}",
                        f"{r['ci_width_inflation']:.2f}"])

    json_path = args.output_root / "per_cell.json"
    json_path.write_text(json.dumps(rows, indent=2))

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print()
    print(f"=== threshold_L2 vs always_verify on SWE-bench cells: IID vs cluster bootstrap CIs ===")
    print(f"{'cell':14} {'gen':14} {'n':>4} {'#repo':>5} {'IID 95%':>22} {'cluster 95%':>22} {'inflation':>9}")
    for r in rows:
        iid_ci = f"[{r['iid_ci_lo']:+.2f},{r['iid_ci_hi']:+.2f}]"
        cl_ci = f"[{r['cluster_ci_lo']:+.2f},{r['cluster_ci_hi']:+.2f}]"
        print(f"{r['cell']:14} {r['generator']:14} {r['n_instances']:>4} {r['n_repos']:>5} {iid_ci:>22} {cl_ci:>22} {r['ci_width_inflation']:>9.2f}x")


if __name__ == "__main__":
    main()
