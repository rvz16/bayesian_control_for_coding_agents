#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = [
    "LCB-hard",
    "LCB-medium",
    "LCB-easy",
    "MBPP+",
    "HumanEval+",
    "SWE-Bench Lite",
    "SWE-Bench Verified",
    "HumanEvalFix",
    "CodeContests",
]

METHODS = [
    ("perplexity", "Perplexity"),
    ("llm_log_seq_prob", "Seq. Prob."),
    ("verbalized_2s_confidence", "Verb."),
    ("tool_success", "Tool Success Rate"),
    ("bayes_state", "Bayes Belief State"),
    ("uhead_confidence", "UHead"),
]

DEFAULT_RUNS = {
    "gpt_oss_20b": {
        "LCB-easy": "/capstor/store/cscs/swissai/a0142/agents_uq/lcb_llm_tool_agent_gpt_oss_20b/2652261_20260630_122731/readable/lcb_easy/metric_scores.csv",
        "LCB-medium": "/capstor/store/cscs/swissai/a0142/agents_uq/lcb_llm_tool_agent_gpt_oss_20b/2652261_20260630_122731/readable/lcb_medium/metric_scores.csv",
        "LCB-hard": "/capstor/store/cscs/swissai/a0142/agents_uq/lcb_llm_tool_agent_gpt_oss_20b/2652261_20260630_122731/readable/lcb_hard/metric_scores.csv",
        "MBPP+": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_nonswe_gpt_oss_20b/2652263_20260630_122732/readable/mbpp/metric_scores.csv",
        "HumanEval+": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_nonswe_gpt_oss_20b/2652263_20260630_122732/readable/humaneval/metric_scores.csv",
        "HumanEvalFix": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_nonswe_gpt_oss_20b/2652263_20260630_122732/readable/humanevalfix/metric_scores.csv",
        "CodeContests": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_nonswe_gpt_oss_20b/2652263_20260630_122732/readable/codecontests/metric_scores.csv",
        "SWE-Bench Lite": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_swe_gpt_oss_20b/2656597_20260630_220737/metric_scores.csv",
        "SWE-Bench Verified": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_swe_verified100_gpt_oss_20b/2661018_20260701_133312/readable/swebench_verified/metric_scores.csv",
    },
    "qwen25_32b": {
        "LCB-easy": "/capstor/store/cscs/swissai/a0142/agents_uq/lcb_llm_tool_agent_qwen25_32b/2652316_20260630_124848/readable/lcb_easy/metric_scores.csv",
        "LCB-medium": "/capstor/store/cscs/swissai/a0142/agents_uq/lcb_llm_tool_agent_qwen25_32b/2652316_20260630_124848/readable/lcb_medium/metric_scores.csv",
        "LCB-hard": "/capstor/store/cscs/swissai/a0142/agents_uq/lcb_llm_tool_agent_qwen25_32b/2652316_20260630_124848/readable/lcb_hard/metric_scores.csv",
        "MBPP+": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_nonswe_qwen25_32b/2652317_20260630_124842/readable/mbpp/metric_scores.csv",
        "HumanEval+": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_nonswe_qwen25_32b/2652317_20260630_124842/readable/humaneval/metric_scores.csv",
        "HumanEvalFix": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_nonswe_qwen25_32b/2652317_20260630_124842/readable/humanevalfix/metric_scores.csv",
        "CodeContests": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_nonswe_qwen25_32b/2652317_20260630_124842/readable/codecontests/metric_scores.csv",
        "SWE-Bench Lite": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_swe_qwen25_32b/2656598_20260630_220804/metric_scores.csv",
        "SWE-Bench Verified": "/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_swe_verified100_qwen25_32b/2661019_20260701_133358/readable/swebench_verified/metric_scores.csv",
    },
}

RUN_ROOT_DATASET_DIRS = {
    "LCB-hard": "lcb_hard",
    "LCB-medium": "lcb_medium",
    "LCB-easy": "lcb_easy",
    "MBPP+": "mbpp",
    "HumanEval+": "humaneval",
    "SWE-Bench Lite": "swebench_lite",
    "SWE-Bench Verified": "swebench_verified",
    "HumanEvalFix": "humanevalfix",
    "CodeContests": "codecontests",
}


def paths_from_run_root(root: Path) -> dict[str, str]:
    return {
        dataset: str(root / "readable" / dirname / "metric_scores.csv")
        for dataset, dirname in RUN_ROOT_DATASET_DIRS.items()
    }


def read_scores(path: str, metric: str) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["score"]] = float(row[metric])
            except (KeyError, TypeError, ValueError):
                pass
    return out


def fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.3f}"


def md_fmt(x: float | None, rank: int | None) -> str:
    text = fmt(x)
    if not text:
        return ""
    if rank == 1:
        return f"**{text}**"
    if rank == 2:
        return f"<u>{text}</u>"
    return text


def ranks_by_column(rows: list[list[float | None]]) -> list[list[int | None]]:
    if not rows:
        return []
    ranks = [[None for _ in row] for row in rows]
    for col in range(len(rows[0])):
        vals = sorted(
            ((row[col], i) for i, row in enumerate(rows) if row[col] is not None),
            reverse=True,
        )
        if vals:
            ranks[vals[0][1]][col] = 1
        if len(vals) > 1:
            ranks[vals[1][1]][col] = 2
    return ranks


def write_table(model: str, paths: dict[str, str], metric: str, out_dir: Path) -> None:
    values = {ds: read_scores(path, metric) for ds, path in paths.items()}
    numeric_rows = []
    for method_key, method_name in METHODS:
        nums = [values.get(ds, {}).get(method_key) for ds in DATASETS]
        present = [x for x in nums if x is not None]
        mean = sum(present) / len(present) if present else None
        numeric_rows.append([method_name, *nums, mean])

    ranks = ranks_by_column([row[1:] for row in numeric_rows])
    csv_rows = [[row[0], *[fmt(x) for x in row[1:]]] for row in numeric_rows]
    md_rows = [
        [row[0], *[md_fmt(x, rank) for x, rank in zip(row[1:], row_ranks)]]
        for row, row_ranks in zip(numeric_rows, ranks)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"uncertainty_table__{model}__{metric}.csv"
    md_path = out_dir / f"uncertainty_table__{model}__{metric}.md"

    header = ["Method", *DATASETS, "Mean"]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    lines = [
        f"### {model}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in md_rows]
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)
    print(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="PRR_05", choices=["PRR_05", "PRR", "spearman"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/orchestration_hypothesis_testing/sim_results"),
    )
    parser.add_argument(
        "--gpt-root",
        type=Path,
        default=None,
        help="Optional run root produced by run_sage_uncertainty_experiments.sh for gpt_oss_20b.",
    )
    parser.add_argument(
        "--qwen-root",
        type=Path,
        default=None,
        help="Optional run root produced by run_sage_uncertainty_experiments.sh for qwen25_32b.",
    )
    args = parser.parse_args()

    runs = dict(DEFAULT_RUNS)
    if args.gpt_root is not None:
        runs["gpt_oss_20b"] = paths_from_run_root(args.gpt_root)
    if args.qwen_root is not None:
        runs["qwen25_32b"] = paths_from_run_root(args.qwen_root)

    for model, paths in runs.items():
        write_table(model, paths, args.metric, args.output_dir)


if __name__ == "__main__":
    main()
