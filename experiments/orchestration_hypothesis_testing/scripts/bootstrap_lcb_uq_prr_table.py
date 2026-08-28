#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = [
    ("Perplexity", "llm_perplexity"),
    ("Seq. Prob.", "llm_log_seq_prob"),
    ("Tool Success Rate", "tool_success"),
    ("Bayes Prob.", "bayes_state"),
]

DELTA_COMPARISONS = [
    ("Bayes - Perplexity", "Bayes Prob.", "Perplexity"),
    ("Bayes - Seq. Prob.", "Bayes Prob.", "Seq. Prob."),
    ("Bayes - Tool Success Rate", "Bayes Prob.", "Tool Success Rate"),
]

DEFAULT_RUNS = {
    "LCB-Medium": Path(
        "/capstor/store/cscs/swissai/a0142/agents_uq/"
        "lcb_llm_tool_agent_gpt_oss_20b/2368762_20260524_153248"
    ),
    "LCB-Hard": Path(
        "/capstor/store/cscs/swissai/a0142/agents_uq/"
        "lcb_llm_tool_agent_gpt_oss_20b/2368761_20260524_153248"
    ),
}


@dataclass(frozen=True)
class DatasetScores:
    name: str
    n: int
    point: dict[str, float]
    bootstrap: dict[str, np.ndarray]
    source: Path


def readable_dir(run_root: Path) -> Path:
    if (run_root / "final_logprob_bayes_quality.csv").exists():
        return run_root
    return run_root / "readable"


def prior_y1(summary_path: Path) -> float:
    summary = json.loads(summary_path.read_text())
    prior = summary.get("prior", {})
    if "prior_Y1" in prior:
        return float(prior["prior_Y1"])
    if "prior_y1" in prior:
        return float(prior["prior_y1"])
    raise KeyError(f"Could not find prior_Y1 in {summary_path}")


def load_dataset(run_root: Path) -> pd.DataFrame:
    rdir = readable_dir(run_root)
    results_path = rdir / "final_logprob_bayes_quality.csv"
    summary_path = rdir / "analysis_summary.json"
    tool_path = run_root / "tool_success_by_instance.csv"

    if not results_path.exists():
        raise FileNotFoundError(results_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not tool_path.exists():
        raise FileNotFoundError(tool_path)

    results = pd.read_csv(results_path)
    prior = prior_y1(summary_path)
    tool = pd.read_csv(tool_path)
    tool_col = "tool_no_final_before_verify_pass_success_rate"

    if "instance_id" in results.columns and "instance_id" in tool.columns:
        results = results.merge(
            tool[["instance_id", tool_col]],
            on="instance_id",
            how="left",
            validate="one_to_one",
        )
        results["tool_success"] = results[tool_col].fillna(prior)
    else:
        if len(tool) != len(results):
            raise ValueError(
                f"Cannot row-align tool success: {len(tool)=}, {len(results)=}"
            )
        results["tool_success"] = tool[tool_col].fillna(prior).to_numpy()

    required = ["quality", *[col for _, col in METHODS]]
    missing = [col for col in required if col not in results.columns]
    if missing:
        raise KeyError(f"Missing required columns in {results_path}: {missing}")

    out = results[required].copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"NaNs in required score columns for {run_root}: {bad}")
    out["quality"] = out["quality"].astype(int)
    return out


def normalize_target(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    min_t = float(np.min(target))
    max_t = float(np.max(target))
    if np.isclose(min_t, max_t):
        min_t -= 1.0
        max_t += 1.0
    return (target - min_t) / (max_t - min_t)


def prediction_rejection_area(
    uncertainty: np.ndarray,
    quality: np.ndarray,
    max_rejection: float,
) -> float:
    # This mirrors lm_polygraph.ue_metrics.PredictionRejectionArea.
    target = normalize_target(quality)
    ue = np.asarray(uncertainty, dtype=float)
    n = len(ue)
    num_rej = int(max_rejection * n)
    if num_rej <= 0:
        raise ValueError(f"max_rejection={max_rejection} rejects zero samples for n={n}")
    sorted_metrics = target[np.argsort(ue)]
    cumsum = np.cumsum(sorted_metrics)[-num_rej:]
    denominators = np.arange((n - num_rej) + 1, n + 1)
    scores = (cumsum / denominators)[::-1]
    return float(np.sum(scores) / num_rej)


def random_score_monte_carlo(
    quality: np.ndarray,
    max_rejection: float,
    num_iter: int,
    seed: int,
) -> float:
    # Same seeded Monte Carlo random baseline as lm_polygraph get_random_scores.
    np.random.seed(seed)
    rand_scores = np.arange(len(quality))
    values = []
    for _ in range(num_iter):
        np.random.shuffle(rand_scores)
        values.append(prediction_rejection_area(rand_scores, quality, max_rejection))
    return float(np.mean(values))


def normalized_prr(
    confidence: np.ndarray,
    quality: np.ndarray,
    max_rejection: float,
    random_iter: int,
    random_seed: int,
) -> float:
    area = prediction_rejection_area(-confidence, quality, max_rejection)
    oracle = prediction_rejection_area(-quality, quality, max_rejection)
    random = random_score_monte_carlo(
        quality,
        max_rejection=max_rejection,
        num_iter=random_iter,
        seed=random_seed,
    )
    if oracle == random:
        return area
    return float((area - random) / (oracle - random))


def normalized_prr_with_baselines(
    confidence: np.ndarray,
    quality: np.ndarray,
    max_rejection: float,
    oracle: float,
    random: float,
) -> float:
    area = prediction_rejection_area(-confidence, quality, max_rejection)
    if oracle == random:
        return area
    return float((area - random) / (oracle - random))


def score_dataset(
    name: str,
    run_root: Path,
    n_bootstrap: int,
    rng: np.random.Generator,
    max_rejection: float,
    random_iter: int,
    random_seed: int,
    sample_fraction: float,
    without_replacement: bool,
) -> DatasetScores:
    data = load_dataset(run_root)
    quality = data["quality"].to_numpy(dtype=float)
    point_oracle = prediction_rejection_area(-quality, quality, max_rejection)
    point_random = random_score_monte_carlo(
        quality,
        max_rejection=max_rejection,
        num_iter=random_iter,
        seed=random_seed,
    )
    point = {
        label: normalized_prr_with_baselines(
            data[col].to_numpy(dtype=float),
            quality,
            max_rejection=max_rejection,
            oracle=point_oracle,
            random=point_random,
        )
        for label, col in METHODS
    }

    boot = {label: np.empty(n_bootstrap, dtype=float) for label, _ in METHODS}
    n = len(data)
    sample_size = max(2, int(round(sample_fraction * n)))
    if without_replacement and sample_size > n:
        raise ValueError(
            f"Cannot sample {sample_size} rows without replacement from n={n}"
        )
    for b in range(n_bootstrap):
        if without_replacement:
            idx = rng.choice(n, size=sample_size, replace=False)
        else:
            idx = rng.integers(0, n, size=sample_size)
        q = quality[idx]
        oracle = prediction_rejection_area(-q, q, max_rejection)
        random = random_score_monte_carlo(
            q,
            max_rejection=max_rejection,
            num_iter=random_iter,
            seed=random_seed,
        )
        for label, col in METHODS:
            boot[label][b] = normalized_prr_with_baselines(
                data[col].to_numpy(dtype=float)[idx],
                q,
                max_rejection=max_rejection,
                oracle=oracle,
                random=random,
            )

    return DatasetScores(
        name=name,
        n=n,
        point=point,
        bootstrap=boot,
        source=run_root,
    )


def ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    lower = 100.0 * alpha / 2.0
    upper = 100.0 * (1.0 - alpha / 2.0)
    lo, hi = np.percentile(values, [lower, upper])
    return float(lo), float(hi)


def fmt_num(x: float) -> str:
    text = f"{x:.3f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def fmt_cell(point: float, bounds: tuple[float, float]) -> str:
    return f"{fmt_num(point)} [{fmt_num(bounds[0])}, {fmt_num(bounds[1])}]"


def fmt_delta_cell(point: float, bounds: tuple[float, float], p_gt_0: float) -> str:
    return f"{fmt_cell(point, bounds)}, P>0={p_gt_0:.3f}"


def write_outputs(
    datasets: dict[str, DatasetScores],
    out_prefix: Path,
    alpha: float,
    n_bootstrap: int,
    max_rejection: float,
    random_iter: int,
    random_seed: int,
    sample_fraction: float,
    without_replacement: bool,
    include_deltas: bool,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    medium = datasets["LCB-Medium"]
    hard = datasets["LCB-Hard"]
    for label, _ in METHODS:
        avg_point = (medium.point[label] + hard.point[label]) / 2.0
        avg_boot = (medium.bootstrap[label] + hard.bootstrap[label]) / 2.0
        for ds in (medium, hard):
            lo, hi = ci(ds.bootstrap[label], alpha)
            rows.append(
                {
                    "method": label,
                    "dataset": ds.name,
                    "n": ds.n,
                    "point": ds.point[label],
                    "ci_low": lo,
                    "ci_high": hi,
                    "source": str(ds.source),
                }
            )
        lo, hi = ci(avg_boot, alpha)
        rows.append(
            {
                "method": label,
                "dataset": "Avg",
                "n": f"{medium.n}+{hard.n}",
                "point": avg_point,
                "ci_low": lo,
                "ci_high": hi,
                "source": "mean(LCB-Medium, LCB-Hard)",
            }
        )

    detail = pd.DataFrame(rows)
    detail.to_csv(out_prefix.with_suffix(".csv"), index=False)

    wide_rows = []
    for label, _ in METHODS:
        avg_point = (medium.point[label] + hard.point[label]) / 2.0
        avg_boot = (medium.bootstrap[label] + hard.bootstrap[label]) / 2.0
        wide_rows.append(
            {
                "Method": label,
                "LCB-Medium": fmt_cell(medium.point[label], ci(medium.bootstrap[label], alpha)),
                "LCB-Hard": fmt_cell(hard.point[label], ci(hard.bootstrap[label], alpha)),
                "Avg": fmt_cell(avg_point, ci(avg_boot, alpha)),
            }
        )
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(out_prefix.with_name(out_prefix.name + "_wide.csv"), index=False)

    lines = [
        "# LCB gpt-oss-20b uncertainty PRR_05 resampling intervals",
        "",
        f"- Metric: normalized PRR with `max_rejection={max_rejection}`.",
        (
            f"- Resampling: percentile {(1 - alpha) * 100:.1f}% interval, "
            f"`n_resamples={n_bootstrap}`, `sample_fraction={sample_fraction}`"
            f", `without_replacement={without_replacement}`."
        ),
        f"- Random baseline: lm-polygraph-style Monte Carlo, `random_iter={random_iter}`, `random_seed={random_seed}`.",
        f"- LCB-Medium: `{medium.source}` (`n={medium.n}`).",
        f"- LCB-Hard: `{hard.source}` (`n={hard.n}`).",
        "",
        "| Method | LCB-Medium | LCB-Hard | Avg |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in wide_rows:
        lines.append(
            f"| {row['Method']} | {row['LCB-Medium']} | {row['LCB-Hard']} | {row['Avg']} |"
        )

    if include_deltas:
        if without_replacement:
            paired_name = "paired subsampling"
            paired_desc = (
                f"Paired subsampling: percentile {(1 - alpha) * 100:.1f}% interval, "
                f"`n_resamples={n_bootstrap}`, `sample_fraction={sample_fraction}`, "
                "`without_replacement=True`."
            )
            paired_suffix = "_paired_resampling_ci"
        else:
            paired_name = "paired bootstrap"
            paired_desc = (
                f"Paired bootstrap: percentile {(1 - alpha) * 100:.1f}% CI, "
                f"`n_bootstrap={n_bootstrap}`."
            )
            paired_suffix = "_paired_bootstrap_ci"

        delta_rows = []
        delta_wide_rows = []
        delta_wide_ci_rows = []
        for comparison, left, right in DELTA_COMPARISONS:
            medium_dist = medium.bootstrap[left] - medium.bootstrap[right]
            hard_dist = hard.bootstrap[left] - hard.bootstrap[right]
            avg_dist = (medium_dist + hard_dist) / 2.0
            cells = {}
            ci_cells = {}
            for ds, dist in ((medium, medium_dist), (hard, hard_dist)):
                point = ds.point[left] - ds.point[right]
                bounds = ci(dist, alpha)
                p_gt_0 = float(np.mean(dist > 0))
                delta_rows.append(
                    {
                        "comparison": comparison,
                        "dataset": ds.name,
                        "n": ds.n,
                        "delta": point,
                        "ci_low": bounds[0],
                        "ci_high": bounds[1],
                        "p_gt_0": p_gt_0,
                        "left": left,
                        "right": right,
                    }
                )
                cells[ds.name] = fmt_delta_cell(point, bounds, p_gt_0)
                ci_cells[ds.name] = fmt_cell(point, bounds)

            avg_point = (
                (medium.point[left] + hard.point[left])
                - (medium.point[right] + hard.point[right])
            ) / 2.0
            avg_bounds = ci(avg_dist, alpha)
            avg_p_gt_0 = float(np.mean(avg_dist > 0))
            delta_rows.append(
                {
                    "comparison": comparison,
                    "dataset": "Avg",
                    "n": f"{medium.n}+{hard.n}",
                    "delta": avg_point,
                    "ci_low": avg_bounds[0],
                    "ci_high": avg_bounds[1],
                    "p_gt_0": avg_p_gt_0,
                    "left": left,
                    "right": right,
                }
            )
            cells["Avg"] = fmt_delta_cell(avg_point, avg_bounds, avg_p_gt_0)
            ci_cells["Avg"] = fmt_cell(avg_point, avg_bounds)
            delta_wide_rows.append(
                {
                    "Comparison": comparison,
                    "LCB-Medium": cells["LCB-Medium"],
                    "LCB-Hard": cells["LCB-Hard"],
                    "Avg": cells["Avg"],
                }
            )
            delta_wide_ci_rows.append(
                {
                    "Comparison": comparison,
                    "LCB-Medium": ci_cells["LCB-Medium"],
                    "LCB-Hard": ci_cells["LCB-Hard"],
                    "Avg": ci_cells["Avg"],
                }
            )

        pd.DataFrame(delta_rows).to_csv(
            out_prefix.with_name(out_prefix.name + "_deltas.csv"),
            index=False,
        )
        pd.DataFrame(delta_wide_ci_rows).to_csv(
            out_prefix.with_name(out_prefix.name + paired_suffix + "_wide.csv"),
            index=False,
        )

        delta_md = [
            f"# {paired_name.capitalize()} intervals",
            "",
            f"- Metric: normalized PRR with `max_rejection={max_rejection}`.",
            f"- {paired_desc}",
            f"- Random baseline: lm-polygraph-style Monte Carlo, `random_iter={random_iter}`, `random_seed={random_seed}`.",
            "- Positive values mean `Bayes Prob.` has higher PRR_05.",
            "",
            "| Comparison | LCB-Medium | LCB-Hard | Avg |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in delta_wide_ci_rows:
            delta_md.append(
                f"| {row['Comparison']} | {row['LCB-Medium']} | {row['LCB-Hard']} | {row['Avg']} |"
            )
        delta_md.append("")
        out_prefix.with_name(out_prefix.name + paired_suffix + ".md").write_text(
            "\n".join(delta_md)
        )

        lines.extend(
            [
                "",
                "## Paired resampling deltas",
                "",
                "Positive delta means `Bayes Prob.` has higher PRR_05.",
                "",
                "| Comparison | LCB-Medium | LCB-Hard | Avg |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in delta_wide_rows:
            lines.append(
                f"| {row['Comparison']} | {row['LCB-Medium']} | {row['LCB-Hard']} | {row['Avg']} |"
            )
    lines.append("")
    out_prefix.with_suffix(".md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for the LCB gpt-oss-20b PRR_05 table."
    )
    parser.add_argument(
        "--medium-run",
        type=Path,
        default=DEFAULT_RUNS["LCB-Medium"],
        help="LCB-Medium run root, containing readable/ and tool_success_by_instance.csv.",
    )
    parser.add_argument(
        "--hard-run",
        type=Path,
        default=DEFAULT_RUNS["LCB-Hard"],
        help="LCB-Hard run root, containing readable/ and tool_success_by_instance.csv.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-rejection", type=float, default=0.5)
    parser.add_argument("--random-iter", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Fraction of each dataset to draw in every resampling replicate.",
    )
    parser.add_argument(
        "--without-replacement",
        action="store_true",
        help="Draw each resampling replicate without replacement.",
    )
    parser.add_argument(
        "--include-deltas",
        action="store_true",
        help="Also write paired method-difference tables.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path(
            "sim_results/lcb_gpt_oss_20b_prr05_bootstrap_2368761_2368762"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.sample_fraction <= 1.0):
        raise ValueError("--sample-fraction must be in (0, 1]")
    rng = np.random.default_rng(args.seed)
    datasets = {
        "LCB-Medium": score_dataset(
            "LCB-Medium",
            args.medium_run,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
            max_rejection=args.max_rejection,
            random_iter=args.random_iter,
            random_seed=args.random_seed,
            sample_fraction=args.sample_fraction,
            without_replacement=args.without_replacement,
        ),
        "LCB-Hard": score_dataset(
            "LCB-Hard",
            args.hard_run,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
            max_rejection=args.max_rejection,
            random_iter=args.random_iter,
            random_seed=args.random_seed,
            sample_fraction=args.sample_fraction,
            without_replacement=args.without_replacement,
        ),
    }
    write_outputs(
        datasets,
        args.out_prefix,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        max_rejection=args.max_rejection,
        random_iter=args.random_iter,
        random_seed=args.random_seed,
        sample_fraction=args.sample_fraction,
        without_replacement=args.without_replacement,
        include_deltas=args.include_deltas,
    )
    print(f"Wrote {args.out_prefix.with_suffix('.md')}")
    print(f"Wrote {args.out_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.out_prefix.with_name(args.out_prefix.name + '_wide.csv')}")


if __name__ == "__main__":
    main()
