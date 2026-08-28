#!/usr/bin/env python3
"""Build notebook-style logprob / Bayesian-quality tables for LCB tool-agent runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CRITIC_MAP = {
    "critic_L0": "L0_syntax",
    "critic_L1": "L1_lint",
    "critic_L2": "L2_public_tests",
    "critic_L3": "L3_llm_review",
}
CRITIC_SHORT = {
    "L0_syntax": "L0",
    "L1_lint": "L1",
    "L2_public_tests": "L2",
    "L3_llm_review": "L3",
}
DEFAULT_KERNEL = {"p_fix_broken": 0.50, "p_break_correct": 0.05}


def configure_hf_cache() -> None:
    override = os.environ.get("ORCH_HF_CACHE_DIR")
    root = Path(override or "/capstor/store/cscs/swissai/a0142/hf_cache")
    # Fall back to the standard local HF cache when the configured root is not
    # usable (e.g. the CSCS /capstor path when running on a laptop).
    if override is None:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            root = Path.home() / ".cache" / "huggingface"
    for name, path in {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "hub",
        "HF_DATASETS_CACHE": root / "datasets",
        "XDG_CACHE_HOME": root / "xdg",
    }.items():
        os.environ.setdefault(name, str(path))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(x: Any) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def logprob_stats(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "llm_perplexity": None,
            "perplexity": None,
            "llm_log_seq_prob": None,
            "n_logprob_tokens": 0,
            "logprobs_supported": False,
        }
    vals = []
    for item in ((row.get("logprobs") or {}).get("content") or []):
        lp = finite_or_none(item.get("logprob") if isinstance(item, dict) else None)
        if lp is not None and lp > -9998:
            vals.append(lp)
    seq = sum(vals) if vals else None
    mean = (seq / len(vals)) if vals else None
    return {
        # Historical name used in analysis.ipynb: this is mean token logprob.
        "llm_perplexity": mean,
        "perplexity": (math.exp(-mean) if mean is not None else None),
        "llm_log_seq_prob": seq,
        "n_logprob_tokens": len(vals),
        "logprobs_supported": bool(vals),
    }


def beta_rate(n_pass: int, n_total: int) -> float:
    return (n_pass + 1.0) / (n_total + 2.0)


def prior_from_train(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("passed") in (True, False)]
    correct = sum(1 for r in usable if r.get("passed") is True)
    return {
        "prior_Y1": beta_rate(correct, len(usable)) if usable else 0.5,
        "prior_calibration_n": len(usable),
        "prior_calibration_correct": correct,
        "prior_smoothing": "Beta(1,1)",
    }


def normalize_theta(j: dict[str, Any]) -> dict[str, dict[str, float]]:
    theta = {}
    for name, row in (j.get("critic_likelihoods") or {}).items():
        p1 = row.get("p_pass_y1", row.get("P_pass_given_Y1"))
        p0 = row.get("p_pass_y0", row.get("P_pass_given_Y0"))
        if p1 is not None and p0 is not None:
            theta[name] = {"p_pass_y1": float(p1), "p_pass_y0": float(p0)}
    return theta


def fit_likelihood_table(records: list[dict[str, Any]], generator: str) -> dict[str, Any]:
    rows = [r for r in records if r.get("Y") in (0, 1)]
    n_y1 = sum(1 for r in rows if r["Y"] == 1)
    likelihoods = {}
    for critic in CRITIC_MAP.values():
        tp = sum(1 for r in rows if r["Y"] == 1 and r.get(critic) is True)
        fn = sum(1 for r in rows if r["Y"] == 1 and r.get(critic) is False)
        fp = sum(1 for r in rows if r["Y"] == 0 and r.get(critic) is True)
        tn = sum(1 for r in rows if r["Y"] == 0 and r.get(critic) is False)
        n_pos = tp + fn
        n_neg = fp + tn
        p1 = beta_rate(tp, n_pos) if n_pos else None
        p0 = beta_rate(fp, n_neg) if n_neg else None
        likelihoods[critic] = {
            "P_pass_given_Y1": p1,
            "P_pass_given_Y0": p0,
            "gap": (p1 - p0) if p1 is not None and p0 is not None else None,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
        }
    return {
        "generator": generator,
        "source": "train_prior_calibration candidates; critics recomputed on train only",
        "n_records": len(records),
        "n_evaluated": len(rows),
        "n_resolved": n_y1,
        "prior_Y1": beta_rate(n_y1, len(rows)) if rows else 0.5,
        "critic_likelihoods": likelihoods,
        "smoothing": "Beta(1,1)",
    }


def fit_train_likelihoods(
    *,
    benchmark: str,
    generator: str,
    split_path: Path,
    train_rows: list[dict[str, Any]],
    output_path: Path,
    private_test_cap: int,
    seed: int,
    platform: str,
    lcb_version: str,
    plus_input_cap: int,
    swe_harness_workers: int,
    include_l3: bool,
) -> dict[str, Any]:
    from calibration import lcb as lcb_calibrate

    sys.modules.setdefault("lcb_calibrate", lcb_calibrate)
    from fitted_live.common import Candidate
    from fitted_live.function_adapters import make_function_adapter
    from fitted_live.swe_adapter import make_swe_adapter

    benchmark = "lcb_medium" if benchmark == "lcb_med" else benchmark
    if benchmark in {"swebench_lite", "swebench_verified"}:
        from calibration import from_spotcheck as calibrate_from_spotcheck

        sys.modules.setdefault("calibrate_from_spotcheck", calibrate_from_spotcheck)
        adapter = make_swe_adapter(
            benchmark=benchmark,
            n_instances=0,
            seed=seed,
            output_dir=output_path.parent,
            harness_workers=swe_harness_workers,
        )
        candidate_kind = "diff"
    else:
        if benchmark == "mbpp":
            from calibration import mbpp as mbpp_calibrate

            sys.modules.setdefault("mbpp_calibrate", mbpp_calibrate)
        elif benchmark == "humaneval":
            from calibration import humaneval as humaneval_calibrate

            sys.modules.setdefault("humaneval_calibrate", humaneval_calibrate)
        elif benchmark == "humanevalfix":
            from calibration import humanevalfix as humanevalfix_calibrate

            sys.modules.setdefault("humanevalfix_calibrate", humanevalfix_calibrate)
        elif benchmark == "codecontests":
            from calibration import codecontests as codecontests_calibrate

            sys.modules.setdefault("codecontests_calibrate", codecontests_calibrate)
        adapter = make_function_adapter(
            benchmark=benchmark,
            n_instances=0,
            seed=seed,
            lcb_version=lcb_version,
            plus_input_cap=plus_input_cap,
            lcb_private_test_cap=private_test_cap,
            platform=platform,
        )
        candidate_kind = "code"

    split = json.loads(split_path.read_text())
    train_ids = {str(x) for x in split.get("train_ids", [])}
    by_train_row = {
        str(r.get("instance_id")): r
        for r in train_rows
        if str(r.get("instance_id")) in train_ids and r.get("passed") in (True, False)
    }
    instances = {adapter.instance_id(inst): inst for inst in adapter.load_instances()}
    records = []
    missing_l3 = []
    for iid in sorted(train_ids):
        row = by_train_row.get(iid)
        inst = instances.get(iid)
        if row is None or inst is None:
            continue
        rec: dict[str, Any] = {
            "instance_id": iid,
            "patch_id": int(row.get("patch_id", 0) or 0),
            "Y": int(bool(row.get("passed"))),
        }
        candidate = Candidate(
            payload=str(row.get("code") or ""),
            raw_text=str(row.get("code") or ""),
            kind=candidate_kind,
        )
        for critic in ("L0_syntax", "L1_lint", "L2_public_tests"):
            try:
                rec[critic] = bool(adapter.run_critic(CRITIC_SHORT[critic], inst, candidate, None).passed)
            except Exception as exc:
                rec[critic] = None
                rec[f"{critic}_error"] = f"{type(exc).__name__}: {exc}"
        if include_l3:
            l3 = row.get("L3_llm_review")
            if l3 not in (True, False):
                for action in row.get("actions", []):
                    if action.get("action") == "critic_L3" and action.get("passed") in (
                        True,
                        False,
                    ):
                        l3 = action["passed"]
                        break
            rec["L3_llm_review"] = l3 if l3 in (True, False) else None
            if rec["L3_llm_review"] is None:
                missing_l3.append(iid)
        records.append(rec)

    if include_l3 and missing_l3:
        examples = ", ".join(missing_l3[:5])
        raise RuntimeError(
            f"missing saved L3 train-calibration results for {len(missing_l3)} "
            f"instances ({examples}); resume the agent with --calibrate-l3 first"
        )

    table = fit_likelihood_table(records, generator)
    output_path.write_text(json.dumps(table, indent=2))
    records_path = output_path.with_name("train_critic_results.jsonl")
    write_jsonl(records_path, records)
    return table


def bayes_update(belief: float, theta: dict[str, float], passed: bool) -> float:
    p1 = theta["p_pass_y1"] if passed else 1.0 - theta["p_pass_y1"]
    p0 = theta["p_pass_y0"] if passed else 1.0 - theta["p_pass_y0"]
    den = p1 * belief + p0 * (1.0 - belief)
    return belief if den <= 1e-12 else (p1 * belief / den)


def kernel_update(belief: float, kernel: dict[str, float]) -> float:
    p_fix = float(kernel.get("p_fix_broken", kernel.get("P_fix_given_broken", 0.50)))
    p_break = float(kernel.get("p_break_correct", kernel.get("P_break_given_correct", 0.05)))
    return belief * (1.0 - p_break) + (1.0 - belief) * p_fix


def action_token(row: dict[str, Any]) -> str:
    action = str(row.get("action"))
    if row.get("skipped") is True:
        if action == "generate":
            return "generate_skipped"
        if action in CRITIC_MAP:
            return f"{CRITIC_MAP[action]}_skipped"
        if action in {"verify", "final_verify"}:
            return f"{action}_skipped"
        return f"{action}_skipped"
    if action == "generate":
        return "generate"
    if action in CRITIC_MAP:
        return f"{CRITIC_MAP[action]}{'+' if bool(row.get('passed')) else '-'}"
    if action in {"verify", "final_verify"}:
        return f"{action}{'+' if bool(row.get('passed')) else '-'}"
    if action == "think" and row.get("reasoning") == "decision parse failed":
        return "think_parse_failed"
    return action


def compact_path(tokens: list[str]) -> str:
    out = []
    for token in tokens:
        if out and out[-1][0] == token:
            out[-1][1] += 1
        else:
            out.append([token, 1])
    return " -> ".join(t if n == 1 else f"{t}x{n}" for t, n in out)


def spearman(xs: list[float], ys: list[int]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and math.isfinite(x)]
    if len(pairs) < 2:
        return None
    x, y = zip(*pairs)
    try:
        import pandas as pd

        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    except Exception:
        return None


def prediction_rejection_area(
    uncertainty: list[float], quality: list[int], max_rejection: float
) -> float | None:
    """lm-polygraph-compatible PredictionRejectionArea for binary quality."""
    pairs = [
        (float(u), int(q))
        for u, q in zip(uncertainty, quality)
        if u is not None and math.isfinite(float(u))
    ]
    if len(pairs) < 2:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    ue = np.array([u for u, _ in pairs])
    target = np.array([q for _, q in pairs])
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (target - min_t) / (max_t - min_t)
    sorted_metrics = target[np.argsort(ue)]
    n = len(sorted_metrics)
    num_rej = int(max_rejection * n)
    if num_rej <= 0:
        return None
    cumsum = np.cumsum(sorted_metrics)[-num_rej:]
    scores = (cumsum / np.arange((n - num_rej) + 1, n + 1))[::-1]
    return float(np.sum(scores) / num_rej)


def prr(conf: list[float], quality: list[int], max_rejection: float) -> float | None:
    pairs = [
        (float(c), int(q))
        for c, q in zip(conf, quality)
        if c is not None and math.isfinite(float(c))
    ]
    if len(pairs) < 2:
        return None
    conf_f = [c for c, _ in pairs]
    quality_f = [q for _, q in pairs]
    area = prediction_rejection_area([-c for c in conf_f], quality_f, max_rejection)
    oracle = prediction_rejection_area([-float(q) for q in quality_f], quality_f, max_rejection)
    try:
        import numpy as np

        rng = np.random.RandomState(42)
        rand_scores = np.arange(len(quality_f))
        random_scores = []
        for _ in range(1000):
            rng.shuffle(rand_scores)
            score = prediction_rejection_area(rand_scores.tolist(), quality_f, max_rejection)
            if score is not None:
                random_scores.append(score)
        random = float(np.mean(random_scores)) if random_scores else None
    except Exception:
        random = sum(quality_f) / len(quality_f)
    if area is None or oracle is None or random is None or abs(oracle - random) < 1e-12:
        return None
    return (area - random) / (oracle - random)


def load_tool_success(path: Path | None, prior: float) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            iid = row.get("instance_id")
            if iid is None:
                continue
            score = finite_or_none(row.get("tool_no_final_before_verify_pass_success_rate"))
            out[str(iid)] = prior if score is None else score
    return out


def load_logprob_stats(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("instance_id")), int(row.get("generation_index", 0)))
            stat = logprob_stats(row)
            stat.update(
                {
                    "prompt_tokens": int(row.get("prompt_tokens") or 0),
                    "completion_tokens": int(row.get("completion_tokens") or 0),
                    "api_cost_usd": float(row.get("api_cost_usd") or 0.0),
                    "candidate_chars": len(row.get("code") or ""),
                    "raw_response_chars": len(row.get("raw_text") or ""),
                }
            )
            out[key] = stat
    return out


def load_verbalized_2s(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        iid = row.get("instance_id")
        if iid is None:
            continue
        out[str(iid)] = {
            "verbalized_2s_confidence": finite_or_none(
                row.get("verbalized_2s_confidence")
            ),
            "verbalized_2s_uncertainty": finite_or_none(
                row.get("verbalized_2s_uncertainty")
            ),
            "verbalized_2s_error": row.get("verbalized_2s_error", ""),
            "verbalized_2s_completion_tokens": int(
                row.get("verbalized_2s_completion_tokens") or 0
            ),
        }
    return out


def load_uhead(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row.get("instance_id")), int(row.get("generation_index", 0))): {
            "uhead_confidence": finite_or_none(row.get("uhead_confidence")),
            "uhead_uncertainty": finite_or_none(row.get("uhead_uncertainty")),
        }
        for row in read_jsonl(path)
    }


def simulate_instance(
    result: dict[str, Any],
    actions: list[dict[str, Any]],
    logprobs: dict[tuple[str, int], dict[str, Any]],
    *,
    generator: str,
    prior: float,
    theta: dict[str, dict[str, float]],
    kernel: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    iid = str(result["instance_id"])
    belief = float(prior)
    gen_idx = 0
    generation_trace = []
    before_final_label = None
    after_final_label = None
    tokens = []

    for row in sorted(actions, key=lambda r: int(r.get("step", 0))):
        action = str(row.get("action"))
        tokens.append(action_token(row))
        if action == "generate":
            if row.get("skipped") is True:
                continue
            before = belief
            belief = kernel_update(belief, kernel)
            stat = logprobs.get((iid, gen_idx), logprob_stats(None))
            generation_trace.append(
                {
                    "benchmark": result.get("benchmark"),
                    "split": "test",
                    "generator": generator,
                    "model_id": result.get("model_id"),
                    "policy": "llm_tool_agent",
                    "instance_id": iid,
                    "patch_idx": gen_idx,
                    "action_step": int(row.get("step", 0)),
                    "bayes_state_at_generation": before,
                    "bayes_state_after_generation": belief,
                    **stat,
                    "is_final_candidate": False,
                    "final_quality": None,
                    "final_bayes_state": None,
                    "final_action": "",
                }
            )
            gen_idx += 1
            continue

        critic = CRITIC_MAP.get(action)
        if critic and row.get("passed") in (True, False) and critic in theta:
            belief = bayes_update(belief, theta[critic], bool(row["passed"]))
            continue

        if action in {"verify", "final_verify"} and row.get("passed") in (True, False):
            before_final_label = belief
            if row.get("passed") is False and action == "verify":
                belief = 0.05
            after_final_label = belief

    if before_final_label is None:
        before_final_label = belief
    if after_final_label is None:
        after_final_label = belief

    if generation_trace:
        generation_trace[-1]["is_final_candidate"] = True
        generation_trace[-1]["final_quality"] = int(bool(result.get("fixed")))
        generation_trace[-1]["final_bayes_state"] = before_final_label
        generation_trace[-1]["final_action"] = result.get("final_action", "")
        final_stats = generation_trace[-1]
    else:
        final_stats = logprob_stats(None)

    final = {
        "benchmark": result.get("benchmark"),
        "split": "test",
        "generator": generator,
        "model_id": result.get("model_id"),
        "policy": "llm_tool_agent",
        "instance_id": iid,
        "llm_perplexity": final_stats.get("llm_perplexity"),
        "perplexity": final_stats.get("perplexity"),
        "llm_log_seq_prob": final_stats.get("llm_log_seq_prob"),
        "uhead_confidence": final_stats.get("uhead_confidence"),
        "uhead_uncertainty": final_stats.get("uhead_uncertainty"),
        "bayes_state": before_final_label,
        "bayes_state_before_final_label": before_final_label,
        "bayes_state_after_final_label": after_final_label,
        "bayes_state_after_generation": (
            generation_trace[-1]["bayes_state_after_generation"] if generation_trace else None
        ),
        "quality": int(bool(result.get("fixed"))),
        "final_action": result.get("final_action", ""),
        "label_source": result.get("final_action", ""),
        "n_logprob_tokens": final_stats.get("n_logprob_tokens", 0),
        "logprobs_supported": bool(final_stats.get("logprobs_supported")),
        "prompt_tokens": int(result.get("prompt_tokens") or 0),
        "completion_tokens": int(result.get("completion_tokens") or 0),
        "n_llm_calls": int(result.get("n_generations") or 0),
        "n_critic_runs": int(result.get("n_critic_runs") or 0),
        "n_full_tests": int(result.get("n_verifications") or 0),
        "wall_clock": float(result.get("wall_clock_s") or 0.0),
        "prior_Y1": prior,
        "actions": actions,
        "generation_trace": generation_trace,
    }
    return generation_trace, final, compact_path(tokens)


def build_metrics(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality = [int(r["quality"]) for r in final_rows]
    specs = [
        ("llm_perplexity", True),
        ("perplexity", False),
        ("llm_log_seq_prob", True),
        ("bayes_state", True),
        ("bayes_state_after_generation", True),
        ("tool_success", True),
        ("verbalized_2s_confidence", True),
        ("uhead_confidence", True),
    ]
    rows = []
    for name, higher_is_better in specs:
        raw = [finite_or_none(r.get(name)) for r in final_rows]
        conf = [x if (x is None or higher_is_better) else -x for x in raw]
        rows.append(
            {
                "score": name,
                "higher_is_better": higher_is_better,
                "spearman": spearman(conf, quality),
                "PRR": prr(conf, quality, 1.0),
                "PRR_05": prr(conf, quality, 0.5),
            }
        )
    return rows


def main() -> None:
    configure_hf_cache()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--benchmark", default="lcb_hard")
    p.add_argument("--generator", default="gpt_oss_120b_local")
    p.add_argument("--likelihoods", type=Path, default=None)
    p.add_argument("--kernel", type=Path, default=None)
    p.add_argument("--tool-success-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--platform", default="leetcode")
    p.add_argument("--lcb-version", default="all")
    p.add_argument("--private-test-cap", type=int, default=12)
    p.add_argument("--plus-input-cap", type=int, default=200)
    p.add_argument("--swe-harness-workers", type=int, default=1)
    p.add_argument(
        "--include-l3-train-likelihood",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit L3 from saved train-calibration reviews (default: enabled).",
    )
    args = p.parse_args()

    stem = f"{args.benchmark}__{args.generator}"
    root = args.run_root
    out_dir = args.output_dir or root
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_results = [r for r in read_jsonl(root / f"{stem}.jsonl") if r.get("split") == "test"]
    results_by_id: dict[str, dict[str, Any]] = {}
    for row in raw_results:
        iid = str(row.get("instance_id"))
        if iid in results_by_id:
            del results_by_id[iid]
        results_by_id[iid] = row
    results = list(results_by_id.values())
    actions = [r for r in read_jsonl(root / f"{stem}.actions.jsonl") if r.get("split") == "test"]
    prior_rows = read_jsonl(root / f"{stem}.train_prior_calibration.jsonl")
    logprobs = load_logprob_stats(root / f"{stem}.generation_logprobs.jsonl")
    uhead = load_uhead(root / f"{stem}.uhead.jsonl")
    for key, score in uhead.items():
        logprobs.setdefault(key, logprob_stats(None)).update(score)
    verbalized_2s = load_verbalized_2s(root / f"{stem}.verbalized_2s.jsonl")
    prior = prior_from_train(prior_rows)
    tool_success_path = args.tool_success_csv
    if tool_success_path is None:
        for candidate in (
            out_dir / "tool_success_by_instance.csv",
            root / "tool_success_by_instance.csv",
            root / "readable" / "tool_success_by_instance.csv",
        ):
            if candidate.exists():
                tool_success_path = candidate
                break
    tool_success = load_tool_success(tool_success_path, float(prior["prior_Y1"]))

    if args.likelihoods:
        table = json.loads(args.likelihoods.read_text())
        theta_source = str(args.likelihoods)
    else:
        likelihood_path = out_dir / "likelihood_tables.json"
        table = fit_train_likelihoods(
            benchmark=args.benchmark,
            generator=args.generator,
            split_path=root / f"{stem}.split.json",
            train_rows=prior_rows,
            output_path=likelihood_path,
            private_test_cap=args.private_test_cap,
            seed=args.seed,
            platform=args.platform,
            lcb_version=args.lcb_version,
            plus_input_cap=args.plus_input_cap,
            swe_harness_workers=args.swe_harness_workers,
            include_l3=args.include_l3_train_likelihood,
        )
        theta_source = str(likelihood_path)
    theta = normalize_theta(table)

    kernel = dict(DEFAULT_KERNEL)
    if args.kernel:
        raw_kernel = json.loads(args.kernel.read_text())
        kernel = raw_kernel.get("kernel_all", raw_kernel)

    actions_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        actions_by_id[str(row.get("instance_id"))].append(row)

    final_rows, traj_rows, path_rows = [], [], []
    for result in results:
        iid = str(result["instance_id"])
        result_actions = result.get("trajectory")
        if not isinstance(result_actions, list) or not result_actions:
            result_actions = actions_by_id.get(iid, [])
        trace, final, path = simulate_instance(
            result,
            result_actions,
            logprobs,
            generator=args.generator,
            prior=float(prior["prior_Y1"]),
            theta=theta,
            kernel=kernel,
        )
        final.update(
            verbalized_2s.get(
                iid,
                {
                    "verbalized_2s_confidence": None,
                    "verbalized_2s_uncertainty": None,
                    "verbalized_2s_error": "missing",
                    "verbalized_2s_completion_tokens": 0,
                },
            )
        )
        final["tool_success"] = (
            tool_success.get(iid, float(prior["prior_Y1"])) if tool_success_path else None
        )
        final_rows.append(final)
        traj_rows.extend(trace)
        path_rows.append(
            {
                "instance_id": iid,
                "policy": "llm_tool_agent",
                "quality": final["quality"],
                "final_action": final["final_action"],
                "bayes_state": final["bayes_state"],
                "bayes_state_after_generation": final["bayes_state_after_generation"],
                "action_path": path,
            }
        )

    compact_rows = [
        {
            "benchmark": r["benchmark"],
            "split": "test",
            "generator": r["generator"],
            "model_id": r["model_id"],
            "policy": r["policy"],
            "instance_id": r["instance_id"],
            "llm_perplexity": r["llm_perplexity"],
            "llm_log_seq_prob": r["llm_log_seq_prob"],
            "uhead_confidence": r["uhead_confidence"],
            "uhead_uncertainty": r["uhead_uncertainty"],
            "bayes_state": r["bayes_state"],
            "quality": r["quality"],
            "bayes_state_after_generation": r["bayes_state_after_generation"],
            "tool_success": r["tool_success"],
            "verbalized_2s_confidence": r["verbalized_2s_confidence"],
            "verbalized_2s_uncertainty": r["verbalized_2s_uncertainty"],
            "verbalized_2s_error": r["verbalized_2s_error"],
            "perplexity": r["perplexity"],
            "final_action": r["final_action"],
        }
        for r in final_rows
    ]
    write_csv(out_dir / "final_logprob_bayes_quality.csv", compact_rows)
    write_jsonl(out_dir / "final_logprob_bayes_quality.jsonl", final_rows)
    write_csv(out_dir / "generation_trajectory_scores.csv", traj_rows)
    write_jsonl(out_dir / "generation_trajectory_scores.jsonl", traj_rows)

    grouped = defaultdict(list)
    for row in path_rows:
        grouped[row["action_path"]].append(row)
    path_table = []
    for i, (path, rows) in enumerate(
        sorted(grouped.items(), key=lambda kv: (-len(kv[1]), -sum(r["quality"] for r in kv[1]) / len(kv[1]))),
        start=1,
    ):
        path_table.append(
            {
                "path_id": f"P{i:02d}",
                "action_path": path,
                "n": len(rows),
                "quality_mean": sum(r["quality"] for r in rows) / len(rows),
                "final_actions": ", ".join(f"{k}:{v}" for k, v in Counter(r["final_action"] for r in rows).items()),
                "bayes_states": ", ".join(f"{x:.6f}" for x in sorted({r["bayes_state"] for r in rows if r["bayes_state"] is not None}, reverse=True)),
                "bayes_states_after_generation": ", ".join(f"{x:.6f}" for x in sorted({r["bayes_state_after_generation"] for r in rows if r["bayes_state_after_generation"] is not None}, reverse=True)),
                "examples": ", ".join(str(r["instance_id"]) for r in rows[:10]),
            }
        )
    write_csv(out_dir / "action_path_analysis.csv", path_table)

    metric_rows = build_metrics(final_rows)
    write_csv(out_dir / "metric_scores.csv", metric_rows)

    summary = {
        "run_root": str(root),
        "output_dir": str(out_dir),
        "n": len(final_rows),
        "quality_mean": sum(r["quality"] for r in final_rows) / len(final_rows) if final_rows else None,
        "final_actions": dict(Counter(r["final_action"] for r in final_rows)),
        "prior": prior,
        "theta_source": theta_source,
        "theta": theta,
        "kernel": kernel,
        "action_counts": dict(Counter(r.get("action") for r in actions)),
        "tool_success_source": str(tool_success_path) if tool_success_path else None,
        "uhead_rows": len(uhead),
        "verbalized_2s": {
            "rows": len(verbalized_2s),
            "parsed": sum(
                1
                for r in verbalized_2s.values()
                if r.get("verbalized_2s_confidence") is not None
            ),
            "errors": dict(
                Counter(r.get("verbalized_2s_error", "") for r in verbalized_2s.values())
            ),
        },
        "outputs": [
            "final_logprob_bayes_quality.csv",
            "final_logprob_bayes_quality.jsonl",
            "generation_trajectory_scores.csv",
            "generation_trajectory_scores.jsonl",
            "action_path_analysis.csv",
            "metric_scores.csv",
            "analysis_summary.json",
        ],
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
