"""Sweep L3 (LLM-review) over multiple reviewer models on existing
calibration data — no patch regeneration, just re-runs reviewers on the
already-saved raw_responses + Y labels.

Output per (output_dir, generator):
  L3_sweep.jsonl — one row per (instance_id, patch_id) with reviewer columns:
    {instance_id, patch_id, Y, L3_<label1>: bool, L3_<label2>: bool, ...}

Resume support: skips rows where every requested reviewer is already filled.
Writes incrementally (line-buffered append) so a kill mid-run is safe.

Usage:
  python lcb_l3_sweep.py \\
    --output-dir data/lcb_calibration_v2 \\
    --generators gpt5_mini,qwen3_coder,haiku45 \\
    --reviewers haiku45=anthropic/claude-haiku-4.5,gpt4omini=openai/gpt-4o-mini,sonnet45=anthropic/claude-sonnet-4.5,gpt5mini=openai/gpt-5-mini \\
    --difficulty hard --platform leetcode \\
    --max-workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibration.lcb import (  # noqa: E402
    extract_code, load_lcb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lcb_l3_sweep")


REVIEW_PROMPT = (
    "You are a senior software engineer reviewing a code submission.\n\n"
    "## Problem\n{problem}\n\n"
    "## Submitted code\n```python\n{code}\n```\n\n"
    "Does this code correctly solve the problem? Respond with exactly one word: "
    "PASS or FAIL. No explanation."
)


def review_once(client, model_id: str, problem: str, code: str) -> tuple[bool, float, dict]:
    prompt = REVIEW_PROMPT.format(problem=problem[:3000], code=code[:6000])
    # gpt-5 family uses reasoning tokens that count against max_tokens; bump
    # to a comfortable budget for those models. Other reviewers don't think.
    max_tok = 200 if "gpt-5" in model_id else 32
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=max_tok,
    )
    text = (resp.choices[0].message.content or "").strip().upper()
    usage = resp.usage
    # Per-reviewer cost approx (matches lcb_calibrate's cost_for_call rates)
    if "haiku" in model_id:
        cost = (usage.prompt_tokens / 1_000_000) * 1.0 + (usage.completion_tokens / 1_000_000) * 5.0
    elif "sonnet" in model_id:
        cost = (usage.prompt_tokens / 1_000_000) * 3.0 + (usage.completion_tokens / 1_000_000) * 15.0
    elif "gpt-4o-mini" in model_id:
        cost = (usage.prompt_tokens / 1_000_000) * 0.15 + (usage.completion_tokens / 1_000_000) * 0.6
    elif "gpt-5-mini" in model_id:
        cost = (usage.prompt_tokens / 1_000_000) * 0.5 + (usage.completion_tokens / 1_000_000) * 4.0
    else:
        cost = (usage.prompt_tokens / 1_000_000) * 1.0 + (usage.completion_tokens / 1_000_000) * 5.0
    passed = ("PASS" in text and "FAIL" not in text)
    info = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
            "raw": text[:32]}
    return passed, cost, info


def parse_reviewers(spec: str) -> dict[str, str]:
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"bad reviewer spec '{part}', expected label=model_id")
        label, model = part.split("=", 1)
        out[label.strip()] = model.strip()
    return out


def load_existing(path: Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows[(str(r["instance_id"]), int(r["patch_id"]))] = r
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--reviewers", required=True, help="label=model_id,label2=model_id2,...")
    parser.add_argument("--difficulty", default="hard",
                        help="LCB difficulty (only used when --source-dataset=lcb)")
    parser.add_argument("--platform", default="leetcode",
                        help="LCB platform (only used when --source-dataset=lcb)")
    parser.add_argument("--lcb-version", default="v1", choices=["v1", "all"])
    parser.add_argument("--source-dataset", default="lcb",
                        choices=["lcb", "mbpp", "humaneval", "swebench_lite", "swebench_verified"],
                        help="Where to load problem statements from (default: lcb)")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force-retry", default="",
                        help="comma-separated reviewer labels to retry even if value already set")
    args = parser.parse_args()
    force_retry = {x.strip() for x in args.force_retry.split(",") if x.strip()}

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    reviewers = parse_reviewers(args.reviewers)
    log.info("reviewers: %s", reviewers)

    # Load problem text for the source dataset (LCB / MBPP+ / HumanEval+ / SWE-bench).
    # Each benchmark uses a different loader and a different field name for the
    # problem statement. We normalize to {instance_id: problem_text}.
    src = (args.source_dataset or "lcb").lower()
    problem_text_by_id: dict[str, str] = {}
    if src == "lcb":
        for p in load_lcb(difficulty=args.difficulty, platform=args.platform, lcb_version=args.lcb_version):
            problem_text_by_id[str(p["question_id"])] = p.get("question_content") or ""
    elif src == "mbpp":
        os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        from datasets import load_dataset
        for p in load_dataset("evalplus/mbppplus", split="test"):
            problem_text_by_id[str(p["task_id"])] = p.get("prompt") or ""
    elif src == "humaneval":
        from evalplus.data import get_human_eval_plus
        for tid, p in get_human_eval_plus().items():
            problem_text_by_id[str(tid)] = p.get("prompt") or ""
    elif src == "swebench_lite":
        os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        from datasets import load_dataset
        for p in load_dataset("princeton-nlp/SWE-bench_Lite", split="test"):
            problem_text_by_id[str(p["instance_id"])] = p.get("problem_statement") or ""
    elif src == "swebench_verified":
        os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        from datasets import load_dataset
        for p in load_dataset("princeton-nlp/SWE-bench_Verified", split="test"):
            problem_text_by_id[str(p["instance_id"])] = p.get("problem_statement") or ""
    else:
        raise SystemExit(f"unknown --source-dataset: {args.source_dataset}")
    log.info("loaded %d problems from %s", len(problem_text_by_id), src)
    # Backwards compat: keep `by_qid` shape with .get('question_content') access pattern
    by_qid = {k: {"question_content": v} for k, v in problem_text_by_id.items()}

    out_dir = args.output_dir.resolve()
    total_cost = 0.0

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        gen_dir = out_dir / gen
        rec_path = gen_dir / "critic_results.jsonl"
        raw_dir = gen_dir / "raw_responses"
        sweep_path = gen_dir / "L3_sweep.jsonl"
        if not rec_path.exists() or not raw_dir.exists():
            log.warning("[%s] missing data, skipping", gen)
            continue

        # Load Y labels from critic_results
        y_by_key: dict[tuple[str, int], int] = {}
        for line in open(rec_path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            y_by_key[(str(r["instance_id"]), int(r["patch_id"]))] = int(r["Y"])

        existing = load_existing(sweep_path)
        log.info("[%s] %d total records, %d already in sweep", gen, len(y_by_key), len(existing))

        # Build work list: (key, problem_text, code, missing_reviewers)
        work = []
        for key, Y in y_by_key.items():
            inst_id, pid = key
            problem = by_qid.get(str(inst_id))
            if problem is None:
                continue
            problem_text = problem.get("question_content", "") or ""
            # HumanEval+ raw filenames are <safe_id>_p<pid>.txt where '/' → '_'
            safe_id = str(inst_id).replace("/", "_")
            raw_path = raw_dir / f"{safe_id}_p{pid}.txt"
            if not raw_path.exists():
                # Fall back to the unsanitized form for non-HumanEval datasets
                raw_path = raw_dir / f"{inst_id}_p{pid}.txt"
                if not raw_path.exists():
                    continue
            code = extract_code(raw_path.read_text())
            existing_row = existing.get(key, {"instance_id": inst_id, "patch_id": pid, "Y": Y})
            # Retry: never-tried (key absent), previously-failed (None), or in force_retry list
            missing = [lbl for lbl in reviewers
                       if f"L3_{lbl}" not in existing_row
                          or existing_row[f"L3_{lbl}"] is None
                          or lbl in force_retry]
            if not missing:
                continue
            work.append((key, problem_text, code, missing, existing_row))

        if not work:
            log.info("[%s] all done, nothing to do", gen)
            continue

        log.info("[%s] %d records × %d reviewers = %d API calls",
                 gen, len(work), len(reviewers),
                 sum(len(m) for _, _, _, m, _ in work))

        out_fp = open(sweep_path, "a", buffering=1)

        def evaluate(key_work):
            key, problem, code, missing, base_row = key_work
            row = dict(base_row)
            local_cost = 0.0
            for lbl in missing:
                model_id = reviewers[lbl]
                try:
                    passed, cost, info = review_once(client, model_id, problem, code)
                    row[f"L3_{lbl}"] = bool(passed)
                    local_cost += cost
                except Exception as e:
                    log.warning("[%s] review failed (%s) for %s_p%d: %s",
                                model_id, lbl, key[0], key[1], e)
                    row[f"L3_{lbl}"] = None
            return row, local_cost, key

        try:
            done = 0
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futures = [ex.submit(evaluate, w) for w in work]
                for fut in as_completed(futures):
                    row, c, key = fut.result()
                    out_fp.write(json.dumps(row) + "\n")
                    out_fp.flush()
                    os.fsync(out_fp.fileno())
                    total_cost += c
                    done += 1
                    if done % 25 == 0:
                        log.info("[%s] %d/%d (cost so far $%.4f)", gen, done, len(work), total_cost)
        finally:
            out_fp.close()

        log.info("[%s] done, sweep at %s, total cost $%.4f", gen, sweep_path, total_cost)

    log.info("ALL DONE, total spent $%.4f", total_cost)


if __name__ == "__main__":
    main()
