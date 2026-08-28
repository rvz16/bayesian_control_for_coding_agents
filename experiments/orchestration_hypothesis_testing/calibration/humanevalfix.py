"""HumanEvalFix calibration for the orchestration controller.

Uses bigcode/humanevalpack (python subset) — the canonical HumanEvalFix
dataset. Each instance is a buggy_solution + tests; the model's job is to
produce a fixed version. The L2/Y split is:
  - L2 (public)  = example_test  (small visible test block)
  - Y  (oracle)  = test           (full hidden test block)
Both are MBPP-style code blocks that define `check(candidate)` and then
call it on entry_point; reuses mbpp_calibrate.run_full_test as the runner.

Per-generator output mirrors the LCB/MBPP/HumanEval+ layout under
<output-dir>/<gen>/{critic_results.jsonl,raw_responses/,...}.

Usage:
  python3 humanevalfix_calibrate.py \\
    --output-dir data/humanevalfix_calibration \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
    --n-instances 100 \\
    --n-patches 3 \\
    --max-cost-usd-per-model 4.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibration.lcb import (  # noqa: E402
    _make_client,
    extract_code, critic_L0_syntax, critic_L1_lint, critic_L3_review,
    cost_for_call, GENERATORS,
)
from calibration.mbpp import run_full_test  # noqa: E402  -- test runner for MBPP-style test blocks
from _common.cost import CostTracker  # noqa: E402
from _common.telemetry import TelemetryLogger  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("hefix_cal")


# ---------- Dataset loader ----------

def load_humanevalfix(n_instances: int, seed: int = 42) -> list[dict]:
    """Load HumanEvalFix (python subset) from bigcode/humanevalpack.

    Returns a flat list of dicts with normalized fields. The dataset has
    164 problems total; if n_instances > 0 we shuffle (seed=42) and take
    the head — same paired-instance discipline as the other calibration
    scripts so the 75/25 split downstream is deterministic per generator.
    """
    from datasets import load_dataset
    ds = load_dataset("bigcode/humanevalpack", "python", split="test")
    out: list[dict] = []
    for r in ds:
        out.append({
            "task_id":           r.get("task_id"),
            "entry_point":       r.get("entry_point"),
            "prompt":            r.get("prompt") or "",
            "declaration":       r.get("declaration") or "",
            "buggy_solution":    r.get("buggy_solution") or "",
            "canonical_solution": r.get("canonical_solution") or "",
            "test":              r.get("test") or "",
            "example_test":      r.get("example_test") or "",
            "test_setup":        r.get("test_setup") or "",
            "bug_type":          r.get("bug_type") or "",
        })
    log.info("loaded %d HumanEvalFix (python) problems", len(out))
    if n_instances and n_instances < len(out):
        random.Random(seed).shuffle(out)
        out = out[:n_instances]
        log.info("sampled %d instances (seed=%d)", n_instances, seed)
    return out


# ---------- Prompt ----------

def build_prompt(problem: dict) -> str:
    """HumanEvalFix prompt: show the buggy function + its docstring +
    example test, ask the model to return a corrected version.

    The model should output a complete function (signature + body) that
    will replace the buggy_solution. Same code-block / docstring discipline
    as humaneval_calibrate.build_prompt so extract_code() recovers it.
    """
    declaration = problem.get("declaration", "").rstrip()
    buggy       = problem.get("buggy_solution", "").rstrip()
    example     = problem.get("example_test", "").rstrip()
    example_section = (
        f"\nThe following example tests illustrate expected behavior:\n"
        f"```python\n{example}\n```\n"
        if example else ""
    )
    return (
        "The Python function below contains one or more bugs. Provide a "
        "corrected, drop-in replacement for the function body that passes "
        "all tests. Return ONLY the complete function (signature included) "
        "inside a single ```python``` code block — no commentary, no "
        "`__main__` block, no example assertions in your output.\n\n"
        "## Buggy function\n"
        f"```python\n{declaration}{buggy}\n```\n"
        f"{example_section}"
    )


# ---------- Test wrappers ----------

def _eval_with_block(code: str, test_block: str, entry_point: str | None,
                     test_setup: str, timeout: int = 10) -> bool:
    """Run `code` + (test_setup + test_block) using mbpp_calibrate.run_full_test.
    test_setup is prepended (HEFix sometimes imports `numpy`, `string`, etc.).
    """
    if not (test_block or "").strip():
        return False
    if test_setup and test_setup.strip():
        test_block = test_setup + "\n\n" + test_block
    return bool(run_full_test(code, test_block, entry_point or None, timeout=timeout))


# ---------- Main pipeline ----------

def calibrate_one_generator(
    gen_key: str, problems: list[dict], n_patches: int, out_dir: Path,
    max_cost_usd: float, client,
) -> None:
    if gen_key not in GENERATORS:
        log.error("unknown generator: %s", gen_key); return
    model_id, label, _base_url = GENERATORS[gen_key]
    gen_client = _make_client(gen_key) if _base_url else client
    log.info("=== %s (%s) — cap $%.2f ===", gen_key, model_id, max_cost_usd)
    gen_dir = out_dir / gen_key
    gen_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = gen_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)
    results_path = gen_dir / "critic_results.jsonl"
    cost = CostTracker(name=gen_key, cap_usd=max_cost_usd,
                       log_path=gen_dir / "cost_log.jsonl")
    dataset_name = out_dir.name  # e.g. "humanevalfix_calibration"
    tele = TelemetryLogger(gen_dir / "action_telemetry.jsonl",
                           dataset=dataset_name, model_name=gen_key)

    # Resume logic: skip (instance, patch) pairs already in critic_results
    done: set[tuple[str, int]] = set()
    records: list[dict] = []
    if results_path.exists():
        for line in open(results_path):
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            done.add((str(r["instance_id"]), int(r["patch_id"])))
            records.append(r)
        log.info("[%s] resuming with %d records persisted", gen_key, len(records))

    out_fp = open(results_path, "a", buffering=1)

    try:
        for inst in problems:
            if cost.capped:
                log.warning("[%s] cap reached, stopping", gen_key); break
            inst_id = inst["task_id"]
            entry   = inst["entry_point"]
            tsetup  = inst["test_setup"]
            tex     = inst["example_test"]  # L2
            tfull   = inst["test"]          # Y
            for pid in range(n_patches):
                if cost.capped: break
                if (str(inst_id), pid) in done:
                    continue
                try:
                    _t0 = time.perf_counter()
                    resp = gen_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": build_prompt(inst)}],
                        temperature=0.7, max_tokens=4000,
                    )
                    _gen_rt = time.perf_counter() - _t0
                    text = resp.choices[0].message.content or ""
                    u = resp.usage
                    c = cost_for_call(model_id, u.prompt_tokens, u.completion_tokens)
                    cost.record(c, prompt_tokens=u.prompt_tokens,
                                completion_tokens=u.completion_tokens,
                                instance_id=inst_id, patch_id=pid)
                    tele.record(action_type="generate", runtime_s=_gen_rt,
                                instance_id=inst_id, patch_id=pid, api_cost_usd=c)
                except Exception as e:
                    log.warning("[%s] gen failed for %s: %s", gen_key, inst_id, e)
                    continue
                safe_id = str(inst_id).replace("/", "_")
                (raw_dir / f"{safe_id}_p{pid}.txt").write_text(text)
                code = extract_code(text)
                # L2 (public): example_test (small visible test) -> bool
                # Y  (oracle): full test block                    -> bool
                try:
                    _t0 = time.perf_counter()
                    l2_ok = _eval_with_block(code, tex,   entry, tsetup, timeout=10)
                    tele.record(action_type="critic_L2",
                                runtime_s=time.perf_counter() - _t0,
                                instance_id=inst_id, patch_id=pid, passed=bool(l2_ok))
                    _t0 = time.perf_counter()
                    y_ok  = _eval_with_block(code, tfull, entry, tsetup, timeout=15)
                    tele.record(action_type="verify",
                                runtime_s=time.perf_counter() - _t0,
                                instance_id=inst_id, patch_id=pid, passed=bool(y_ok))
                    Y     = 1 if y_ok else 0
                    _t0 = time.perf_counter()
                    l0    = critic_L0_syntax(code)
                    tele.record(action_type="critic_L0",
                                runtime_s=time.perf_counter() - _t0,
                                instance_id=inst_id, patch_id=pid, passed=bool(l0))
                    _t0 = time.perf_counter()
                    l1    = critic_L1_lint(code)
                    tele.record(action_type="critic_L1",
                                runtime_s=time.perf_counter() - _t0,
                                instance_id=inst_id, patch_id=pid, passed=bool(l1))
                except Exception as e:
                    log.warning("[%s] critic eval failed for %s_p%d: %s",
                                gen_key, inst_id, pid, e)
                    continue
                l3 = None
                if not cost.capped:
                    try:
                        _t0 = time.perf_counter()
                        l3_pass, l3_cost = critic_L3_review(
                            inst.get("prompt", "")[:3000], code, client)
                        cost.record(l3_cost, prompt_tokens=0, completion_tokens=0,
                                    instance_id=inst_id, patch_id=pid,
                                    extra={"kind": "L3_review"})
                        tele.record(action_type="critic_L3",
                                    runtime_s=time.perf_counter() - _t0,
                                    instance_id=inst_id, patch_id=pid,
                                    passed=bool(l3_pass), api_cost_usd=l3_cost)
                        l3 = l3_pass
                    except Exception as e:
                        log.warning("[%s] L3 failed for %s_p%d: %s",
                                    gen_key, inst_id, pid, e)
                rec = {
                    "generator": gen_key,
                    "instance_id": inst_id,
                    "patch_id": pid,
                    "Y": int(Y),
                    "L0_syntax": bool(l0),
                    "L1_lint": bool(l1),
                    "L2_public_tests": bool(l2_ok),
                    "L3_llm_review": l3,
                    "diff_chars": len(code),
                    # Note: HEFix L2/Y are bool (no pass-rate fractions)
                    "y_pass_rate": float(Y),
                    "l2_pass_rate": (1.0 if l2_ok else 0.0),
                    "bug_type": inst.get("bug_type", ""),
                }
                records.append(rec)
                out_fp.write(json.dumps(rec) + "\n")
                out_fp.flush(); os.fsync(out_fp.fileno())
                done.add((str(inst_id), pid))
                if len(records) % 10 == 0:
                    log.info("[%s] %d records, cost $%.4f",
                             gen_key, len(records), cost.total_usd)
    finally:
        out_fp.close()
        tele.close()

    summary = {
        "model": gen_key, "cap_usd": cost.cap_usd, "total_usd": cost.total_usd,
        "n_calls": cost.n_calls, "remaining_usd": cost.remaining,
        "capped": cost.capped,
    }
    (gen_dir / "cost_summary.json").write_text(json.dumps(summary, indent=2))


# ---------- Likelihoods ----------

def build_likelihoods(records: list[dict], gen_key: str) -> dict:
    n = len(records)
    n_y1 = sum(1 for r in records if r["Y"] == 1)
    n_y0 = n - n_y1
    def likes_for(field: str) -> dict:
        TP = sum(1 for r in records if r["Y"] == 1 and r.get(field))
        FP = sum(1 for r in records if r["Y"] == 0 and r.get(field))
        FN = n_y1 - TP
        TN = n_y0 - FP
        p1 = (TP + 1) / (n_y1 + 2)
        p0 = (FP + 1) / (n_y0 + 2)
        return {"P_pass_given_Y1": p1, "P_pass_given_Y0": p0,
                "gap": p1 - p0, "TP": TP, "FN": FN, "FP": FP, "TN": TN}
    return {
        "generator": gen_key,
        "n_records": n, "n_evaluated": n, "n_resolved": n_y1,
        "prior_Y1": (n_y1 + 1) / (n + 2),
        "critic_likelihoods": {
            "L0_syntax":       likes_for("L0_syntax"),
            "L1_lint":         likes_for("L1_lint"),
            "L2_public_tests": likes_for("L2_public_tests"),
            "L3_llm_review":   likes_for("L3_llm_review"),
        },
        "smoothing": "Beta(1,1)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--n-instances", type=int, default=100)
    parser.add_argument("--n-patches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost-usd-per-model", default="4.0",
                        help="single float OR key=val,key=val,...")
    args = parser.parse_args()

    # Auto-load .env walking up the tree (matches lcb_calibrate's pattern)
    try:
        from dotenv import load_dotenv
        for env_path in [ROOT / ".env", ROOT.parent / ".env",
                         ROOT.parent.parent / ".env",
                         ROOT.parent.parent.parent / ".env"]:
            if env_path.exists() and env_path.stat().st_size > 0:
                load_dotenv(env_path, override=False)
    except ImportError:
        pass

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set"); sys.exit(1)
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    cap_str = args.max_cost_usd_per_model
    cap_default = 4.0
    cap_map: dict[str, float] = {}
    if "=" in cap_str:
        for pair in cap_str.split(","):
            k, v = pair.split("=", 1)
            cap_map[k.strip()] = float(v)
    else:
        cap_default = float(cap_str)

    problems = load_humanevalfix(args.n_instances, args.seed)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        cap = cap_map.get(gen, cap_default)
        calibrate_one_generator(gen, problems, args.n_patches, out_dir, cap, client)
        rec_path = out_dir / gen / "critic_results.jsonl"
        if rec_path.exists():
            recs = [json.loads(l) for l in open(rec_path) if l.strip()]
            tables = build_likelihoods(recs, gen)
            (out_dir / gen / "likelihood_tables.json").write_text(
                json.dumps(tables, indent=2))
            log.info("[%s] likelihood tables written", gen)
            cl = tables["critic_likelihoods"]
            log.info("  prior_Y1 = %.3f", tables["prior_Y1"])
            for k, v in cl.items():
                log.info("  %s: P(pass|Y=1)=%.3f P(pass|Y=0)=%.3f gap=%.3f "
                         "(TP=%d FP=%d TN=%d FN=%d)",
                         k, v["P_pass_given_Y1"], v["P_pass_given_Y0"], v["gap"],
                         v["TP"], v["FP"], v["TN"], v["FN"])


if __name__ == "__main__":
    main()
