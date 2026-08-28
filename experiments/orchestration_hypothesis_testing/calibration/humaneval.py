"""HumanEval+ calibration for the orchestration controller.

Uses evalplus.data.get_human_eval_plus() (same loader as ThinkBooster's
human_eval_plus_evaluator.py) to get clean L2/Y split:
  - base_input  (≈7 inputs per problem, original HumanEval) → L2_public_tests
  - plus_input  (≈1000 inputs per problem, EvalPlus extras)
  - Y = pass on base_input AND plus_input combined (full PLUS suite)

Per-generator output mirrors the LCB/MBPP layout under <output-dir>/<gen>/.

Usage:
  python3 humaneval_calibrate.py \\
    --output-dir data/humaneval_calibration \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
    --n-instances 100 \\
    --n-patches 3 \\
    --plus-input-cap 200 \\
    --max-cost-usd-per-model gpt5_mini=4.0,qwen3_coder=4.0,haiku45=4.0,sonnet45=15.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
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
from _common.cost import CostTracker  # noqa: E402
from _common.telemetry import TelemetryLogger as _ActionTelemetry  # noqa: E402
import time as _time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("humaneval_cal")


# ---------- Dataset loader ----------

def load_humaneval_plus(n_instances: int, plus_input_cap: int, seed: int = 42) -> list[dict]:
    """Load HumanEval+ via evalplus library. Returns flat dicts."""
    # HF_HOME falls back to the system default (~/.cache/huggingface) when unset.
    # Override via env var if a different cache location is desired.
    from evalplus.data import get_human_eval_plus
    raw = get_human_eval_plus()
    log.info("loaded %d HumanEval+ problems (%d capped plus_input)", len(raw), plus_input_cap)
    out = []
    for task_id, r in raw.items():
        plus = r.get("plus_input") or []
        if plus_input_cap > 0 and len(plus) > plus_input_cap:
            plus = plus[:plus_input_cap]
        canon_full = r["prompt"] + r["canonical_solution"]
        out.append({
            "task_id": task_id,
            "prompt": r["prompt"],
            "entry_point": r["entry_point"],
            "canonical_full": canon_full,
            "base_input": r.get("base_input") or [],
            "plus_input": plus,
            "atol": r.get("atol") or 0,
        })
    if n_instances and n_instances < len(out):
        import random
        random.Random(seed).shuffle(out)
        out = out[:n_instances]
        log.info("sampled %d instances (seed=%d)", n_instances, seed)
    return out


# ---------- Test runner ----------

def run_test_inputs(model_code: str, entry_point: str, canon_full_code: str,
                    inputs: list, atol: float, timeout: int = 30) -> tuple[int, int]:
    """Run model code against given inputs, comparing each output to the
    canonical solution's output. Returns (n_pass, n_total).

    Both model and canonical are exec'd in subprocess. Canonical is namespaced
    so model's `entry_point` definition takes precedence at the import line.
    """
    n = len(inputs)
    if n == 0 or not model_code.strip():
        return 0, n
    prog = f"""import sys, math

# === Model code ===
{model_code}
try:
    _MODEL_FN = {entry_point}
except NameError:
    print(0); sys.exit(0)

# === Canonical in isolated namespace ===
_canon_ns = {{}}
exec({canon_full_code!r}, _canon_ns)
_CANON_FN = _canon_ns.get({entry_point!r})
if _CANON_FN is None:
    print(0); sys.exit(0)

_inputs = {inputs!r}
_atol = {atol}

def _eq(a, b, atol):
    try:
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(_eq(x, y, atol) for x, y in zip(a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(_eq(a[k], b[k], atol) for k in a)
        if isinstance(a, set) and isinstance(b, set):
            return a == b
        if isinstance(a, float) or isinstance(b, float):
            try:
                return math.isclose(a, b, abs_tol=atol if atol else 1e-9)
            except Exception:
                return False
        return a == b
    except Exception:
        return False

n_pass = 0
for inp in _inputs:
    try:
        m_out = _MODEL_FN(*inp)
        c_out = _CANON_FN(*inp)
        if _eq(m_out, c_out, _atol):
            n_pass += 1
    except Exception:
        pass
print(n_pass)
"""
    try:
        r = subprocess.run([sys.executable, "-c", prog],
                           capture_output=True, timeout=timeout, text=True)
        if r.returncode != 0:
            return 0, n
        try:
            tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "0"
            return int(tail), n
        except (ValueError, IndexError):
            return 0, n
    except subprocess.TimeoutExpired:
        return 0, n


# ---------- Prompt ----------

def build_prompt(problem: dict) -> str:
    """The HumanEval+ prompt is the function signature + docstring. Ask the
    model to provide a complete implementation. EvalPlus methodology uses an
    instruction prefix and wraps the prompt in a code block."""
    raw = problem["prompt"].strip()
    return (
        "Please provide a self-contained Python script that solves the "
        "following problem in a markdown code block:\n"
        f"```python\n{raw}\n```\n"
        "\nOutput ONLY the complete function (signature included) inside a single "
        "```python``` code block. Do NOT include the example assertions or a "
        "`__main__` block in your output."
    )


# ---------- Main pipeline ----------

def calibrate_one_generator(
    gen_key: str, problems: list[dict], n_patches: int, out_dir: Path,
    max_cost_usd: float, client,
) -> None:
    if gen_key not in GENERATORS:
        log.error("unknown generator: %s", gen_key); return
    model_id, label, _base_url = GENERATORS[gen_key]
    # Per-generator generation client: vLLM for qwen25_32b, OpenRouter otherwise.
    gen_client = _make_client(gen_key) if _base_url else client
    log.info("=== %s (%s) — cap $%.2f ===", gen_key, model_id, max_cost_usd)
    gen_dir = out_dir / gen_key
    gen_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = gen_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)
    results_path = gen_dir / "critic_results.jsonl"
    cost = CostTracker(name=gen_key, cap_usd=max_cost_usd, log_path=gen_dir / "cost_log.jsonl")
    tele = _ActionTelemetry(gen_dir / "action_telemetry.jsonl", "humaneval", gen_key)

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
            entry = inst["entry_point"]
            canon_full = inst["canonical_full"]
            base_in = inst["base_input"]
            plus_in = inst["plus_input"]
            atol = inst["atol"]
            for pid in range(n_patches):
                if cost.capped: break
                if (str(inst_id), pid) in done:
                    continue
                try:
                    _t0 = _time.perf_counter()
                    resp = gen_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": build_prompt(inst)}],
                        temperature=0.7, max_tokens=4000,
                    )
                    _gen_dt = _time.perf_counter() - _t0
                    text = resp.choices[0].message.content or ""
                    usage = resp.usage
                    c = cost_for_call(model_id, usage.prompt_tokens, usage.completion_tokens)
                    cost.record(c, prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                instance_id=inst_id, patch_id=pid)
                    tele.record(instance_id=inst_id, patch_id=pid,
                                action_type="generate", runtime_s=_gen_dt,
                                api_cost_usd=c)
                except Exception as e:
                    log.warning("[%s] gen failed for %s: %s", gen_key, inst_id, e)
                    continue
                safe_id = inst_id.replace("/", "_")
                (raw_dir / f"{safe_id}_p{pid}.txt").write_text(text)
                code = extract_code(text)
                # L2 = base inputs only (visible)
                # Y = base + plus inputs (full PLUS test suite)
                try:
                    _t0 = _time.perf_counter()
                    l2_pass, l2_total = run_test_inputs(code, entry, canon_full, base_in, atol)
                    _l2_dt = _time.perf_counter() - _t0
                    l2_ok = (l2_pass == l2_total) and l2_total > 0
                    tele.record(instance_id=inst_id, patch_id=pid,
                                action_type="critic_L2", runtime_s=_l2_dt,
                                passed=bool(l2_ok))
                    _t0 = _time.perf_counter()
                    full_in = list(base_in) + list(plus_in)
                    y_pass, y_total = run_test_inputs(code, entry, canon_full, full_in, atol)
                    _ver_dt = _time.perf_counter() - _t0
                    Y = 1 if (y_pass == y_total) and y_total > 0 else 0
                    tele.record(instance_id=inst_id, patch_id=pid,
                                action_type="verify", runtime_s=_ver_dt,
                                passed=bool(Y))
                    _t0 = _time.perf_counter()
                    l0 = critic_L0_syntax(code)
                    tele.record(instance_id=inst_id, patch_id=pid,
                                action_type="critic_L0",
                                runtime_s=_time.perf_counter() - _t0,
                                passed=bool(l0))
                    _t0 = _time.perf_counter()
                    l1 = critic_L1_lint(code)
                    tele.record(instance_id=inst_id, patch_id=pid,
                                action_type="critic_L1",
                                runtime_s=_time.perf_counter() - _t0,
                                passed=bool(l1))
                except Exception as e:
                    log.warning("[%s] critic eval failed for %s_p%d: %s", gen_key, inst_id, pid, e)
                    continue
                l3 = None
                if not cost.capped:
                    try:
                        _t0 = _time.perf_counter()
                        l3_pass, l3_cost = critic_L3_review(
                            inst.get("prompt", "")[:3000], code, client)
                        _l3_dt = _time.perf_counter() - _t0
                        cost.record(l3_cost, prompt_tokens=0, completion_tokens=0,
                                    instance_id=inst_id, patch_id=pid,
                                    extra={"kind": "L3_review"})
                        l3 = l3_pass
                        tele.record(instance_id=inst_id, patch_id=pid,
                                    action_type="critic_L3", runtime_s=_l3_dt,
                                    passed=bool(l3_pass), api_cost_usd=l3_cost)
                    except Exception as e:
                        log.warning("[%s] L3 failed for %s_p%d: %s", gen_key, inst_id, pid, e)
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
                    "y_pass_rate": (y_pass / max(y_total, 1)),
                    "l2_pass_rate": (l2_pass / max(l2_total, 1)),
                    "n_base_inputs": l2_total,
                    "n_plus_inputs": len(plus_in),
                }
                records.append(rec)
                out_fp.write(json.dumps(rec) + "\n")
                out_fp.flush()
                os.fsync(out_fp.fileno())
                done.add((str(inst_id), pid))
                if len(records) % 10 == 0:
                    log.info("[%s] %d records, cost $%.4f", gen_key, len(records), cost.total_usd)
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
            "L0_syntax": likes_for("L0_syntax"),
            "L1_lint": likes_for("L1_lint"),
            "L2_public_tests": likes_for("L2_public_tests"),
            "L3_llm_review": likes_for("L3_llm_review"),
        },
        "smoothing": "Beta(1,1)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--n-instances", type=int, default=100)
    parser.add_argument("--n-patches", type=int, default=3)
    parser.add_argument("--plus-input-cap", type=int, default=200,
                        help="cap on plus_input length per problem (HumanEval+ has ~1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost-usd-per-model", default="4.0")
    parser.add_argument("--extend-existing", action="store_true",
                        help="If <output-dir>/sample.json exists, keep its instance_ids "
                             "at the head of the sampled list, then append new "
                             "candidates from the full HumanEval+ pool to reach --n-instances. "
                             "Resume logic in calibrate_one_generator skips already-"
                             "completed (instance, patch) pairs.")
    args = parser.parse_args()

    # Auto-load OPENROUTER_API_KEY from a .env file walking up the tree
    # (same chain as lcb_calibrate.py).
    try:
        from dotenv import load_dotenv
        for env_path in [ROOT / ".env", ROOT.parent / ".env",
                         ROOT.parent.parent / ".env",
                         ROOT.parent.parent.parent / ".env"]:
            if env_path.exists() and env_path.stat().st_size > 0:
                load_dotenv(env_path, override=False)
    except ImportError:
        pass  # dotenv optional; rely on already-exported env vars

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

    all_problems = load_humaneval_plus(0, args.plus_input_cap, args.seed)
    import random as _rng
    _rng.Random(args.seed).shuffle(all_problems)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample.json"
    if args.extend_existing and sample_path.exists():
        existing = json.loads(sample_path.read_text())
        existing_ids = [str(s["task_id"]) for s in existing]
        existing_set = set(existing_ids)
        by_id = {str(p["task_id"]): p for p in all_problems}
        head = [by_id[i] for i in existing_ids if i in by_id]
        if len(head) != len(existing_ids):
            missing = [i for i in existing_ids if i not in by_id]
            log.warning("%d existing IDs not in current HumanEval+ pool. Examples: %s",
                          len(missing), missing[:3])
        tail = [p for p in all_problems if str(p["task_id"]) not in existing_set]
        all_problems = head + tail
        log.info("extend-existing: %d existing + %d new (target n=%d)",
                 len(head), len(tail), args.n_instances)
    problems = all_problems[: args.n_instances]
    sample_path.write_text(json.dumps(
        [{"task_id": p["task_id"]} for p in problems], indent=2))
    log.info("sampled %d instances (seed=%d) → sample.json", len(problems), args.seed)

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        cap = cap_map.get(gen, cap_default)
        calibrate_one_generator(gen, problems, args.n_patches, out_dir, cap, client)
        rec_path = out_dir / gen / "critic_results.jsonl"
        if rec_path.exists():
            recs = [json.loads(l) for l in open(rec_path) if l.strip()]
            tables = build_likelihoods(recs, gen)
            (out_dir / gen / "likelihood_tables.json").write_text(json.dumps(tables, indent=2))
            log.info("[%s] likelihood tables written", gen)
            cl = tables["critic_likelihoods"]
            log.info("  prior_Y1 = %.3f", tables["prior_Y1"])
            for k, v in cl.items():
                log.info("  %s: P(pass|Y=1)=%.3f P(pass|Y=0)=%.3f gap=%.3f (TP=%d FP=%d TN=%d FN=%d)",
                         k, v["P_pass_given_Y1"], v["P_pass_given_Y0"], v["gap"],
                         v["TP"], v["FP"], v["TN"], v["FN"])


if __name__ == "__main__":
    main()
