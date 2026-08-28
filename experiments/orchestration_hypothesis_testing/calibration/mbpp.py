"""MBPP+ calibration for the orchestration controller.

Reuses lcb_calibrate.py's pipeline but with MBPP+'s schema:
  - test_list (3 visible assertions per problem) → L2_public_tests critic
  - test (full PLUS suite, hidden) → Y ground truth
  - Free-function output style (no class Solution)

Per-generator output goes under <output-dir>/<gen>/ matching the LCB layout:
  critic_results.jsonl, likelihood_tables.json, raw_responses/, cost_log.jsonl, cost_summary.json

Usage:
  python3 mbpp_calibrate.py \\
    --output-dir data/mbpp_calibration \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
    --n-instances 100 \\
    --n-patches 3 \\
    --max-cost-usd-per-model gpt5_mini=3.0,qwen3_coder=3.0,haiku45=3.0,sonnet45=15.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibration.lcb import (  # noqa: E402
    _make_client,
    extract_code,
    critic_L0_syntax,
    critic_L1_lint,
    critic_L3_review,
    cost_for_call,
    GENERATORS,
)
from _common.cost import CostTracker  # noqa: E402
from _common.telemetry import TelemetryLogger as _ActionTelemetry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mbpp_cal")


# ---------- Dataset loader ----------

def load_mbpp_plus(n_instances: int, seed: int = 42) -> list[dict]:
    # HF_HOME falls back to the system default (~/.cache/huggingface) when unset.
    # Override via env var if a different cache location is desired.
    from datasets import load_dataset
    ds = load_dataset("evalplus/mbppplus", split="test")
    log.info("loaded %d MBPP+ problems", len(ds))
    rows = list(ds)
    if n_instances and n_instances < len(rows):
        import random
        random.Random(seed).shuffle(rows)
        rows = rows[:n_instances]
        log.info("sampled %d instances (seed=%d)", n_instances, seed)
    return rows


# ---------- Test runners ----------

_LENIENT_HELPER = '''
import math as __math__
def __lenient_eq__(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): return False
        return all(__lenient_eq__(x, y) for x, y in zip(a, b))
    if isinstance(a, set) and isinstance(b, set):
        return a == b
    if isinstance(a, set) or isinstance(b, set):
        try:
            return set(a) == set(b)
        except TypeError:
            return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()): return False
        return all(__lenient_eq__(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return __math__.isclose(a, b, abs_tol=1e-6)
        except TypeError:
            return a == b
    return a == b
'''


def run_assertions(code: str, asserts: list[str], timeout: int = 5) -> tuple[int, int]:
    """Run a list of assertion strings against `code`. Returns (n_pass, n_total).

    Two-phase semantics to match MBPP+'s built-in `assertion()` helper used by
    the full PLUS test (Y):
      1. Try the assertion as written (strict ==). Pass if it doesn't raise.
      2. If strict fails AND the assertion is `assert X == Y`, retry with
         `assert __lenient_eq__(X, Y)` which tolerates list/tuple/set type
         coercion and float atol. This matches MBPP+'s `assertion(out, exp, atol)`
         helper that wraps comparisons in `set(out) == set(exp)` plus
         np.allclose-style tolerance.

    Each assertion is run in its own subprocess so a hang or import error in
    one test doesn't poison the others.
    """
    if not asserts or not code.strip():
        return 0, len(asserts)
    n_pass = 0
    for a in asserts:
        # Phase 1: strict (matches the original test_list semantics)
        prog = code + "\n" + a + "\n"
        try:
            r = subprocess.run([sys.executable, "-c", prog],
                               capture_output=True, timeout=timeout, text=True)
            if r.returncode == 0:
                n_pass += 1
                continue
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            pass
        # Phase 2: lenient fallback for `assert X == Y` to match Y's semantics
        m = re.match(r"^\s*assert\s+(.*?)\s*==\s*(.*?)\s*(?:,\s*.*?)?\s*$",
                     a.strip(), re.DOTALL)
        if not m:
            continue
        left, right = m.group(1), m.group(2)
        prog2 = (code + "\n" + _LENIENT_HELPER + "\n"
                  + f"assert __lenient_eq__({left}, {right})\n")
        try:
            r2 = subprocess.run([sys.executable, "-c", prog2],
                                capture_output=True, timeout=timeout, text=True)
            if r2.returncode == 0:
                n_pass += 1
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
    return n_pass, len(asserts)


def run_full_test(code: str, test_block: str, entry_point: str | None,
                   timeout: int = 10) -> bool:
    """Run the MBPP+ `test` field against `code`. MBPP+ tests are
    self-running scripts that call the function inline at module level
    (no `check(candidate)` indirection like HumanEval+). We just exec
    `code + test_block` and check that nothing raised.

    Returns True iff the test executes without raising.
    """
    if not code.strip() or not test_block.strip():
        return False
    prog = code + "\n" + test_block + "\n"
    try:
        r = subprocess.run([sys.executable, "-c", prog],
                           capture_output=True, timeout=timeout, text=True)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# ---------- Prompt ----------

def build_prompt(problem: dict) -> str:
    nl = problem.get("prompt", "")
    examples = problem.get("test_list", []) or []
    sig_hint = ""
    # Use the canonical solution's signature line as a hint for the function name
    canon = problem.get("code") or ""
    m = re.search(r"^\s*def\s+\w+\s*\([^)]*\)[^:]*:", canon, re.M)
    if m:
        sig_hint = m.group(0).rstrip(":")

    parts = [f"# Task\n{nl}\n"]
    if sig_hint:
        parts.append(f"\nThe function signature is:\n```python\n{sig_hint}:\n    ...\n```\n")
    if examples:
        parts.append("\nExamples (your solution must satisfy these):\n```python\n"
                     + "\n".join(examples[:3]) + "\n```\n")
    parts.append(
        "\nWrite a complete, self-contained Python solution that defines the "
        "function. Output ONLY the code in a single ```python``` code block. "
        "Do NOT include the example assertions or a `__main__` block in your output.\n"
    )
    return "".join(parts)


# ---------- Main pipeline ----------

def calibrate_one_generator(
    gen_key: str, problems: list[dict], n_patches: int, out_dir: Path,
    max_cost_usd: float, client,
) -> None:
    if gen_key not in GENERATORS:
        log.error("unknown generator: %s", gen_key)
        return
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
    dataset_name = out_dir.name  # e.g. "mbpp_calibration"
    tele = _ActionTelemetry(gen_dir / "action_telemetry.jsonl", dataset_name, gen_key)

    # Resume support
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
            inst_id = str(inst["task_id"])
            test_list = inst.get("test_list") or []
            test_block = inst.get("test") or ""
            entry_point = inst.get("entry_point")  # None for MBPP — we infer
            for pid in range(n_patches):
                if cost.capped: break
                if (inst_id, pid) in done:
                    continue
                # Generate
                try:
                    _t0 = time.perf_counter()
                    resp = gen_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": build_prompt(inst)}],
                        temperature=0.7, max_tokens=4000,
                    )
                    _gen_rt = time.perf_counter() - _t0
                    text = resp.choices[0].message.content or ""
                    usage = resp.usage
                    c = cost_for_call(model_id, usage.prompt_tokens, usage.completion_tokens)
                    cost.record(c, prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                instance_id=inst_id, patch_id=pid)
                    tele.record(instance_id=inst_id, patch_id=pid, action_type="generate",
                                runtime_s=_gen_rt, api_cost_usd=c)
                except Exception as e:
                    log.warning("[%s] gen failed for %s: %s", gen_key, inst_id, e)
                    continue
                (raw_dir / f"{inst_id}_p{pid}.txt").write_text(text)
                code = extract_code(text)
                # Critics
                try:
                    _t0 = time.perf_counter()
                    l0 = critic_L0_syntax(code)
                    tele.record(instance_id=inst_id, patch_id=pid, action_type="critic_L0",
                                runtime_s=time.perf_counter() - _t0, passed=bool(l0))
                    _t0 = time.perf_counter()
                    l1 = critic_L1_lint(code)
                    tele.record(instance_id=inst_id, patch_id=pid, action_type="critic_L1",
                                runtime_s=time.perf_counter() - _t0, passed=bool(l1))
                    _t0 = time.perf_counter()
                    l2_pass, l2_total = run_assertions(code, test_list)
                    l2_ok = (l2_pass == l2_total) and l2_total > 0
                    tele.record(instance_id=inst_id, patch_id=pid, action_type="critic_L2",
                                runtime_s=time.perf_counter() - _t0, passed=l2_ok)
                    _t0 = time.perf_counter()
                    Y = run_full_test(code, test_block, entry_point)
                    tele.record(instance_id=inst_id, patch_id=pid, action_type="verify",
                                runtime_s=time.perf_counter() - _t0, passed=bool(Y))
                except Exception as e:
                    log.warning("[%s] critic eval failed for %s_p%d: %s", gen_key, inst_id, pid, e)
                    continue
                l3 = None
                if not cost.capped:
                    try:
                        _t0 = time.perf_counter()
                        l3_pass, l3_cost = critic_L3_review(
                            inst.get("prompt", "")[:3000], code, client)
                        tele.record(instance_id=inst_id, patch_id=pid, action_type="critic_L3",
                                    runtime_s=time.perf_counter() - _t0,
                                    passed=bool(l3_pass), api_cost_usd=l3_cost)
                        cost.record(l3_cost, prompt_tokens=0, completion_tokens=0,
                                    instance_id=inst_id, patch_id=pid,
                                    extra={"kind": "L3_review"})
                        l3 = l3_pass
                    except Exception as e:
                        log.warning("[%s] L3 failed for %s_p%d: %s", gen_key, inst_id, pid, e)
                rec = {
                    "generator": gen_key,
                    "instance_id": inst_id,
                    "patch_id": pid,
                    "Y": int(bool(Y)),
                    "L0_syntax": bool(l0),
                    "L1_lint": bool(l1),
                    "L2_public_tests": bool(l2_ok),
                    "L3_llm_review": l3,
                    "diff_chars": len(code),
                    "l2_pass_rate": (l2_pass / max(l2_total, 1)),
                }
                records.append(rec)
                out_fp.write(json.dumps(rec) + "\n")
                out_fp.flush()
                os.fsync(out_fp.fileno())
                done.add((inst_id, pid))
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost-usd-per-model", default="3.0",
                        help="single float OR key=val,...")
    parser.add_argument("--extend-existing", action="store_true",
                        help="If <output-dir>/sample.json exists, keep its instance_ids "
                             "at the head of the sampled list, then append new "
                             "candidates from the full MBPP+ pool to reach --n-instances. "
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

    # Parse cap
    cap_str = args.max_cost_usd_per_model
    cap_default = 3.0
    cap_map: dict[str, float] = {}
    if "=" in cap_str:
        for pair in cap_str.split(","):
            k, v = pair.split("=", 1)
            cap_map[k.strip()] = float(v)
    else:
        cap_default = float(cap_str)

    # Load FULL MBPP+ pool then shuffle with seed (so we can rearrange
    # for --extend-existing before truncating to --n-instances).
    all_problems = load_mbpp_plus(0, seed=args.seed)  # sentinel 0 = all
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
            log.warning("%d existing IDs not in current MBPP+ pool. Examples: %s",
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
        # Build likelihoods after each generator finishes
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
