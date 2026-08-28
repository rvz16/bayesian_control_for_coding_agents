"""CodeContests calibration for the orchestration controller.

Uses deepmind/code_contests (test split, ~165 problems). Each instance is
a competitive-programming problem with stdin/stdout test cases:
  - public_tests    -> L2 (small visible set)
  - private_tests + generated_tests -> Y (large hidden set, capped at 30)

Test runner is subprocess-based: code is run as a python program, stdin
is fed each test input, stdout is compared line-by-line against the
expected output. Mirrors the _run_stdio_tests helper in
iter_refine_real_baselines.py.

Per-generator output mirrors the LCB/MBPP/HumanEval+ layout under
<output-dir>/<gen>/{critic_results.jsonl,raw_responses/,...}.

NOTE: deepmind/code_contests is ~13 GB total (all splits). The naive
`load_dataset("deepmind/code_contests", split="test")` greedily pulls
ALL the train shards too, even when only test is requested. This loader
sidesteps that by directly fetching the single 63 MB test parquet via
hf_hub_download. First-run download is ~20s; cached thereafter.

Usage:
  python3 codecontests_calibrate.py \\
    --output-dir data/codecontests_calibration \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
    --n-instances 100 \\
    --n-patches 3 \\
    --max-cost-usd-per-model 5.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
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
from _common.cost import CostTracker  # noqa: E402
from _common.telemetry import TelemetryLogger  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cc_cal")


# Caps to keep oracle eval bounded (CodeContests can have hundreds of
# private/generated tests per problem; running them all per-patch blows
# up wall-clock without meaningful information gain past ~30 samples).
PUBLIC_TEST_CAP   = 10   # L2 evaluation
ORACLE_TEST_CAP   = 30   # Y evaluation (private + generated, deduped)
PER_TEST_TIMEOUT  = 5    # seconds


# ---------- Dataset loader ----------

def load_codecontests(n_instances: int, seed: int = 42) -> list[dict]:
    """Load CodeContests test split via deepmind/code_contests on HF.

    Each problem has:
      name           : unique string ID (used as instance_id)
      description    : natural-language problem statement
      public_tests   : dict {"input": [...], "output": [...]}
      private_tests  : dict {"input": [...], "output": [...]}
      generated_tests: dict {"input": [...], "output": [...]}
      solutions      : dict {"language": [...], "solution": [...]} -- ignored
                       (we don't use the bundled solutions; generators write
                       their own)
    """
    # Direct parquet fetch — pulls just the 63 MB test shard, not the
    # full 13 GB repo. load_dataset() greedy-pulls all train shards even
    # when split="test"; this is the known workaround (see
    # README in this package).
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    parquet_path = hf_hub_download(
        repo_id="deepmind/code_contests",
        filename="data/test-00000-of-00001-9c49eeff30aacaa8.parquet",
        repo_type="dataset",
    )
    tbl = pq.read_table(parquet_path,
                        columns=["name", "description", "public_tests",
                                 "private_tests", "generated_tests"])
    rows = tbl.to_pylist()
    out: list[dict] = []
    for r in rows:
        # Normalize: keep only the fields the calibration + iter pipelines need
        out.append({
            "name":            r.get("name"),
            "description":     r.get("description") or "",
            "public_tests":    dict(r.get("public_tests") or {}),
            "private_tests":   dict(r.get("private_tests") or {}),
            "generated_tests": dict(r.get("generated_tests") or {}),
        })
    log.info("loaded %d CodeContests test problems (parquet-only loader)", len(out))
    if n_instances and n_instances < len(out):
        random.Random(seed).shuffle(out)
        out = out[:n_instances]
        log.info("sampled %d instances (seed=%d)", n_instances, seed)
    return out


# ---------- Test runner (stdin/stdout) ----------

def run_stdio_tests(code: str, inputs: list, outputs: list,
                    timeout: int = PER_TEST_TIMEOUT) -> tuple[int, int]:
    """Run `code` as a subprocess; for each (input_str, expected_output_str)
    pair, pipe input to stdin and compare stripped stdout (line-by-line)
    against expected.

    Returns (n_pass, n_total). Caps at min(len(inputs), len(outputs)) to
    handle malformed pairs. Mirrors _run_stdio_tests in iter_refine
    so calibration and iter use the SAME test runner.

    NB: CodeContests answers are sometimes whitespace-sensitive or have
    multiple valid outputs. This runner does a simple line-by-line
    rstrip+compare which may report false negatives. Acceptable for
    calibration -- the controller learns from whatever signal the
    inline critic provides.
    """
    total = min(len(inputs), len(outputs))
    if total == 0:
        return 0, 0
    n_pass = 0
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        tf.write(code); tf.flush(); tf.close()
        for inp, exp in zip(inputs[:total], outputs[:total]):
            try:
                result = subprocess.run(
                    [sys.executable, tf.name],
                    input=str(inp), capture_output=True, text=True,
                    timeout=timeout)
                actual_lines   = (result.stdout or "").strip().splitlines()
                expected_lines = (str(exp) or "").strip().splitlines()
                if [a.rstrip() for a in actual_lines] == [e.rstrip() for e in expected_lines]:
                    n_pass += 1
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
    finally:
        try: os.unlink(tf.name)
        except Exception: pass
    return n_pass, total


# ---------- Prompt ----------

def build_prompt(problem: dict) -> str:
    """CodeContests prompt: show the problem statement (capped) + one
    public input/output example, ask for a self-contained Python program
    that reads from stdin and writes to stdout."""
    desc = (problem.get("description") or "").strip()[:6000]
    pt   = problem.get("public_tests") or {}
    example = ""
    if isinstance(pt, dict) and pt.get("input") and pt.get("output"):
        example = (
            "\n## Example test\n"
            f"Input:\n```\n{pt['input'][0][:500]}\n```\n"
            f"Expected output:\n```\n{pt['output'][0][:500]}\n```\n"
        )
    return (
        "Solve this competitive programming problem. Output a self-contained "
        "Python 3 program that reads the input from stdin and writes the "
        "answer to stdout. Use efficient algorithms; problems may have large "
        "input sizes. Return ONLY the complete program inside a single "
        "```python``` code block — no commentary, no `__main__` guard.\n\n"
        "## Problem\n"
        f"{desc}\n"
        f"{example}"
    )


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
    dataset_name = out_dir.name  # e.g. "codecontests_calibration"
    tele = TelemetryLogger(gen_dir / "action_telemetry.jsonl",
                           dataset=dataset_name, model_name=gen_key)

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
            inst_id = inst["name"]
            pt = inst.get("public_tests")   or {}
            ot = inst.get("private_tests")  or {}
            gt = inst.get("generated_tests") or {}
            # L2 (public) inputs/outputs -- capped at PUBLIC_TEST_CAP
            l2_inputs  = list(pt.get("input",  []))[:PUBLIC_TEST_CAP]
            l2_outputs = list(pt.get("output", []))[:PUBLIC_TEST_CAP]
            # Y (oracle) = private + generated, capped at ORACLE_TEST_CAP
            y_in_full  = list(ot.get("input", []))  + list(gt.get("input",  [])) + l2_inputs
            y_out_full = list(ot.get("output", [])) + list(gt.get("output", [])) + l2_outputs
            y_inputs  = y_in_full[:ORACLE_TEST_CAP]
            y_outputs = y_out_full[:ORACLE_TEST_CAP]

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
                # Sanitize for filesystem (CodeContests names can contain spaces / slashes)
                safe_id = str(inst_id).replace("/", "_").replace(" ", "_")
                (raw_dir / f"{safe_id}_p{pid}.txt").write_text(text)
                code = extract_code(text)
                try:
                    _t0 = time.perf_counter()
                    l2_pass, l2_total = run_stdio_tests(code, l2_inputs, l2_outputs)
                    l2_ok = (l2_total > 0) and (l2_pass == l2_total)
                    tele.record(action_type="critic_L2",
                                runtime_s=time.perf_counter() - _t0,
                                instance_id=inst_id, patch_id=pid, passed=bool(l2_ok))
                    _t0 = time.perf_counter()
                    y_pass, y_total = run_stdio_tests(code, y_inputs, y_outputs)
                    Y = 1 if (y_total > 0) and (y_pass == y_total) else 0
                    tele.record(action_type="verify",
                                runtime_s=time.perf_counter() - _t0,
                                instance_id=inst_id, patch_id=pid, passed=bool(Y))
                    _t0 = time.perf_counter()
                    l0 = critic_L0_syntax(code)
                    tele.record(action_type="critic_L0",
                                runtime_s=time.perf_counter() - _t0,
                                instance_id=inst_id, patch_id=pid, passed=bool(l0))
                    _t0 = time.perf_counter()
                    l1 = critic_L1_lint(code)
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
                            (inst.get("description") or "")[:3000], code, client)
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
                    "y_pass_rate": (y_pass / max(y_total, 1)),
                    "l2_pass_rate": (l2_pass / max(l2_total, 1)),
                    "n_public_tests":  l2_total,
                    "n_oracle_tests":  y_total,
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
    parser.add_argument("--extend-existing", action="store_true",
                        help="If <output-dir>/sample.json exists, keep its instance_ids "
                             "at the head of the sampled list, then append new "
                             "candidates from the full CodeContests pool to reach --n-instances. "
                             "Resume logic in calibrate_one_generator skips already-"
                             "completed (instance, patch) pairs.")
    parser.add_argument("--max-cost-usd-per-model", default="5.0",
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
    cap_default = 5.0
    cap_map: dict[str, float] = {}
    if "=" in cap_str:
        for pair in cap_str.split(","):
            k, v = pair.split("=", 1)
            cap_map[k.strip()] = float(v)
    else:
        cap_default = float(cap_str)

    all_problems = load_codecontests(0, args.seed)  # sentinel 0 = all
    import random as _rng
    _rng.Random(args.seed).shuffle(all_problems)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample.json"
    if args.extend_existing and sample_path.exists():
        existing = json.loads(sample_path.read_text())
        existing_ids = [str(s["name"]) for s in existing]
        existing_set = set(existing_ids)
        by_id = {str(p["name"]): p for p in all_problems}
        head = [by_id[i] for i in existing_ids if i in by_id]
        if len(head) != len(existing_ids):
            missing = [i for i in existing_ids if i not in by_id]
            log.warning("%d existing IDs not in current CodeContests pool. Examples: %s",
                          len(missing), missing[:3])
        tail = [p for p in all_problems if str(p["name"]) not in existing_set]
        all_problems = head + tail
        log.info("extend-existing: %d existing + %d new (target n=%d)",
                 len(head), len(tail), args.n_instances)
    problems = all_problems[: args.n_instances]
    sample_path.write_text(json.dumps(
        [{"name": p["name"]} for p in problems], indent=2))
    log.info("sampled %d instances (seed=%d) → sample.json", len(problems), args.seed)

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
