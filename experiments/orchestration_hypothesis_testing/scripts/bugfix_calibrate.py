"""Single-shot calibration for the bug-fixing Table 4 rows.

This mirrors the other benchmark calibration scripts, but for:
  - HumanEvalFix
  - CodeContests

Output layout per generator:
  <output-dir>/<gen>/
    critic_results.jsonl
    likelihood_tables.json
    raw_responses/
    cost_log.jsonl
    cost_summary.json

Critic-slot mapping for the existing Table 4 policy code:
  L0_syntax       <- syntax critic
  L2_public_tests <- early partial-test critic
  L3_llm_review   <- mid partial-test critic

`L1_lint` is recorded for audits, but the policy layer intentionally ignores it.

Example:
  python3 scripts/bugfix_calibrate.py \\
    --benchmark humanevalfix \\
    --output-dir data/humanevalfix_calibration \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45,qwen25_7b,qwen25_32b,gpt_oss_20b \\
    --n-instances 100 --n-patches 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bugfix_table4_common import (  # noqa: E402
    build_initial_prompt,
    build_likelihood_tables,
    evaluate_candidate,
    extract_code,
    get_failure_output,
    get_initial_source,
    list_task_ids,
    safe_stem,
)
from _common.cost import CostTracker  # noqa: E402
from calibration.lcb import (  # noqa: E402
    GENERATORS,
    OPENROUTER_KEY_NAMES,
    _make_client,
    canonical_generator_key,
    cost_for_call,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bugfix_calibrate")

DEFAULT_GENERATORS = ",".join([
    "gpt5_mini",
    "qwen3_coder",
    "haiku45",
    "sonnet45",
    "qwen25_7b",
    "qwen25_32b",
    "gpt_oss_20b",
])


def load_env_chain() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None

    for env_path in [
        ROOT / ".env",
        ROOT.parent / ".env",
        ROOT.parent.parent / ".env",
        ROOT.parent.parent.parent / ".env",
        ROOT.parent.parent.parent.parent / ".env",
    ]:
        if env_path.exists() and env_path.stat().st_size > 0:
            if load_dotenv is not None:
                load_dotenv(env_path, override=False)
                continue
            for raw_line in env_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or key in os.environ:
                    continue
                if value and value[0] in {"'", '"'}:
                    try:
                        parsed = shlex.split(f"dummy={value}", posix=True)
                    except ValueError:
                        parsed = [f"dummy={value.strip('\"')}"]
                    value = parsed[0].split("=", 1)[1] if parsed else value
                else:
                    value = value.split(" #", 1)[0].strip()
                os.environ[key] = value


def validate_provider_env(generators: list[str]) -> None:
    needs_openrouter = any(GENERATORS[gen][2] is None for gen in generators)
    if needs_openrouter and not any(os.environ.get(name, "").strip() for name in OPENROUTER_KEY_NAMES):
        raise SystemExit(
            "OpenRouter API key is not set. Expected one of: "
            "OPENROUTER_API_KEY, OPEN_ROUTER_API_KEY, OPEN_ROUTER."
        )


def parse_cap_map(raw: str, generators: list[str]) -> dict[str, float]:
    if "=" not in raw:
        flat = float(raw)
        return {gen: flat for gen in generators}
    out: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, val = pair.split("=", 1)
        out[canonical_generator_key(key)] = float(val)
    for gen in generators:
        out.setdefault(gen, 5.0)
    return out


def load_or_create_sample(sample_path: Path, benchmark: str, n_instances: int, seed: int) -> list[str]:
    if sample_path.exists():
        raw = json.loads(sample_path.read_text())
        if raw and isinstance(raw[0], dict):
            return [str(row["instance_id"]) for row in raw]
        return [str(x) for x in raw]
    task_ids = list_task_ids(benchmark)
    random.Random(seed).shuffle(task_ids)
    if n_instances > 0:
        task_ids = task_ids[:n_instances]
    sample_path.write_text(json.dumps([{"instance_id": tid} for tid in task_ids], indent=2))
    return task_ids


def run_one_generator(
    *,
    benchmark: str,
    gen_key: str,
    task_ids: list[str],
    n_patches: int,
    temperature: float,
    max_tokens: int,
    out_dir: Path,
    max_cost_usd: float,
) -> None:
    if gen_key not in GENERATORS:
        raise SystemExit(f"unknown generator: {gen_key}")
    model_id, _label, _base_url = GENERATORS[gen_key]
    client = _make_client(gen_key)

    gen_dir = out_dir / gen_key
    gen_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = gen_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)
    results_path = gen_dir / "critic_results.jsonl"
    cost = CostTracker(name=gen_key, cap_usd=max_cost_usd, log_path=gen_dir / "cost_log.jsonl")

    done: set[tuple[str, int]] = set()
    records: list[dict] = []
    if results_path.exists():
        for line in open(results_path):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            done.add((str(rec["instance_id"]), int(rec["patch_id"])))
            records.append(rec)
        log.info("[%s/%s] resuming with %d records", benchmark, gen_key, len(records))

    out_fp = open(results_path, "a", buffering=1)
    try:
        for task_id in task_ids:
            if cost.capped:
                log.warning("[%s/%s] cost cap reached", benchmark, gen_key)
                break
            initial_source = get_initial_source(benchmark, task_id)
            if not initial_source.strip():
                log.warning("[%s/%s] %s has empty source, skipping", benchmark, gen_key, task_id)
                continue
            test_output = get_failure_output(benchmark, task_id, initial_source)
            prompt = build_initial_prompt(benchmark, task_id, initial_source, test_output)
            stem = safe_stem(task_id)

            for patch_id in range(n_patches):
                if cost.capped:
                    break
                if (task_id, patch_id) in done:
                    continue
                try:
                    resp = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text = resp.choices[0].message.content or ""
                    usage = resp.usage
                    call_cost = cost_for_call(
                        model_id,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    )
                    cost.record(
                        call_cost,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        instance_id=task_id,
                        patch_id=patch_id,
                    )
                except Exception as exc:
                    log.warning("[%s/%s] generation failed for %s_p%d: %s",
                                benchmark, gen_key, task_id, patch_id, exc)
                    continue

                (raw_dir / f"{stem}_p{patch_id}.txt").write_text(text)
                code = extract_code(text, initial_source)
                eval_rec = evaluate_candidate(benchmark, task_id, code)
                rec = {
                    "benchmark": benchmark,
                    "generator": gen_key,
                    "instance_id": task_id,
                    "patch_id": patch_id,
                    "Y": eval_rec["Y"],
                    "L0_syntax": eval_rec["L0_syntax"],
                    "L1_lint": eval_rec["L1_lint"],
                    "L2_public_tests": eval_rec["L2_public_tests"],
                    "L3_llm_review": eval_rec["L3_llm_review"],
                    "diff_chars": len(code),
                    "oracle_detail": eval_rec["oracle_detail"],
                    "l2_detail": eval_rec["l2_detail"],
                    "l3_detail": eval_rec["l3_detail"],
                }
                records.append(rec)
                out_fp.write(json.dumps(rec) + "\n")
                out_fp.flush()
                os.fsync(out_fp.fileno())
                done.add((task_id, patch_id))
                if len(records) % 10 == 0:
                    log.info("[%s/%s] %d records, cost $%.4f",
                             benchmark, gen_key, len(records), cost.total_usd)
    finally:
        out_fp.close()

    tables = build_likelihood_tables(records, gen_key, benchmark)
    (gen_dir / "likelihood_tables.json").write_text(json.dumps(tables, indent=2))
    (gen_dir / "cost_summary.json").write_text(json.dumps(cost.snapshot(), indent=2))
    log.info("[%s/%s] wrote %d records", benchmark, gen_key, len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["humanevalfix", "codecontests"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", default=DEFAULT_GENERATORS)
    parser.add_argument("--n-instances", type=int, default=0, help="0 = all available")
    parser.add_argument("--n-patches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-cost-usd-per-model", default="5.0")
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    generators = [canonical_generator_key(g) for g in args.generators.split(",") if g.strip()]
    load_env_chain()
    validate_provider_env(generators)
    caps = parse_cap_map(args.max_cost_usd_per_model, generators)
    sample_path = out_dir / "sample.json"
    task_ids = load_or_create_sample(sample_path, args.benchmark, args.n_instances, args.seed)

    log.info("benchmark=%s  tasks=%d  generators=%s",
             args.benchmark, len(task_ids), ",".join(generators))
    for gen_key in generators:
        run_one_generator(
            benchmark=args.benchmark,
            gen_key=gen_key,
            task_ids=task_ids,
            n_patches=args.n_patches,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            out_dir=out_dir,
            max_cost_usd=caps[gen_key],
        )


if __name__ == "__main__":
    main()
