#!/usr/bin/env python3
"""Run live GrFt/DPFt controllers on orchestration benchmarks.

This is the live counterpart to replay policy comparison.  It reuses existing
benchmark code for generation prompts, cheap critics, and verifiers, while the
controller loop uses fitted likelihoods from ``<calibration-dir>/<gen>/
likelihood_tables.json``.

Policy name mapping:
  greedy_fitted -> GrFt
  dp_fitted     -> DPFt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from cost_tracker import CostTracker  # noqa: E402
from fitted_live.common import CRITIC_FIELDS, Candidate, load_jsonl_keys, safe_stem  # noqa: E402
from fitted_live.function_adapters import make_function_adapter  # noqa: E402
from fitted_live.swe_adapter import make_swe_adapter  # noqa: E402
from lcb_calibrate import GENERATORS, _make_client, cost_for_call  # noqa: E402


BENCH_DEFAULTS = {
    "lcb_hard": {"n": 102, "calibration": "data/lcb_calibration_v2"},
    "lcb_medium": {"n": 207, "calibration": "data/lcb_calibration_medium"},
    "lcb_med": {"n": 207, "calibration": "data/lcb_calibration_medium"},
    "lcb_easy": {"n": 135, "calibration": "data/lcb_calibration_easy"},
    "mbpp": {"n": 378, "calibration": "data/mbpp_calibration"},
    "humaneval": {"n": 164, "calibration": "data/humaneval_calibration"},
    "swebench_lite": {"n": 300, "calibration": "data/swebench_lite"},
    "swebench_verified": {"n": 500, "calibration": "data/swebench_verified"},
}

POLICIES = ("greedy_fitted", "dp_fitted")


def load_controller_classes():
    try:
        from lcb_compare import BayesianController, GreedyController
        from run_baseline_vs_controller import CostModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency {exc.name!r}. Install the experiment "
            "runtime first, e.g. numpy/scipy plus the benchmark packages."
        ) from exc
    return BayesianController, GreedyController, CostModel


def canonical_generator_arg(raw: str) -> str:
    key = raw.strip()
    if not key:
        raise SystemExit("empty generator key")
    if key in GENERATORS:
        return key
    alt = key.replace("-", "_")
    if alt in GENERATORS:
        return alt
    for gen_key, (model_id, _label, _base_url) in GENERATORS.items():
        if key == model_id:
            return gen_key
    raise SystemExit(f"unknown generator {raw!r}; known: {', '.join(sorted(GENERATORS))}")


@dataclass
class LiveResult:
    benchmark: str
    generator: str
    policy: str
    instance_id: str
    fixed: bool
    reward: float
    decision_cost: float
    utility: float
    api_cost_usd: float
    n_llm_calls: int
    n_critic_runs: int
    n_full_tests: int
    final_action: str
    wall_clock: float
    actions: list[dict[str, Any]]
    prior_Y1: float
    kernel_source: str


def _load_env_chain() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    candidates = [ROOT / ".env", PROJECT_ROOT / ".env"]
    cur = ROOT.parent
    for _ in range(6):
        candidates.append(cur / ".env")
        if cur.parent == cur:
            break
        cur = cur.parent
    for env_path in candidates:
        if env_path.exists() and env_path.stat().st_size > 0:
            load_dotenv(env_path, override=False)


def parse_cap_map(raw: str, generators: list[str]) -> dict[str, float]:
    if "=" not in raw:
        flat = float(raw)
        return {g: flat for g in generators}
    out: dict[str, float] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        k, v = pair.split("=", 1)
        out[canonical_generator_arg(k)] = float(v)
    return {g: out.get(g, 10.0) for g in generators}


def parse_kernel_file_map(raw: str | None) -> dict[str, Path]:
    if not raw:
        return {}
    out: dict[str, Path] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise SystemExit("--kernel-file must be gen=/path/to/transition_kernel.json[,gen=...]")
        k, v = pair.split("=", 1)
        out[canonical_generator_arg(k)] = Path(v.strip()).expanduser().resolve()
    return out


def default_calibration_dir(benchmark: str) -> Path:
    b = "lcb_medium" if benchmark == "lcb_med" else benchmark
    try:
        return ROOT / BENCH_DEFAULTS[b]["calibration"]
    except KeyError as exc:
        raise SystemExit(f"unknown benchmark {benchmark!r}") from exc


def default_n(benchmark: str) -> int:
    b = "lcb_medium" if benchmark == "lcb_med" else benchmark
    return int(BENCH_DEFAULTS[b]["n"])


def load_likelihoods(calibration_dir: Path, gen: str) -> dict:
    path = calibration_dir / gen / "likelihood_tables.json"
    if not path.exists():
        raise FileNotFoundError(f"missing fitted likelihoods: {path}")
    data = json.loads(path.read_text())
    likes = data.get("critic_likelihoods") or {}
    # Drop critics with incomplete tables. This matters for SWE cells where L2
    # is not part of the fitted action space.
    cleaned = {}
    for name, row in likes.items():
        if row.get("P_pass_given_Y1") is None or row.get("P_pass_given_Y0") is None:
            continue
        cleaned[name] = row
    data["critic_likelihoods"] = cleaned
    return data


def filter_likelihoods_for_benchmark(likes: dict, benchmark: str) -> dict:
    """Keep only critics that the live adapter can actually execute."""
    out = dict(likes)
    critics = dict(likes.get("critic_likelihoods") or {})
    if benchmark.startswith("swebench"):
        critics.pop("L2_public_tests", None)
    out["critic_likelihoods"] = critics
    return out


def preflight_likelihoods(calibration_dir: Path, generators: list[str]) -> None:
    missing = [
        calibration_dir / gen / "likelihood_tables.json"
        for gen in generators
        if not (calibration_dir / gen / "likelihood_tables.json").exists()
    ]
    if not missing:
        return
    lines = "\n".join(f"  - {p}" for p in missing)
    raise SystemExit(
        "Missing fitted likelihood tables. Run the benchmark calibration first, "
        "then rerun live GrFt/DPFt.\n"
        f"Expected files:\n{lines}"
    )


def load_kernel(
    prior: float,
    kernel_dir: Path | None,
    kernel_file: Path | None,
    gen: str,
) -> tuple[dict, str]:
    path = kernel_file
    if path is None and kernel_dir is not None:
        path = kernel_dir / gen / "transition_kernel.json"
    if path is not None:
        if path.exists():
            full = json.loads(path.read_text())
            if "kernel_all" in full:
                return {"kernel_all": full["kernel_all"]}, f"measured:{path}"
            return {"kernel_all": full}, f"measured:{path}"
    return {
        "kernel_all": {
            "P_fix_given_broken": prior,
            "P_break_given_correct": 1 - prior,
        }
    }, "iid_synth"


def bayes_update(belief: float, field: str, observed_pass: bool, likes: dict) -> float:
    row = likes["critic_likelihoods"][field]
    if observed_pass:
        num = row["P_pass_given_Y1"] * belief
        den = num + row["P_pass_given_Y0"] * (1 - belief)
    else:
        num = (1 - row["P_pass_given_Y1"]) * belief
        den = num + (1 - row["P_pass_given_Y0"]) * (1 - belief)
    return belief if den <= 1e-12 else (num / den)


def fallback_action(
    belief: float,
    prior: float,
    cost: CostModel,
    can_generate: bool,
) -> str:
    q_verify = cost.reward * belief - cost.c_ver
    q_generate = -math.inf
    if can_generate:
        q_generate = -cost.c_gen + cost.reward * prior - cost.c_ver
    if q_verify >= 0 and q_verify >= q_generate:
        return "verify"
    if q_generate > 0:
        return "generate"
    return "give_up"


def make_adapter(args, output_dir: Path):
    benchmark = "lcb_medium" if args.benchmark == "lcb_med" else args.benchmark
    if benchmark in {"lcb_hard", "lcb_medium", "lcb_easy", "mbpp", "humaneval"}:
        return make_function_adapter(
            benchmark=benchmark,
            n_instances=args.n_instances,
            seed=args.seed,
            lcb_version=args.lcb_version,
            plus_input_cap=args.plus_input_cap,
            lcb_private_test_cap=args.lcb_private_test_cap,
        )
    if benchmark in {"swebench_lite", "swebench_verified"}:
        return make_swe_adapter(
            benchmark=benchmark,
            n_instances=args.n_instances,
            seed=args.seed,
            output_dir=output_dir,
            harness_workers=args.swe_harness_workers,
        )
    raise SystemExit(f"unknown benchmark: {args.benchmark}")


def generate_candidate(
    adapter,
    instance: dict,
    previous: Candidate | None,
    actions: list[dict[str, Any]],
    gen_client,
    model_id: str,
    gen_key: str,
    policy: str,
    llm_call_idx: int,
    tracker: CostTracker,
    raw_dir: Path,
    temperature: float,
) -> tuple[Candidate | None, float, int, int]:
    inst_id = adapter.instance_id(instance)
    if tracker.capped:
        tracker.note_skipped()
        return None, 0.0, 0, 0
    prompt = adapter.build_prompt(instance, previous, actions)
    resp = gen_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=4000,
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    api_cost = cost_for_call(model_id, prompt_tokens, completion_tokens)
    tracker.record(
        api_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        instance_id=inst_id,
        patch_id=llm_call_idx,
        extra={"kind": "generate", "policy": policy},
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{safe_stem(inst_id)}__{policy}__g{llm_call_idx}.txt"
    raw_path.write_text(text)
    candidate = adapter.extract_candidate(instance, text)
    return candidate, api_cost, prompt_tokens, completion_tokens


def run_one(
    adapter,
    instance: dict,
    benchmark: str,
    gen_key: str,
    policy: str,
    gen_client,
    reviewer_client,
    model_id: str,
    likes: dict,
    kernel: dict,
    kernel_source: str,
    cost: CostModel,
    tracker: CostTracker,
    run_dir: Path,
    max_generations: int,
    max_actions: int,
    temperature: float,
) -> LiveResult:
    start = time.perf_counter()
    inst_id = adapter.instance_id(instance)
    prior = float(likes.get("prior_Y1", 0.5))
    raw_dir = run_dir / gen_key / "raw_responses"
    BayesianController, GreedyController, _CostModel = load_controller_classes()

    if policy == "dp_fitted":
        controller = BayesianController(prior, likes, kernel, cost, horizon=max_generations)
    elif policy == "greedy_fitted":
        controller = GreedyController(prior, likes, cost)
    else:
        raise ValueError(f"unknown policy: {policy}")

    actions: list[dict[str, Any]] = []
    candidate: Candidate | None = None
    belief = prior
    api_cost_usd = 0.0
    decision_cost = 0.0
    reward = 0.0
    fixed = False
    final_action = "exhausted"
    n_llm_calls = 0
    n_critic_runs = 0
    n_full_tests = 0
    used_critics: set[str] = set()

    for action_idx in range(max_actions):
        can_generate = n_llm_calls < max_generations

        if candidate is None:
            if not can_generate:
                final_action = "give_up"
                break
            candidate, c_usd, p_tok, c_tok = generate_candidate(
                adapter, instance, None, actions, gen_client, model_id, gen_key,
                policy, n_llm_calls, tracker, raw_dir, temperature,
            )
            if candidate is None:
                final_action = "api_cost_cap"
                break
            n_llm_calls += 1
            api_cost_usd += c_usd
            decision_cost += cost.c_gen
            belief = prior
            used_critics = set()
            actions.append({
                "step": action_idx,
                "action": "generate",
                "belief": belief,
                "api_cost_usd": c_usd,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "candidate_chars": len(candidate.payload),
                **candidate.metadata,
            })
            continue

        step = min(max(n_llm_calls - 1, 0), max_generations - 1)
        action = controller.select_action(belief, step)
        if action in {"L0", "L2", "L3"}:
            field = CRITIC_FIELDS[action]
            if field not in likes.get("critic_likelihoods", {}) or action in used_critics:
                action = fallback_action(belief, prior, cost, can_generate)
        elif action == "generate" and not can_generate:
            action = fallback_action(belief, prior, cost, False)

        if action in {"give_up", "bail_out"}:
            final_action = "give_up"
            actions.append({"step": action_idx, "action": "give_up", "belief": belief})
            break

        if action == "verify":
            run_id = safe_stem(f"{benchmark}__{gen_key}__{policy}__{inst_id}__v{n_full_tests}", 180)
            vr = adapter.verify(instance, candidate, run_id)
            n_full_tests += 1
            decision_cost += cost.c_ver
            fixed = bool(vr.passed)
            reward = cost.reward if fixed else 0.0
            final_action = "verify_pass" if fixed else "verify_fail"
            actions.append({
                "step": action_idx,
                "action": "verify",
                "belief": belief,
                "passed": bool(vr.passed),
                "detail": vr.detail,
            })
            break

        if action == "generate":
            prev = candidate
            if policy == "dp_fitted":
                belief = controller._generate_next_belief(belief)
            else:
                belief = prior
            candidate, c_usd, p_tok, c_tok = generate_candidate(
                adapter, instance, prev, actions, gen_client, model_id, gen_key,
                policy, n_llm_calls, tracker, raw_dir, temperature,
            )
            if candidate is None:
                final_action = "api_cost_cap"
                break
            n_llm_calls += 1
            api_cost_usd += c_usd
            decision_cost += cost.c_gen
            used_critics = set()
            actions.append({
                "step": action_idx,
                "action": "generate",
                "belief": belief,
                "api_cost_usd": c_usd,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "candidate_chars": len(candidate.payload),
                **candidate.metadata,
            })
            continue

        if action in {"L0", "L2", "L3"}:
            obs = adapter.run_critic(action, instance, candidate, reviewer_client)
            if action == "L0":
                decision_cost += cost.c_L0
            elif action == "L2":
                decision_cost += cost.c_L2
            else:
                decision_cost += cost.c_L3
            api_cost_usd += obs.api_cost_usd
            if obs.api_cost_usd:
                tracker.record(
                    obs.api_cost_usd,
                    prompt_tokens=obs.prompt_tokens,
                    completion_tokens=obs.completion_tokens,
                    instance_id=inst_id,
                    patch_id=n_llm_calls - 1,
                    extra={"kind": action, "policy": policy},
                )
            n_critic_runs += 1
            used_critics.add(action)
            if obs.passed is not None:
                belief = bayes_update(belief, CRITIC_FIELDS[action], bool(obs.passed), likes)
            actions.append({
                "step": action_idx,
                "action": action,
                "passed": obs.passed,
                "belief": belief,
                "detail": obs.detail,
                "api_cost_usd": obs.api_cost_usd,
            })
            continue

        final_action = f"unknown_action:{action}"
        break

    utility = reward - decision_cost
    return LiveResult(
        benchmark=benchmark,
        generator=gen_key,
        policy=policy,
        instance_id=inst_id,
        fixed=fixed,
        reward=reward,
        decision_cost=decision_cost,
        utility=utility,
        api_cost_usd=api_cost_usd,
        n_llm_calls=n_llm_calls,
        n_critic_runs=n_critic_runs,
        n_full_tests=n_full_tests,
        final_action=final_action,
        wall_clock=time.perf_counter() - start,
        actions=actions,
        prior_Y1=prior,
        kernel_source=kernel_source,
    )


def aggregate(results: list[dict[str, Any]], policies: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_records": len(results), "policies": {}}
    for policy in policies:
        rows = [r for r in results if r.get("policy") == policy]
        if not rows:
            continue
        out["policies"][policy] = {
            "n": len(rows),
            "fix_rate": sum(1 for r in rows if r.get("fixed")) / len(rows),
            "mean_utility": sum(float(r.get("utility", 0.0)) for r in rows) / len(rows),
            "mean_decision_cost": sum(float(r.get("decision_cost", 0.0)) for r in rows) / len(rows),
            "mean_api_cost_usd": sum(float(r.get("api_cost_usd", 0.0)) for r in rows) / len(rows),
            "mean_llm_calls": sum(int(r.get("n_llm_calls", 0)) for r in rows) / len(rows),
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, choices=sorted(BENCH_DEFAULTS))
    p.add_argument("--generators", required=True)
    p.add_argument("--policies", default="greedy_fitted,dp_fitted")
    p.add_argument("--calibration-dir", type=Path, default=None)
    p.add_argument("--kernel-dir", type=Path, default=None,
                   help="Optional measured-kernel dir with <gen>/transition_kernel.json")
    p.add_argument("--kernel-file", default=None,
                   help="Optional gen=path[,gen=path] transition-kernel mapping")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data" / "fitted_live")
    p.add_argument("--n-instances", type=int, default=0,
                   help="0 means benchmark default/paper draw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lcb-version", default="all", choices=["v1", "all"])
    p.add_argument("--lcb-private-test-cap", type=int, default=12,
                   help="LCB private verifier cap; 0 runs all private tests")
    p.add_argument("--plus-input-cap", type=int, default=200)
    p.add_argument("--max-generations", type=int, default=3)
    p.add_argument("--max-actions", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-api-cost-usd-per-model", default="20.0")
    p.add_argument("--swe-harness-workers", type=int, default=1)
    p.add_argument("--c-gen", type=float, default=5.0)
    p.add_argument("--c-l0", type=float, default=1.0)
    p.add_argument("--c-l2", type=float, default=2.0)
    p.add_argument("--c-l3", type=float, default=5.0)
    p.add_argument("--c-ver", type=float, default=30.0)
    p.add_argument("--reward", type=float, default=100.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _load_env_chain()

    benchmark = "lcb_medium" if args.benchmark == "lcb_med" else args.benchmark
    generators = []
    for raw_gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        gen = canonical_generator_arg(raw_gen)
        if gen not in generators:
            generators.append(gen)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for policy in policies:
        if policy not in POLICIES:
            raise SystemExit(f"unknown policy {policy!r}; allowed: {', '.join(POLICIES)}")
    _BayesianController, _GreedyController, CostModel = load_controller_classes()
    if args.n_instances <= 0:
        args.n_instances = default_n(benchmark)
    calibration_dir = (args.calibration_dir or default_calibration_dir(benchmark)).resolve()
    kernel_files = parse_kernel_file_map(args.kernel_file)
    preflight_likelihoods(calibration_dir, generators)
    output_dir = (args.output_dir / benchmark).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = make_adapter(args, output_dir)
    instances = adapter.load_instances()
    sample_path = output_dir / "sample.json"
    sample_path.write_text(json.dumps([
        {"instance_id": adapter.instance_id(inst)} for inst in instances
    ], indent=2))

    cost = CostModel(
        c_gen=args.c_gen,
        c_L0=args.c_l0,
        c_L2=args.c_l2,
        c_L3=args.c_l3,
        c_ver=args.c_ver,
        reward=args.reward,
    )
    caps = parse_cap_map(args.max_api_cost_usd_per_model, generators)
    all_results: list[dict[str, Any]] = []

    for gen in generators:
        gen_dir = output_dir / gen
        gen_dir.mkdir(parents=True, exist_ok=True)
        results_path = gen_dir / "fitted_live_results.jsonl"
        done = load_jsonl_keys(results_path)
        likes = filter_likelihoods_for_benchmark(load_likelihoods(calibration_dir, gen), benchmark)
        prior = float(likes.get("prior_Y1", 0.5))
        kernel, kernel_source = load_kernel(
            prior,
            args.kernel_dir.resolve() if args.kernel_dir else None,
            kernel_files.get(gen),
            gen,
        )
        tracker = CostTracker(
            name=gen,
            cap_usd=caps[gen],
            log_path=gen_dir / "fitted_live_cost_log.jsonl",
        )
        model_id = GENERATORS[gen][0]
        gen_client = _make_client(gen)
        reviewer_client = _make_client(None)

        existing_rows = []
        if results_path.exists():
            for line in results_path.read_text().splitlines():
                if line.strip():
                    try:
                        existing_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        all_results.extend(existing_rows)

        with open(results_path, "a", buffering=1) as out_fp:
            for i, inst in enumerate(instances, 1):
                inst_id = adapter.instance_id(inst)
                for policy in policies:
                    if (inst_id, policy) in done:
                        continue
                    if tracker.capped:
                        print(f"[{gen}] API cost cap reached; stopping generator")
                        break
                    print(f"[{benchmark}/{gen}/{policy}] {i}/{len(instances)} {inst_id}", flush=True)
                    try:
                        result = run_one(
                            adapter=adapter,
                            instance=inst,
                            benchmark=benchmark,
                            gen_key=gen,
                            policy=policy,
                            gen_client=gen_client,
                            reviewer_client=reviewer_client,
                            model_id=model_id,
                            likes=likes,
                            kernel=kernel,
                            kernel_source=kernel_source,
                            cost=cost,
                            tracker=tracker,
                            run_dir=output_dir,
                            max_generations=args.max_generations,
                            max_actions=args.max_actions,
                            temperature=args.temperature,
                        )
                    except Exception as exc:
                        result = LiveResult(
                            benchmark=benchmark,
                            generator=gen,
                            policy=policy,
                            instance_id=inst_id,
                            fixed=False,
                            reward=0.0,
                            decision_cost=0.0,
                            utility=0.0,
                            api_cost_usd=0.0,
                            n_llm_calls=0,
                            n_critic_runs=0,
                            n_full_tests=0,
                            final_action=f"exception:{type(exc).__name__}",
                            wall_clock=0.0,
                            actions=[{"error": str(exc)}],
                            prior_Y1=prior,
                            kernel_source=kernel_source,
                        )
                    row = asdict(result)
                    out_fp.write(json.dumps(row) + "\n")
                    out_fp.flush()
                    os.fsync(out_fp.fileno())
                    all_results.append(row)
                    done.add((inst_id, policy))
                    status = "OK" if row["fixed"] else "no"
                    print(
                        f"  -> fixed={status} utility={row['utility']:.2f} "
                        f"api=${row['api_cost_usd']:.4f} final={row['final_action']}",
                        flush=True,
                    )
                if tracker.capped:
                    break

        gen_rows = [r for r in all_results if r.get("generator") == gen]
        (gen_dir / "fitted_live_summary.json").write_text(
            json.dumps(aggregate(gen_rows, policies), indent=2)
        )
        (gen_dir / "fitted_live_cost_summary.json").write_text(
            json.dumps(tracker.snapshot(), indent=2)
        )

    (output_dir / "fitted_live_summary.json").write_text(
        json.dumps(aggregate(all_results, policies), indent=2)
    )
    print(f"Saved live fitted results under {output_dir}")


if __name__ == "__main__":
    main()
