#!/usr/bin/env python
"""Re-run CodeContests with measured-kernel DP variants.

Adds two new variants to the existing comparison:
    dp_hand_kernel   = DP + hand-tuned theta + measured kernel
    dp_fitted_kernel = DP + fitted theta     + measured kernel

Same 82 held-out tasks as run_codecontests_full.py. Resume-safe.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abbo.realworld.agents.bayes_agent import DPPlanner
from abbo.realworld.agents.code_contests import (
    CC_CRITIC_LIKELIHOODS, list_task_ids,
)
from abbo.realworld.agents.llm_provider import build_llm_config_from_env
from abbo.realworld.agents.simple_agent import AgentCostConfig

# Import functions from the existing runner
sys.path.insert(0, str(ROOT / "scripts"))
from run_codecontests_full import (
    FITTED_THETA, SPLIT_SEED, N_TRAIN, PRIOR,
    MAX_GENERATORS, MAX_VERIFICATIONS, LLM_MODEL,
    run_dp, serialize, load_existing, save_progress,
)

# Measured kernel from sim_results/transition_kernels.json
MEASURED_KERNEL_CC = {
    "p_fix_broken":   0.071,   # measured from 168 transitions in our prior CC run
    "p_break_correct": 0.05,   # literature prior (matches supervisor deck's 0.06)
}

RESULTS_PATH = ROOT / "sim_results" / "codecontests_kernel_endtoend.json"

VARIANTS = ("dp_hand_kernel", "dp_fitted_kernel")


def main():
    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    test_ids = all_ids[N_TRAIN:]
    print(f"Held-out: {len(test_ids)} tasks")
    print(f"Measured kernel: {MEASURED_KERNEL_CC}")

    state = load_existing(RESULTS_PATH)
    results = state.setdefault("results", {})

    costs = AgentCostConfig()
    # Two new planners: same theta as before, but with the measured kernel.
    dp_hand_kernel = DPPlanner(
        costs, MAX_GENERATORS, MAX_VERIFICATIONS,
        critic_likelihoods=CC_CRITIC_LIKELIHOODS,
        transition_kernel=MEASURED_KERNEL_CC,
    ); dp_hand_kernel.solve()
    dp_fitted_kernel = DPPlanner(
        costs, MAX_GENERATORS, MAX_VERIFICATIONS,
        critic_likelihoods=FITTED_THETA,
        transition_kernel=MEASURED_KERNEL_CC,
    ); dp_fitted_kernel.solve()

    llm_cfg = build_llm_config_from_env(
        default_provider="openrouter",
        default_model=LLM_MODEL,
        default_base_url="https://openrouter.ai/api",
        default_temperature=0.1,
        default_max_tokens=8192,  # was 2048; CC patches with CoT got truncated
        default_timeout=120,
    )
    print(f"LLM provider={llm_cfg.provider} model={llm_cfg.model} base_url={llm_cfg.base_url}")

    total = len(test_ids) * len(VARIANTS)
    done = sum(1 for tid in test_ids for v in VARIANTS if results.get(f"{tid}|{v}"))
    print(f"\nResume: {done}/{total} pairs already done.\n")

    started = time.time()
    for i, tid in enumerate(test_ids):
        elapsed = time.time() - started
        rate = (i + 1) / max(0.001, elapsed)
        eta_min = (len(test_ids) - i - 1) / max(0.0001, rate) / 60
        print(f"\n[{i+1}/{len(test_ids)}] task={tid}  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta_min:.1f}min")
        for v in VARIANTS:
            key = f"{tid}|{v}"
            if results.get(key):
                continue
            try:
                if v == "dp_hand_kernel":
                    r = run_dp(tid, CC_CRITIC_LIKELIHOODS, "hand_kernel",
                               llm_cfg, costs, dp_hand_kernel,
                               MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR)
                elif v == "dp_fitted_kernel":
                    r = run_dp(tid, FITTED_THETA, "fitted_kernel",
                               llm_cfg, costs, dp_fitted_kernel,
                               MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR)
                else:
                    continue
            except Exception as e:
                print(f"  [{v}] EXCEPTION: {e}")
                continue
            results[key] = serialize(r)
            tag = "OK" if r.fixed else "no"
            print(f"  {v:<22} fix={tag}  cost={r.total_cost:5.1f}  "
                  f"llm={r.n_llm_calls}  crit={r.n_critic_runs}  "
                  f"toks={r.completion_tokens}  wc={r.wall_clock:.1f}s  "
                  f"final={r.final_action}")
            save_progress(RESULTS_PATH, state)

    state["llm_model"] = LLM_MODEL
    state["measured_kernel"] = MEASURED_KERNEL_CC
    save_progress(RESULTS_PATH, state)

    # Final aggregate (combine with prior runs from codecontests_full_endtoend.json)
    print("\n=== Final aggregate (kernel variants) ===")
    R = 100
    from collections import defaultdict
    by_v = defaultdict(list)
    for rec in results.values():
        by_v[rec["variant"]].append(rec)
    for v in VARIANTS:
        rs = by_v.get(v, [])
        if not rs: continue
        n = len(rs)
        fix = sum(1 for r in rs if r["fixed"]) / n * 100
        c = sum(r["total_cost"] for r in rs) / n
        u = sum((R if r["fixed"] else 0) - r["total_cost"] for r in rs) / n
        print(f"{v:<22} n={n:>3}  fix={fix:>5.1f}%  cost={c:>6.2f}  Ū_π={u:>+8.2f}")
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
