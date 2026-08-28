"""Bayesian POMDP agent adapter for the HumanEvalFix benchmark.

Mirrors `_run_dp_loop` from `bayes_agent.py` but uses the HumanEvalFix harness:
    - source file is `solution.py` (one function, not a module)
    - verify/critic run via the `humaneval_fix` subprocess helpers
    - critic set is {syntax, lint, early, mid} (4 critics, not 8)

Reuses DPPlanner, bayes_update, generator_transition unchanged.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from rich.console import Console

from abbo.realworld.agents.bayes_agent import (
    BayesAgentResult,
    DPPlanner,
    bayes_update,
    generator_transition,
)
from abbo.realworld.agents.code_fixer import extract_code_from_response
from abbo.realworld.agents.humaneval_fix import (
    HE_ARM_PROMPTS,
    HE_CRITIC_LIKELIHOODS,
    HE_CRITIC_NAMES,
    get_buggy_source,
    run_critic,
    run_full_test,
    get_full_test_script,
)
from abbo.realworld.agents.llm_provider import LLMConfig, call_llm
from abbo.realworld.agents.simple_agent import AgentCostConfig

console = Console()


def run_bayes_agent_he(
    task_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: AgentCostConfig | None = None,
    max_generators: int = 3,
    max_verifications: int = 2,
    prior: float = 0.1,
    critic_likelihoods: dict | None = None,
    policy_name: str | None = None,
    verbose: bool = True,
) -> BayesAgentResult:
    """Run the Bellman-DP Bayesian agent on one HumanEvalFix task.

    Args:
        task_id: e.g. "Python/0"
        critic_likelihoods: override for HE_CRITIC_LIKELIHOODS (e.g. calibrated).
        policy_name: result label, e.g. "bayes_he_handtuned" vs "bayes_he_calibrated".
    """
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = AgentCostConfig()

    lk_table = critic_likelihoods if critic_likelihoods is not None else HE_CRITIC_LIKELIHOODS

    buggy_source = get_buggy_source(task_id)
    result = BayesAgentResult(bug_id=task_id)
    if policy_name:
        result.policy_name = policy_name
    belief = prior
    result.belief_timeline.append(belief)
    start_time = time.perf_counter()

    planner = DPPlanner(
        cost_config,
        max_generators=max_generators,
        max_verifications=max_verifications,
        critic_likelihoods=lk_table,
    )
    planner.solve()
    if verbose:
        console.print(
            f"  [dim]DP solved: {planner.cache_size} states, "
            f"initial belief={belief:.3f}[/dim]"
        )

    gen_left = max_generators
    ver_left = max_verifications
    crit_used: frozenset[str] = frozenset()
    step = 0
    max_steps = 20

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "solution.py"
        src_file.write_text(buggy_source)
        current_source = buggy_source

        while step < max_steps:
            action, value = planner.choose_action(
                belief, gen_left, crit_used, ver_left
            )
            belief_before = belief

            if verbose:
                console.print(
                    f"  Step {step}: b={belief:.3f} -> "
                    f"[yellow]{action}[/yellow] (V={value:.1f})"
                )

            # ---- BAIL OUT ----
            if action == "bail_out":
                result.action_log.append({
                    "step": step, "action": "bail_out",
                    "belief_before": belief, "belief_after": belief,
                    "value": value, "cost": 0,
                })
                if verbose:
                    console.print(f"  [dim]BAIL OUT: V={value:.1f}[/dim]")
                break

            # ---- VERIFY ----
            elif action == "verify":
                all_pass, detail = run_full_test(workdir, task_id)
                result.test_runs += 1
                result.total_cost += cost_config.c_full_test
                ver_left -= 1

                result.action_log.append({
                    "step": step, "action": "verify",
                    "belief_before": belief, "belief_after": belief,
                    "result": "pass" if all_pass else "fail",
                    "detail": detail[:200],
                    "value": value, "cost": cost_config.c_full_test,
                })
                step += 1
                result.belief_timeline.append(belief)

                if all_pass:
                    result.fixed = True
                    result.reward = cost_config.reward
                    if verbose:
                        console.print("  [green]VERIFIED: all asserts pass![/green]")
                    break
                else:
                    belief = 0.05
                    if verbose:
                        console.print(f"  [red]VERIFY FAILED[/red]")
                    result.belief_timeline.append(belief)

            # ---- GENERATE ----
            elif action.startswith("generate:"):
                arm = action.split(":", 1)[1]

                # Get failing test output for the prompt
                _, test_output = run_full_test(workdir, task_id)

                prompt = HE_ARM_PROMPTS[arm].format(
                    source_code=current_source,
                    test_output=test_output,
                    test_code=get_full_test_script(task_id),
                )
                if verbose:
                    console.print(f"    Calling LLM ({llm_config.model})...")
                try:
                    llm_resp = call_llm(prompt, llm_config)
                except Exception as e:
                    # Abort gracefully if the endpoint times out
                    if verbose:
                        console.print(f"  [red]LLM error: {type(e).__name__}: {e}[/red]")
                    result.action_log.append({
                        "step": step, "action": action, "error": str(e),
                        "belief_before": belief_before, "belief_after": belief,
                    })
                    break
                fixed_code = extract_code_from_response(llm_resp.text)
                src_file.write_text(fixed_code)
                current_source = fixed_code
                result.llm_calls += 1
                result.total_cost += cost_config.c_llm_call
                gen_left -= 1

                belief = generator_transition(belief, arm)

                result.action_log.append({
                    "step": step, "action": action,
                    "belief_before": belief_before, "belief_after": belief,
                    "value": value, "cost": cost_config.c_llm_call,
                })
                step += 1
                result.belief_timeline.append(belief)

            # ---- CRITIC ----
            elif action.startswith("critic:"):
                critic_name = action.split(":", 1)[1]
                if critic_name not in HE_CRITIC_NAMES:
                    raise ValueError(f"Unknown critic: {critic_name}")

                passed, detail = run_critic(workdir, critic_name, task_id)
                result.critic_runs += 1
                result.total_cost += cost_config.c_critic_test
                crit_used = crit_used | frozenset([critic_name])

                belief = bayes_update(
                    belief, critic_name, passed, likelihoods=lk_table,
                )

                if verbose:
                    status = "[green]pass[/green]" if passed else "[red]fail[/red]"
                    console.print(f"    {status} -> b={belief:.3f}")

                result.action_log.append({
                    "step": step, "action": action,
                    "belief_before": belief_before, "belief_after": belief,
                    "passed": passed, "detail": detail[:120],
                    "value": value, "cost": cost_config.c_critic_test,
                })
                step += 1
                result.belief_timeline.append(belief)

            else:
                raise ValueError(f"Unknown action: {action}")

    result.steps = step
    result.wall_clock_seconds = time.perf_counter() - start_time
    result.utility = result.reward - result.total_cost
    return result
