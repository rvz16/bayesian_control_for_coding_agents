"""Counter-factual simulator for the Bayesian POMDP agent.

Replays held-out patches through the agent decision loop using *pre-collected*
critic outcomes (no LLM, no Docker, no subprocess). Lets us isolate the
behavioural effect of switching from hand-tuned to fitted theta_hat without
re-running the full benchmark.

Inputs per patch:
    - critic_outcomes: {critic_name -> bool}   (passed/failed, already measured)
    - Y: bool                                  (ground-truth correctness)

Inputs per simulation:
    - theta: critic likelihood table (hand-tuned, fitted, or flat)
    - agent_kind: "dp" | "greedy"
    - costs: AgentCostConfig

Outputs per patch:
    - actions taken (with belief trajectory)
    - final_decision: "verify_pass" | "verify_fail" | "bail"
    - fixed: bool       (verify_pass on Y=1)
    - total_cost: float
    - n_critics_called

The 'generate' action is intentionally disabled (max_generators=0) — these
patches are fixed candidates, not LLM-produced ones. So the action set
collapses to {critic_c, verify, bail}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update
from abbo.realworld.agents.simple_agent import AgentCostConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatchOutcomes:
    """One held-out patch with its critic outcomes and ground-truth Y."""
    bug_id: str
    arm: str                                # "no_fix" / "gold_fix" / etc.
    critic_outcomes: dict[str, bool]        # {critic_name: passed}
    Y: bool                                 # ground truth: was the patch correct?


@dataclass
class SimAction:
    """One action in the simulated agent run."""
    step: int
    action: str                             # "critic:NAME" | "verify" | "bail"
    belief_before: float
    belief_after: float
    cost: float
    extra: str = ""                         # e.g. "pass" / "fail" for critic outcomes


@dataclass
class SimResult:
    """Outcome of simulating one patch through one agent variant."""
    bug_id: str
    arm: str
    Y: bool
    final_decision: str                     # "verify_pass" | "verify_fail" | "bail"
    fixed: bool                             # True iff (verify_pass and Y=1)
    total_cost: float
    n_critics_called: int
    actions: list[SimAction] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    """Per-(theta, agent_kind) aggregated metrics across all held-out patches."""
    label: str                              # e.g. "dp_fitted"
    n_patches: int
    fix_rate: float                         # # fixed / n
    avg_critics_called: float
    avg_cost: float
    bail_rate: float
    verify_rate: float
    wasted_verify_rate: float               # verify on a Y=0 patch
    avg_utility: float                      # reward * fixed - cost
    total_utility: float


# ---------------------------------------------------------------------------
# DP simulation (re-uses DPPlanner with generate action disabled)
# ---------------------------------------------------------------------------

def simulate_dp(
    patch: PatchOutcomes,
    theta: dict[str, dict[str, float]],
    costs: AgentCostConfig,
    prior: float = 0.5,
    max_verifications: int = 1,
    planner: DPPlanner | None = None,
) -> SimResult:
    """Run the DP-optimal agent on one held-out patch.

    Args:
        patch: pre-collected critic outcomes + ground-truth Y.
        theta: critic likelihood table to plug into DPPlanner.
        costs: AgentCostConfig (c_critic_test, c_full_test, reward).
        prior: starting belief, default 0.5 (uninformed).
        max_verifications: typically 1; >1 allows post-failed-verify recovery.
        planner: optional pre-solved planner to avoid re-solving for each patch.

    Returns:
        SimResult with action log, final decision, fixed flag, cost.
    """
    if planner is None:
        planner = DPPlanner(
            costs=costs,
            max_generators=0,                # no LLM in simulation
            max_verifications=max_verifications,
            critic_likelihoods=theta,
        )
        planner.solve()

    belief = prior
    crit_used: frozenset[str] = frozenset()
    ver_left = max_verifications
    actions: list[SimAction] = []
    total_cost = 0.0
    step = 0
    max_steps = 20  # safety

    while step < max_steps:
        action, _value = planner.choose_action(
            belief=belief,
            gen_left=0,
            crit_used=crit_used,
            ver_left=ver_left,
        )

        belief_before = belief

        if action == "bail_out":
            actions.append(SimAction(
                step=step, action="bail",
                belief_before=belief_before, belief_after=belief,
                cost=0.0,
            ))
            return SimResult(
                bug_id=patch.bug_id, arm=patch.arm, Y=patch.Y,
                final_decision="bail", fixed=False,
                total_cost=total_cost,
                n_critics_called=sum(1 for a in actions if a.action.startswith("critic:")),
                actions=actions,
            )

        if action == "verify":
            total_cost += costs.c_full_test
            ver_left -= 1
            extra = "pass" if patch.Y else "fail"
            actions.append(SimAction(
                step=step, action="verify",
                belief_before=belief_before, belief_after=belief,
                cost=costs.c_full_test, extra=extra,
            ))
            # In our minimal setup max_verifications==1, so episode ends here.
            # If patch is correct, agent collected reward; otherwise just paid.
            return SimResult(
                bug_id=patch.bug_id, arm=patch.arm, Y=patch.Y,
                final_decision="verify_pass" if patch.Y else "verify_fail",
                fixed=patch.Y,
                total_cost=total_cost,
                n_critics_called=sum(1 for a in actions if a.action.startswith("critic:")),
                actions=actions,
            )

        if action.startswith("critic:"):
            critic_name = action.split(":", 1)[1]
            if critic_name not in patch.critic_outcomes:
                raise ValueError(
                    f"Critic {critic_name!r} chosen by DP but not in "
                    f"patch.critic_outcomes (keys={list(patch.critic_outcomes)})"
                )
            passed = patch.critic_outcomes[critic_name]
            belief = bayes_update(belief, critic_name, passed, likelihoods=theta)
            total_cost += costs.c_critic_test
            crit_used = crit_used | frozenset([critic_name])
            actions.append(SimAction(
                step=step, action=f"critic:{critic_name}",
                belief_before=belief_before, belief_after=belief,
                cost=costs.c_critic_test, extra="pass" if passed else "fail",
            ))
            step += 1
            continue

        if action.startswith("generate:"):
            # We disabled generate via max_generators=0 — this should be
            # unreachable. Treat as bail to be safe and break out.
            actions.append(SimAction(
                step=step, action="bail",
                belief_before=belief_before, belief_after=belief,
                cost=0.0, extra=f"unexpected_{action}",
            ))
            return SimResult(
                bug_id=patch.bug_id, arm=patch.arm, Y=patch.Y,
                final_decision="bail", fixed=False,
                total_cost=total_cost,
                n_critics_called=sum(1 for a in actions if a.action.startswith("critic:")),
                actions=actions,
            )

        raise RuntimeError(f"Unknown action from DPPlanner: {action!r}")

    # max_steps hit — treat as bail
    return SimResult(
        bug_id=patch.bug_id, arm=patch.arm, Y=patch.Y,
        final_decision="bail", fixed=False,
        total_cost=total_cost,
        n_critics_called=sum(1 for a in actions if a.action.startswith("critic:")),
        actions=actions,
    )


# ---------------------------------------------------------------------------
# Greedy simulation (one-step lookahead)
# ---------------------------------------------------------------------------

def _q_one_step_critic(
    belief: float,
    critic: str,
    theta: dict[str, dict[str, float]],
    costs: AgentCostConfig,
) -> float:
    """One-step Q value of calling `critic`, then verify-or-bail.

    Greedy treats whatever state results from this critic as terminal —
    it computes the value as max(verify, bail) at each resulting belief.
    """
    lk = theta[critic]
    p_pass = lk["p_pass_y1"] * belief + lk["p_pass_y0"] * (1 - belief)
    b_pass = bayes_update(belief, critic, True, likelihoods=theta)
    b_fail = bayes_update(belief, critic, False, likelihoods=theta)

    # After observing outcome, Greedy's terminal is max(verify, bail).
    v_pass = max(0.0, -costs.c_full_test + b_pass * costs.reward)
    v_fail = max(0.0, -costs.c_full_test + b_fail * costs.reward)

    return -costs.c_critic_test + p_pass * v_pass + (1 - p_pass) * v_fail


def simulate_greedy(
    patch: PatchOutcomes,
    theta: dict[str, dict[str, float]],
    costs: AgentCostConfig,
    prior: float = 0.5,
) -> SimResult:
    """Run the Bayesian Greedy agent on one held-out patch.

    Decision rule at each step:
        Q_bail   = 0
        Q_verify = -c_verify + b * R
        Q_critic = -c_critic + E[ max(verify, bail) at b' ]    (per critic)

    Argmax. Stop on bail or verify.
    """
    belief = prior
    crit_used: set[str] = set()
    actions: list[SimAction] = []
    total_cost = 0.0
    step = 0
    max_steps = 20

    while step < max_steps:
        belief_before = belief

        Q_bail = 0.0
        Q_verify = -costs.c_full_test + belief * costs.reward

        Q_critics: dict[str, float] = {}
        for c in theta.keys():
            if c in crit_used:
                continue
            if c not in patch.critic_outcomes:
                # Critic in theta but no outcome for it on this patch — skip.
                continue
            Q_critics[c] = _q_one_step_critic(belief, c, theta, costs)

        best_critic, best_critic_q = (
            max(Q_critics.items(), key=lambda x: x[1])
            if Q_critics else (None, -math.inf)
        )

        # Pick argmax over {bail, verify, best critic}.
        choices: list[tuple[str, float]] = [
            ("bail", Q_bail),
            ("verify", Q_verify),
        ]
        if best_critic is not None:
            choices.append((f"critic:{best_critic}", best_critic_q))

        action, _q = max(choices, key=lambda x: x[1])

        if action == "bail":
            actions.append(SimAction(
                step=step, action="bail",
                belief_before=belief_before, belief_after=belief,
                cost=0.0,
            ))
            return SimResult(
                bug_id=patch.bug_id, arm=patch.arm, Y=patch.Y,
                final_decision="bail", fixed=False,
                total_cost=total_cost,
                n_critics_called=len(crit_used),
                actions=actions,
            )

        if action == "verify":
            total_cost += costs.c_full_test
            actions.append(SimAction(
                step=step, action="verify",
                belief_before=belief_before, belief_after=belief,
                cost=costs.c_full_test,
                extra="pass" if patch.Y else "fail",
            ))
            return SimResult(
                bug_id=patch.bug_id, arm=patch.arm, Y=patch.Y,
                final_decision="verify_pass" if patch.Y else "verify_fail",
                fixed=patch.Y,
                total_cost=total_cost,
                n_critics_called=len(crit_used),
                actions=actions,
            )

        # critic action
        critic_name = action.split(":", 1)[1]
        passed = patch.critic_outcomes[critic_name]
        belief = bayes_update(belief, critic_name, passed, likelihoods=theta)
        total_cost += costs.c_critic_test
        crit_used.add(critic_name)
        actions.append(SimAction(
            step=step, action=f"critic:{critic_name}",
            belief_before=belief_before, belief_after=belief,
            cost=costs.c_critic_test, extra="pass" if passed else "fail",
        ))
        step += 1

    # All critics used and neither verify nor bail picked — degenerate, bail.
    actions.append(SimAction(
        step=step, action="bail",
        belief_before=belief, belief_after=belief, cost=0.0,
        extra="all_critics_exhausted",
    ))
    return SimResult(
        bug_id=patch.bug_id, arm=patch.arm, Y=patch.Y,
        final_decision="bail", fixed=False,
        total_cost=total_cost,
        n_critics_called=len(crit_used),
        actions=actions,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    results: list[SimResult],
    label: str,
    costs: AgentCostConfig,
) -> AggregateMetrics:
    """Compute per-variant summary metrics across all held-out patches."""
    n = len(results)
    if n == 0:
        return AggregateMetrics(
            label=label, n_patches=0, fix_rate=0.0, avg_critics_called=0.0,
            avg_cost=0.0, bail_rate=0.0, verify_rate=0.0,
            wasted_verify_rate=0.0, avg_utility=0.0, total_utility=0.0,
        )
    n_fixed = sum(1 for r in results if r.fixed)
    total_cost = sum(r.total_cost for r in results)
    total_critics = sum(r.n_critics_called for r in results)
    n_bail = sum(1 for r in results if r.final_decision == "bail")
    n_verify = sum(1 for r in results if r.final_decision.startswith("verify"))
    n_wasted = sum(1 for r in results if r.final_decision == "verify_fail")
    total_utility = sum(
        (costs.reward if r.fixed else 0.0) - r.total_cost for r in results
    )
    return AggregateMetrics(
        label=label,
        n_patches=n,
        fix_rate=n_fixed / n,
        avg_critics_called=total_critics / n,
        avg_cost=total_cost / n,
        bail_rate=n_bail / n,
        verify_rate=n_verify / n,
        wasted_verify_rate=n_wasted / n,
        avg_utility=total_utility / n,
        total_utility=total_utility,
    )


# ---------------------------------------------------------------------------
# Top-level batch runner
# ---------------------------------------------------------------------------

def run_simulation_grid(
    patches: list[PatchOutcomes],
    theta_tables: dict[str, dict[str, dict[str, float]]],
    costs: AgentCostConfig | None = None,
    prior: float = 0.5,
    max_verifications: int = 1,
) -> dict[str, AggregateMetrics]:
    """Run all (agent_kind × theta) combinations on the same held-out patches.

    Args:
        patches: list of held-out PatchOutcomes (each has critic outcomes + Y).
        theta_tables: {"hand": <hand_lk>, "fitted": <fitted_lk>, ...}.
        costs: AgentCostConfig; defaults to canonical (c_full_test=5,
            c_critic_test=1, reward=100).
        prior: starting belief.
        max_verifications: typically 1.

    Returns:
        dict mapping "<agent>_<theta_label>" -> AggregateMetrics.
    """
    if costs is None:
        costs = AgentCostConfig()

    results: dict[str, AggregateMetrics] = {}

    for theta_label, theta in theta_tables.items():
        # Pre-solve DP once per theta — same patches reuse the cache.
        dp_planner = DPPlanner(
            costs=costs,
            max_generators=0,
            max_verifications=max_verifications,
            critic_likelihoods=theta,
        )
        dp_planner.solve()

        dp_results: list[SimResult] = []
        greedy_results: list[SimResult] = []
        for p in patches:
            dp_results.append(simulate_dp(
                p, theta, costs, prior=prior,
                max_verifications=max_verifications,
                planner=dp_planner,
            ))
            greedy_results.append(simulate_greedy(
                p, theta, costs, prior=prior,
            ))

        results[f"dp_{theta_label}"] = aggregate(
            dp_results, f"dp_{theta_label}", costs,
        )
        results[f"greedy_{theta_label}"] = aggregate(
            greedy_results, f"greedy_{theta_label}", costs,
        )

    return results


def format_grid(metrics: dict[str, AggregateMetrics]) -> str:
    """Pretty-print the metrics grid as a text table."""
    header = (
        f"{'variant':<22} {'n':>3} {'fix':>6} {'crit/p':>7} {'cost':>7} "
        f"{'bail':>6} {'ver':>6} {'wasted':>7} {'util':>8}"
    )
    lines = [header, "-" * len(header)]
    for label, m in metrics.items():
        lines.append(
            f"{label:<22} {m.n_patches:>3} "
            f"{m.fix_rate:>6.3f} {m.avg_critics_called:>7.2f} "
            f"{m.avg_cost:>7.2f} {m.bail_rate:>6.3f} "
            f"{m.verify_rate:>6.3f} {m.wasted_verify_rate:>7.3f} "
            f"{m.avg_utility:>8.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: convert CalibrationSample list -> PatchOutcomes list
# ---------------------------------------------------------------------------

def patches_from_samples(
    samples,                                # list[CalibrationSample-like]
) -> list[PatchOutcomes]:
    """Group a flat list of (bug_id, arm, critic, passed, Y) samples into one
    PatchOutcomes per (bug_id, arm) — i.e. one held-out patch per arm.

    Compatible with the CalibrationSample dataclass used by HumanEvalFix /
    CodeContests / SWE-bench harnesses (they all share the same field names).
    """
    by_patch: dict[tuple[str, str], dict] = {}
    for s in samples:
        key = (s.bug_id, s.arm)
        if key not in by_patch:
            by_patch[key] = {
                "bug_id": s.bug_id,
                "arm": s.arm,
                "critic_outcomes": {},
                "Y": s.patch_correct,
            }
        by_patch[key]["critic_outcomes"][s.critic] = s.critic_passed
    return [
        PatchOutcomes(
            bug_id=v["bug_id"], arm=v["arm"],
            critic_outcomes=v["critic_outcomes"], Y=v["Y"],
        )
        for v in by_patch.values()
    ]
