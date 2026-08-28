"""Baseline-vs-controller comparison on the calibrated SWE-bench Lite data.

Replays the iter_records trajectory (30 instances × 5 patches each) under
different orchestration policies, all seeing the same patches + critic
outcomes. Apples-to-apples comparison.

Inputs per generator:
  data/spot_check_n50/<gen>/iter_records.jsonl   — (instance, step, Y, L0, L1, L3, diff)
  data/spot_check_n50/<gen>/likelihood_tables.json — P(z|Y) per critic + prior
  data/spot_check_n50/<gen>/transition_kernel.json — P(fix|broken), P(break|correct)

Policies compared:
  - Always verify first patch
  - Best-of-N (regenerate K times, verify last)
  - Threshold(L0) — run L0, verify if PASS, else regen
  - Threshold(L3) — run L3, verify if PASS, else regen
  - Fixed Pipeline — run L0 + L3, verify if both PASS, else regen
  - Bayesian controller — backward induction

Cost model (declared, simple):
  c_gen=10, c_L0=1, c_L3=5, c_ver=30, reward=100

Output: <gen>/policy_comparison.json with utility / pass-rate / cost
per policy.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    c_gen: float = 5.0   # generation API call
    c_L0: float = 1.0    # cheap critic
    c_L2: float = 2.0    # public-tests critic (LCB)
    c_L3: float = 5.0    # LLM review critic
    c_ver: float = 30.0  # full SWE-bench Docker eval
    reward: float = 100.0  # correct patch passes verifier


# ---------------------------------------------------------------------------
# Trajectory data
# ---------------------------------------------------------------------------

def load_trajectories(records_path: Path) -> dict[str, list[dict]]:
    """Group iter_records by instance, sort by step."""
    by_inst: dict[str, list[dict]] = {}
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            by_inst.setdefault(r["instance_id"], []).append(r)
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in by_inst.items()}


# ---------------------------------------------------------------------------
# Bayesian controller (lightweight implementation tailored to our schema)
# ---------------------------------------------------------------------------

class BayesianController:
    """Discretized-belief backward-induction controller.

    State = (belief, step). Actions = {generate, critic_L0, critic_L3,
    verify, give_up}. Forward step rules reflect our data:
      - generate: belief moves via the transition kernel
      - critic L_k: belief Bayes-updated on observed z_k
      - verify: episode ends, reward = 100 * Y, cost = c_ver
      - give_up: episode ends, reward = 0
    """

    def __init__(self, prior: float, like_tables: dict, kernel: dict, cost: CostModel,
                 horizon: int = 5, n_belief: int = 51) -> None:
        self.prior = prior
        self.likes = like_tables["critic_likelihoods"]
        self.kernel = kernel["kernel_all"]
        self.cost = cost
        self.horizon = horizon
        self.beliefs = np.linspace(0.0, 1.0, n_belief)
        # Will fill self.policy[step][belief_idx] -> action_str
        self._build()

    def _bayes_update(self, b: float, critic: str, observed_pass: bool) -> float:
        l = self.likes[critic]
        if observed_pass:
            num = l["P_pass_given_Y1"] * b
            den = num + l["P_pass_given_Y0"] * (1 - b)
        else:
            num = (1 - l["P_pass_given_Y1"]) * b
            den = num + (1 - l["P_pass_given_Y0"]) * (1 - b)
        return num / max(den, 1e-12)

    def _generate_next_belief(self, b: float) -> float:
        p_fix = self.kernel["P_fix_given_broken"]
        p_break = self.kernel["P_break_given_correct"]
        return b * (1 - p_break) + (1 - b) * p_fix

    def _value(self, b: float, step: int, V_next: np.ndarray) -> tuple[float, str]:
        """Return (best Q-value, best action) at (b, step)."""
        cm = self.cost
        # Action: verify (terminal)
        Q_verify = cm.reward * b - cm.c_ver
        # Action: give_up (terminal)
        Q_giveup = 0.0
        # Action: generate (advances to next step with new belief)
        if step + 1 >= self.horizon:
            Q_generate = -cm.c_gen + Q_giveup  # next step would be give-up
        else:
            new_b = self._generate_next_belief(b)
            new_idx = int(np.argmin(np.abs(self.beliefs - new_b)))
            Q_generate = -cm.c_gen + V_next[new_idx]
        # Action: critic L_k (observe outcome, then choose next action this step)
        def Q_critic(name: str, c_critic: float) -> float:
            l = self.likes[name]
            p_pass = l["P_pass_given_Y1"] * b + l["P_pass_given_Y0"] * (1 - b)
            b_pass = self._bayes_update(b, name, True)
            b_fail = self._bayes_update(b, name, False)
            # After critic, recurse on the same step (no advance) — pick best
            # of {verify, give_up, generate} given the new belief.
            def best_after(b_new: float) -> float:
                opts = [cm.reward * b_new - cm.c_ver, 0.0]
                if step + 1 < self.horizon:
                    new_b = self._generate_next_belief(b_new)
                    idx = int(np.argmin(np.abs(self.beliefs - new_b)))
                    opts.append(-cm.c_gen + V_next[idx])
                return max(opts)
            return -c_critic + p_pass * best_after(b_pass) + (1 - p_pass) * best_after(b_fail)
        actions = {
            "verify": Q_verify, "give_up": Q_giveup, "generate": Q_generate,
        }
        # Critics are conditional on being in the likelihood tables
        # (SWE-bench Lite has L0_syntax + L3_llm_review; LCB adds L2_public_tests)
        if "L0_syntax" in self.likes:
            actions["L0"] = Q_critic("L0_syntax", cm.c_L0)
        if "L2_public_tests" in self.likes:
            actions["L2"] = Q_critic("L2_public_tests", cm.c_L2)
        if "L3_llm_review" in self.likes:
            actions["L3"] = Q_critic("L3_llm_review", cm.c_L3)
        best_a = max(actions, key=lambda a: actions[a])
        return actions[best_a], best_a

    def _build(self) -> None:
        n_b = len(self.beliefs)
        V = np.zeros(n_b)  # terminal: 0 (treated as give_up baseline)
        # At terminal step, only verify or give_up are meaningful
        cm = self.cost
        for i, b in enumerate(self.beliefs):
            V[i] = max(cm.reward * b - cm.c_ver, 0.0)
        self.policy = [None] * self.horizon
        # Backward induction from horizon-1 down to 0
        for step in range(self.horizon - 1, -1, -1):
            new_V = np.zeros(n_b)
            best_actions = []
            for i, b in enumerate(self.beliefs):
                q, a = self._value(b, step, V)
                new_V[i] = q
                best_actions.append(a)
            self.policy[step] = best_actions
            V = new_V

    def select_action(self, b: float, step: int) -> str:
        idx = int(np.argmin(np.abs(self.beliefs - b)))
        return self.policy[step][idx]


# ---------------------------------------------------------------------------
# Policy simulators
# ---------------------------------------------------------------------------

def simulate_policy(traj: list[dict], policy_fn, cost: CostModel,
                    *, max_actions: int = 64) -> dict:
    """Run a stateful policy on a trajectory. Returns (reward, cost, actions).

    Critic actions (L0/L2/L3) don't advance step/patch_idx and don't terminate;
    a degenerate policy that keeps recommending the same critic can stall the
    loop. This happens at extreme cost ratios (e.g. c_ver >> reward) where the
    Bayesian DP finds critics weakly positive while every advancing/terminal
    action is negative. To stay robust without changing normal-regime behavior
    we cap total actions per simulation and force `give_up` if hit.
    """
    state = {
        "step": 0,
        "patch_idx": 0,
        "max_patches": len(traj),
        "verified": False,
        "given_up": False,
        "actions": [],
    }
    cum_cost = 0.0
    reward = 0.0
    while not (state["verified"] or state["given_up"]) and state["step"] < len(traj):
        if len(state["actions"]) >= max_actions:
            # Hit the safety cap (almost certainly a critic-cycle in a degenerate
            # cost regime). Force give_up — caller still gets a finite utility.
            state["given_up"] = True
            state["actions"].append("give_up_cap")
            break
        rec = traj[state["patch_idx"]]
        action = policy_fn(state, rec)
        state["actions"].append(action)
        if action == "verify":
            cum_cost += cost.c_ver
            reward = cost.reward * rec["Y"]
            state["verified"] = True
        elif action == "give_up":
            state["given_up"] = True
        elif action == "generate":
            cum_cost += cost.c_gen
            state["patch_idx"] += 1
            state["step"] += 1
            if state["patch_idx"] >= len(traj):
                state["given_up"] = True
        elif action == "L0":
            cum_cost += cost.c_L0
        elif action == "L2":
            cum_cost += cost.c_L2
        elif action == "L3":
            cum_cost += cost.c_L3
    return {"reward": reward, "cost": cum_cost,
            "utility": reward - cum_cost,
            "verified": state["verified"],
            "n_actions": len(state["actions"])}


# Baseline factories

def policy_always_verify(state, rec):
    return "verify"

def make_threshold(critic: str):
    def _p(state, rec):
        if f"{critic}_done" not in state:
            state[f"{critic}_done"] = True
            return critic
        # observed critic outcome
        passed = rec.get(critic if critic == "L0_syntax" or critic == "L3_llm_review" else None)
        # We re-key: critic is stored in rec under L0_syntax / L3_llm_review
        return "verify" if rec.get(_critic_field(critic)) else ("generate" if state["patch_idx"] + 1 < 5 else "give_up")
    return _p

def _critic_field(c: str) -> str:
    return {"L0": "L0_syntax", "L3": "L3_llm_review"}.get(c, c)


def policy_threshold_L0(state, rec):
    if not state.get("L0_done"):
        state["L0_done"] = True
        return "L0"
    state["L0_done"] = False  # reset for next patch
    if rec.get("L0_syntax"):
        return "verify"
    if state["patch_idx"] + 1 < state.get("max_patches", 5):
        return "generate"
    return "give_up"


def policy_threshold_L3(state, rec):
    if not state.get("L3_done"):
        state["L3_done"] = True
        return "L3"
    state["L3_done"] = False
    if rec.get("L3_llm_review"):
        return "verify"
    if state["patch_idx"] + 1 < state.get("max_patches", 5):
        return "generate"
    return "give_up"


def policy_fixed_pipeline(state, rec):
    if not state.get("L0_done"):
        state["L0_done"] = True
        return "L0"
    if not state.get("L3_done"):
        state["L3_done"] = True
        return "L3"
    state["L0_done"] = False
    state["L3_done"] = False
    if rec.get("L0_syntax") and rec.get("L3_llm_review"):
        return "verify"
    if state["patch_idx"] + 1 < state.get("max_patches", 5):
        return "generate"
    return "give_up"


def policy_best_of_N(N: int):
    def _p(state, rec):
        if state["patch_idx"] + 1 < N:
            return "generate"
        return "verify"
    return _p


def make_bayesian_policy(controller: BayesianController):
    def _p(state, rec):
        # Update belief based on observations seen this step's patch
        b = state.get("belief", controller.prior)
        a = controller.select_action(b, state["step"])
        if a == "L0":
            obs_pass = bool(rec.get("L0_syntax"))
            state["belief"] = controller._bayes_update(b, "L0_syntax", obs_pass)
            return "L0"
        if a == "L2":
            obs_pass = bool(rec.get("L2_public_tests"))
            state["belief"] = controller._bayes_update(b, "L2_public_tests", obs_pass)
            return "L2"
        if a == "L3":
            obs_pass = bool(rec.get("L3_llm_review"))
            state["belief"] = controller._bayes_update(b, "L3_llm_review", obs_pass)
            return "L3"
        if a == "generate":
            state["belief"] = controller._generate_next_belief(b)
        return a
    return _p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_generator(gen: str, out_dir: Path, cost: CostModel) -> dict:
    gen_dir = out_dir / gen
    traj = load_trajectories(gen_dir / "iter_records.jsonl")
    likes = json.loads((gen_dir / "likelihood_tables.json").read_text())
    kernel = json.loads((gen_dir / "transition_kernel.json").read_text())
    prior = likes.get("prior_Y1", 0.5)

    controller = BayesianController(prior, likes, kernel, cost)

    policies = {
        "always_verify": policy_always_verify,
        "best_of_N=3": policy_best_of_N(3),
        "best_of_N=5": policy_best_of_N(5),
        "threshold_L0": policy_threshold_L0,
        "threshold_L3": policy_threshold_L3,
        "fixed_pipeline": policy_fixed_pipeline,
        "bayesian": make_bayesian_policy(controller),
    }

    results = {}
    for name, fn in policies.items():
        utilities, costs, rewards, verified = [], [], [], []
        for inst, t in traj.items():
            r = simulate_policy(t, fn, cost)
            utilities.append(r["utility"])
            costs.append(r["cost"])
            rewards.append(r["reward"])
            verified.append(r["verified"])
        results[name] = {
            "mean_utility": float(np.mean(utilities)),
            "mean_cost": float(np.mean(costs)),
            "mean_reward": float(np.mean(rewards)),
            "pass_rate": float(np.mean([r > 0 for r in rewards])),
            "verify_rate": float(np.mean(verified)),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    args = parser.parse_args()

    cost = CostModel()
    out_dir = args.output_dir.resolve()

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        print(f"\n=== {gen} ===")
        results = run_one_generator(gen, out_dir, cost)
        # Write
        (out_dir / gen / "policy_comparison.json").write_text(json.dumps(results, indent=2))
        # Print table
        print(f"  {'policy':<18} {'utility':>9} {'pass_rate':>10} {'cost':>8} {'verify%':>9}")
        # Sort by utility desc
        for name, r in sorted(results.items(), key=lambda x: -x[1]["mean_utility"]):
            print(f"  {name:<18} {r['mean_utility']:>9.2f} {r['pass_rate']*100:>9.1f}% "
                  f"{r['mean_cost']:>8.2f} {r['verify_rate']*100:>8.1f}%")


if __name__ == "__main__":
    main()
