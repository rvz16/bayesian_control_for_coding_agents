"""Exact symbolic computation for the synthetic benchmark.

All arithmetic uses ``fractions.Fraction`` to avoid floating-point noise.
The dynamic program computes optimal expected utility and extracts the
decision tree for Bayesian decision-theoretic orchestration.

Expected results for the default environment:
- Bayes optimal expected utility: 391/3
- One-test MAP expected utility: 268/3
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from abbo.synthetic.env import SyntheticEnv

# ---------------------------------------------------------------------------
# Convenience types
# ---------------------------------------------------------------------------
Belief = dict[str, Fraction]  # {class_label: probability}


def uniform_prior(env: SyntheticEnv) -> Belief:
    """Return a uniform prior over the hidden classes."""
    n = len(env.classes)
    return {c: Fraction(1, n) for c in env.classes}


# ---------------------------------------------------------------------------
# Bayesian posterior update
# ---------------------------------------------------------------------------

def posterior_update(
    belief: Belief,
    test_id: str,
    obs_fail: bool,
    env: SyntheticEnv,
) -> Belief:
    """Exact Bayes update: P(H | z, T) proportional to P(z | H, T) * P(H).

    Args:
        belief: Current belief distribution over hidden classes.
        test_id: Which diagnostic test was run.
        obs_fail: True if the test failed, False if it passed.
        env: The synthetic environment.

    Returns:
        Updated belief distribution (normalised).
    """
    unnorm: dict[str, Fraction] = {}
    for cls in env.classes:
        p_fail = env.get_fail_prob_fraction(cls, test_id)
        likelihood = p_fail if obs_fail else (Fraction(1) - p_fail)
        unnorm[cls] = likelihood * belief[cls]

    total = sum(unnorm.values())
    return {cls: unnorm[cls] / total for cls in env.classes}


# ---------------------------------------------------------------------------
# Marginal observation probability
# ---------------------------------------------------------------------------

def obs_probability(
    belief: Belief,
    test_id: str,
    obs_fail: bool,
    env: SyntheticEnv,
) -> Fraction:
    """P(z | belief, T) = sum_k P(z | k, T) * belief[k]."""
    total = Fraction(0)
    for cls in env.classes:
        p_fail = env.get_fail_prob_fraction(cls, test_id)
        likelihood = p_fail if obs_fail else (Fraction(1) - p_fail)
        total += likelihood * belief[cls]
    return total


# ---------------------------------------------------------------------------
# Terminal value (no more tests left)
# ---------------------------------------------------------------------------

def terminal_value(belief: Belief, env: SyntheticEnv) -> Fraction:
    """V_terminal(b) = R * max_k b[k] - C_patch - C_verifier.

    This is the expected utility of choosing the MAP patch and verifying.
    """
    r = Fraction(env.reward).limit_denominator(1000)
    c_patch = Fraction(env.c_patch).limit_denominator(1000)
    c_ver = Fraction(env.c_verifier).limit_denominator(1000)
    max_b = max(belief.values())
    return r * max_b - c_patch - c_ver


# ---------------------------------------------------------------------------
# Dynamic programming solver
# ---------------------------------------------------------------------------

def _dp_value(
    belief: Belief,
    remaining_tests: int,
    env: SyntheticEnv,
) -> tuple[Fraction, str | None]:
    """Compute V(belief, remaining_tests) and the optimal first action.

    Returns:
        (value, best_test) where best_test is None if stopping is optimal
        and a test_id string otherwise.
    """
    c_test = Fraction(env.c_test).limit_denominator(1000)

    # Option 0: stop testing, go to MAP patch + verify
    v_stop = terminal_value(belief, env)
    best_val = v_stop
    best_action: str | None = None

    if remaining_tests <= 0:
        return best_val, best_action

    # Option: run each available test
    for test_id in env.tests:
        # Expected value of running this test
        ev = Fraction(0)
        for obs_fail in [True, False]:
            p_z = obs_probability(belief, test_id, obs_fail, env)
            if p_z == 0:
                continue
            new_belief = posterior_update(belief, test_id, obs_fail, env)
            v_child, _ = _dp_value(new_belief, remaining_tests - 1, env)
            ev += p_z * v_child
        v_test = -c_test + ev

        if v_test > best_val:
            best_val = v_test
            best_action = test_id

    return best_val, best_action


def solve_dp(env: SyntheticEnv, max_tests: int | None = None) -> Fraction:
    """Solve the DP and return the optimal expected utility from uniform prior.

    Args:
        env: Synthetic environment.
        max_tests: Override for max diagnostic tests (default: env.max_tests).

    Returns:
        Exact expected utility as a Fraction.
    """
    if max_tests is None:
        max_tests = env.max_tests
    prior = uniform_prior(env)
    val, _ = _dp_value(prior, max_tests, env)
    return val


# ---------------------------------------------------------------------------
# Decision tree extraction
# ---------------------------------------------------------------------------

def extract_tree(
    env: SyntheticEnv, max_tests: int | None = None
) -> dict[str, Any]:
    """Extract the full optimal decision tree.

    Returns a nested dict describing the tree.  Each node is either:
    - ``{"action": "test", "test": "T1", "fail": <subtree>, "pass": <subtree>}``
    - ``{"action": "patch", "patch": "PA"}``
    """
    if max_tests is None:
        max_tests = env.max_tests
    prior = uniform_prior(env)
    return _extract_node(prior, max_tests, env)


def _extract_node(
    belief: Belief,
    remaining: int,
    env: SyntheticEnv,
) -> dict[str, Any]:
    _, best_test = _dp_value(belief, remaining, env)

    if best_test is None or remaining <= 0:
        # Terminal: choose MAP patch
        map_cls = max(belief, key=lambda c: belief[c])
        patch = env.patch_for_class(map_cls)
        return {
            "action": "patch",
            "patch": patch,
            "target_class": map_cls,
            "belief": {c: str(belief[c]) for c in env.classes},
        }

    # Run the best test
    fail_belief = posterior_update(belief, best_test, True, env)
    pass_belief = posterior_update(belief, best_test, False, env)

    return {
        "action": "test",
        "test": best_test,
        "belief": {c: str(belief[c]) for c in env.classes},
        "fail": _extract_node(fail_belief, remaining - 1, env),
        "pass": _extract_node(pass_belief, remaining - 1, env),
    }


# ---------------------------------------------------------------------------
# One-test MAP baseline (exact)
# ---------------------------------------------------------------------------

def one_test_map_utility(
    env: SyntheticEnv, test_id: str = "T1"
) -> Fraction:
    """Exact expected utility of the one-test MAP policy.

    Policy: run *test_id* once, Bayes update, choose MAP patch, verify.
    """
    prior = uniform_prior(env)
    c_test = Fraction(env.c_test).limit_denominator(1000)

    ev = Fraction(0)
    for obs_fail in [True, False]:
        p_z = obs_probability(prior, test_id, obs_fail, env)
        post = posterior_update(prior, test_id, obs_fail, env)
        tv = terminal_value(post, env)
        ev += p_z * tv

    return -c_test + ev


# ---------------------------------------------------------------------------
# Convenience: compute all exact utilities
# ---------------------------------------------------------------------------

def compute_exact_utilities(
    env: SyntheticEnv | None = None,
) -> dict[str, Fraction]:
    """Compute exact expected utilities for default policies.

    Returns dict with keys ``'bayes_optimal'`` and ``'one_test_map'``.
    The Bayes optimal should be 391/3 and one-test MAP should be 268/3.
    """
    if env is None:
        env = SyntheticEnv()
    return {
        "bayes_optimal": solve_dp(env),
        "one_test_map": one_test_map_utility(env, "T1"),
    }
