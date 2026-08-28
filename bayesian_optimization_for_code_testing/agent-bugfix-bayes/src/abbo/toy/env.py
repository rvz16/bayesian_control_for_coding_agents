"""Toy executable benchmark environment.

Wraps the router_app package with bug injection, diagnostics, patches,
and a full verifier — preserving the same information structure as the
synthetic benchmark (3 classes, 3 diagnostics, target confusion matrix).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from abbo.synthetic.env import SyntheticEnv
from abbo.toy.router_app.routing import Route, Router


# Canonical test suites partitioned by sensitivity
# Each diagnostic runs a subset of tests.  Controlled randomness is added
# to approximate the target failure matrix.
PRIORITY_TESTS = [
    "test_priority_highest_wins",
    "test_priority_single_match",
    "test_priority_admin_subpath",
    "test_priority_order_independence",
    "test_priority_negative_values",
    "test_priority_equal_takes_last",
]

PATTERN_TESTS = [
    "test_pattern_prefix_match",
    "test_pattern_exact_match",
    "test_pattern_no_match",
    "test_pattern_admin_prefix",
    "test_pattern_partial_no_match",
    "test_pattern_slash_sensitivity",
    "test_pattern_regex_support",
]

CATEGORY_TESTS = [
    "test_category_filter_api",
    "test_category_filter_admin",
    "test_category_filter_infra",
    "test_category_filter_empty",
    "test_category_list",
    "test_category_single",
    "test_category_no_cross_contamination",
]

ALL_TESTS = PRIORITY_TESTS + PATTERN_TESTS + CATEGORY_TESTS


def _make_router() -> Router:
    """Create a router with the standard test fixture routes."""
    r = Router()
    r.add_route(Route("/api/users", "users_handler", priority=10, category="api"))
    r.add_route(Route("/api/users", "users_v2_handler", priority=20, category="api"))
    r.add_route(Route("/api/orders", "orders_handler", priority=10, category="api"))
    r.add_route(Route("/admin", "admin_handler", priority=5, category="admin"))
    r.add_route(Route("/admin/settings", "settings_handler", priority=15, category="admin"))
    r.add_route(Route("/health", "health_handler", priority=1, category="infra"))
    r.add_route(Route("/metrics", "metrics_handler", priority=1, category="infra"))
    return r


def _run_test(router: Router, test_name: str) -> bool:
    """Run a single named test against *router*.  Returns True if passes."""
    try:
        if test_name == "test_priority_highest_wins":
            route = router.match("/api/users")
            return route is not None and route.handler == "users_v2_handler"
        elif test_name == "test_priority_single_match":
            route = router.match("/api/orders")
            return route is not None and route.handler == "orders_handler"
        elif test_name == "test_priority_admin_subpath":
            route = router.match("/admin/settings")
            return route is not None and route.handler == "settings_handler"
        elif test_name == "test_priority_order_independence":
            r2 = Router()
            r2.inject_bug("A") if router._bug_a else None
            r2.inject_bug("B") if router._bug_b else None
            r2.inject_bug("C") if router._bug_c else None
            r2.add_route(Route("/x", "low", priority=1))
            r2.add_route(Route("/x", "high", priority=100))
            r2.add_route(Route("/x", "mid", priority=50))
            return r2.match("/x").handler == "high"
        elif test_name == "test_priority_negative_values":
            r2 = Router()
            r2.inject_bug("A") if router._bug_a else None
            r2.inject_bug("B") if router._bug_b else None
            r2.inject_bug("C") if router._bug_c else None
            r2.add_route(Route("/p", "neg", priority=-5))
            r2.add_route(Route("/p", "pos", priority=5))
            return r2.match("/p").handler == "pos"
        elif test_name == "test_priority_equal_takes_last":
            r2 = Router()
            r2.inject_bug("A") if router._bug_a else None
            r2.add_route(Route("/eq", "first", priority=10))
            r2.add_route(Route("/eq", "second", priority=10))
            result = r2.match("/eq")
            return result is not None and result.priority == 10
        elif test_name == "test_pattern_prefix_match":
            route = router.match("/api/users/123")
            return route is not None and "users" in route.handler
        elif test_name == "test_pattern_exact_match":
            route = router.match("/health")
            return route is not None and route.handler == "health_handler"
        elif test_name == "test_pattern_no_match":
            return router.match("/unknown/path") is None
        elif test_name == "test_pattern_admin_prefix":
            route = router.match("/admin/dashboard")
            return route is not None and route.category == "admin"
        elif test_name == "test_pattern_partial_no_match":
            return router.match("/heal") is None
        elif test_name == "test_pattern_slash_sensitivity":
            r2 = Router()
            r2.inject_bug("B") if router._bug_b else None
            r2.add_route(Route("/api", "api_root", priority=1))
            route = r2.match("/api/deep/path")
            return route is not None and route.handler == "api_root"
        elif test_name == "test_pattern_regex_support":
            r2 = Router()
            r2.inject_bug("B") if router._bug_b else None
            r2.add_route(Route(r"/item/\d+", "item_handler", priority=1))
            return r2.match("/item/42") is not None and r2.match("/item/abc") is None
        elif test_name == "test_category_filter_api":
            routes = router.get_routes_by_category("api")
            return len(routes) == 3 and all(r.category == "api" for r in routes)
        elif test_name == "test_category_filter_admin":
            routes = router.get_routes_by_category("admin")
            return len(routes) == 2 and all(r.category == "admin" for r in routes)
        elif test_name == "test_category_filter_infra":
            routes = router.get_routes_by_category("infra")
            return len(routes) == 2
        elif test_name == "test_category_filter_empty":
            return len(router.get_routes_by_category("nonexistent")) == 0
        elif test_name == "test_category_list":
            cats = router.list_categories()
            return "api" in cats and "admin" in cats and "infra" in cats
        elif test_name == "test_category_single":
            r2 = Router()
            r2.inject_bug("C") if router._bug_c else None
            r2.add_route(Route("/a", "h1", category="x"))
            r2.add_route(Route("/b", "h2", category="y"))
            result = r2.get_routes_by_category("x")
            return len(result) == 1 and result[0].handler == "h1"
        elif test_name == "test_category_no_cross_contamination":
            r2 = Router()
            r2.inject_bug("C") if router._bug_c else None
            r2.add_route(Route("/a", "h1", category="alpha"))
            r2.add_route(Route("/b", "h2", category="beta"))
            alpha = r2.get_routes_by_category("alpha")
            return all(r.category == "alpha" for r in alpha) and len(alpha) == 1
        return True
    except Exception:
        return False


# Diagnostic definitions: which tests to run + noise parameters
DIAGNOSTIC_SPEC = {
    "T1": {
        "primary_tests": PRIORITY_TESTS,      # sensitive to Bug A
        "secondary_tests": PATTERN_TESTS[:2],  # slight sensitivity to others
        "primary_weight": 0.85,
    },
    "T2": {
        "primary_tests": PATTERN_TESTS,
        "secondary_tests": CATEGORY_TESTS[:2],
        "primary_weight": 0.85,
    },
    "T3": {
        "primary_tests": CATEGORY_TESTS,
        "secondary_tests": PRIORITY_TESTS[:2],
        "primary_weight": 0.85,
    },
}


class ToyEnv:
    """Toy executable benchmark environment.

    Wraps the Router with bug injection and diagnostics that approximate
    the target confusion matrix from the synthetic benchmark.
    """

    def __init__(self) -> None:
        self.classes = ["A", "B", "C"]
        self.tests = ["T1", "T2", "T3"]
        self._current_bug: str | None = None

    def inject_bug(self, bug_class: str) -> None:
        """Inject a bug family into the router."""
        self._current_bug = bug_class

    def remove_bug(self) -> None:
        """Remove the currently injected bug."""
        self._current_bug = None

    def _make_buggy_router(self) -> Router:
        """Create a router with the current bug injected."""
        router = _make_router()
        if self._current_bug:
            router.inject_bug(self._current_bug)
        return router

    def run_diagnostic(
        self, test_id: str, rng: np.random.Generator
    ) -> bool:
        """Run a diagnostic.  Returns True if the diagnostic *fails*.

        The diagnostic runs a subset of tests and adds controlled randomness
        to approximate the target failure matrix:
          T1: P(fail|A)~0.9, P(fail|B)~0.2, P(fail|C)~0.2
          (and cyclically for T2, T3)
        """
        router = self._make_buggy_router()
        spec = DIAGNOSTIC_SPEC[test_id]

        # Run primary tests
        primary_failures = sum(
            1 for t in spec["primary_tests"] if not _run_test(router, t)
        )
        primary_total = len(spec["primary_tests"])

        # Run secondary tests
        secondary_failures = sum(
            1 for t in spec["secondary_tests"] if not _run_test(router, t)
        )
        secondary_total = len(spec["secondary_tests"])

        # Compute failure score
        w = spec["primary_weight"]
        if primary_total > 0 and secondary_total > 0:
            score = (
                w * (primary_failures / primary_total)
                + (1 - w) * (secondary_failures / secondary_total)
            )
        elif primary_total > 0:
            score = primary_failures / primary_total
        else:
            score = 0.0

        # Add controlled noise to approximate target probabilities
        # If any primary tests fail, this diagnostic's domain is hit
        if primary_failures > 0:
            # Bug is in this diagnostic's domain: fail with P~0.9
            fail = rng.random() < 0.9
        elif secondary_failures > 0:
            # Some cross-sensitivity
            fail = rng.random() < 0.35
        else:
            # Bug is not in this domain: fail with P~0.2
            fail = rng.random() < 0.2

        return fail

    def run_full_suite(self) -> tuple[bool, dict[str, bool]]:
        """Run all tests.  Returns (all_pass, per_test_results)."""
        router = self._make_buggy_router()
        results = {}
        for test_name in ALL_TESTS:
            results[test_name] = _run_test(router, test_name)
        all_pass = all(results.values())
        return all_pass, results

    def apply_patch(self, patch_id: str) -> bool:
        """Apply a patch.  Returns True if it fixes the current bug."""
        patch_map = {"PA": "A", "PB": "B", "PC": "C"}
        target = patch_map.get(patch_id)
        return target == self._current_bug

    def as_synthetic_env(self) -> SyntheticEnv:
        """Create a SyntheticEnv with the same parameters for policy reuse."""
        return SyntheticEnv()

    def sample_hidden(self, rng: np.random.Generator) -> str:
        """Sample a hidden bug class uniformly."""
        return str(rng.choice(self.classes))
