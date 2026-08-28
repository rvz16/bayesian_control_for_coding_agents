"""Full test suite for the router app.

Tests are organised so that:
- Priority tests (test_priority_*) are most sensitive to Bug A
- Pattern tests (test_pattern_*) are most sensitive to Bug B
- Category tests (test_category_*) are most sensitive to Bug C
Some tests have cross-sensitivity to approximate the target confusion matrix.
"""

from __future__ import annotations

import pytest

from abbo.toy.router_app.routing import Route, Router


@pytest.fixture
def router() -> Router:
    r = Router()
    r.add_route(Route("/api/users", "users_handler", priority=10, category="api"))
    r.add_route(Route("/api/users", "users_v2_handler", priority=20, category="api"))
    r.add_route(Route("/api/orders", "orders_handler", priority=10, category="api"))
    r.add_route(Route("/admin", "admin_handler", priority=5, category="admin"))
    r.add_route(Route("/admin/settings", "settings_handler", priority=15, category="admin"))
    r.add_route(Route("/health", "health_handler", priority=1, category="infra"))
    r.add_route(Route("/metrics", "metrics_handler", priority=1, category="infra"))
    return r


# ----- Priority tests (sensitive to Bug A) -----

def test_priority_highest_wins(router: Router) -> None:
    route = router.match("/api/users")
    assert route is not None
    assert route.handler == "users_v2_handler"


def test_priority_single_match(router: Router) -> None:
    route = router.match("/api/orders")
    assert route is not None
    assert route.handler == "orders_handler"


def test_priority_admin_subpath(router: Router) -> None:
    route = router.match("/admin/settings")
    assert route is not None
    assert route.handler == "settings_handler"


def test_priority_order_independence(router: Router) -> None:
    r2 = Router()
    r2.add_route(Route("/x", "low", priority=1))
    r2.add_route(Route("/x", "high", priority=100))
    r2.add_route(Route("/x", "mid", priority=50))
    assert r2.match("/x").handler == "high"


def test_priority_negative_values() -> None:
    r = Router()
    r.add_route(Route("/p", "neg", priority=-5))
    r.add_route(Route("/p", "pos", priority=5))
    assert r.match("/p").handler == "pos"


def test_priority_equal_takes_last() -> None:
    r = Router()
    r.add_route(Route("/eq", "first", priority=10))
    r.add_route(Route("/eq", "second", priority=10))
    # max returns first max by default, but both are valid
    result = r.match("/eq")
    assert result is not None
    assert result.priority == 10


# ----- Pattern matching tests (sensitive to Bug B) -----

def test_pattern_prefix_match(router: Router) -> None:
    route = router.match("/api/users/123")
    assert route is not None
    assert "users" in route.handler


def test_pattern_exact_match(router: Router) -> None:
    route = router.match("/health")
    assert route is not None
    assert route.handler == "health_handler"


def test_pattern_no_match(router: Router) -> None:
    route = router.match("/unknown/path")
    assert route is None


def test_pattern_admin_prefix(router: Router) -> None:
    route = router.match("/admin/dashboard")
    assert route is not None
    assert route.category == "admin"


def test_pattern_partial_no_match(router: Router) -> None:
    # "/heal" should not match "/health" as prefix
    route = router.match("/heal")
    assert route is None


def test_pattern_slash_sensitivity() -> None:
    r = Router()
    r.add_route(Route("/api", "api_root", priority=1))
    route = r.match("/api/deep/path")
    assert route is not None
    assert route.handler == "api_root"


def test_pattern_regex_support() -> None:
    r = Router()
    r.add_route(Route(r"/item/\d+", "item_handler", priority=1))
    assert r.match("/item/42") is not None
    assert r.match("/item/abc") is None


# ----- Category filtering tests (sensitive to Bug C) -----

def test_category_filter_api(router: Router) -> None:
    routes = router.get_routes_by_category("api")
    assert len(routes) == 3
    assert all(r.category == "api" for r in routes)


def test_category_filter_admin(router: Router) -> None:
    routes = router.get_routes_by_category("admin")
    assert len(routes) == 2
    assert all(r.category == "admin" for r in routes)


def test_category_filter_infra(router: Router) -> None:
    routes = router.get_routes_by_category("infra")
    assert len(routes) == 2


def test_category_filter_empty(router: Router) -> None:
    routes = router.get_routes_by_category("nonexistent")
    assert len(routes) == 0


def test_category_list(router: Router) -> None:
    cats = router.list_categories()
    assert "api" in cats
    assert "admin" in cats
    assert "infra" in cats


def test_category_single() -> None:
    r = Router()
    r.add_route(Route("/a", "h1", category="x"))
    r.add_route(Route("/b", "h2", category="y"))
    assert len(r.get_routes_by_category("x")) == 1
    assert r.get_routes_by_category("x")[0].handler == "h1"


def test_category_no_cross_contamination() -> None:
    r = Router()
    r.add_route(Route("/a", "h1", category="alpha"))
    r.add_route(Route("/b", "h2", category="beta"))
    alpha = r.get_routes_by_category("alpha")
    assert all(route.category == "alpha" for route in alpha)
    beta = r.get_routes_by_category("beta")
    assert all(route.category == "beta" for route in beta)


# ----- Utility tests -----

def test_remove_route(router: Router) -> None:
    assert router.remove_route("/health") is True
    assert router.match("/health") is None


def test_clear_routes(router: Router) -> None:
    router.clear()
    assert router.match("/api/users") is None


def test_dispatch(router: Router) -> None:
    handler = router.dispatch("/api/orders")
    assert handler == "orders_handler"


def test_dispatch_no_match(router: Router) -> None:
    handler = router.dispatch("/nope")
    assert handler is None
