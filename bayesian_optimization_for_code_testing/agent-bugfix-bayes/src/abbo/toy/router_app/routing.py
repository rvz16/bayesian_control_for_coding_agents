"""A small request-routing package used as the toy benchmark target.

This module implements a simple priority-based request router with pattern
matching and category filtering.  Three injection points allow controlled
bugs to be introduced at runtime:

- Bug A: breaks priority-based route selection (numeric comparison)
- Bug B: breaks pattern matching (string prefix matching)
- Bug C: breaks category filtering (dict lookup)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Route:
    """A single route definition."""

    pattern: str
    handler: str
    priority: int = 0
    category: str = "default"


class Router:
    """Priority-based request router with pattern matching and categories.

    Attributes:
        _routes: Internal list of registered routes.
        _bug_a: When True, priority comparison is inverted (Bug A).
        _bug_b: When True, pattern matching uses wrong logic (Bug B).
        _bug_c: When True, category lookup is broken (Bug C).
    """

    def __init__(self) -> None:
        self._routes: list[Route] = []
        self._bug_a: bool = False
        self._bug_b: bool = False
        self._bug_c: bool = False

    def add_route(self, route: Route) -> None:
        """Register a new route."""
        self._routes.append(route)

    def match(self, path: str) -> Route | None:
        """Find the best matching route for *path*.

        Routes are matched by checking if the route pattern is a prefix of
        *path* or matches as a regex.  Among matching routes the one with
        the highest priority wins.
        """
        candidates: list[Route] = []
        for route in self._routes:
            if self._matches(route.pattern, path):
                candidates.append(route)

        if not candidates:
            return None

        # Bug A: inverts priority comparison
        if self._bug_a:
            return min(candidates, key=lambda r: r.priority)
        return max(candidates, key=lambda r: r.priority)

    def _matches(self, pattern: str, path: str) -> bool:
        """Check if *pattern* matches *path*."""
        # Bug B: uses suffix instead of prefix matching
        if self._bug_b:
            return path.endswith(pattern)
        # Normal: prefix match or full regex
        if path.startswith(pattern):
            return True
        try:
            return bool(re.fullmatch(pattern, path))
        except re.error:
            return False

    def dispatch(self, path: str) -> str | None:
        """Dispatch a request to the matching handler.  Returns handler name."""
        route = self.match(path)
        return route.handler if route is not None else None

    def get_routes_by_category(self, category: str) -> list[Route]:
        """Return all routes in *category*."""
        # Bug C: returns routes NOT in the category
        if self._bug_c:
            return [r for r in self._routes if r.category != category]
        return [r for r in self._routes if r.category == category]

    def list_categories(self) -> list[str]:
        """Return sorted unique categories."""
        return sorted({r.category for r in self._routes})

    def remove_route(self, pattern: str) -> bool:
        """Remove the first route with the given pattern.  Returns True if found."""
        for i, r in enumerate(self._routes):
            if r.pattern == pattern:
                self._routes.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all routes."""
        self._routes.clear()

    # --- Bug injection API (used by ToyEnv) ---

    def inject_bug(self, bug_class: str) -> None:
        """Inject a specific bug family."""
        self._bug_a = bug_class == "A"
        self._bug_b = bug_class == "B"
        self._bug_c = bug_class == "C"

    def clear_bugs(self) -> None:
        """Remove all injected bugs."""
        self._bug_a = False
        self._bug_b = False
        self._bug_c = False
