"""Patch definitions for the toy benchmark.

Each patch is a function that modifies the Router to fix a specific bug family.
"""

from __future__ import annotations

from abbo.toy.router_app.routing import Router


def apply_patch_a(router: Router) -> None:
    """Fix Bug A: restore correct priority comparison."""
    router._bug_a = False


def apply_patch_b(router: Router) -> None:
    """Fix Bug B: restore correct pattern matching."""
    router._bug_b = False


def apply_patch_c(router: Router) -> None:
    """Fix Bug C: restore correct category filtering."""
    router._bug_c = False


PATCHES = {
    "PA": apply_patch_a,
    "PB": apply_patch_b,
    "PC": apply_patch_c,
}


def get_patch(patch_id: str):
    """Return the patch function for the given patch ID."""
    return PATCHES[patch_id]
