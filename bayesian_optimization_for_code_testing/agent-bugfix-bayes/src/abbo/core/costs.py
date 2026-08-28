"""Cost model loading and utility computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from abbo.utils.io import load_yaml


@dataclass
class CostModel:
    """Parameterised cost model for a benchmark."""

    c_test: float
    c_patch: float
    c_verifier: float
    reward: float
    c_generator: float = 0.0
    c_critic: float = 0.0
    unit: str = "abstract"

    def utility(self, success: bool, n_tests: int, n_patches: int,
                n_verifiers: int, n_generators: int = 0,
                n_critics: int = 0) -> float:
        """Compute the scalar utility for an episode outcome."""
        r = self.reward if success else 0.0
        cost = (
            self.c_test * n_tests
            + self.c_patch * n_patches
            + self.c_verifier * n_verifiers
            + self.c_generator * n_generators
            + self.c_critic * n_critics
        )
        return r - cost


def load_cost_model(config: dict[str, Any]) -> CostModel:
    """Build a CostModel from a config dict's ``costs`` section."""
    costs = config.get("costs", config)
    return CostModel(
        c_test=float(costs.get("test", 1)),
        c_patch=float(costs.get("patch", 3)),
        c_verifier=float(costs.get("verifier", 20)),
        reward=float(costs.get("reward", 200)),
        c_generator=float(costs.get("generator", 0)),
        c_critic=float(costs.get("critic", 0)),
    )


def load_cost_models_from_file(path: str) -> dict[str, CostModel]:
    """Load all cost models from the cost_models.yaml file."""
    raw = load_yaml(path)
    models = {}
    for name, spec in raw.get("cost_models", {}).items():
        c = spec["costs"]
        models[name] = CostModel(
            c_test=float(c.get("test", 1)),
            c_patch=float(c.get("patch", 3)),
            c_verifier=float(c.get("verifier", 20)),
            reward=float(c.get("reward", 200)),
            unit=spec.get("unit", "abstract"),
        )
    return models
