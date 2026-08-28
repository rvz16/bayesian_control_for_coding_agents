"""Allure integration for experiment auditability.

Provides an AllureLogger that creates structured steps and attachments
for each episode, following the suite hierarchy:
  parent suite: benchmark name
  suite: policy name
  sub-suite: project/repository
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from abbo.core.types import EpisodeResult


class AllureLogger:
    """Logs episode results as Allure-compatible artifacts.

    This class writes structured JSON files that can be consumed by
    allure-pytest or processed into Allure reports.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def log_episode(
        self,
        episode: EpisodeResult,
        benchmark: str,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Log a single episode as an Allure-compatible artifact.

        Creates a JSON file with steps, attachments, and metadata.
        """
        episode_id = f"{benchmark}_{episode.policy_name}_{episode.hidden_class}"
        artifact = {
            "name": episode_id,
            "benchmark": benchmark,
            "policy": episode.policy_name,
            "hidden_class": episode.hidden_class,
            "steps": self._build_steps(episode),
            "attachments": self._build_attachments(episode, config),
            "result": "passed" if episode.patch_correct else "failed",
            "utility": episode.utility,
        }

        out_path = self.results_dir / f"{episode_id}.json"
        out_path.write_text(json.dumps(artifact, indent=2, default=str))
        return out_path

    def _build_steps(self, episode: EpisodeResult) -> list[dict[str, Any]]:
        steps = [{"name": "setup_bug", "status": "passed"}]

        for action in episode.actions:
            steps.append({
                "name": f"{action.kind.value}: {action.label}",
                "status": "passed",
                "observation": str(action.observation) if action.observation else None,
                "cost": action.cost,
            })

        steps.append({
            "name": "summarize_outcome",
            "status": "passed" if episode.patch_correct else "failed",
            "utility": episode.utility,
        })
        return steps

    def _build_attachments(
        self,
        episode: EpisodeResult,
        config: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        attachments = []

        if config:
            attachments.append({
                "name": "config.json",
                "type": "application/json",
                "content": json.dumps(config, indent=2, default=str),
            })

        attachments.append({
            "name": "episode_manifest.json",
            "type": "application/json",
            "content": json.dumps({
                "policy": episode.policy_name,
                "hidden_class": episode.hidden_class,
                "patch_chosen": episode.patch_chosen,
                "patch_correct": episode.patch_correct,
                "utility": episode.utility,
                "n_actions": len(episode.actions),
            }, indent=2, default=str),
        })

        if episode.belief_timeline:
            attachments.append({
                "name": "posterior_timeline.json",
                "type": "application/json",
                "content": json.dumps(episode.belief_timeline, indent=2, default=str),
            })

        attachments.append({
            "name": "cost_breakdown.json",
            "type": "application/json",
            "content": json.dumps({
                "diagnostic": episode.costs.diagnostic_cost,
                "patch": episode.costs.patch_cost,
                "verifier": episode.costs.verifier_cost,
                "generator": episode.costs.generator_cost,
                "critic": episode.costs.critic_cost,
                "total": episode.costs.total_cost,
            }, indent=2),
        })

        return attachments
