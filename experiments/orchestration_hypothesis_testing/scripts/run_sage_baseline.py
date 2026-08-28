#!/usr/bin/env python
"""SAGE baseline runner with TTS self-consistency (no clarification).

Builds on `different_agents/v4/lcb_llm_tool_agent.py` (colleague's adapter
that wires our critics/verifier as Sage tools). Differences from colleague:

- TTSCandidateGenerator: draws N=5 LLM samples per decision (temp 0.7),
  groups by unique action → Sage sees frequency-aggregated candidates
  instead of one single-chat candidate.

Clarification is intentionally disabled (max_questions=0, tau_exec=0):
this is automated benchmark evaluation, there's no human to answer
questions, and self-clarification adds cost without policy signal on
deterministic test sets. The result is a "Sage runtime + TTS UQ +
majority-vote policy" baseline — fair to compare against our BDP because
both observe the same critics/verifier.

Usage:
    python run_sage_baseline.py \\
        --benchmark lcb_hard \\
        --generator haiku45 \\
        --n-instances 20 \\
        --tts-samples 5 \\
        --output sim_results/sage_baseline/lcb_hard_haiku45_n20.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Import the colleague's adapter machinery
from different_agents.v4 import lcb_llm_tool_agent as base  # noqa: E402
from sage_agent import (  # noqa: E402
    ExecutionResult, ToolCall, ToolCallCandidate, ToolSchema,
)
from sage_agent.langgraph import SAGEGraph, SAGEGraphConfig  # noqa: E402


# ---------------------------------------------------------------------------
# TTSCandidateGenerator — N-sample self-consistency for action distribution
# ---------------------------------------------------------------------------

@dataclass
class TTSCandidateGenerator:
    """Draw N samples from the LLM, return one ToolCallCandidate per
    unique action. Sage's BeliefState will assign uniform weights, so
    a 5/5 unanimous sample yields one candidate (prob 1.0 → execute);
    a 3/2 split yields two candidates (prob 0.5 each → clarify)."""

    deps: base.AgentDeps
    state: base.AgentState
    n_samples: int = 5
    temperature: float = 0.7  # higher than 0.0 → diversity across samples

    def generate_candidates(
        self,
        _user_input: str,
        _observations: list[str],
        _tool_schemas: dict[str, ToolSchema],
    ) -> list[ToolCallCandidate]:
        # Initial step: no candidate exists yet → force generate (same as base)
        if not (self.state.get("candidate_payload") or ""):
            self.state.update({
                "chosen_action": "generate",
                "chosen_reasoning": "auto: no candidate exists",
                "decision_raw": "",
            })
            return [ToolCallCandidate(
                tool_name="generate",
                arguments={"reasoning": "auto: no candidate exists"},
            )]

        # Draw N decision samples from the LLM
        samples: list[tuple[str, str]] = []
        usage_pt = usage_ct = usage_cost = 0
        last_raw = ""
        for _ in range(self.n_samples):
            raw, pt, ct, cost, _lp = base._chat(
                self.deps,
                base._decision_messages(self.state),
                temperature=self.temperature,
                max_tokens=self.deps.max_tokens_decision,
            )
            action, reasoning = base._parse_decision(raw)
            if not action:
                action, reasoning = base.fallback_decision_after_parse_failure(
                    self.state)
            samples.append((action, reasoning))
            usage_pt += pt
            usage_ct += ct
            usage_cost += cost
            last_raw = raw

        # Frequency-aggregate per unique action
        counter = Counter(a for a, _ in samples)
        reasoning_by_action = {a: r for a, r in samples}
        top_action, top_count = counter.most_common(1)[0]

        # State book-keeping (use top action as the "chosen" one for downstream)
        updates = base._merge_usage(self.state, usage_pt, usage_ct, usage_cost)
        updates.update({
            "chosen_action": top_action,
            "chosen_reasoning": reasoning_by_action[top_action],
            "decision_raw": last_raw,
            "n_decisions": int(self.state.get("n_decisions", 0)) + 1,
            "tts_action_freq": dict(counter),
            "tts_top_action_prob": top_count / self.n_samples,
            "tts_n_samples": self.n_samples,
        })
        self.state.update(updates)

        # Build one ToolCallCandidate per unique action.
        # Order most-frequent first so best_idx points to it when probabilities tie.
        return [
            ToolCallCandidate(
                tool_name=action,
                arguments={"reasoning": reasoning_by_action[action]},
            )
            for action, _count in counter.most_common()
        ]


# ---------------------------------------------------------------------------
# Paper-config episode runner
# ---------------------------------------------------------------------------

def run_sage_paper_episode(
    state: base.AgentState,
    deps: base.AgentDeps,
    tts_samples: int = 5,
    tts_temperature: float = 0.7,
) -> base.AgentState:
    """Run an episode with Sage's paper-config Algorithm 1 (EVPI on)."""
    tool_schemas = base.sage_tool_schemas()
    while not state.get("done") and int(state.get("step", 0)) < state["max_steps"]:
        if (
            not (state.get("candidate_payload") or "")
            and int(state.get("n_generations", 0)) >= state["max_generations"]
        ):
            state.update({"done": True, "final_action": "no_candidate"})
            break

        graph = SAGEGraph(
            tool_schemas=tool_schemas,
            candidate_generator=TTSCandidateGenerator(
                deps=deps, state=state,
                n_samples=tts_samples, temperature=tts_temperature,
            ),
            question_generator=base.NoQuestionGenerator(),  # Step 4 will upgrade
            question_asker=base.NoQuestionAsker(),
            tool_executor=base.SageActionToolExecutor(deps, state),
            constraint_extractor=None,
            config=SAGEGraphConfig(
                # Automated benchmark run: no human to answer questions,
                # so clarification is disabled and the planner is forced to
                # execute the top candidate from the TTS distribution.
                tau_exec=0.0,        # force "confident" gate always passes
                alpha=0.0,
                max_questions=0,     # no clarification rounds
                enable_escalation=False,
                recursion_limit=8,
            ),
        )
        graph.run("Choose and execute the next LiveCodeBench tool action.")

    if not state.get("done") and int(state.get("step", 0)) >= state["max_steps"]:
        state.update({"done": True, "final_action": "max_steps"})
    return state


# ---------------------------------------------------------------------------
# Thin main() wrapping colleague's main but using the paper-config runner
# ---------------------------------------------------------------------------

def _patch_runner() -> None:
    """Replace base.run_sage_agent_episode with our paper-config version."""
    base.run_sage_agent_episode = run_sage_paper_episode  # type: ignore[attr-defined]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # Forward all the existing flags by delegating to base.main, but inject
    # extras for TTS config.
    p.add_argument("--tts-samples", type=int, default=5,
                   help="N TTS samples per decision step (default 5).")
    p.add_argument("--tts-temperature", type=float, default=0.7,
                   help="LLM temperature for TTS sampling (default 0.7).")
    p.add_argument("--passthrough", nargs=argparse.REMAINDER, default=[],
                   help="All other CLI args forwarded to lcb_llm_tool_agent.main")
    ns = p.parse_args()

    # Configure the runner override
    _patch_runner()
    # The patched runner reads tts params via closure-injection. Stash them
    # on the base module so the patched function picks them up.
    base._tts_samples = ns.tts_samples  # type: ignore[attr-defined]
    base._tts_temperature = ns.tts_temperature  # type: ignore[attr-defined]

    # Forward to base.main() with remaining argv. Build sys.argv shape.
    new_argv = [sys.argv[0]] + ns.passthrough
    sys.argv = new_argv
    base.main()


if __name__ == "__main__":
    main()
