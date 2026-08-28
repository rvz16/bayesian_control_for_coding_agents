"""Agent comparison: Simple Agent vs Bayesian POMDP on real bugs.

Creates rich Allure reports with belief timelines, cost breakdowns,
action traces, and diffs. Run with::

    pytest tests/test_agent_comparison.py --alluredir allure-results -v
    allure serve allure-results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import allure
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.llm_provider import (
    LLMConfig,
    is_ollama_available,
    is_openai_endpoint_available,
)
from abbo.realworld.agents.real_bugs import BUGGY_SOURCES_REAL, BUG_METADATA
from abbo.realworld.agents.simple_agent import (
    AgentCostConfig,
    run_simple_agent,
    run_simple_agent_escalating,
)
from abbo.realworld.agents.bayes_agent import run_bayes_agent

# --- LLM backend selection ---
# Primary: Innopolis hosted gpt-oss-120b via OpenAI-compatible endpoint.
# Fallback: local Ollama with qwen2.5:7b.
INNOPOLIS_BASE = "https://openai.global.ai.innopolis.university"
INNOPOLIS_MODEL = "gpt-oss-120b"

if is_openai_endpoint_available(INNOPOLIS_BASE):
    LLM = LLMConfig(
        provider="openai",
        model=INNOPOLIS_MODEL,
        base_url=INNOPOLIS_BASE,
        temperature=0.3,
        max_tokens=8192,
        timeout=600,
    )
    _backend_ok = True
    _backend_name = f"Innopolis ({INNOPOLIS_MODEL})"
elif is_ollama_available():
    LLM = LLMConfig(model="qwen2.5:7b", temperature=0.3)
    _backend_ok = True
    _backend_name = "Ollama (qwen2.5:7b)"
else:
    LLM = LLMConfig()
    _backend_ok = False
    _backend_name = "none"

pytestmark = pytest.mark.skipif(
    not _backend_ok,
    reason=f"No LLM backend available — check {INNOPOLIS_BASE} or start Ollama",
)

COSTS = AgentCostConfig()
ALL_BUGS = sorted(BUGGY_SOURCES_REAL.keys())


def _attach_result(r, request=None):
    """Attach agent result as JSON to Allure and collect for summary."""
    meta = BUG_METADATA.get(r.bug_id, {})
    d = {
        "policy": r.policy_name,
        "llm_backend": _backend_name,
        "bug_id": r.bug_id,
        "difficulty": meta.get("difficulty", "?"),
        "fixed": r.fixed,
        "total_cost": r.total_cost,
        "reward": r.reward,
        "utility": r.utility,
        "llm_calls": r.llm_calls,
        "test_runs": getattr(r, "test_runs", 0),
        "critic_runs": getattr(r, "critic_runs", 0),
        "wall_seconds": round(r.wall_clock_seconds, 2),
    }
    if hasattr(r, "belief_timeline") and r.belief_timeline:
        d["belief_timeline"] = [round(b, 4) for b in r.belief_timeline]
    if hasattr(r, "action_log"):
        d["action_log"] = r.action_log
    if hasattr(r, "attempt_log"):
        d["attempt_log"] = r.attempt_log
    allure.attach(
        json.dumps(d, indent=2, default=str),
        name="result.json",
        attachment_type=allure.attachment_type.JSON,
    )
    if r.final_diff:
        allure.attach(r.final_diff, name="fix.diff", attachment_type=allure.attachment_type.TEXT)
    # Store on test node for summary collection
    if request is not None:
        request.node._agent_result_dict = d
    return d


# ---------------------------------------------------------------------------
# Simple Agent tests (one per bug)
# ---------------------------------------------------------------------------

@allure.parent_suite("agent_comparison")
@allure.suite("simple_agent")
@pytest.mark.parametrize("bug_id", ALL_BUGS)
def test_simple_agent(bug_id: str, request) -> None:
    """Simple Agent (retry loop) on each bug."""
    meta = BUG_METADATA[bug_id]
    allure.dynamic.sub_suite(meta["difficulty"])
    allure.dynamic.title(f"Simple Agent | {bug_id} ({meta['difficulty']})")
    allure.dynamic.description(f"Bug: {meta['description']}")

    with allure.step("Generate fix (max 3 retries)"):
        result = run_simple_agent(bug_id, LLM, COSTS, max_retries=3)

    with allure.step(f"{'FIXED' if result.fixed else 'NOT FIXED'} | cost={result.total_cost}"):
        _attach_result(result, request)

    assert result.total_cost > 0


@allure.parent_suite("agent_comparison")
@allure.suite("simple_agent_escalating")
@pytest.mark.parametrize("bug_id", ALL_BUGS)
def test_simple_agent_escalating(bug_id: str, request) -> None:
    """Simple Agent with escalating prompts on each bug."""
    meta = BUG_METADATA[bug_id]
    allure.dynamic.sub_suite(meta["difficulty"])
    allure.dynamic.title(f"Escalating Agent | {bug_id} ({meta['difficulty']})")
    allure.dynamic.description(f"Bug: {meta['description']}")

    with allure.step("Escalating prompts: direct → localized → test-guided"):
        result = run_simple_agent_escalating(bug_id, LLM, COSTS, max_retries=3)

    with allure.step(f"{'FIXED' if result.fixed else 'NOT FIXED'} | cost={result.total_cost}"):
        _attach_result(result, request)

    assert result.total_cost > 0


# ---------------------------------------------------------------------------
# Bayesian POMDP Agent tests — DP (Bellman recursion)
# ---------------------------------------------------------------------------

@allure.parent_suite("agent_comparison")
@allure.suite("bayes_dp")
@pytest.mark.parametrize("bug_id", ALL_BUGS)
def test_bayes_agent_dp(bug_id: str, request) -> None:
    """Bayesian POMDP Agent (full Bellman DP) on each bug."""
    meta = BUG_METADATA[bug_id]
    allure.dynamic.sub_suite(meta["difficulty"])
    allure.dynamic.title(f"Bayesian DP | {bug_id} ({meta['difficulty']})")
    allure.dynamic.description(f"Bug: {meta['description']}. Using full Bellman DP recursion.")

    with allure.step("Run Bayesian DP agent"):
        result = run_bayes_agent(bug_id, LLM, COSTS, use_dp=True)

    with allure.step(f"{'FIXED' if result.fixed else 'NOT FIXED'} | cost={result.total_cost}"):
        _attach_result(result, request)

    if result.belief_timeline:
        with allure.step("Belief timeline"):
            allure.attach(
                json.dumps({
                    "timeline": [round(b, 4) for b in result.belief_timeline],
                    "peak": round(max(result.belief_timeline), 4),
                    "final": round(result.belief_timeline[-1], 4),
                }, indent=2),
                name="belief_timeline.json",
                attachment_type=allure.attachment_type.JSON,
            )

    assert result.total_cost > 0


# ---------------------------------------------------------------------------
# Bayesian POMDP Agent tests — Greedy (one-step lookahead)
# ---------------------------------------------------------------------------

@allure.parent_suite("agent_comparison")
@allure.suite("bayes_greedy")
@pytest.mark.parametrize("bug_id", ALL_BUGS)
def test_bayes_agent_greedy(bug_id: str, request) -> None:
    """Bayesian POMDP Agent (greedy lookahead) on each bug."""
    meta = BUG_METADATA[bug_id]
    allure.dynamic.sub_suite(meta["difficulty"])
    allure.dynamic.title(f"Bayesian Greedy | {bug_id} ({meta['difficulty']})")
    allure.dynamic.description(f"Bug: {meta['description']}. Greedy one-step lookahead.")

    with allure.step("Run Bayesian greedy agent"):
        result = run_bayes_agent(bug_id, LLM, COSTS, use_dp=False)

    with allure.step(f"{'FIXED' if result.fixed else 'NOT FIXED'} | cost={result.total_cost}"):
        _attach_result(result, request)

    if result.belief_timeline:
        with allure.step("Belief timeline"):
            allure.attach(
                json.dumps({
                    "timeline": [round(b, 4) for b in result.belief_timeline],
                    "peak": round(max(result.belief_timeline), 4),
                    "final": round(result.belief_timeline[-1], 4),
                }, indent=2),
                name="belief_timeline.json",
                attachment_type=allure.attachment_type.JSON,
            )

    assert result.total_cost > 0


# ---------------------------------------------------------------------------
# Shared results collector (used by summary + charts)
# ---------------------------------------------------------------------------

_collected_results: list[dict] = []


@pytest.fixture(autouse=True)
def _collect_result(request):
    """After each agent test, stash its result for the summary charts."""
    yield
    # _attach_result stores the dict on the test node
    rd = getattr(request.node, "_agent_result_dict", None)
    if rd:
        _collected_results.append(rd)


# ---------------------------------------------------------------------------
# Summary comparison with publication-quality charts
# ---------------------------------------------------------------------------

@allure.parent_suite("agent_comparison")
@allure.suite("summary")
@allure.title("Aggregate comparison: all agents x all bugs")
def test_comparison_summary() -> None:
    """Generate aggregate statistics and charts from individual test results.

    Must run LAST (pytest runs it last because it's defined last).
    Uses results collected by the autouse fixture above.
    """
    import io
    from collections import defaultdict

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    rows = _collected_results
    if not rows:
        pytest.skip("No individual test results collected — run with -k 'not summary' first")

    # --- Attach raw data ---
    allure.attach(
        json.dumps(rows, indent=2, default=str),
        name="all_results.json",
        attachment_type=allure.attachment_type.JSON,
    )

    # --- Aggregate by policy ---
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)

    LABELS = {
        "simple_agent": "Simple\n(retry)",
        "simple_agent_escalating": "Simple\n(escalating)",
        "bayes_greedy": "Bayesian\n(greedy)",
        "bayes_pomdp": "Bayesian\n(DP)",
    }
    COLORS = {
        "simple_agent": "#e74c3c",
        "simple_agent_escalating": "#e67e22",
        "bayes_greedy": "#2ecc71",
        "bayes_pomdp": "#3498db",
    }
    order = ["simple_agent", "simple_agent_escalating", "bayes_greedy", "bayes_pomdp"]
    policies = [p for p in order if p in by_policy]

    agg = {}
    for p in policies:
        pr = by_policy[p]
        n = len(pr)
        fixed = sum(1 for r in pr if r["fixed"])
        agg[p] = {
            "n": n,
            "fixed": fixed,
            "fix_rate": fixed / n if n else 0,
            "avg_cost": sum(r["total_cost"] for r in pr) / n,
            "avg_utility": sum(r["utility"] for r in pr) / n,
            "total_utility": sum(r["utility"] for r in pr),
        }

    with allure.step("Aggregate statistics"):
        allure.attach(
            json.dumps({p: {k: (f"{v:.0%}" if k == "fix_rate" else round(v, 1))
                            for k, v in agg[p].items()} for p in policies}, indent=2),
            name="aggregate_by_policy.json",
            attachment_type=allure.attachment_type.JSON,
        )

    # --- Chart 1: Fix rate comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = [LABELS.get(p, p) for p in policies]
    colors = [COLORS.get(p, "#999") for p in policies]

    ax = axes[0]
    rates = [agg[p]["fix_rate"] * 100 for p in policies]
    bars = ax.bar(labels, rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fix Rate (%)", fontsize=12)
    ax.set_title("Fix Rate by Agent", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # --- Chart 2: Average cost ---
    ax = axes[1]
    costs = [agg[p]["avg_cost"] for p in policies]
    bars = ax.bar(labels, costs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Average Cost", fontsize=12)
    ax.set_title("Average Cost per Bug", fontsize=13, fontweight="bold")
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{cost:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # --- Chart 3: Average utility ---
    ax = axes[2]
    utils = [agg[p]["avg_utility"] for p in policies]
    bars = ax.bar(labels, utils, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Average Utility", fontsize=12)
    ax.set_title("Average Utility (Reward - Cost)", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    for bar, u in zip(bars, utils):
        ypos = bar.get_height() + 1 if u >= 0 else bar.get_height() - 3
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{u:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle(f"Agent Comparison — {len(rows)} tests, {len(ALL_BUGS)} bugs\n"
                 f"LLM: {_backend_name}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    allure.attach(buf.getvalue(), name="comparison_overview.png",
                  attachment_type=allure.attachment_type.PNG)

    # --- Chart 4: Fix rate by difficulty ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    difficulties = ["easy", "medium", "hard"]
    x = range(len(difficulties))
    width = 0.2
    for i, p in enumerate(policies):
        pr = by_policy[p]
        diff_rates = []
        for d in difficulties:
            d_rows = [r for r in pr if r.get("difficulty") == d]
            rate = sum(1 for r in d_rows if r["fixed"]) / len(d_rows) * 100 if d_rows else 0
            diff_rates.append(rate)
        offset = (i - len(policies) / 2 + 0.5) * width
        bars = ax2.bar([xi + offset for xi in x], diff_rates, width,
                       label=LABELS.get(p, p).replace("\n", " "),
                       color=COLORS.get(p, "#999"), edgecolor="black", linewidth=0.5)
        for bar, rate in zip(bars, diff_rates):
            if rate > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.capitalize() for d in difficulties], fontsize=12)
    ax2.set_ylabel("Fix Rate (%)", fontsize=12)
    ax2.set_title("Fix Rate by Bug Difficulty", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig2.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    allure.attach(buf2.getvalue(), name="fix_rate_by_difficulty.png",
                  attachment_type=allure.attachment_type.PNG)

    # --- Chart 5: Cost vs Utility scatter per bug ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for p in policies:
        pr = by_policy[p]
        costs_p = [r["total_cost"] for r in pr]
        utils_p = [r["utility"] for r in pr]
        marker = "o" if "simple" in p else "^"
        ax3.scatter(costs_p, utils_p, label=LABELS.get(p, p).replace("\n", " "),
                    color=COLORS.get(p, "#999"), marker=marker, s=60, alpha=0.7,
                    edgecolors="black", linewidth=0.5)
    ax3.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax3.set_xlabel("Total Cost", fontsize=12)
    ax3.set_ylabel("Utility (Reward - Cost)", fontsize=12)
    ax3.set_title("Cost vs Utility per Bug Instance", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    fig3.tight_layout()
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    allure.attach(buf3.getvalue(), name="cost_vs_utility_scatter.png",
                  attachment_type=allure.attachment_type.PNG)

    assert len(rows) > 0
