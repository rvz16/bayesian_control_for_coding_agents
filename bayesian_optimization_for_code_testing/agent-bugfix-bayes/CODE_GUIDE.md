# Bayesian POMDP for Adaptive Code Bug Fixing — Code Guide

**Repository:** [github.com/rvz16/agents_with_uncertainty_research](https://github.com/rvz16/agents_with_uncertainty_research)

**Base path:** `bayesian_optimization_for_code_testing/agent-bugfix-bayes/`

---

## Core Algorithm

### Bayesian Agent (`bayes_agent.py`)

The main POMDP implementation — Bellman recursion over belief states.

| Component | Description | Link |
|---|---|---|
| `DPPlanner` class | Full Bellman DP recursion, pre-computes optimal policy over ~19K states | [bayes_agent.py#L155](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/bayes_agent.py#L155) |
| `bayes_update()` | Posterior belief update: `b' = P(z∣Y=1)·b / [P(z∣Y=1)·b + P(z∣Y=0)·(1-b)]` | [bayes_agent.py#L100](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/bayes_agent.py#L100) |
| `generator_transition()` | Belief transition after LLM generates a fix: `b' = b*(1-p10) + (1-b)*p01` | [bayes_agent.py#L115](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/bayes_agent.py#L115) |
| `_run_dp_loop()` | Executes optimal policy: observe → update belief → choose best action | [bayes_agent.py#L331](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/bayes_agent.py#L331) |
| `_run_greedy_loop()` | One-step greedy lookahead baseline (Bayes Greedy agent) | [bayes_agent.py#L485](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/bayes_agent.py#L485) |

### Simple Agents (`simple_agent.py`)

Baseline agents without Bayesian belief tracking.

| Component | Description | Link |
|---|---|---|
| `run_simple_agent()` | Retry same prompt 3x, always run full tests | [simple_agent.py#L137](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/simple_agent.py#L137) |
| `run_simple_agent_escalating()` | Change prompt strategy each retry, still always full tests | [simple_agent.py#L233](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/simple_agent.py#L233) |

---

## Bug Definitions (`real_bugs.py`)

20 single-line Python bugs in a log parser module (easy/medium/hard).

| Component | Description | Link |
|---|---|---|
| `LOG_PARSER_CLEAN` | Correct source code (ground truth) | [real_bugs.py#L19](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/real_bugs.py#L19) |
| `BUGGY_SOURCES_REAL` | 20 buggy variants (single-line mutations) | [real_bugs.py#L109](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/real_bugs.py#L109) |
| `REAL_CRITIC_TESTS` | 4 cheap diagnostic test subsets (parsing, filtering, aggregation, context) | search for `REAL_CRITIC_TESTS` in [real_bugs.py](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/real_bugs.py) |
| `REAL_ARM_PROMPTS` | 4 LLM prompt templates (generator arms) | search for `REAL_ARM_PROMPTS` in [real_bugs.py](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/real_bugs.py) |

---

## Infrastructure

### LLM Provider (`llm_provider.py`)

| Component | Description | Link |
|---|---|---|
| `call_llm()` | Unified LLM interface (Ollama + OpenAI-compatible) | [llm_provider.py#L41](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/llm_provider.py#L41) |
| `_call_openai()` | OpenAI-compatible endpoint support (vLLM, TGI) | [llm_provider.py#L104](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/src/abbo/realworld/agents/llm_provider.py#L104) |

---

## Tests

| Test file | Count | Description | Link |
|---|---|---|---|
| `test_agent_comparison.py` | 81 | Full comparison: 4 agents x 20 bugs + summary charts | [test_agent_comparison.py](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/tests/test_agent_comparison.py) |
| `test_bayes_model.py` | 33 | DP planner unit tests (discretization, Bellman values, policy) | [test_bayes_model.py](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/tests/test_bayes_model.py) |
| `test_real_bugs.py` | 54 | Bug validation (clean source, mutations, critic subsets, prompts) | [test_real_bugs.py](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/tests/test_real_bugs.py) |

### Key test functions

| Function | Description | Link |
|---|---|---|
| `test_simple_agent()` | Simple agent on all 20 bugs | [test_agent_comparison.py#L111](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/tests/test_agent_comparison.py#L111) |
| `test_simple_agent_escalating()` | Escalating agent on all 20 bugs | [test_agent_comparison.py#L130](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/tests/test_agent_comparison.py#L130) |
| `test_bayes_agent_dp()` | Bayesian DP agent on all 20 bugs | [test_agent_comparison.py#L153](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/tests/test_agent_comparison.py#L153) |
| `test_bayes_agent_greedy()` | Bayesian Greedy agent on all 20 bugs | [test_agent_comparison.py#L188](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/tests/test_agent_comparison.py#L188) |
| `test_comparison_summary()` | Generates matplotlib charts (fix rate, cost, utility) | [test_agent_comparison.py#L240](https://github.com/rvz16/agents_with_uncertainty_research/blob/main/bayesian_optimization_for_code_testing/agent-bugfix-bayes/tests/test_agent_comparison.py#L240) |

---

## How to Run

```bash
# Unit tests (fast, no LLM needed)
pytest tests/test_bayes_model.py tests/test_real_bugs.py -v

# Full agent comparison (requires Ollama with qwen2.5:7b)
pytest tests/test_agent_comparison.py --alluredir allure-results -v

# View results in Allure
allure serve allure-results
```

---

## Key Formula

```
V(b, s) = max_a [ -C_a + E[ V(b', s') ] ]
```

Where `b` = belief that current fix is correct, `s` = remaining resources (generators, critics, verifications).

The `DPPlanner` discretizes belief into 101 grid points and solves ~19,392 states via memoized recursion in milliseconds.
