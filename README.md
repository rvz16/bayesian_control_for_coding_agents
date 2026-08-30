<div align="center">
  <h1>Bayesian Control for Coding Agents</h1>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.24453-b31b1b.svg)](https://arxiv.org/abs/2606.24453)

[Setup](#setup) | [Reproducing the paper](#reproducing-the-paper) | [Models & benchmarks](#models-and-benchmarks) | [Baseline policies](#baseline-policies) | [Citation](#citation)

---

Code release for the paper **"Bayesian Control for Coding Agents"**.

We formulate LLM code-generation orchestration as cost-sensitive sequential
hypothesis testing over candidate correctness, and derive two Bayesian
belief-state controllers (`bayesian_greedy`, `bayesian_DP`) from the Bellman
equation of the underlying POMDP. This repository contains the calibration,
policy, and evaluation code used to reproduce the empirical results across
six generators and nine coding benchmarks, plus the uncertainty-quantification
experiment with the SAGE agent.

## Repository layout

```
experiments/orchestration_hypothesis_testing/
    _common/         cost, critic, generator, kernel, and logprob primitives
    calibration/     per-benchmark calibration of prior, critic likelihoods,
                     and refinement-transition kernel (LCB, MBPP+, HumanEval+,
                     HumanEvalFix, CodeContests, SWE-Bench Lite/Verified)
    iter/            Self-Refine, Reflexion, and Bayesian iterative refinement
    scripts/         entry points:
                       spot_check_generators.py — single-shot calibration draws
                       run_fitted_live.py       — online Bayesian policy
                                                  evaluation (greedy/DP)
                       run_sage_baseline.py     — SAGE self-consistency baseline
                       score_sage_uhead.py      — post-hoc SAGE uncertainty
                       bootstrap_lcb_uq_prr_table.py  — Table 1 (PRR)
                       experiment2_uq_bayes_critic.py — critic-belief evaluation
                       aggregate_trajectory_uq.py     — DeepSeek/OpenRouter logprobs
                       fitted_live/             — live-policy adapters
    iter/replay_baselines.py  — cached-artifact replay of stateless baselines
                                (always_verify, best_of_N, gate(Cr_*),
                                 fixed_pipeline, self_refine, reflexion)
    analysis/        controller, regime-map sweeps, sensitivity, bootstrap CI
    tools/, paper/, tests/

bayesian_optimization_for_code_testing/agent-bugfix-bayes/
    src/, scripts/, configs/, tests/  — Bayesian bug-fixing agent (ABBO)

sage_agent/          SAGE agent used in the uncertainty-quantification
                     experiment (Section 6.6 of the paper)

different_agents/v4/ LangGraph SAGE-v4 harness and LCB LLM-tool agent
```

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional environment variables:

- `OPENROUTER_API_KEY` — for closed-API generators (gpt-5-mini, qwen3-coder,
  claude-haiku-4.5, claude-sonnet-4.5) and the LLM critic
- `HF_HOME` — HuggingFace cache directory (defaults to `~/.cache/huggingface`)
- `SWEBENCH_NAMESPACE=none` — force local Docker builds for the SWE-Bench harness
- `HF_TOKEN` — optional, for higher HuggingFace Hub rate limits

Copy `.env.example` (if provided) to `.env` and fill in the keys.

## Reproducing the paper

The full pipeline runs in three stages: **generate → calibrate → evaluate**.

### 1. Generate calibration candidates

For each `(benchmark, generator)` cell, draw `n=3` independent single-shot
patches with a fixed seed:

```bash
python experiments/orchestration_hypothesis_testing/scripts/spot_check_generators.py \
    --dataset <benchmark_name> \
    --generators <gen1,gen2,...> \
    --n-instances <N> \
    --n-patches 3 \
    --seed 42 \
    --output-dir data/<benchmark>_calibration \
    --max-cost-usd-per-model 30
```

Datasets: `princeton-nlp/SWE-bench_Lite`, `princeton-nlp/SWE-bench_Verified`,
`livecodebench/code_generation_lite`, `evalplus/mbppplus`,
`evalplus/humanevalplus`, `bigcode/humanevalpack`,
`deepmind/code_contests`.

Generators (from `_common/generators.py`): `gpt5_mini`, `qwen3_coder`,
`haiku45`, `sonnet45`, `qwen25_32b`, `gpt_oss_20b`.

### 2. Calibrate the Bayesian controllers

Compute per-cell priors, critic likelihoods, and the refinement-transition
kernel from the 75% calibration split; the disjoint 25% split is held out for
evaluation.

```bash
python experiments/orchestration_hypothesis_testing/calibration/from_spotcheck.py \
    --output-dir data/<benchmark>_calibration \
    --generators <gen1,gen2,...> \
    --dataset <benchmark_name> \
    --max-cost-usd-per-model 5
```

`--output-dir` points at the calibration draws produced in step 1;
`from_spotcheck.py` reads the pre-generated patches, runs the syntax/public-test
/LLM critics, and writes the prior, critic likelihoods, and transition kernel
under the same directory. Use `--skip-l3` to skip the LLM critic.

### 3. Evaluate policies

**Bayesian controllers** (`bayesian_greedy`, `bayesian_DP`) — evaluate online
on the 25% held-out split with the calibrated prior/likelihoods/kernel:

```bash
python experiments/orchestration_hypothesis_testing/scripts/run_fitted_live.py \
    --benchmark <benchmark_name> \
    --generators <gen1,gen2,...> \
    --policies greedy_fitted,dp_fitted \
    --calibration-dir data/<benchmark>_calibration \
    --output-dir data/<benchmark>_fitted_live \
    --c-gen 10 --c-l0 1 --c-l2 2 --c-l3 5 --c-ver 30 --reward 100
```

**Baseline policies** (`always_verify`, `best_of_N`, `gate(Cr_*)`,
`fixed_pipeline`, `self_refine`, `reflexion`) — cached-artifact replay over
the same held-out split:

```bash
python experiments/orchestration_hypothesis_testing/iter/replay_baselines.py \
    --calibration-dir data/<benchmark>_calibration \
    --generators <gen1,gen2,...>
```

> **Evaluator caps disclosure.** `run_fitted_live.py` defaults to
> `--lcb-private-test-cap 12` (LiveCodeBench uses the first 12 private tests
> per problem) and `--plus-input-cap 200` (EvalPlus HumanEval+/MBPP+ use the
> first 200 PLUS inputs). Override at the command line if you want the full
> hidden suites.

### Additional experiments

- **SAGE uncertainty quantification** (paper Table 1 / Section 6.6):

  ```bash
  python experiments/orchestration_hypothesis_testing/scripts/run_sage_baseline.py ...
  python experiments/orchestration_hypothesis_testing/scripts/score_sage_uhead.py ...
  python experiments/orchestration_hypothesis_testing/scripts/bootstrap_lcb_uq_prr_table.py
  ```

- **Cost-regime sweep** (paper Figure 4):

  ```bash
  python experiments/orchestration_hypothesis_testing/analysis/cver_sensitivity_sweep.py
  ```

## Models and benchmarks

Six generators (see Appendix A.1 of the paper):

| Access     | Model                       |
| ---------- | --------------------------- |
| Closed-API | gpt-5-mini                  |
|            | qwen3-coder                 |
|            | claude-haiku-4.5            |
|            | claude-sonnet-4.5           |
| Open-weight| Qwen2.5-Coder-32B-Instruct  |
|            | gpt-oss-20b                 |

Nine benchmarks (see Appendix A.2):

- Function-level synthesis: LCB-hard (102), LCB-medium (207), LCB-easy (135),
  MBPP+ (378), HumanEval+ (164), CodeContests (165)
- Repository-level patch generation: SWE-Bench Lite (300), SWE-Bench Verified (500)
- Bug-fixing: HumanEvalFix (164)

## Baseline policies

Ten policies, grouped into two families:

- **Stateless per-patch**: `always_verify`, `best_of_3`, `gate(Cr_syn)`,
  `gate(Cr_test)`, `gate(Cr_llm)`, `fixed_pipeline`
- **Iterative refinement**: `self_refine`, `reflexion`
- **Bayesian** (this work): `bayesian_greedy`, `bayesian_DP`

Utility for a policy π is `U(π) = R·Y_T − Σ_t C(a_t)` with `R=100`.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{bayesiancontrol2026,
  title     = {Bayesian Control for Coding Agents},
  author    = {Papamarkou, Theodore and Smirnov, Vladislav and Mazanov, Viktor and Vazhentsev, Artem and Nakov, Preslav and Baldwin, Timothy and Shelmanov, Artem},
  booktitle = {Preprint},
  year      = {2026},
  eprint    = {2606.24453},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/pdf/2606.24453}
}
```

## License

MIT — see [LICENSE](LICENSE).
