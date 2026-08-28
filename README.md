# Bayesian Control for Coding Agents

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
                       run_fitted_live.py       — online policy evaluation
                       run_sage_baseline.py     — SAGE self-consistency baseline
                       score_sage_uhead.py      — post-hoc SAGE uncertainty
                       bootstrap_lcb_uq_prr_table.py  — Table 1 (PRR)
                       experiment2_uq_bayes_critic.py — critic-belief evaluation
                       aggregate_trajectory_uq.py     — DeepSeek/OpenRouter logprobs
                       fitted_live/             — live-policy adapters
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
`evalplus/humanevalplus`, `Muennighoff/humanevalpack`,
`deepmind/code_contests`.

Generators (from `_common/generators.py`): `gpt5_mini`, `qwen3_coder`,
`haiku45`, `sonnet45`, `qwen25_32b`, `gpt_oss_20b`.

### 2. Calibrate the Bayesian controllers

Compute per-cell priors, critic likelihoods, and the refinement-transition
kernel from the 75% calibration split; the disjoint 25% split is held out for
evaluation.

```bash
python experiments/orchestration_hypothesis_testing/calibration/from_spotcheck.py \
    --data-dir data/<benchmark>_calibration \
    --generators <gen1,gen2,...> \
    --dataset <benchmark_name>
```

### 3. Evaluate policies

Replay the ten policies (see paper Table 7) over the 25% held-out split:

```bash
python experiments/orchestration_hypothesis_testing/scripts/run_fitted_live.py \
    --config configs/<benchmark>_<generator>.yaml
```

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
  MBPP+ (378), HumanEval+ (164)
- Repository-level patch generation: SWE-Bench Lite (300), SWE-Bench Verified (500)
- Bug-fixing: HumanEvalFix (164), CodeContests (165)

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
