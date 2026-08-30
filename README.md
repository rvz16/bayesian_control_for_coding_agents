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

All commands assume you `cd experiments/orchestration_hypothesis_testing/` first.

### Prerequisites

- **`.env`** at repo root with `OPENROUTER_API_KEY=sk-or-v1-...` (auto-loaded
  by the entry scripts).
- **Docker or Podman** for SWE-Bench Phase 2 (harness eval) on `x86_64`.
  Podman users:
  ```bash
  export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
  export SWEBENCH_PODMAN_COMPAT=1
  ```
- **Local vLLM** endpoints for open-weight generators (optional): `qwen25_32b`
  on port `8003`, `gpt_oss_20b` on the port you serve it on.

### One-command reproduction — function-level synthesis + SWE-Bench

The end-to-end launcher `scripts/run_all_fitted_live.sh` runs calibration
followed by the fitted Bayesian controllers (`greedy_fitted`, `dp_fitted`)
across all synthesis + SWE benchmarks:

```bash
cd experiments/orchestration_hypothesis_testing

bash scripts/run_all_fitted_live.sh
# environment knobs (all optional):
#   TASKS=lcb_hard,lcb_medium,lcb_easy,mbpp,humaneval,swebench_lite,swebench_verified
#   MODE=all|calibrate|live
#   CALIB_N=100
#   GEN_CAPS=qwen3_coder=4.0,haiku45=4.0,sonnet45=15.0,gpt5_mini=4.0
#   LIVE_CAPS=qwen3_coder=20.0,haiku45=20.0,sonnet45=50.0,gpt5_mini=20.0
#   USE_PODMAN=1
```

### SWE-Bench full pipeline (Lite + Verified) — per generator

The 10-step pipeline used to produce the SWE-Bench cells in Table 7 (shown
here for `qwen3_coder`; repeat with `--generators sonnet45`, `haiku45`,
`gpt5_mini`):

```bash
cd experiments/orchestration_hypothesis_testing

# 1. Calibration draws — SWE-Bench Lite (300 instances × 3 patches)
python scripts/spot_check_generators.py \
    --dataset princeton-nlp/SWE-bench_Lite \
    --n-instances 300 --n-patches 3 \
    --generators qwen3_coder \
    --output-dir data/swebench_lite_calibration_full \
    --max-cost-usd-per-model qwen3_coder=15

# 2. Calibration draws — SWE-Bench Verified (500 instances × 3 patches)
python scripts/spot_check_generators.py \
    --dataset princeton-nlp/SWE-bench_Verified \
    --n-instances 500 --n-patches 3 \
    --generators qwen3_coder \
    --output-dir data/swebench_verified_calibration_full \
    --max-cost-usd-per-model qwen3_coder=25

# 3. Self-Refine trajectory — Lite (5 refinement steps)
python iter/refine_swe.py --method selfrefine \
    --dataset princeton-nlp/SWE-bench_Lite \
    --src-dir data/swebench_lite_calibration_full \
    --output-dir data/swebench_lite_realbaselines_selfrefine_full \
    --generators qwen3_coder \
    --n-instances 300 --steps 5 --max-workers 1

# 4. Evaluate SR trajectory — Lite
python iter/eval_steps.py \
    --gen qwen3_coder \
    --data-dir data/swebench_lite_realbaselines_selfrefine_full \
    --dataset princeton-nlp/SWE-bench_Lite \
    --n-steps 5

# 5-6. Self-Refine trajectory + eval — Verified (500 instances)
python iter/refine_swe.py --method selfrefine \
    --dataset princeton-nlp/SWE-bench_Verified \
    --src-dir data/swebench_verified_calibration_full \
    --output-dir data/swebench_verified_realbaselines_selfrefine_full \
    --generators qwen3_coder \
    --n-instances 500 --steps 5 --max-workers 1
python iter/eval_steps.py \
    --gen qwen3_coder \
    --data-dir data/swebench_verified_realbaselines_selfrefine_full \
    --dataset princeton-nlp/SWE-bench_Verified \
    --n-steps 5

# 7-8. Reflexion trajectory + eval — Lite
python iter/refine_swe.py --method reflexion \
    --dataset princeton-nlp/SWE-bench_Lite \
    --src-dir data/swebench_lite_calibration_full \
    --output-dir data/swebench_lite_realbaselines_reflexion_full \
    --generators qwen3_coder \
    --n-instances 300 --steps 5 --max-workers 1
python iter/eval_steps.py \
    --gen qwen3_coder \
    --data-dir data/swebench_lite_realbaselines_reflexion_full \
    --dataset princeton-nlp/SWE-bench_Lite \
    --n-steps 5

# 9-10. Reflexion trajectory + eval — Verified
python iter/refine_swe.py --method reflexion \
    --dataset princeton-nlp/SWE-bench_Verified \
    --src-dir data/swebench_verified_calibration_full \
    --output-dir data/swebench_verified_realbaselines_reflexion_full \
    --generators qwen3_coder \
    --n-instances 500 --steps 5 --max-workers 1
python iter/eval_steps.py \
    --gen qwen3_coder \
    --data-dir data/swebench_verified_realbaselines_reflexion_full \
    --dataset princeton-nlp/SWE-bench_Verified \
    --n-steps 5
```

The same 10 steps produce all SWE-Bench cells for the other generators —
substitute `--generators sonnet45` (with a larger `--max-cost-usd-per-model`
cap, e.g. `sonnet45=50`), `haiku45`, or `gpt5_mini`.

### Function-level synthesis pipeline (LCB, MBPP+, HumanEval+)

Same three-phase shape (calibration draws → critic scoring → policy replay)
but the oracle is a subprocess test runner, not the Docker harness:

```bash
cd experiments/orchestration_hypothesis_testing

# 1. Calibration draws (per benchmark)
python scripts/spot_check_generators.py \
    --dataset livecodebench/code_generation_lite \
    --n-instances 200 --n-patches 3 \
    --generators qwen3_coder,haiku45,sonnet45,gpt5_mini,qwen25_32b \
    --output-dir data/lcb_calibration_full \
    --max-cost-usd-per-model qwen3_coder=4,haiku45=4,sonnet45=15,gpt5_mini=4

# 2. Critic scoring + likelihoods + transition kernel
python calibration/from_spotcheck.py \
    --output-dir data/lcb_calibration_full \
    --generators qwen3_coder,haiku45,sonnet45,gpt5_mini,qwen25_32b \
    --dataset livecodebench/code_generation_lite

# 3. Bayesian controllers online (greedy_fitted, dp_fitted)
python scripts/run_fitted_live.py \
    --benchmark lcb_medium \
    --generators qwen3_coder,haiku45,sonnet45,gpt5_mini,qwen25_32b \
    --policies greedy_fitted,dp_fitted \
    --calibration-dir data/lcb_calibration_full \
    --output-dir data/lcb_fitted_live
```

Repeat with `--benchmark lcb_hard`, `lcb_easy`, `mbpp`, `humaneval` (see
`run_all_fitted_live.sh` for the exhaustive loop).

### Bug-fixing pipeline (HumanEvalFix)

HumanEvalFix uses the same `iter/refine_swe.py` + `iter/eval_steps.py`
pattern via `--dataset bigcode/humanevalpack`. CodeContests is handled by
the ABBO pipeline in `bayesian_optimization_for_code_testing/agent-bugfix-bayes/`.

### Uncertainty-quantification experiment (Table 1, Section 6.6)

```bash
cd experiments/orchestration_hypothesis_testing

# Assumes a running OpenAI-compatible vLLM endpoint for gpt-oss-20b:
#   GENERATOR_KEY=gpt_oss_20b_local
#   GPT_OSS_20B_BASE_URL=http://127.0.0.1:8004/v1

bash scripts/run_sage_uncertainty_experiments.sh
# environment knobs:
#   BENCHMARKS=lcb_hard,lcb_medium,...
#   GENERATOR_KEY=gpt_oss_20b_local
#   TRAIN_FRACTION=0.25
#   PLUS_INPUT_CAP=200
```

Aggregation into the PRR table:

```bash
python scripts/bootstrap_lcb_uq_prr_table.py
```

### Cost-regime sweep (Figure 4)

```bash
python analysis/cver_sensitivity_sweep.py
```

### Evaluator caps disclosure

- **LiveCodeBench**: `--lcb-private-test-cap 12` (first 12 private tests per problem).
- **HumanEval+ / MBPP+**: `--plus-input-cap 200` (first 200 PLUS inputs).
- **CodeContests**: public tests capped at 10, final label uses the first 30
  available tests.

Override at the command line if you want the full hidden suites.

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
