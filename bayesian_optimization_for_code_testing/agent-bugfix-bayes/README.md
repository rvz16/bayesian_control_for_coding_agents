# agent-bugfix-bayes

A research artifact comparing **Bayesian decision-theoretic orchestration** against fixed heuristic orchestration for agent-driven bug fixing. The Bayesian component here is *sequential decision-making and hypothesis testing over bug-fixing actions* — it is not generic Gaussian-process Bayesian optimization. The controller maintains a posterior belief over whether the current candidate patch is correct and uses value-of-information calculations to decide whether to run another diagnostic, generate a new patch, or commit to expensive verification.

## Quick Start

```bash
# 1. Set up environment
bash scripts/setup_env.sh
source .venv/bin/activate

# 2. Run synthetic benchmark (exact + Monte Carlo)
bash scripts/run_synthetic.sh

# 3. Run toy benchmark
python -m abbo.cli calibrate-toy --trials 10000
python -m abbo.cli run-toy --episodes 10000 --seed 7

# 4. Run tests
pytest -q --alluredir allure-results --clean-alluredir
```

## Benchmarks

### Synthetic Benchmark

Exact reproduction of the paper's synthetic environment with three hidden bug classes, three diagnostic tests, and three targeted patches. Computes exact expected utilities using symbolic arithmetic (`fractions.Fraction`) and validates via Monte Carlo simulation.

```bash
# Exact symbolic computation
python -m abbo.cli run-synthetic-exact

# Monte Carlo simulation (100k episodes)
python -m abbo.cli run-synthetic --episodes 100000 --seed 7
```

**Expected results:**
- Bayes optimal expected utility: 391/3 ≈ 130.33
- One-test MAP expected utility: 268/3 ≈ 89.33

### Toy Benchmark

A real executable codebase (request router) with three injected bug families preserving the same information structure as the synthetic benchmark.

```bash
# Calibrate diagnostic failure matrix
python -m abbo.cli calibrate-toy --trials 10000

# Run evaluation
python -m abbo.cli run-toy --episodes 10000 --seed 7
```

### BugsInPy Pilot (Real-World)

Two-stage architecture: (A) build a candidate bank with live LLM calls, then (B) replay offline for fair policy comparison.

```bash
# Build candidate bank (requires API keys and BugsInPy installation)
python -m abbo.cli build-candidate-bank --benchmark bugsinpy --config configs/bugsinpy_pilot.yaml

# Fit Bayesian model from calibration split
python -m abbo.cli fit-bayes-model --benchmark bugsinpy --config configs/bugsinpy_pilot.yaml

# Run offline evaluation (no live calls)
python -m abbo.cli run-benchmark --benchmark bugsinpy --policy all

# Generate tables and plots
python -m abbo.cli summarize-results --benchmark bugsinpy
```

## Policies Compared

| Policy | Description |
|--------|-------------|
| `bayes` | Bayesian decision-theoretic controller (value-of-information) |
| `h1_one_shot` | Direct fix once, then verify |
| `h2_fixed_workflow` | trigger_tests → localized_fix → verify |
| `h3_two_stage_fixed` | direct_fix → trigger_tests → conditional regen → verify |
| `h4_threshold` | Hand-tuned threshold policy |
| `h5_single_critic_then_map` | One cheap critic, best generator arm, verify |

## Allure Reports

```bash
# Run tests with Allure output
pytest -q --alluredir allure-results --clean-alluredir

# View report (requires allure CLI)
allure serve allure-results
```

## Project Structure

```
agent-bugfix-bayes/
  configs/          # YAML configuration files
  src/abbo/         # Main package
    core/           # Types, costs, metrics, statistics
    synthetic/      # Synthetic benchmark (exact + MC)
    toy/            # Toy executable codebase benchmark
    realworld/      # BugsInPy/Defects4J adapters, candidate bank, calibration
  tests/            # Pytest test suite
  scripts/          # Shell scripts for running experiments
  data/             # Raw data, candidate banks, splits
  results/          # Generated results (CSV, JSON, plots)
  allure-results/   # Allure test report data
```

## Known Limitations

- BugsInPy adapter requires a local BugsInPy installation and compatible Python versions per project.
- Defects4J adapter is optional and requires Java/JDK setup.
- Candidate bank building requires LLM API keys; evaluation is fully offline.
- The Bayesian controller uses a discretized belief grid (resolution 0.01) which may lose precision for extreme beliefs.
- Transition model estimates (p01, p10) are global; per-repo conditioning is not yet implemented.

## Reproducibility

- All random operations use explicit seeds passed through numpy RandomState/Generator.
- Exact symbolic results use `fractions.Fraction` to avoid floating-point drift.
- Evaluation never makes live API calls — it replays pre-built candidate banks.
- All configuration is in YAML files under `configs/`.
- Bootstrap confidence intervals use fixed seeds for reproducibility.
