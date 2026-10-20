# Diversity of the Bayesian controllers' action traces

CPU-only replay of the paper's cached runs (6 generators x 9 benchmarks = 54 cells, 3 cached patches per
task, 25% eval split). It asks how varied the action sequences of Bayes-greedy (BG) and Bayes-DP (BDP) are,
whether simple fixes or per-task beliefs change that, and whether BDP beats simple rules. No LLM, API or GPU
calls. Each subfolder has a `SUMMARY.md` with the full method, numbers and caveats.

## Headline findings

- **The paper controller is close to a lookup table.** BDP's first action is constant within every cell
  (it varies in 0/54 cells); 30/54 cells have a single action sequence, and there are 18 distinct BDP
  sequences pooled over all cells. BDP's trace equals one simple rule's on every task in 52/54 cells (ignoring repeated critic calls; 41/54 strictly).
- **BDP never beats the best per-cell rule:** 0/54 wins (44 ties, 10 losses), in every one of the 31
  variants of `diversity_fixes/`. It beats the best single rule over all cells (gate(syn)) by +12.1.
- **Per-task beliefs make it diverse, not better.** Starting from a per-task/per-patch belief built from
  free signals (patch length, task features), the first action varies in 36–38/54 cells and pooled
  sequences rise from 18 (paper controller) or 30 (the matched no-signal base `calK_const`) to 74 (`calK_jointcomb`). Utility vs train-picked fair rules moves by about
  +0.1 per task.
- **SWE at intermediate verifier cost is the one place BDP pays.** At C_ver 10–20, BDP beats the
  signal-aware rules by +1.6…+2.9 with slow-regime critic costs, +1.3…+2.2 with all critics at 1 (pooled CIs exclude 0). BG with the same beliefs does as well, so the gain
  comes from the per-task belief, not from DP lookahead.
- **BG bug.** `make_greedy_policy` never updated the belief on `generate`, so a new cached patch inherited
  the previous patch's post-FAIL belief while `select_action` valued a fresh patch at the prior. BG then
  regenerated until patches ran out without checking them (12.2% of BG episodes, 47% on LCB-medium).
  Resetting to the prior raises BG from +9.75 to +11.73 vs always_verify (paper regime, 54 cells), and the
  number of settings where DP beats BG by more than 10 drops from 13 to 9.

## Code changes outside this folder

- `analysis/lcb_compare.py`: `make_greedy_policy(..., reset_on_generate=True)` resets the belief to the
  prior on `generate`. `reset_on_generate=False` reproduces the camera-ready BG numbers;
  `action_traces/scripts/replay_traces.py` uses it so the replay matches the paper.
- `analysis/controller.py`: `simulate_policy` also returns the action list (`"actions"`).

## Layout

| Path | Contents |
|---|---|
| `action_traces/` | Replay with the repo's own controllers; protocol inference, paper reproduction check, trace diversity, rules vs BDP |
| `diversity_fixes/` | Re-implemented engine with switchable fixes (0a BG reset, 0b critic masking, 0c BDP reset) and a repair action; regression gate |
| `per_task_prior/` | Per-task / per-patch starting beliefs from free signals; calibration and conditional-independence checks |
| `swe_cver_sweep/` | Verifier-cost sweep (C_ver 5–90) on SWE and the other benchmarks |
| `scripts/make_traces_md.py`, `AGENT_TRACES.md` | Readable example traces per benchmark |
| `cache/cc_test_desc.json`, `per_task_prior/cache/lcb_meta.jsonl` | Third-party problem statements and LCB metadata; **not committed**, fetched locally (see Reproduce) |
| `fetch_artifacts.py` | Downloads the pinned W&B artifacts |

Large or regenerable outputs (`traces.jsonl`, `runs/`, `runs_fineq/`, the big `summary.csv` files,
`per_task_prior/signals.csv`, `*.pkl`) are not committed; the commands below regenerate them.

## Data provenance

- **Cached runs:** W&B project `nlpresearch.group/orchestration-hypothesis-testing`. Calibration,
  Self-Refine and Reflexion versions per cell are pinned (with digests) in `action_traces/versions.json`;
  the max-overlap Self-Refine versions used by `step1mx_*` are in `diversity_fixes/chain_overlap.csv`.
  All are paper-era (created on or before 2026-06-22); the 2026-08 SWE-Verified expansion is not used.
- **Task metadata** (`per_task_prior` only): public HF datasets `evalplus/mbppplus`,
  `bigcode/humanevalpack`, `deepmind/code_contests`, `princeton-nlp/SWE-bench_Lite`,
  `princeton-nlp/SWE-bench_Verified` from the local HF hub cache (`HF_HUB_CACHE`); LCB metadata is
  fetched into `per_task_prior/cache/lcb_meta.jsonl` (not committed: it holds full problem statements).
- **Paper values:** parsed from the camera-ready LaTeX (`full_results.tex`, `prior_pass_rate.tex`,
  `transition_kernel.tex`), only for the reproduction check.

## Reproduce

Environment variables:
- `BCCA_DATA_DIR`: artifact directory (default `<repo root>/data/diversity_artifacts`, git-ignored).
- `PAPER_TEX_DIR`: camera-ready LaTeX folder. Optional; without it `run_all.py` skips the paper
  comparison and keeps the committed `versions.json` / `reproduction_check.csv`.

Python deps: numpy, scipy, scikit-learn, pyarrow, matplotlib, wandb (fetch only), datasets (traces doc only).
Run from this directory; every step is CPU-only.

```bash
python3 fetch_artifacts.py                       # 180 artifacts, ~100 MB; --check verifies, --diagnostics adds
                                                 # the extra versions used by the protocol-search scripts
# 1. Paper-controller replay (writes traces.jsonl, needed by the gate)
python3 -B action_traces/scripts/run_all.py action_traces

# 2. Fixes (from diversity_fixes/)
cd diversity_fixes
python3 -B scripts/run_gate.py regression_gate.json   # must PASS: 47,304 episodes reproduced
python3 -B scripts/run_variants.py runs
python3 -B scripts/analyze.py runs .
python3 -B scripts/repair_effect.py runs step1_vs_step0.csv
python3 -B scripts/chain_overlap.py chain_overlap.csv
python3 -B scripts/validate_dp.py dp_consistency_check.json
cd ..

# 3. Per-task beliefs (from per_task_prior/)
cd per_task_prior
python3 -B ../diversity_fixes/scripts/run_gate.py logs/regression_gate.json
python3 -I -B scripts/signals.py                  # signals.csv, auroc.csv (needs the HF datasets above)
python3 -I -B scripts/run_ptp.py runs             # skips existing (FORCE=1 to redo)
python3 -I -B scripts/check_const.py runs logs/const_signal_check.json
python3 -I -B scripts/check_joint.py logs/joint_dp_check.json
python3 -I -B scripts/ci_check.py "$PWD/ci_check.csv"
python3 -I -B scripts/analyze_ptp.py runs .
cd ..

# 4. Verifier-cost sweep (from swe_cver_sweep/)
cd swe_cver_sweep
python3 -I -B scripts/sweep.py "$PWD/runs" all
python3 -I -B scripts/sweep.py "$PWD/runs_fineq" swe_fineq
python3 -I -B scripts/check_repro.py runs ../per_task_prior/runs logs/repro_check.json
python3 -I -B scripts/analyze_sweep.py runs . ../per_task_prior runs_fineq
python3 -I -B scripts/plot_sweep.py .
python3 -I -B scripts/examples.py .
cd ..

# 5. Example-trace document
python3 -B scripts/make_traces_md.py              # rewrites AGENT_TRACES.md
```

Before running `per_task_prior/`, fetch `per_task_prior/cache/lcb_meta.jsonl` with
`per_task_prior/scripts/fetch_lcb_meta.py`; before `scripts/make_traces_md.py`, fetch `cache/cc_test_desc.json`
with `scripts/fetch_cc_desc.py` (both network; third-party problem statements, kept out of git).
`action_traces/scripts/{repro,prior_test,kernel_test,diag_colleague}.py` are the protocol-search
diagnostics; they need `PAPER_TEX_DIR` and `fetch_artifacts.py --diagnostics`.

## Caveats

- **The protocol was inferred, not found in code.** No repo script produces `full_results.tex`; the split,
  Laplace prior, Beta(1,1) likelihoods, Self-Refine kernel and cost regimes were inferred by matching the
  paper tables. BG and BDP match the paper in 43/54 cells; the other 11 are included and flagged
  (`reproduces_paper_BG_BDP` in `action_traces/versions.json`), and results are reported for both sets.
- The oracle rule per cell is picked on eval with hindsight (an upper bound for rules); the
  train-picked per-benchmark rule is the fair comparison.
- `file:line` references in the `SUMMARY.md` files point to the paper code at `d8876f5`, before the
  `reset_on_generate` change.
