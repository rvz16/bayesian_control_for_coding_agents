# Per-task starting beliefs for BG / BDP (diversity study, Step 2 Phase A)

CPU-only replay of the paper's cached data: no LLM, API or GPU calls. The protocol, data versions and engine are
those of `../action_traces/` and `../diversity_fixes/`; the engine is imported, not copied.

## Bottom line

1. **The free signals are real (GO).** Patch length, task-text features and a learned task-difficulty score all
   predict Y on the eval split in many cells. The combined task score reaches a pooled AUROC of 0.68 [0.63, 0.73],
   about the same as the paid LLM-judge critic (0.67).
2. **Diversity: yes.** A per-task / per-patch starting belief breaks the "constant first action":
   - First action varies within the cell in **36–38 of 54 cells** (0/54 before).
   - Distinct action sequences per cell: mean 4.1–4.3 (2.0 before).
   - Cells with a single sequence: 12–15 (22 before).
   - Distinct sequences pooled over all cells: 71–74 (30 before).
3. **Utility: no measurable gain over fair rules.** Compared with the calK base, per-cell differences stay within
   ±1 and the averages are below 1. The best variant, `calK_jointcomb`, gives:
   - +0.17 vs the same controller without signals;
   - +0.10 vs the per-benchmark rule chosen on train;
   - +0.11 vs the signal-thresholded rule chosen on train.

   Cells where the CI excludes 0 are roughly balanced between wins and losses. In the paper regime, **no cell** has
   the signal making BDP beat both fair rules by more than 1 with a CI that excludes 0.
4. **Naive Bayes is wrong here.** The signals are not conditionally independent of the critics given Y. Under naive
   Bayes, BDP becomes overconfident and loses utility on slow-regime SWE. Fitting the joint likelihood fixes this.
5. **Phase B (GPU sequence-probability scoring) is not worth running** to decide BDP vs rules.
   - A synthetic free signal with AUROC 0.95 buys BDP only +0.06 vs the thresholded rule over all cells (+0.50 on
     the Phase-B cells).
   - The total headroom of perfect information over the current BDP is only 4.9 utility per task. See §7.

## 0. Checks

| Check | Result |
|---|---|
| diversity_fixes regression gate, re-run with all flags off (`logs/regression_gate.json`) | **PASS**: 47,304/47,304 episodes; 0 cell-Δ mismatches |
| Constant (single-bin) signal reproduces the base exactly (`logs/const_signal_check.json`) | **PASS**: `calK_const` ≡ `step0_calK` and `g201_const` ≡ `step0_g201`. Each base: 5,656 BG/BDP episodes (actions, beliefs to 4 dp, patch, observations, utility) and 546 per-instance utility vectors, 0 mismatches |
| Joint DP with neutralised joint likelihoods equals the engine DP (`logs/joint_dp_check.json`) | **PASS**: 2,828/2,828 episodes |

## 1. Signals (`signals.csv`, `auroc.csv`, `cache/match_report.json`)

All signals are observable before any paid action.

**Excluded:** gold/reference solutions, gold and test patches, hidden-test contents, FAIL_TO_PASS lists,
`y_pass_rate`, `l2_pass_rate`, and the `*_detail` fields.

**Patch-level signals** (from `critic_results`):
- `log_diff_chars`.
- `empty_diff`: diff_chars = 0, which always means Y = 0. It covers 0–74% of a cell's patches, e.g. 74% on
  SWE-Verified/gpt-5-mini and 43% on LCB-hard/gpt-oss.
- `L1_lint`, treated as free.

**Task-level signals** (from the public datasets; LCB metadata was streamed from `code_generation_lite`
test.jsonl…test6.jsonl with test cases discarded):
- All benchmarks: statement length, number of public tests.
- LCB: contest date, release, starter-code length, number of `<=` constraint lines.
- CodeContests: cf_rating, difficulty, problem index, tags, time and memory limits.
- SWE: repo, hints length, has_hints, created_at, code blocks, traceback. SWE-Verified also has its difficulty
  annotation.
- HumanEvalFix: bug_type, failure_symptoms.
- Hidden-test-suite sizes appear only as `t_priv_*` columns and never feed a controller.
- Match rate: 100% on all 9 benchmarks.

**Learned scores** (TF-IDF and features + logistic regression, C = 1, generator one-hot):

| Score | Inputs |
|---|---|
| `tasklr` | Numeric and categorical task features |
| `difficulty` | TF-IDF 1–2-grams of the task text |
| `combtask` | Both of the above |

**Leakage rule.** For a target cell, the model is trained on the patches of **all** generators of the benchmark,
but only on tasks in that cell's **train** split.
- Splits coincide across generators on 7 benchmarks, but not on SWE-Verified (200 vs 500 ids; 616
  eval-in-train pairs across generators) or CodeContests (104–165 ids; 480 pairs). Every model is therefore refit
  per target cell.
- Train-split scores are 5-fold out-of-fold. Eval scores come from the model fit on all train tasks.
- The orientation of each raw feature is chosen on train.

**AUROC vs Y on eval patches.** CIs: task-cluster bootstrap, 2,000 resamples. Pooled AUROC is stratified within
cells.

| Signal | Cells with CI > 0.5 (/54) | Median cell AUROC | Pooled, 54 cells | Pooled, 43 matched |
|---|---|---|---|---|
| log_diff_chars | 26 (4 have CI < 0.5) | 0.654 | 0.631 [0.602, 0.658] | 0.612 [0.581, 0.642] |
| empty_diff | 17 | 0.500 | 0.596 [0.583, 0.610] | 0.576 |
| L1_lint | 5 | 0.500 | 0.515 [0.511, 0.519] | 0.506 |
| tasklr | 23 | 0.600 | 0.656 [0.609, 0.701] | 0.662 |
| difficulty | 17 | 0.580 | 0.634 [0.597, 0.672] | 0.640 |
| combtask | 23 | 0.581 | **0.679 [0.629, 0.730]** | 0.680 |
| *paid critics:* L0 / L2 / L3 | 20 / 35 (of 43) / 32 | — | 0.601 / 0.764 / 0.666 | — |

**Pooled AUROC by benchmark:**

| Benchmark | combtask | Patch length |
|---|---|---|
| LCB-easy | **0.90** | 0.45 |
| LCB-med | 0.81 | 0.69 |
| SWE-Lite | 0.75 | 0.60 |
| SWE-Verified | 0.76 | 0.56 |
| CodeContests | 0.64 | 0.68 |
| HumanEvalFix | 0.55 | 0.70 |
| MBPP+ | 0.51 | 0.69 |
| HumanEval+ | 0.46 | 0.62 |

**Best single task features** (cells with CI > 0.5): SWE `n_code_blocks` 9/12; LCB `n_constraints_lines` 8/18;
LCB starter-code length 8/18.

**Verdict: GO.**

## 2. Integration

**Signal model** (`scripts/prior_model.py`):
- K quantile bins are fitted on train: the task score over train tasks; non-empty log length plus a dedicated empty
  bin; lint as pass / fail / missing.
- P(bin | Y) is Beta(1,1)-smoothed on train patches.

**Execution:**
- At the start of an episode: b = prior ⊕ task bin ⊕ patch bin.
- On regenerate:
  - calK base: push b through the calibration fresh-patch kernel, then ⊕ the new patch's bin.
  - g201 base: reset to the task posterior b_T, then ⊕ the new patch's bin.

**Planning** (`scripts/ptp_engine.py`):
- Same DP state and grid as the engine.
- Q_generate takes the expectation over the next patch's bin.
- In g201 mode the DP is solved per distinct b_T.
- BG receives the same beliefs.

**Variants:**
- Signal sets: `len`, `lint0` (lint free), `task` (tasklr), `diff` (difficulty), `comb` (combtask + length + lint).
- `comb_lint1`: lint as a critic costing 1.
- `comb_K3` / `comb_K5`: number of bins.
- `calKg201`: calK on a 201-point grid.
- `joint*`: see §3.
- `synth*`: see §7.

## 3. Calibration and conditional independence (`calibration.csv`, `ci_check.csv`)

**Starting belief b0 on eval patches** (54 cells pooled):

| Signal set | Brier | ECE |
|---|---|---|
| Prior only | 0.158 | 0.023 |
| comb | 0.121 | 0.015 |
| jointcomb | 0.124 | 0.018 |
| len | 0.139 | — |
| task | 0.144 | — |

The median per-cell AUROC of b0 is 0.72 for comb.

**Belief at verify time (BDP).** The signals make it slightly *worse*:

| Variant | Brier | ECE |
|---|---|---|
| calK | 0.099 | 0.019 |
| comb | 0.114 | 0.035 |
| jointcomb | 0.100 | 0.028 |

**Naive Bayes is rejected** (G-test within Y strata on train, p < 0.01):

| Pair | Cells rejected |
|---|---|
| Length bin vs L3 | 33/52 |
| Length bin vs L0 | 23/39 |
| Length bin vs L2 | 15/38 |
| Task bin vs L3 | 20/52 |
| Task bin vs L2 | 15/38 |
| Task bin vs length bin | 42/54 |

**Consequence.** On slow-regime SWE, naive-Bayes `comb` verifies at a mean belief of 0.94–0.97 while only 50–80%
of those patches are correct. Utility drops by −12.52 [−21.0, −5.1] on SWE-Lite/haiku and −4.92 on
SWE-Verified/gpt-5-mini.

**Fix: the `joint` variants.**
- They model P(length bin | Y, task bin) and P(critic | Y, length bin).
- Critic likelihoods are shrunk towards the pooled likelihood with 2 pseudo-counts.
- The DP state carries the length bin.
- This restores give_up on slow SWE and gives the best utility of all variants.

## 4. Results, paper regime (`variant_table.csv`, `rules_vs_bdp.csv`)

**Column key:**
- *1st act. varies*: cells (of 54) where BDP's first action is not constant.
- *Seq./cell*: distinct sequences per cell, median / mean / max.
- *Cells 1 seq.*: cells with a single distinct sequence.
- *Pooled*: distinct sequences pooled over all cells.
- *Entropy*: median / max, in bits.
- *vs base BDP*, *vs (ii)*, *vs (iii)*: mean U(BDP) − U(comparator), with (number of cells whose CI is above 0 /
  below 0).
- *(ii)*: per-benchmark base rule chosen on train.
- *(iii)*: per-benchmark rule chosen on train from the base rules plus signal-thresholded rules. A thresholded
  rule follows rule A if the patch's free-signal posterior is ≥ τ_q, else rule B, with τ from train quantiles
  0.25 / 0.5 / 0.75.

**All 54 cells (BDP):**

| Variant | 1st act. varies | Seq./cell | Cells 1 seq. | Pooled | Entropy | Δ vs always_verify | vs base BDP | vs oracle rule | vs (ii) | vs (iii) |
|---|---|---|---|---|---|---|---|---|---|---|
| calK_const (= step0_calK) | 0 | 2/2.00/4 | 22 | 30 | 0.47/1.70 | +14.90 | 0 | −1.03 | −0.07 (3/5) | −0.07 (3/5) |
| calK_len | 27 | 2/3.48/11 | 16 | 68 | 0.92/3.05 | +14.81 | −0.09 (6/4) | −1.12 | −0.16 (6/6) | −0.25 (7/8) |
| calK_lint0 | 5 | 2/2.20/6 | 21 | 37 | 0.47/2.31 | +14.93 | +0.03 (1/0) | −1.00 | −0.04 (3/5) | −0.33 (2/7) |
| calK_task | 22 | 2/2.80/10 | 18 | 46 | 0.91/2.64 | +15.00 | +0.10 (5/7) | −0.94 | +0.02 (5/7) | +0.11 (3/5) |
| calK_diff | 22 | 2/3.11/12 | 17 | 55 | 0.94/3.25 | +14.82 | −0.09 (9/7) | −1.12 | −0.16 (8/5) | −0.31 (6/6) |
| calK_comb | 38 | 3/4.13/14 | 12 | 71 | 1.15/3.38 | +14.56 | −0.35 (11/4) | −1.38 | −0.42 (7/7) | −0.47 (8/10) |
| calK_comb_lint1 | 38 | 3/4.24/14 | 12 | 84 | 1.21/3.38 | +14.60 | −0.31 (9/6) | −1.34 | −0.38 (7/6) | −0.33 (7/8) |
| calK_comb_K3 | 35 | mean 3.89 | 14 | 72 | — | +14.66 | −0.25 | −1.28 | −0.32 | −0.32 |
| calK_comb_K5 | 40 | mean 4.46 | 12 | 73 | — | +14.52 | −0.39 | −1.42 | −0.46 | −0.66 |
| **calK_jointcomb** | 36 | 3/4.31/20 | 15 | 74 | 1.17/4.18 | **+15.07** | +0.17 (10/7) | −0.86 | +0.10 (8/6) | +0.11 (6/7) |
| calK_jointlen | 22 | 2/3.31/12 | 18 | 61 | 0.94/3.13 | +15.11 | +0.21 (10/6) | −0.82 | +0.14 (6/5) | +0.05 (10/6) |
| calKg201_comb | 34 | 3/3.33/12 | 14 | 46 | 0.95/3.24 | +14.76 | −0.20 (14/5) | −1.18 | −0.22 (11/4) | −0.26 (9/8) |
| g201_const (= step0_g201) | 0 | 2/2.44/5 | 15 | 29 | 0.84/1.88 | +12.73 | 0 | −3.20 | −2.24 (2/14) | −2.24 (2/14) |
| g201_comb | 32 | 4/4.94/13 | 10 | 69 | 1.40/3.43 | +13.70 | +0.97 (19/3) | −2.24 | −1.28 (6/11) | −1.32 (4/21) |
| g201_jointcomb | 35 | 4.5/5.41/16 | 11 | 74 | 1.49/3.40 | +13.74 | +1.01 (10/7) | −2.20 | −1.24 (3/11) | −1.23 (1/19) |

The g201 len / lint0 / task / diff rows (Δ vs always_verify +12.48 / +12.74 / +13.12 / +13.61) are in the CSV.

**43 paper-matched cells (BDP):**

| Variant | 1st act. varies | Mean seq./cell | Cells 1 seq. | Pooled | Δ vs always_verify | vs base BDP | vs oracle rule | vs (ii) | vs (iii) |
|---|---|---|---|---|---|---|---|---|---|
| calK_const | 0 | 1.77 | 21 | 23 | +16.65 | 0 | −0.81 | −0.27 | −0.27 |
| calK_comb | 28 | 3.30 | 12 | 42 | +16.43 | −0.23 | −1.03 | −0.49 | −0.25 |
| calK_jointcomb | 27 | 3.40 | 14 | 38 | +16.88 | +0.23 | −0.58 | −0.04 | +0.20 |

**Per benchmark (calK_jointcomb, BDP):**
- Distinct sequences per cell: LCB-med 8.2, CodeContests 7.5, HumanEvalFix 6.0, LCB-hard 5.8, HumanEval+ 4.0,
  LCB-easy 3.7, MBPP+ 1.7. SWE stays at 1 (give_up).
- Δ vs base BDP stays within ±0.8 on every benchmark. Largest: LCB-med +0.76, HumanEvalFix +0.54.
- Fair rules still win clearly on CodeContests: by 5.6 for the oracle rule and 3.6 for (ii). This is driven by
  qwen3-coder (−13.4 vs gate_test_giveup, unchanged from the base).

**BG:**
- calK_comb: first action varies in 27 cells; mean 2.43 sequences per cell; +0.05 vs base BG.
- calK_jointcomb: −0.25 vs base BG.

**Cells where BDP beats both (ii) and (iii) with CI excluding 0 (paper regime):**
- calK_comb:
  - CodeContests/Qwen2.5-32B: +0.24 / +0.24.
  - HumanEval+/sonnet: +0.58 / +0.37.
  - HumanEvalFix/sonnet (not a matched cell): +9.22 / +0.29.
- calK_jointcomb:
  - HumanEval+/haiku: +0.51 / +0.24.
  - HumanEval+/sonnet: +0.85 / +0.58.
- calK_jointlen: also LCB-easy/gpt-5-mini, +10.15 / +10.15. The no-signal base already shows this, so the signal is
  not the cause.

No gain larger than 1 comes from the signal. The only margins above 1 with CI excluding 0 also exist without
signals: LCB-easy/gpt-5-mini, and HumanEvalFix/sonnet against (ii).

**Secondary: SWE in the fast regime** (12 cells, calK_jointcomb):
- Mean +0.48 vs base BDP, +0.62 vs (ii), +0.28 vs (iii).
- BDP beats both rules with CI above 0 in 6/12 cells. Largest: SWE-Verified/gpt-5-mini, +3.14 [+2.50, +3.76],
  where the base alone already gives +2.40.
- SWE-Lite/gpt-oss is worse: −4.67 vs (ii).

**Regenerate: task posterior vs calK.**
- Resetting to the task posterior (g201 + signal) recovers about 1.0 of the 2.2 lost by resetting to b0:
  +12.73 → +13.70, against +14.90 for calK.
- The task signal captures only part of the correlation between patches. Eval median cell, P(next pass | fail):

  | Estimate | Value |
  |---|---|
  | Observed | 0.08 |
  | Task posterior ⊕ patch bin | 0.31 |
  | Prior | 0.41 |

- The task posterior is closer to the observed rate than the prior in 42/53 cells, but calK remains the better
  regenerate model.

## 5. Example traces (calK_jointcomb; b = belief after the step; `examples.json`)

**LCB-medium/gpt-5-mini** (matched cell; prior .283). First actions over 52 tasks: L2 43, verify 4, L3 3,
generate 2.

| Task | Signals | Starting belief | Actions | Outcome |
|---|---|---|---|---|
| 2792 | combtask −0.12 (top task bin); 177-char patch | b_T .619, b0 .651 | L2 FAIL → .001 → give_up | U −1 |
| 2854 | top task bin; 1,105-char patch | b_T .619, b0 .852 | **verify** immediately | Y = 0, U −5. Mean over the 4 tasks verified first: +70 |
| 3498 | low task bin; **empty** patch | b_T .141, b0 .038 | L3 FAIL → .010 → give_up | U −1 |
| 2834 | lowest task bin; short patch | b_T .141, b0 .047 | **regenerate first** → .052 → L2 PASS → .189 → verify | Y = 0, U −16 |

**LCB-medium/haiku** (prior .173):

| Task | Starting belief | Actions | Outcome / tasks |
|---|---|---|---|
| 3498 | b_T .008 | **give_up** at once | U 0; 13 tasks |
| 2792 | b_T .522 | L3 PASS → L2 FAIL → give_up | 7 tasks |
| 2854 | b0 .610 | verify | 2 tasks |

**HumanEvalFix/gpt-oss** (prior .833):

| Task | Starting belief | Actions | Outcome |
|---|---|---|---|
| Python/108 | b0 .80 | **verify** at once | 21 tasks |
| Python/110 | b0 .75 | L3 FAIL → .28 → regen → L3 FAIL → regen → verify | Y = 1, U +73 |
| Python/103 | b0 .93 | L2 PASS → verify | U +94 |

## 6. Answers

- **Does the per-task prior make BDP diverse?** Yes.
  - The first action varies within 36–38/54 cells (27–28/43 matched), against 0 before.
  - Distinct sequences per cell roughly double.
- **Does it buy utility over fair rules?** No.
  - Best case: +0.10 / +0.11 per task vs the train-chosen per-benchmark rule and the signal-thresholded rule.
  - Wins and losses are balanced.
  - No signal-driven gain above 1 has a CI excluding 0 in the paper regime.
  - The oracle rule per cell still beats BDP by 0.86 on average.
- **Why the effect is small.** A free prior can at most save the cheap critic or the verify call it replaces: 1–5
  in the fast regime. On slow SWE, a belief has to exceed about 0.9 to justify C_ver = 90.
  - Perfect information over BDP is worth only 4.88 per task on average: 5.59 outside SWE, 2.40 on slow SWE.

## 7. Phase B: is GPU sequence-probability scoring worth it?

**Sizing experiment** (`calK_synth*`):
- A **synthetic** patch-level score s = Δ·Y + N(0, 1) is built for a target AUROC a. It is built from the outcome
  on purpose and is used only to size the potential gain.
- It enters as a free signal (K = 5), and the thresholded rules get it too.
- Realised eval AUROC (median): 0.59 / 0.69 / 0.80 / 0.90 / 0.95.

| Synthetic AUROC | All 54 cells: Δ vs base | vs (ii) | vs (iii) (CI wins / losses) | 22 Phase-B cells: vs base | vs (iii) |
|---|---|---|---|---|---|
| 0.6 | +0.05 | −0.03 | −0.06 (9/5) | +0.08 | +0.11 |
| 0.7 | +0.01 | −0.06 | +0.03 (8/7) | +0.28 | +0.39 |
| 0.8 | +0.14 | +0.07 | −0.11 (6/10) | +0.62 | +0.15 |
| 0.9 | +0.25 | +0.18 | −0.20 (8/13) | +0.82 | +0.18 |
| 0.95 | +0.44 | +0.37 | +0.06 (9/13) | +1.01 | +0.50 |

**Real signals on the 22 Phase-B cells** (gpt-oss, Qwen2.5-32B, HumanEvalFix):
- The free signals already reach a b0 AUROC of median 0.73 (comb) and 0.71 (length).
- calK_jointlen on these cells: +0.39 vs base, +0.18 vs (iii).

**What Phase B would have to deliver.** A signal with AUROC above 0.95 that is conditionally independent of the
critics. Even then it would give about +0.5 over the thresholded rule, which would get the same score.

**Recommendation.** Do not run Phase B for the BDP-vs-rules question. Run it only to show that model confidence
is an informative free observation; it is likely correlated with L3, given §3.

## 8. Caveats

1. **Inherited protocol.** This study inherits the inferred protocol and the 11 non-matching cells of
   `action_traces`. Every result is reported both for all 54 cells and for the 43 matched cells.
2. **Small residual in-sample effect.** Learned scores are out-of-fold on train, but the bin edges, P(bin | Y) and
   the thresholded rules' τ are fitted on those same out-of-fold train scores.
3. **Thresholded rules.**
   - They switch per cached patch, using the same free-signal posterior that BDP starts from.
   - (iii) is chosen from about 216 combinations on train and can overfit. Example: LCB-hard
     `q0.25:gate_syn|never_submit` does worse on eval than the plain rule.
4. **"CI excludes 0" counts include tiny effects** (|d| < 0.5) where both policies act almost identically. Margins
   above 1 are reported separately.
5. **Shrinkage in the joint model.** Critic likelihoods are shrunk (M = 2) towards the pooled likelihood. Bins with
   no Y = 1 train patches (e.g. empty diffs) therefore keep the pooled values; this has no effect because b ≈ 0
   there.
6. **The synthetic-AUROC study is the favourable case for Phase B.** It assumes a signal that is conditionally
   independent of the critics given Y.
7. **Data sources.** LCB task metadata was streamed from the HF hub (test2–6.jsonl; only metadata kept, in
   `cache/lcb_meta.jsonl`). The other datasets came from the local HF cache.

## Reproduce

Run from this directory:

```
python3 -B ../diversity_fixes/scripts/run_gate.py logs/regression_gate.json
python3 -I -B scripts/fetch_lcb_meta.py          # network; metadata only
python3 -I -B scripts/signals.py                  # signals.csv, auroc.csv
python3 -I -B scripts/run_ptp.py runs             # all variants; skips existing (FORCE=1 to redo)
python3 -I -B scripts/check_const.py runs logs/const_signal_check.json
python3 -I -B scripts/check_joint.py logs/joint_dp_check.json
python3 -I -B scripts/ci_check.py "$PWD/ci_check.csv"
python3 -I -B scripts/analyze_ptp.py runs .
```
