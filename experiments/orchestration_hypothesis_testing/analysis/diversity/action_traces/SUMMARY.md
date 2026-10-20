# Action traces of BG / BDP on the paper's cached data

Every number below is produced by `scripts/run_all.py` (CPU only, no LLM calls), using artifacts from the W&B
project `nlpresearch.group/orchestration-hypothesis-testing`. Repo: `rvz16/bayesian_control_for_coding_agents`
@ `d8876f5`.

## 1. Method

**Decision code is unchanged.** The script imports the original controllers:
- `analysis/controller.py`: `BayesianController`, `make_bayesian_policy`, `policy_always_verify`,
  `policy_fixed_pipeline`.
- `analysis/lcb_compare.py`: `GreedyController`, `make_greedy_policy`.

The only patch is `simulate_policy_traced`, a line-for-line copy of `controller.py:simulate_policy` (L177-229).
It additionally records, read-only, the actions, the belief after each step, the patch index and the observed
outcome.

**Protocol.** No repo script produces `full_results.tex`, so the protocol was inferred by matching
`prior_pass_rate.tex`, `transition_kernel.tex` and `full_results.tex`:
- **Split:** sort the instance ids, apply `np.random.default_rng(42).permutation`, take the first `int(0.75n)` as
  train.
- **Prior:** Laplace `(k+1)/(n+2)` on the train patches. It matches 49/54 paper priors exactly; the other 5 are
  within 0.007.
- **Critic likelihoods:** Beta(1,1) on the train patches. Critics: L0 = syntax, L2 = public tests (not on SWE),
  L3 = LLM judge.
- **BDP kernel:** Self-Refine corpus, 75% cohort with the same split and `round`. This reproduces
  `transition_kernel.tex`.
- **Controller:** H = 3, 51-point belief grid, 3 cached patches per instance.
- **Cost regimes:**
  - SWE: slow (C_ver = 90; critics 1 / 2 / 5).
  - All other benchmarks: fast (C_ver = 5; critics 1).
  - Both: C_gen = 10, R = 100.
- **Eval pool:** the full 25% eval split. The caption says results are restricted to instances with both
  Self-Refine and Reflexion results, but applying that breaks the matches (e.g. LCB-med gpt-5-mini needs n = 52).
- **Artifact versions:** per cell in `versions.json`. All were created on or before 2026-06-22; none of the
  2026-08 SWE-Verified expansion data is used.

## 2. Reproduction check (`reproduction_check.csv`)

**BG and BDP both match in 43/54 cells:**
- 11/12 SWE cells.
- LCB-medium: all closed-generator cells. LCB-easy: all cells. LCB-hard: haiku, sonnet, Qwen2.5-32B.
- MBPP+ and HumanEval+: all cells.
- HumanEvalFix: gpt-5-mini, qwen3-coder, Qwen2.5-32B, gpt-oss.
- CodeContests: qwen3-coder, Qwen2.5-32B.

**Mismatches, with the best diagnosis for each:**
- **LCB-hard gpt-5-mini** (−7.6/+0.3 vs −14.46/+1.69): only one calibration version exists and its prior matches,
  so the paper's run likely used different data.
- **LCB-hard qwen3-coder** (+2.4 vs +2.27): one patch is missing (305/306).
- **gpt-oss on LCB-hard / LCB-med / CodeContests** (paper −17.0/+4.6; −7.2/+2.3; 0.0/+11.7): 3–5 re-uploads per
  cell; none reproduces the paper.
- **SWE-Lite haiku, BDP only** (67.6 vs 67.33): BG matches; no version makes BDP act.
- **HumanEvalFix haiku and sonnet:** the prior matches but the critic likelihoods must differ (BG gives 0.0
  instead of −13.3 / −17.6).
- **CodeContests gpt-5-mini / haiku / sonnet:** calibration v0 (112 instances) is close but not exact.

**Paper baseline columns, same protocol:**

| Baseline | Matching cells |
|---|---|
| gate(syn) | 43/54 |
| gate(llm) | 36/54 |
| fixed pipeline (`controller.py:policy_fixed_pipeline`; L0 then L3, AND gate) | 36/54 |
| gate(test) | 26/42 |

## 3. Headline diversity (paper regime, 2,828 episodes)

**BG and BDP repeat the same few sequences:**
- The first action is constant within every cell for both BG and BDP (54/54).
- **BDP distinct sequences per cell:** 1 in 30/54 cells (29/43 matched), 2 in 17, 3 in 3, 4 in 4. Median 1, mean
  1.65, maximum 4.
- **BDP entropy:** median 0 bits, maximum 1.70.
- **BG distinct sequences per cell:** 1 in 33/54, 2 in 20, 3 in 1. Maximum entropy 1.10 bits.

**Pooled over all cells:**

| | BG | BDP |
|---|---|---|
| Distinct action sequences | 9 | 18 |
| Counting outcomes | 13 | 24 |
| Matched cells only | 9 | 10 |

**Distinct BDP sequences per benchmark:**

| Benchmark | BDP sequences |
|---|---|
| CodeContests | 10 |
| HumanEvalFix | 9 |
| LCB-hard | 7 |
| LCB-med | 6 |
| LCB-easy | 4 |
| MBPP+ | 1 |
| HumanEval+ | 1 |
| SWE-Lite | 1 |
| SWE-Verified | 1 |

**BDP mostly acts like a simple rule:**
- SWE: BG and BDP both give up immediately on 100% of tasks.
- MBPP+ and HumanEval+: both verify immediately on 100% of tasks.
- BG and BDP produce identical traces on every task in 33/54 cells.
- BDP's trace equals one simple rule's on every task in 52/54 cells (ignoring repeated critic calls). The
  exceptions are HumanEvalFix gpt-5-mini and haiku.
- For contrast, gate(test) and the fixed pipeline produce 3–5 distinct sequences per benchmark.

**Most diverse cells:**
- **All cells:** LCB-hard gpt-oss (4 sequences, 1.70 bits); CodeContests gpt-oss (4, 1.41); CodeContests gpt-5-mini
  (3, 1.24). None of these is a matched cell.
- **Matched cells only:** HumanEvalFix gpt-5-mini (3, 0.44).
- **Fast regime** (not used in the paper for SWE): SWE-Verified gpt-oss (4, 1.80).

## 4. Example traces (b = belief after the step)

**HumanEvalFix gpt-5-mini** (prior .854; kernel P_fix = .84):
- 38/41 tasks: L2 PASS → .997 → verify Y=1. Utility +94.
- 1/41 tasks: L2 FAIL → .019 → regen → .843 → L2 PASS → .997 → verify. Utility +83.
- 2/41 tasks: L2 FAIL → regen → .843 → L2 FAIL → .017 → regen → .843 → blind verify Y=0. Utility −27.

**SWE-Verified gpt-oss** (fast regime; prior .295):
- 62/125 tasks: L3 PASS → .484 → L0 PASS → .580 → L0 PASS → .670 → verify.
- 21/125 tasks: L3 FAIL → .113 → regen → .252 → L3 FAIL → .093 → regen → .237 → L0 FAIL → .003 → give_up.
- 22/125 tasks: same as the previous one until patch 3, then L0 PASS → .313 → L0 PASS → .401 → verify Y=1.
  Utility +71.

**LCB-medium gpt-5-mini:**
- BDP, 30/52 tasks: L2 FAIL → .004 → give_up. Utility −1.
- BG, 29/52 tasks: L2 FAIL → .004 → regen ×3, belief stuck at .004, patches run out, no verify. Utility −31.

## 5. Cross-check vs the colleague's analysis

### (a) The colleague's mismatched cells

| Cell | Colleague | This replay | Diagnosis |
|---|---|---|---|
| SWE-Verified gpt-5-mini | +31.5 | v3 +30.65, v2 +29.67; paper-era v1 +74.00 | used the post-paper 489-instance expansion |
| SWE-Verified sonnet | +37.2 | v3 +36.34; paper-era v2 +66.00 | same |
| LCB-easy gpt-5-mini | +23.9 | paper protocol +0.62; Reflexion kernel +10.76; slow regime +27.12 | not reproduced exactly |
| CodeContests gpt-oss | — | −1.07; +8.93 with the restricted pool | unresolved |
| LCB-med Qwen2.5-32B | −5.0 | v2 0.00, matching the paper | not reproduced |

### (b) U(BDP) − U(rule)

Same eval split; paired bootstrap with 2,000 resamples.

| Comparison | BDP wins by >1 | Ties (\|d\|≤1) | BDP worse | Mean d | CI excludes 0 |
|---|---|---|---|---|---|
| Oracle rule per cell | **0/54** | 44 | 10 | −0.95 | 0 for BDP, 6 for the rule |
| Same, matched cells only | 0/43 | 36 | 7 | — | — |
| Best single rule over all 54 cells (gate(syn)) | 33/54 | 19 | 2 | **+12.13** | 41 for BDP, 0 for the rule |
| Per-benchmark rule picked on train | 4/54 | 44 | 6 | −0.01 | 4 vs 3 |
| Per-cell rule picked on train | 3/54 | 45 | 6 | −0.18 | — |

- The single-rule comparison covers only rules defined on all 54 cells; gate(test) does not exist on SWE.
- **Per-benchmark rules picked on train:**

  | Benchmark | Rule |
  |---|---|
  | LCB, CodeContests | gate(test) → give_up |
  | HumanEval+ | gate(test) |
  | MBPP+, HumanEvalFix | always_verify |
  | SWE | never_submit |

- **Worst BDP gaps:**
  - LCB-easy gpt-5-mini: −10.15 [−21.0, −0.4] vs gate(test).
  - CodeContests sonnet: −10.50 vs gate(llm).
- **Conclusion:** BDP recovers the per-regime rule without hand-picking, but never beats it.

### (c) Unique traces

Colleague: BG 8 / BDP 26 over 3,072 episodes. This replay: 9 / 18 over 2,828 episodes, or 13 / 24 counting
outcomes.

| Benchmark | Distinct traces |
|---|---|
| SWE | 1 (confirmed) |
| MBPP+, HumanEval+ | 1 action sequence (2 counting outcomes) |
| HumanEvalFix | BG 1, BDP 9 |
| LCB-hard | BG 4, BDP 7 |
| LCB-med | BG 5, BDP 6 |
| LCB-easy | BG 3, BDP 4 |
| CodeContests | BG 5, BDP 10 |

### (d) Structural limits (file:line at `d8876f5`)

1. **Per-cell prior.** Every task starts at the same b0 (`controller.py:303`, `lcb_compare.py:110`), and the
   policy is a deterministic lookup (`controller.py:168-170`), so the first action is constant.
2. **Short horizon.** H = 3 (`lcb_compare.py:180`; class default 5 at `controller.py:80`).
3. **3 cached patches.** The episode ends when they run out (`controller.py:198,218`).
4. **`max_actions = 64`** (`controller.py:178,199`). Never hit.
5. **Regenerate draws a fresh, independent cached patch** (`controller.py:215-219`), yet BDP pushes the old belief
   through the refinement kernel (`controller.py:317-318`). With P_fix = .84 on HumanEvalFix, this produces the
   blind regen → verify.
6. **Coarse planning.** 51-point grid with nearest-neighbour lookup (`controller.py:86,117,131,169`). The critic
   lookahead assumes a terminal or generate action follows (`controller.py:127-134`).
7. **Re-querying a critic double-counts evidence.** A critic can be re-queried on the same patch, and its identical
   cached verdict is Bayes-updated again (`controller.py:303-315`; `lcb_compare.py:110-123`). Affects 3.3% of BDP
   and 1.2% of BG episodes.

## 6. Caveats

1. **BG has no belief reset on generate** (`lcb_compare.py:108-124`). After a FAIL it regenerates until the
   patches run out and never inspects the new ones: 12.2% of BG episodes, 47% on LCB-medium. This drives BG's −15
   to −19, and it contradicts the paper's description of BG. The recorded beliefs are the controller's internal
   state as coded.
2. **The protocol was inferred, not found in code** (`scripts/prior_test.py`, `kernel_test.py`, `repro.py`).
   Using the Reflexion kernel instead breaks LCB-easy and HumanEval+ cells.
3. **The 11 non-matching cells are included but flagged** (`reproduces_paper_BG_BDP` in `versions.json`).
   Statistics are given for all 54 cells and for the 43 matched cells.
4. **"Distinct trace" means a distinct action sequence.** Counts that include outcomes are given separately.
5. **`summary.csv` has both regimes for every cell** (`regime` and `paper_regime` columns).
