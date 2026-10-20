# Verifier-cost sweep on SWE (diversity study follow-up)

CPU-only replay of the paper's cached data.
- **Engine:** the per_task_prior engine, imported unchanged (`run_ptp.run_cell`), with extra cost regimes added to
  `protocol.REGIMES`.
- **Fixed costs:** C_gen 10, R 100.
- **Variants:**
  - `calK_jointcomb`: per-task / per-patch free signals with joint likelihoods.
  - `calK_const`: no signal.
- **Critic costs:**
  - Primary: syn 1, llm 5 (SWE has no L2).
  - Secondary: all critics cost 1.

## Bottom line

### 1. There is a band where BDP pays: C_ver ≈ 10–20

It appears under both critic-cost settings. Pooled over the 12 SWE cells, `calK_jointcomb` BDP beats all three fair
rules with CIs excluding 0 throughout the band:

| Fair rule | BDP margin in the band |
|---|---|
| (ii) base rule picked on train per (benchmark, C_ver) | +2.1 to +3.5 |
| (iii) base + signal-thresholded rules | +1.6 to +2.9 |
| (iii-fine) the same with a 19-quantile threshold grid | +1.5 to +2.2 |

- **Per cell:** BDP beats both (ii) and (iii) by more than 1 (CI > 0) in 4–5 of 12 cells. It loses (CI < 0) in at
  most 1 cell.
- **Best point:** C_ver = 20, where BDP − (iii) = +2.87 [+1.77, +3.98].

### 2. The gain comes from the per-task signal, not from planning

- **Without signals** (`calK_const`), BDP is within ±0.7 of the rules over C_ver 5–30, except C_ver 20, where it
  is −1.5 (CI < 0).
- **With the signal,** BDP gains +2.1 to +5.0 over `calK_const` in the band.
- **BG with the same beliefs does as well:** BDP − BG is −0.07 to −0.25 in the band, and every CI contains 0. DP
  lookahead adds nothing here.

What the data supports: a calibrated per-task belief, combined with cost-aware stop/verify decisions, pays at
intermediate verifier cost. It does not support "planning pays".

### 3. Perfect-information headroom peaks later than the band

| C_ver | Perfect − BDP | Perfect − (ii) |
|---|---|---|
| 5 | 8.1 | 8.9 |
| 30 | 13.9 | **18.6** (peak) |
| 45 | **14.7** (peak) | 17.3 |
| 90 | 2.6 | 2.6 |

- **Capture:** in the band, BDP captures 18–23% of the headroom over (ii), and 25% at C_ver 30.
- **At C_ver 30–45,** headroom is largest, but BDP no longer beats (iii).
- **At C_ver ≥ 60,** BDP is net-negative vs never_submit: −1.4 pooled at C_ver 60, and −4.5 [−7.6, −1.4] on
  SWE-Verified. These losses come from overconfident verifies.

### 4. Diversity collapses only at the slow extreme

- **C_ver 5–45:** the first action varies in 12/12 cells, with 3.7–4.3 distinct sequences per cell (max 6–10).
- **C_ver 90:** a single give_up sequence in every cell.
- **Give-up share rises smoothly:** .31 at C_ver 5, .46 at 20, .68 at 45, 1.0 at 90.

### 5. No band on the other benchmarks

Non-SWE screen: 42 cells, calK_jointcomb, all critics cost 1.
- **Margins:** BDP − (ii) and BDP − (iii) stay within [−0.24, +0.53] at every C_ver.
- **Headroom:** flat at 5.4–6.5.
- **Largest headroom:** CodeContests 14.0 at C_ver 5 (where BDP loses 3.6 to the rules); LCB-hard 10–11 at C_ver
  20–60.
- **Win/loss count:** wins never outnumber losses. The best case, C_ver 60, has 6 cells winning by more than 1 and
  11 losing.

## Checks (`logs/`)

| Check | Result |
|---|---|
| Sweep at cv90_s and cv5_f reproduces per_task_prior's slow and fast regimes exactly | **PASS**. 132 (variant, cell) pairs; per-instance eval utilities for all policies (1,356 vectors), train means, 12,096 thresholded-rule vectors, 14,912 BG/BDP traces and summary rows; 0 mismatches. |
| Re-implemented fair-rule selection reproduces per_task_prior's `rules_vs_bdp.csv` | **PASS**. 132 rows identical. |
| Fast regime reproduces per_task_prior's SWE result | **PASS**. cv5_f: +0.62 vs (ii), +0.28 vs (iii), 6/12 cells. |
| Perfect-info oracle | Non-SWE headroom 5.59 matches per_task_prior exactly. Slow SWE gives **2.63**, not the 2.40 per_task_prior stated (no script exists for the 2.40). Here slow SWE equals 10 × P(first patch correct), because BDP gives up on every task. |

## Definitions

- **Regimes:** C_ver ∈ {5, 10, 15, 20, 30, 45, 60, 75, 90}.
  - `cv{C}_s`: critic costs L0 1, L2 2, L3 5.
  - `cv{C}_f`: all critics cost 1.
- **Base rules:** always_verify, never_submit, gate_syn, gate_llm, fixed_pipeline, gate_syn_giveup,
  gate_llm_giveup. Non-SWE cells also have gate_test and gate_test_giveup.
- **(ii):** the base rule with the best mean train utility over the benchmark's 6 cells, chosen per (benchmark,
  C_ver, critic set).
- **(iii):** the same selection over the base rules plus the thresholded rules (q, A, B). A thresholded rule
  follows A if the patch's free-signal posterior is ≥ the train quantile q ∈ {.25, .5, .75}, and B otherwise.
- **(iii-fine):** q ∈ {.05, …, .95}.
- **Perfect-information oracle:** knows Y for every cached patch and pays no critic cost.
  - U = max(0, max over k < 3 with Y_k = 1 of 100 − C_ver − 10k).
  - Headroom = U(perfect) − U(policy).
- **CIs:**
  - Per cell: paired bootstrap over eval tasks (2,000 resamples, seed 42).
  - Pooled: mean of per-cell means, with tasks bootstrapped within each cell.

## Per-C_ver table: SWE pooled, calK_jointcomb, primary critics (`curves.csv`)

Column key:
- **Cells beating both:** cells where BDP beats both (ii) and (iii) with CI > 0, then those with margin > 1, then
  those also > 1 against (iii-fine).
- **Cells losing:** cells where BDP loses with CI < 0.
- **Sequences:** distinct sequences per cell, mean / max.

| C_ver | U BDP | U perfect | BDP − (ii) | BDP − (iii) | BDP − (iii-fine) | BDP − BG | Perfect − BDP | Perfect − (ii) | Cells beating both (CI / >1 / >1 vs iii-fine) | Cells losing | Sequences | Give-up share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 22.36 | 30.49 | +0.76 [+0.04, +1.48] | +0.37 [−0.37, +1.09] | +0.20 | −0.47 | 8.13 | 8.89 | 8 / 4 / 1 | 1 | 4.25 / 10 | .31 |
| **10** | 19.50 | 28.84 | **+2.10 [+1.40, +2.74]** | **+1.57 [+0.76, +2.48]** | +1.48 [+0.67, +2.38] | −0.21 | 9.34 | 11.44 | 7 / 4 / 3 | 1 | 3.67 / 7 | .37 |
| **15** | 16.60 | 27.18 | **+2.49 [+1.60, +3.35]** | **+1.92 [+1.05, +2.81]** | +1.80 [+0.83, +2.87] | −0.25 | 10.59 | 13.08 | 7 / 5 / 2 | 0 | 3.75 / 6 | .42 |
| **20** | 14.10 | 25.53 | **+3.49 [+2.43, +4.53]** | **+2.87 [+1.77, +3.98]** | +2.22 [+1.16, +3.29] | −0.07 | 11.44 | 14.93 | 6 / 5 / 3 | 0 | 4.33 / 7 | .46 |
| 30 | 8.31 | 22.23 | +4.71 [+3.32, +6.10] | +0.95 [−0.50, +2.33] | +0.79 | +0.14 | 13.92 | **18.63** | 1 / 1 / 1 | 1 | 4.17 / 8 | .57 |
| 45 | 2.54 | 17.27 | +2.54 [+0.54, +4.52] | −0.49 [−1.96, +1.04] | −0.79 | +0.01 | **14.74** | 17.27 | 0 / 0 / 0 | 4 | 3.83 / 6 | .68 |
| 60 | −1.37 | 12.32 | −1.37 [−3.25, +0.36] | −0.45 | −1.82 [−3.22, −0.31] | −0.74 | 13.69 | 12.31 | 1 / 1 / 0 | 2 | 3.67 / 6 | .76 |
| 75 | −0.90 | 7.36 | −0.90 | −0.90 | −0.40 | 0.00 | 8.26 | 7.36 | 1 / 1 / 0 | 1 | 2.08 / 5 | .93 |
| 90 | 0.00 | 2.63 | 0 | 0 | 0 | 0 | 2.63 | 2.63 | 0 | 0 | 1 / 1 | 1.0 |

**Without signals** (`calK_const`, primary critics), BDP − (ii):

| C_ver | 5 | 10 | 15 | 20 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|---|
| BDP − (ii) | −0.02 | +0.00 | +0.23 | −1.52 [−2.73, −0.30] | +0.37 | −2.42 | −2.87 |

The first action never varies without signals.

**Secondary critics** (all critics cost 1) show the same band. All CIs are above 0, and 4–6 of 12 cells beat both
rules by more than 1.

| C_ver | BDP − (ii) | BDP − (iii) |
|---|---|---|
| 10 | +2.10 | +1.27 |
| 15 | +2.59 | +2.02 |
| 20 | +3.35 | +2.20 |

**Per benchmark:**

| Benchmark | C_ver range where BDP − (iii) has CI > 0 | BDP − (iii) across that range |
|---|---|---|
| SWE-Verified | 10–30 | +1.47 / +2.18 / +3.23 / +3.28 |
| SWE-Lite | 10–20 | +1.68 / +1.65 / +2.50 |

- **Recurring winners:** SWE-Verified qwen25_32b, sonnet45 and gpt_oss_20b; SWE-Lite sonnet45, qwen25_32b and
  gpt5_mini.
- **Recurring loser:** SWE-Lite gpt_oss_20b at C_ver ≤ 10 (−1.03 vs (ii) at C_ver 10).

## Example traces (`examples.json`; calK_jointcomb, primary critics; b = belief after the step)

Shown per first action: the most common sequence, picked by lowest instance id. Traces were not chosen by outcome.

### SWE-Verified / gpt-oss-20b, C_ver 20

- Prior .295; 125 eval tasks.
- BDP − (ii) = +3.67 [+2.22, +5.08]; BDP − (iii) = +3.39 [+1.89, +4.82].
- First actions: verify 61, give_up 33, L0 22, generate 9. Seven distinct sequences.

| First action | Example task | Signals → belief | Steps | Utility | Group of tasks with this first action |
|---|---|---|---|---|---|
| give_up | django-10554 | lowest task bin, empty patch: b_T .042, b0 .008 | give_up | 0 (rule (ii): −1) | 33 tasks: mean 0 vs −13.1 for (ii) |
| verify | astropy-13033 | top task bin, 1,445-char patch: b_T .509, b0 .38 | blind verify, Y = 0 | −20 | 61 tasks: mean +24.3 vs +23.3 for (ii) |
| L0 | astropy-14598 | empty first patch: b0 .018 | L0 FAIL → give_up | −1 | 22 tasks |
| generate | django-13121 | b0 .18 | regenerate → next patch empty → L0 FAIL → give_up | −11 (rule (ii): −21) | 9 tasks: mean −24.9 vs −21.0 for (ii); this group loses |

### SWE-Lite / qwen2.5-32B, C_ver 15

BDP − (ii) = +5.21; BDP − (iii) = +2.85.

| Example task | Belief | Steps | Utility |
|---|---|---|---|
| astropy-14182 | b0 .106 | give_up | 0 (rule (ii): −16); 33 tasks follow this |
| django-10914 | b0 .34 | verify, Y = 1 | +85; 24 tasks follow this |
| django-14580 | empty patch | L0 FAIL → give_up | −1 |
| django-11797 | b0 .25 | L3 FAIL → .069 → give_up | −5 (rule (ii): −16) |

## Caveats

1. **BG matches BDP in the band.** BG with the same per-task beliefs does as well, so the result shows the value of
   the belief, not of the DP.
2. **The fair rules could be stronger.** (iii) uses coarse thresholds picked per benchmark; (iii-fine) shrinks the
   margins by 0.1–0.7, but every CI stays above 0. Per-cell or cost-aware-threshold rules were not tested.
3. **Above C_ver 45, BDP is net-negative on SWE-Verified.** Beliefs near the verify threshold are not calibrated
   enough.
4. **Protocol limits are inherited:** an inferred protocol, 3 cached patches, H = 3. SWE-Lite haiku does not match
   the paper.
5. **Many comparisons:** 9 C_ver values × 2 critic sets × 12 cells. Treat the pooled curves as the primary evidence
   and the per-cell win counts as descriptive.
6. **A minor quirk:** on empty diffs, BDP still pays for L0 (cost 1) before giving up, because the empty bin's
   critic likelihoods are shrunk toward the pooled ones.

## Reproduce (from this directory; ~15 s, plus ~3 min for the fine-quantile run)

```
python3 -I -B scripts/sweep.py "$PWD/runs" all
python3 -I -B scripts/sweep.py "$PWD/runs_fineq" swe_fineq
python3 -I -B scripts/check_repro.py runs ../per_task_prior/runs logs/repro_check.json
python3 -I -B scripts/analyze_sweep.py runs . ../per_task_prior runs_fineq
python3 -I -B scripts/plot_sweep.py .
python3 -I -B scripts/examples.py .
```
