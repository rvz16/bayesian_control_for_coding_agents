# Diversity fixes for BG / BDP (Step 0 + Step 1)

CPU-only replay of the paper's cached data. Reference protocol and data: `../action_traces/SUMMARY.md`.

## Method

**Engine:** `scripts/engine.py` re-implements the paper code at `d8876f5`, with one flag per fix. Paper file paths
are relative to `experiments/orchestration_hypothesis_testing/analysis/`.

**Step 0 fixes:**
- **0a.** BG resets its belief to b0 on generate (bug at `lcb_compare.py:108-124`).
- **0b.** A critic already observed on the current patch is masked until the patch changes.
  - The mask is part of the DP state, with full within-patch lookahead.
  - `s0b_lookahead1` keeps the paper's one-critic lookahead (`controller.py:125-134`).
- **0c.** Generate resets the belief to b0, in both execution and DP planning (paper: `controller.py:317-318`).
- `step0` turns on all three. `step0_g201` also uses a 201-point grid.

**Step 1 = step0 + a `repair` action + a cached-patch budget.**
- **`repair`:** moves to the next recorded step of the task's Self-Refine chain.
  - A chain is used only if its step 0 matches calibration patch p0: same Y, same L0–L3 verdicts, same size
    (`chains.py:41`).
  - Repair is available only while a recorded next step exists.
- **`generate`:** takes the next fresh cached patch and resets the belief to b0.
- **Fitting:** the repair kernel and the continuation probability ρ(d) are fitted on train chains only
  (`chains.py:97`).
- **Settings:** C_rep = 10, with 5 and 15 as sensitivity checks. Horizon H5/H6 on a 201-point grid, plus H3 on a
  51-point grid.
- **Extra variants:**
  - `_L1`: lint critic, cost 1.
  - `_rfx`: Reflexion chains.
  - `_obs`: the agent observes whether the patch changed.
  - `_chlik`: critic likelihoods on repaired patches fitted on train chain steps ≥ 1.
- **Chain versions:**
  - `step1_*` uses the action_traces versions: 22 replayable cells.
  - `step1mx_*` uses, per cell, the paper-era version with the most p0-rooted chains (`chains.py:167`): 37
    replayable cells.
  - **`step1mx_H5` is the main variant.**

**Protocol:** identical to action_traces.
- Split 75/25 with rng(42). Laplace prior. Beta(1,1) likelihoods.
- SWE uses the slow regime: C_ver 90, critics 1/2/5. All other benchmarks use the fast regime: C_ver 5, critics 1.
- C_gen 10, R 100. Results are on the eval split.

**Rules:** as in `run_all.py:97-137`.
- Oracle rule per cell, chosen on eval.
- Best single rule, chosen on eval.
- Per-benchmark rule, chosen on train.
- Paired bootstrap: 2,000 resamples, seed 42.
- In Step 1, BDP also competes against repair rules: `gate_{syn,test,llm}_rep_{verify,giveup,regen}` and
  `repair{1..4}_verify`.

## Checks

- **Regression gate: PASS.** With all fixes off, all 47,304 reference episodes are reproduced exactly: actions,
  beliefs to 4 dp, patch, observations, utility and cost. All cell Δs also match. Turning 0c on gives 2,676
  mismatches, which shows the gate is sensitive.
- **DP self-consistency:** the planned V_root matches a Monte-Carlo estimate under the planner's own model within
  about 1–2 SE. Worst case: CodeContests/qwen3-coder at step 1, +0.79 to +0.89 (about 2.1–2.3 SE).

## Coverage of repair chains (`chain_overlap.csv`)

**Replayable cells:**

| Chain source | Replayable cells | Of which paper-matched |
|---|---|---|
| Self-Refine, max-overlap versions | 37/54 | 27 |
| Self-Refine, action_traces versions | 22/54 | — |
| Reflexion | 20/54 | — |

**No coverage on:**
- MBPP+ and HumanEval+.
- HumanEvalFix: Qwen2.5-32B and gpt-oss.
- SWE-Lite: haiku.
- SWE-Verified: gpt-5-mini and sonnet.

**Caveats on coverage:**
- 9 of the replayable cells are SWE. There, slow-regime BDP gives up immediately, so they carry no signal.
- Train P_fix is 0.007–0.09 in most cells. It is larger only on:
  - HumanEvalFix: 0.18–0.84.
  - gpt-oss: 0.24–0.31.
  - SWE-Lite gpt-5-mini: 0.20.
  - CodeContests gpt-oss: 0.40 (n = 24).
- Many "repairs" return the identical patch: up to 97% of transitions on LCB-easy.

## Results: BDP, paper regime, all 54 cells

Column key:
- **Sequences per cell:** median/mean/max.
- **vs oracle rule:** cells BDP wins (by >1, CI > 0) / ties (|d| ≤ 1) / loses (< −1) / cells where the rule wins
  with CI excluding 0 / mean difference.
- **vs per-benchmark rule (train):** cells BDP wins / cells BDP loses / mean difference.

| Variant | Sequences per cell | Cells with 1 sequence | Pooled sequences | Entropy (median/max) | Δ vs always_verify | vs oracle rule | vs per-benchmark rule (train) |
|---|---|---|---|---|---|---|---|
| baseline | 1/1.65/4 | 30 | 18 | 0/1.70 | +14.96 | 0/44/10/6/−0.95 | 1/6/−0.01 |
| s0a | = baseline | | | | | | |
| s0b | 1/1.80/6 | 28 | 32 | 0/1.70 | +14.99 | 0/43/11/7/−0.92 | 1/7/+0.02 |
| s0c | 2/2.33/5 | 15 | 27 | 0.72/1.88 | +12.58 | 0/26/28/25/−3.34 | 2/25/−2.40 |
| **step0** | 2/2.61/8 | 15 | 45 | 0.88/2.21 | +12.83 | 0/29/25/21/−3.09 | 2/24/−2.15 |
| step0_g201 | 2/2.44/5 | 15 | 29 | 0.84/1.88 | +12.73 | 0/29/25/20/−3.18 | 2/22/−2.24 |
| step1mx_H3g51 | 2/2.63/8 | 15 | 46 | 0.88/2.21 | +12.92 | 0/25/29/19/−3.57 | 2/25/−2.23 |
| **step1mx_H5** (H6 identical) | 2/2.52/7 | 15 | 34 | 0.84/2.07 | +12.90 | 0/26/28/18/−3.59 | 2/22/−2.25 |
| step1mx_H5, C_rep 5 / 15 | mean 2.61 / 2.46 | 15 | 39 / 30 | — | +12.85 / +12.82 | −3.85 / −3.52 (mean) | −2.37 / −2.26 (mean) |
| step1mx_H5_L1 / _obs / _chlik | ≈ step1mx_H5 | 15 | 35 / 34 / 32 | — | +12.89 / +12.90 / +12.91 | ≈ −3.58 (mean) | ≈ −2.25 (mean) |
| step1_H5 / step1_H5_rfx | mean 2.52 / 2.48 | 15 | 34 / 33 | — | +12.90 / +12.88 | −3.58 / −3.49 (mean) | −2.25 / −2.27 (mean) |
| *diag:* step0_calK | 2/2.00/4 | 22 | 30 | 0.47/1.70 | +14.90 | 0/43/11/9/−1.01 | 2/9/−0.07 |
| *diag:* step1mxK_H5 | 2/2.06/7 | 23 | 38 | 0.33/2.07 | +15.06 | 0/39/15/7/−1.43 | 2/7/−0.09 |

**Behaviour:**
- **First action:** for BDP it is constant within every cell in every variant (0/54 cells vary). For BG it varies
  only with C_rep = 5, in 2 cells.
- **Mean episode length:**

  | Variant | Mean length |
  |---|---|
  | baseline | 1.68 |
  | step0 | 2.81 |
  | step1mx_H5 | 2.76 |
  | s0c alone | 3.15 (critic re-queries) |

- **Action mix (step0):** L2 32.5%, verify 21%, give_up 18%, generate 13%, L3 12.6%, L0 3%.
- **Repair share (step1):** 0.1% of actions, or 0.6% at C_rep = 5.
- **Traces identical to one rule on every task** (strict, counting critic re-queries): 41 cells in the baseline,
  29 in step0, 33 in step1mx_H5.

**BDP vs the best single rule:** +12.13 (baseline), +9.99 (step0), +7.82 (step1mx_H5). In step 1 that rule is
`gate_syn_rep_giveup`.

**BG:**
- 0a raises BG from +9.75 to +11.73 vs always_verify, and shrinks its gap to the oracle rule from −6.17 to −4.19.
- In step 1, BG reaches +13.06 and uses repair in 0.9% of actions.

**43 paper-matched cells (BDP):**

| Variant | Cells with 1 sequence | Pooled sequences | Δ vs always_verify | vs oracle rule | vs per-benchmark rule (mean) |
|---|---|---|---|---|---|
| baseline | 29 | 10 | +16.71 | 0/36/7/4/−0.75 | −0.22 |
| step0 | 14 | 24 | +14.20 | 0/22/21/18/−3.26 | −2.73 |
| step1mx_H5 | 14 | 24 | +14.22 | 0/23/20/15/−3.43 | −2.81 |
| step0_calK | 21 | 23 | +16.65 | mean −0.80 | −0.27 |

**Replayable cells only:**
- **37 cells, step1mx_H5:**
  - Sequences per cell: mean 2.59; 10 cells with 1 sequence.
  - Δ vs always_verify: +13.64.
  - vs oracle rule: 0/17/20/12/−4.47. vs per-benchmark rule: mean −2.67.
  - Repair: 0.2% of actions.
- **22-cell action_traces subset:** mean 2.95 sequences per cell; oracle-rule gap −4.22.

**Repair effect** (`step1_vs_step0.csv`: step1mx_H5 minus step0_g201, paired by task):
- Mean difference over the 37 cells: +0.19.
- Only 3 cells use repair at all, and no cell's CI excludes 0.
  - HumanEvalFix/gpt-5-mini: +5.10 [+0.00, +12.88], with 0.10 repairs per episode.
  - CodeContests/gpt-oss: +2.48 [−0.07, +7.50].
- At C_rep = 5, the only significant cell is negative: LCB-med/Qwen2.5-32B, −1.54 [−2.38, −0.85].

## Answers

1. **Step 0 increases diversity:** cells with a single sequence drop from 30 to 15, and pooled sequences rise from
   18 to 45.
   - Almost all of it comes from 0c; 0a and 0b barely change BDP.
   - **It costs utility:** −2.13 per cell on average vs always_verify, and up to −15.65 on LCB-med/haiku
     (18.71 → 3.06).
   - **Cause:** fresh cached patches of the same task are correlated. In 54/54 cells, P(next patch passes | this one
     failed) is below b0 (median 0.113 vs 0.42). Resetting to b0 therefore over-values FAIL→regenerate loops.
2. **Step 1 adds no measurable diversity or utility.**
   - Repair is rarely worth it at these fix rates (P_fix < 0.1 in most cells) when it costs the same as
     regenerating.
   - In the one cell where it pays (HumanEvalFix/gpt-5-mini), BDP ties the simple rule "gate(test), FAIL → repair
     until PASS, then verify".
   - Adding the repair rules makes the rule baselines stronger: the oracle gap goes from −3.09 to −3.59.
3. **BDP never beats the best per-cell rule** by more than 1 with a CI excluding 0. This holds in all 31 variants
   and in every subset (54, 43 and the replayable cells).
   - Against the per-benchmark rule picked on train, it wins in only these cells:
     - LCB-easy/gpt-5-mini: +10.15 [+0.41, +21.00] vs `gate_test_giveup` (step0 and step1).
     - HumanEvalFix/sonnet: +7.90 [+0.24, +17.25] vs `always_verify` (baseline and step0).
     - HumanEvalFix/haiku: +7.76 [+0.76, +17.49] vs `gate_test_rep_verify` (step1).
   - Only the LCB-easy cell is paper-matched.
4. **Diagnostic (not in the spec): `calK`.** Instead of resetting to b0, generate uses a regenerate kernel fitted
   on consecutive calibration patches. This keeps baseline utility (+14.90) and adds modest diversity (22 cells with
   a single sequence), but still wins 0 cells against the oracle rule.

## Example traces (b = belief after the step)

**Step 0, LCB-hard/gpt-5-mini** (not paper-matched; prior .209; 8 distinct sequences):
- 12/26 tasks: L0 PASS → .307 → L2 PASS → .659 → verify. Utility +93 if correct, −7 if not; mean +59.7.
- 7/26 tasks: L0 FAIL → .013 → regen → .209 → L0 FAIL → regen → L2 FAIL → .007 → give_up. Utility −23.
- 1/26 tasks: … regen ×2 → L2 PASS → .536 → verify, correct. Utility +72.

**Step 1, HumanEvalFix/gpt-5-mini** (prior .854; P_fix .84):
- 38/41 tasks: L2 PASS → verify. Utility +94.
- 2/41 tasks: L2 FAIL → .019 → repair (patch changed) → .843 → L2 PASS → .997 → verify. Utility +83.
- 1/41 tasks: two repairs, then verify. Utility +72.

**Step 1, LCB-hard/gpt-oss:**
- L2 FAIL → .008 → repair returns the *identical* patch → .242 (kernel push) → L3 FAIL → .100 → regen ×2 →
  give_up. Utility −34.
- This is a kernel-model artifact: the kernel moves the belief even though the patch did not change.

**s0c alone:** L2 PASS is queried three times on the same patch (.619 → .870 → .965) → verify. This is why 0b has
to go together with 0c.

## Caveats

1. **0c is misspecified for cached patches** (`engine.py:223` `gen_belief`; paper `controller.py:317-318`). The
   step-0 utility loss comes from this, not from the masking in 0b.
2. **The repair kernel model ignores whether the patch changed** (`engine.py:283,349`).
   - In execution, the critic mask is kept on an unchanged patch (`engine.py:143`); in planning it is reset.
   - The `_obs` model handles both cases, and its aggregates are identical.
3. **Critic likelihoods shift on repaired patches.** Example: for LCB-med/gpt-5-mini, P(L2 pass | Y=0) is .24 on
   calibration patches but .62 on chain steps. `_chlik` corrects for this and changes nothing material.
4. **Chain versions:** the max-overlap versions differ from the action_traces versions in 18 cells (all are
   paper-era). On cells without replayable chains, step1 = step0 + H5/grid 201 + budget, and the all-54 numbers
   include those cells.
5. **Chain end is only weakly informative about Y:** pooled P(Y=1 | last step) is .45 vs .50 for other steps.
   ρ(d) does not condition on Y.
6. **H doesn't bind.** H counts patch-producing actions; with 3 fresh patches and repair almost never chosen, H5
   and H6 produce identical traces.
7. **The oracle rule is chosen on eval with hindsight**, so it is an upper bound for rules. The per-benchmark rule
   chosen on train is the fair comparison.
8. **Inherited from action_traces:** the protocol is inferred, and 11/54 cells do not reproduce the paper.
9. **Reflexion's replayable results** (Δ vs always_verify +6.58) cover a different set of 20 cells and are not
   comparable to the rest.

## Reproduce

Run from this directory:

```
python3 -B scripts/run_gate.py regression_gate.json
python3 -B scripts/run_variants.py runs
python3 -B scripts/analyze.py runs .
python3 -B scripts/repair_effect.py runs step1_vs_step0.csv
```
