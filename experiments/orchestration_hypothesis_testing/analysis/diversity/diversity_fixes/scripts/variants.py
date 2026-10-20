"""Variant definitions. Step 0 runs at the paper setting (H=3, 51-point grid). Step 1 = step0 + repair."""

def F(**kw):
    base = dict(f0a=False, f0b=False, f0c=False, lookahead="paper1", repair=False, repair_model="kernel",
                budget=False, calK=False, rep_likes=False, H=3, n_b=51, n_fresh=3)
    base.update(kw)
    return base

S0 = dict(f0a=True, f0b=True, f0c=True, lookahead="full")
S1 = dict(S0, repair=True, budget=True)
S0K = dict(S0, calK=True)            # diagnostic: data-fitted fresh-patch kernel instead of reset-to-b0
S1K = dict(S1, calK=True)
SR_AT = ("selfrefine", "at")        # action_traces Self-Refine version (cells.ITER)
SR_MX = ("selfrefine", "maxov")     # paper-era Self-Refine version with most chains rooted on p0
RX_AT = ("reflexion", "at")

VARIANTS = {
    "baseline":        dict(flags=F()),
    "s0a":             dict(flags=F(f0a=True)),
    "s0b":             dict(flags=F(f0b=True, lookahead="full")),
    "s0b_lookahead1":  dict(flags=F(f0b=True, lookahead="paper1")),
    "s0c":             dict(flags=F(f0c=True)),
    "step0":           dict(flags=F(**S0)),
    "step0_g201":      dict(flags=F(**S0, n_b=201)),
    # Step 1, action_traces chain versions
    "step1_H5":        dict(flags=F(**S1, H=5, n_b=201), chains=SR_AT),
    "step1_H6":        dict(flags=F(**S1, H=6, n_b=201), chains=SR_AT),
    "step1_H3g51":     dict(flags=F(**S1, H=3, n_b=51), chains=SR_AT),
    "step1_H5_crep5":  dict(flags=F(**S1, H=5, n_b=201), chains=SR_AT, c_rep=5.0),
    "step1_H5_crep15": dict(flags=F(**S1, H=5, n_b=201), chains=SR_AT, c_rep=15.0),
    "step1_H5_L1":     dict(flags=F(**S1, H=5, n_b=201), chains=SR_AT, L1=True),
    "step1_H5_obs":    dict(flags=F(**dict(S1, repair_model="obs"), H=5, n_b=201), chains=SR_AT),
    "step1_H5_rfx":    dict(flags=F(**S1, H=5, n_b=201), chains=RX_AT),
    # Step 1, chain versions chosen for maximal rooting on calibration p0 (paper era, <= 2026-08-31)
    "step1mx_H5":      dict(flags=F(**S1, H=5, n_b=201), chains=SR_MX),
    "step1mx_H6":      dict(flags=F(**S1, H=6, n_b=201), chains=SR_MX),
    "step1mx_H3g51":   dict(flags=F(**S1, H=3, n_b=51), chains=SR_MX),
    "step1mx_H5_crep5":  dict(flags=F(**S1, H=5, n_b=201), chains=SR_MX, c_rep=5.0),
    "step1mx_H5_crep15": dict(flags=F(**S1, H=5, n_b=201), chains=SR_MX, c_rep=15.0),
    "step1mx_H5_obs":  dict(flags=F(**dict(S1, repair_model="obs"), H=5, n_b=201), chains=SR_MX),
    "step1mx_H5_L1":   dict(flags=F(**S1, H=5, n_b=201), chains=SR_MX, L1=True),
    # Step 1 + critic likelihoods on repaired patches fitted on train chain steps >= 1 (not in the spec; see engine)
    "step1mx_H5_chlik": dict(flags=F(**dict(S1, rep_likes=True), H=5, n_b=201), chains=SR_MX),
    "step1mx_H6_chlik": dict(flags=F(**dict(S1, rep_likes=True), H=6, n_b=201), chains=SR_MX),
    "step1_H5_chlik":   dict(flags=F(**dict(S1, rep_likes=True), H=5, n_b=201), chains=SR_AT),
    # Diagnostic (not requested): replace reset-to-b0 by the calibration fresh-patch kernel
    "s0c_calK":        dict(flags=F(calK=True)),
    "step0_calK":      dict(flags=F(**S0K)),
    "step1K_H5":       dict(flags=F(**S1K, H=5, n_b=201), chains=SR_AT),
    "step1mxK_H5":     dict(flags=F(**S1K, H=5, n_b=201), chains=SR_MX),
    "step1mxK_H6":     dict(flags=F(**S1K, H=6, n_b=201), chains=SR_MX),
    "step1mxK_H5_obs": dict(flags=F(**dict(S1K, repair_model="obs"), H=5, n_b=201), chains=SR_MX),
}
