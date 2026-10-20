"""Verifier-cost sweep: re-run per_task_prior's run_cell (imported, not copied) at custom cost regimes.

Regimes (added to protocol.REGIMES at import, so make_policies sees them):
  cv{C}_s : C_ver = C, critics L0 1 / L2 2 / L3 5  (slow-regime critic costs; cv90_s == 'slow')
  cv{C}_f : C_ver = C, critics all 1               (fast-regime critic costs;  cv5_f  == 'fast')
  C_gen 10, R 100 throughout.
One job per (regime, variant, cell); output RUNS/<regime>/<variant>/<bench>__<gen>.pkl, same schema as
per_task_prior/runs (keys use the regime name). Skips existing files (FORCE=1 to redo).

Mode `swe_fineq` (robustness for rule (iii)): calK_jointcomb on the 12 SWE cells with the thresholded-rule
quantile grid widened from (0.25, 0.5, 0.75) to 0.05..0.95 step 0.05 (run_ptp.QS patched), written to RUNS_DIR
(use a separate dir, e.g. runs_fineq).

usage: sweep.py RUNS_DIR [swe|nonswe|all|swe_fineq]
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PTP = os.path.abspath(os.path.join(HERE, "..", "..", "per_task_prior", "scripts"))
sys.path.insert(0, PTP)

RUNS_ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "..", "runs"))
_argv = sys.argv
sys.argv = [os.path.join(PTP, "run_ptp.py"), RUNS_ROOT]   # run_ptp reads RUNS=argv[1] at import
import run_ptp  # noqa: E402
sys.argv = _argv

from protocol import REGIMES  # noqa: E402  (same dict object runner.make_policies indexes)
from cells import cell_keys  # noqa: E402

CVERS = (5, 10, 15, 20, 30, 45, 60, 75, 90)
for c in CVERS:
    REGIMES[f"cv{c}_s"] = dict(c_gen=10.0, c_L0=1.0, c_L2=2.0, c_L3=5.0, c_ver=float(c), reward=100.0)
    REGIMES[f"cv{c}_f"] = dict(c_gen=10.0, c_L0=1.0, c_L2=1.0, c_L3=1.0, c_ver=float(c), reward=100.0)
assert REGIMES["cv90_s"] == REGIMES["slow"] and REGIMES["cv5_f"] == REGIMES["fast"]

VARS = ("calK_jointcomb", "calK_const")
FINE_QS = tuple(round(0.05 * i, 2) for i in range(1, 20))
if "swe_fineq" in sys.argv[2:3]:
    run_ptp.QS = FINE_QS


def job(j):
    regime, vname, b, g = j
    run_ptp.regimes_for = lambda bench, _r=regime: [_r]
    run_ptp.RUNS = os.path.join(RUNS_ROOT, regime)
    _, _, gg, st = run_ptp.run_cell((vname, b, g))
    return regime, vname, b, gg, st


def jobs(which):
    out = []
    swe = [(b, g) for b, g in cell_keys() if b.startswith("swe")]
    non = [(b, g) for b, g in cell_keys() if not b.startswith("swe")]
    if which in ("swe", "all"):
        # reproduction regimes first, then primary (slow critics), then secondary (fast critics)
        regs = ["cv90_s", "cv5_f"] + [f"cv{c}_s" for c in CVERS if c != 90] + [f"cv{c}_f" for c in CVERS if c != 5]
        out += [(r, v, b, g) for r in regs for v in VARS for b, g in swe]
    if which in ("nonswe", "all"):
        regs = ["cv5_f"] + [f"cv{c}_f" for c in CVERS if c != 5]
        out += [(r, v, b, g) for r in regs for v in VARS for b, g in non]
    if which == "swe_fineq":
        regs = [f"cv{c}_s" for c in CVERS] + [f"cv{c}_f" for c in CVERS]
        out += [(r, "calK_jointcomb", b, g) for r in regs for b, g in swe]
    return out


if __name__ == "__main__":
    import time
    from multiprocessing import Pool
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    J = jobs(which)
    t0 = time.time()
    print(f"{len(J)} jobs", flush=True)
    with Pool(min(12, os.cpu_count() or 4)) as P:
        for n, (r, v, b, gg, st) in enumerate(P.imap_unordered(job, J, chunksize=1), 1):
            print(f"{n}/{len(J)} {time.time() - t0:.0f}s {st} {r} {v} {b} {gg}", flush=True)
    print("ALL DONE", flush=True)
