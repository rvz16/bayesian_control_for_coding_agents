"""Search kernel source / rounding / pool restriction per cell for the variant that matches full_results.tex.
usage: repro.py [OUT.pkl]   (needs PAPER_TEX_DIR)"""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(__file__))
from cells import cell_keys, load_cell
from paper import full_results
import replay_traces as rt
fr = full_results()
variants = list(itertools.product(["selfrefine","reflexion"], ["int","round"], [True, False]))
res = {}
for (b, g) in cell_keys():
    gg = g.replace("_local","")
    C = load_cell(b, g)
    pb, pd = fr[(b,gg)]["BG"], fr[(b,gg)]["BDP"]
    row = []
    for kf, cr, rp in variants:
        cell = rt.build_cell(C["cal"], C["sr"], C["rx"], b, kernel_from=kf, cal_rounding=cr, restrict_pool=rp)
        out = rt.run_cell(cell, b, rt.paper_regime(b), include_rules=False)
        u = {k: sum(e["utility"] for e in v)/len(v) for k, v in out.items()}
        dbg, ddp = u["bayesian_greedy"]-u["always_verify"], u["bayesian_DP"]-u["always_verify"]
        okb, okd = abs(round(dbg,1)-pb) < 0.051, abs(round(ddp,1)-pd) < 0.051
        row.append((kf, cr, rp, len(cell["pool"]), round(dbg,2), round(ddp,2), okb, okd))
    res[(b,gg)] = row
    best = [r for r in row if r[6] and r[7]]
    print(f"{b:13s} {gg:12s} paper BG {pb:+6.1f} BDP {pd:+6.1f} | " + " ".join(f"[{r[0][:2]}{r[1][0]}{'P' if r[2] else '-'} n{r[3]} {r[4]:+.1f}/{r[5]:+.1f}{'*' if r[6] and r[7] else ''}]" for r in row))
if len(sys.argv) > 1:  # optional: pickle the per-cell variant table, e.g. repro.py /tmp/repro_variants.pkl
    import pickle; pickle.dump(res, open(sys.argv[1], "wb"))
