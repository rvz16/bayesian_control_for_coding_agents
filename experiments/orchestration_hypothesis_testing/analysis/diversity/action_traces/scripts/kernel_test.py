import sys, os, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import *; from paper import CR, GEN_MAP, BENCH_MAP
from splits import split_variants
# parse kernel table
K = {}; bench=None
for line in open(CR+"transition_kernel.tex"):
    m = re.match(r"^\s*(.*?)\s*&\s*\\texttt\{([^}]*)\}\s*&(.*)\\\\", line)
    if not m or m.group(2) not in GEN_MAP: continue
    if m.group(1).strip(): bench = BENCH_MAP[m.group(1).strip()]
    vals = [c.strip().strip("$") for c in m.group(3).split("&")]
    f = lambda s: None if s=="--" else float(s)
    K[(bench, GEN_MAP[m.group(2)])] = {"selfrefine": (f(vals[0]), f(vals[1])), "reflexion": (f(vals[2]), f(vals[3]))}
def kernel(recs, ids=None):
    by = {}
    for r in recs:
        if ids is not None and str(r["instance_id"]) not in ids: continue
        by.setdefault(str(r["instance_id"]), []).append(r)
    c = {"00":0,"01":0,"10":0,"11":0}
    for i, rs in by.items():
        rs.sort(key=lambda r: r["step"])
        for a, b in zip(rs, rs[1:]):
            if a.get("Y") is None or b.get("Y") is None: continue
            c[f"{int(a['Y'])}{int(b['Y'])}"] += 1
    pf = (c["01"]+1)/(c["00"]+c["01"]+2); pb = (c["10"]+1)/(c["10"]+c["11"]+2)
    return pf, pb
if __name__ == "__main__":
    for (b,g), meth in K.items():
        for m,(tf,tb) in meth.items():
            if tf is None: continue
            hits=[]
            for (n,v,d,c) in versions(b,g,f"{m}_records"):
                R = load_records(d); ids=[str(r["instance_id"]) for r in R]
                for sname,(tr,ev) in split_variants(ids).items():
                    pf,pb = kernel(R,set(tr))
                    if abs(round(pf,3)-tf)<0.0015 and abs(round(pb,3)-tb)<0.0015: hits.append(f"{v}/{sname}")
                pf,pb=kernel(R)
                if abs(round(pf,3)-tf)<0.0015 and abs(round(pb,3)-tb)<0.0015: hits.append(f"{v}/ALL")
                # cal-train ids split
                for (cn,cv,cd,cc) in versions(b,g,"critic_results"):
                    cids=[str(r["instance_id"]) for r in load_records(cd)]
                    tr,_=split_variants(cids)["npG_sorted_int"]
                    pf,pb=kernel(R,set(tr))
                    if abs(round(pf,3)-tf)<0.0015 and abs(round(pb,3)-tb)<0.0015: hits.append(f"{v}/calTrain{cv}")
            print(b,g,m,tf,tb,hits[:8])
