import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import *; from paper import priors
pr = priors()
def npg_split(ids):
    s = sorted(set(ids)); o = np.random.default_rng(42).permutation(len(s)); nt = int(len(s)*0.75)
    return [s[i] for i in o[:nt]], [s[i] for i in o[nt:]]
for b in BENCHES:
    for g in GENS:
        pp = pr[(b,g)]; out=[]
        for (cn,cv,cd,cc) in versions(b,g,"critic_results"):
            R = load_records(cd); ids=[str(r["instance_id"]) for r in R]
            tr, ev = npg_split(ids); trs=set(tr)
            T=[r for r in R if str(r["instance_id"]) in trs]; k=sum(1 for r in T if r["Y"]); n=len(T)
            A=sum(1 for r in R if r["Y"])/len(R)
            tags=[]
            for nm,p in [("all",A),("tr",k/n),("trLap",(k+1)/(n+2))]:
                if abs(round(p,3)-pp)<0.0015: tags.append(nm)
            out.append(f"{cn.replace(b+'_','').replace('_critic_results','')}:{cv}({cc[:10]}) all={A:.3f} tr={k/n:.3f} n={n} {','.join(tags)}")
        print(b,g,pp," | ".join(out))
