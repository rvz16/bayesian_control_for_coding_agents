import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from cells import load_cell, art_dir
from data import load_records, load_json
import replay_traces as rt
def run(b, g, cal=None, sr=None, rx=None, kf="selfrefine", regime=None, insample=False, kernel_override=None, restrict=False):
    C = load_cell(b, g, cal, sr, rx)
    cell = rt.build_cell(C["cal"], C["sr"], C["rx"], b, kernel_from=kf, restrict_pool=restrict)
    if insample:
        allrecs = [r for i in cell["by"] for r in cell["by"][i]]
        f = ["L0_syntax","L3_llm_review"] if b.startswith("swe") else ["L0_syntax","L2_public_tests","L3_llm_review"]
        cell["prior"], cell["likes"] = rt.fit_likelihoods(allrecs, f)
    if kernel_override: cell["kernel"] = kernel_override
    out = rt.run_cell(cell, b, regime or rt.paper_regime(b), include_rules=False)
    u = {k: sum(e["utility"] for e in v)/len(v) for k, v in out.items()}
    return round(u["bayesian_greedy"]-u["always_verify"],2), round(u["bayesian_DP"]-u["always_verify"],2), len(cell["pool"]), round(cell["prior"],3)
def kfile(b,g,m,v):
    d=art_dir(f"{b}_{g}_{m}_kernel", v); j=load_json(d); return j.get("kernel_all", j)
cases = {
 "SWE-V gpt5_mini (paper +74.0; colleague +31.5)": [("paper cal v1",dict(b="swe_verified",g="gpt5_mini")),("cal v3 (489 inst, Aug-29)",dict(b="swe_verified",g="gpt5_mini",cal="v3")),("cal v2 (482 inst, Aug-28)",dict(b="swe_verified",g="gpt5_mini",cal="v2"))],
 "SWE-V sonnet45 (paper +66.0; colleague +37.2)": [("paper cal v2",dict(b="swe_verified",g="sonnet45")),("cal v3 (489 inst, Aug-30)",dict(b="swe_verified",g="sonnet45",cal="v3")),("cal v1 (500 inst, May-25)",dict(b="swe_verified",g="sonnet45",cal="v1"))],
 "LCB-easy gpt5_mini (paper +0.6; colleague +23.9)": [("paper (SR kernel, 75% fit)",dict(b="lcb_easy",g="gpt5_mini")),("Reflexion kernel",dict(b="lcb_easy",g="gpt5_mini",kf="reflexion")),
      ("full SR kernel artifact (100% iter)",dict(b="lcb_easy",g="gpt5_mini",kernel_override=None,_k=("lcb_easy","gpt5_mini","selfrefine","v0"))),
      ("full Rfx kernel artifact",dict(b="lcb_easy",g="gpt5_mini",_k=("lcb_easy","gpt5_mini","reflexion","v0"))),
      ("in-sample fit (all instances)",dict(b="lcb_easy",g="gpt5_mini",insample=True)),("slow regime",dict(b="lcb_easy",g="gpt5_mini",regime="slow"))],
 "CC gpt_oss_20b (paper +11.7; colleague -0.3)": [("our protocol",dict(b="codecontests",g="gpt_oss_20b_local")),("Reflexion kernel",dict(b="codecontests",g="gpt_oss_20b_local",kf="reflexion")),
      ("pool restricted to SR&Rfx",dict(b="codecontests",g="gpt_oss_20b_local",restrict=True)),("in-sample fit",dict(b="codecontests",g="gpt_oss_20b_local",insample=True)),
      ("full SR kernel artifact",dict(b="codecontests",g="gpt_oss_20b_local",_k=("codecontests","gpt_oss_20b_local","selfrefine","v0")))],
 "LCB-med qwen25_32b (paper 0.0; colleague -5.0)": [("paper cal v2",dict(b="lcb_medium",g="qwen25_32b")),("cal v0 (May-17, prior 0.25)",dict(b="lcb_medium",g="qwen25_32b",cal="v0")),("cal v1 (May-24, 168 inst)",dict(b="lcb_medium",g="qwen25_32b",cal="v1")),
      ("Reflexion kernel",dict(b="lcb_medium",g="qwen25_32b",kf="reflexion")),("in-sample fit",dict(b="lcb_medium",g="qwen25_32b",insample=True))],
}
res = {}
for title, L in cases.items():
    print("==", title); res[title] = []
    for lab, kw in L:
        kw = dict(kw); k = kw.pop("_k", None)
        if k: kw["kernel_override"] = kfile(*k)
        r = run(**kw); res[title].append((lab, r)); print(f"   {lab:40s} BG {r[0]:+7.2f}  BDP {r[1]:+7.2f}  n_eval {r[2]}  prior {r[3]}")
json.dump(res, open(sys.argv[1], "w"), indent=1)
