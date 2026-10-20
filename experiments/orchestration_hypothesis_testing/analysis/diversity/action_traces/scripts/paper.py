"""Paper ground truth parsed from camera_ready tex.

Set PAPER_TEX_DIR to the camera-ready LaTeX folder (full_results.tex, prior_pass_rate.tex,
transition_kernel.tex). Only the paper-reproduction checks need it."""
import os, re
CR = os.environ.get("PAPER_TEX_DIR")
if not CR:
    raise ImportError("PAPER_TEX_DIR is not set: point it at the camera-ready LaTeX folder "
                      "(full_results.tex, prior_pass_rate.tex, transition_kernel.tex)")
CR = os.path.join(CR, "")
BENCH_MAP = {"LCB-hard":"lcb_hard","LCB-medium":"lcb_medium","LCB-easy":"lcb_easy","MBPP+":"mbpp","HumanEval+":"humaneval",
             "SWE-Bench Lite":"swe_lite","SWE-Bench Verified":"swe_verified","HumanEvalFix":"humanevalfix","CodeContests":"codecontests"}
GEN_MAP = {"gpt-5-mini":"gpt5_mini","qwen3-coder":"qwen3_coder","claude-haiku-4.5":"haiku45","claude-sonnet-4.5":"sonnet45",
           "Qwen2.5-Coder-32B":"qwen25_32b","gpt-oss-20b":"gpt_oss_20b"}
COLS = ["BoN","t_syn","t_test","t_llm","FP","SR","Rfx","BG","BDP"]
def _rows(path):
    bench = None
    for line in open(path):
        m = re.match(r"^\s*(.*?)\s*&\s*\\texttt\{([^}]*)\}\s*&(.*)\\\\", line)
        if not m or m.group(2) not in GEN_MAP: continue
        b = m.group(1).strip()
        if b: bench = BENCH_MAP[b]
        yield bench, GEN_MAP[m.group(2)], [c.strip() for c in m.group(3).split("&")]
def num(s):
    s = s.replace("\\mathbf{","").replace("}","").replace("$","").strip()
    if s in ("--",""): return None
    return float(s)
def full_results():
    out = {}
    for b,g,cells in _rows(CR+"full_results.tex"):
        out[(b,g)] = dict(zip(COLS,[num(c) for c in cells]))
    return out
def priors():
    out = {}
    for b,g,cells in _rows(CR+"prior_pass_rate.tex"):
        out[(b,g)] = num(cells[0])
    return out
if __name__ == "__main__":
    fr = full_results(); pr = priors()
    print(len(fr), len(pr))
    for k in fr: print(k, fr[k]["BG"], fr[k]["BDP"], pr.get(k))

def prior_cis():
    out = {}
    for b,g,cells in _rows(CR+"prior_pass_rate.tex"):
        m = re.search(r"\[([\d.]+),\\,([\d.]+)\]", cells[1])
        out[(b,g)] = (num(cells[0]), float(m.group(1)), float(m.group(2)))
    return out
