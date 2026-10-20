"""chain_overlap.csv: how Self-Refine / Reflexion chains relate to the calibration patches, per cell."""
import sys
sys.dont_write_bytecode = True
import csv, json, os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import cell_keys, CAL
from chains import ident, all_versions
from runner import _cell, chain_data

OUT = sys.argv[1]
AT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "action_traces", "versions.json")
atv = json.load(open(AT))["cells"]
rows = []
for (b, g) in cell_keys():
    gg = g.replace("_local", "")
    C, cell = _cell(b, g, False)
    likes = cell["likes"]["critic_likelihoods"]
    row = dict(bench=b, gen=gg, calibration=f"{b}_{g}_critic_results:{CAL[(b, g)]}", n_cal_instances=len(cell["by"]),
               n_train=len(cell["train_ids"]), n_eval=len(cell["pool"]), matched_paper_cell=atv[f"{b}/{gg}"]["reproduces_paper_BG_BDP"],
               paper_kernel_P_fix=atv[f"{b}/{gg}"]["kernel_selfrefine_75pct"]["P_fix"],
               paper_kernel_P_break=atv[f"{b}/{gg}"]["kernel_selfrefine_75pct"]["P_break"])
    for tag, (method, mode) in {"sr_at": ("selfrefine", "at"), "sr_mx": ("selfrefine", "maxov"), "rx_at": ("reflexion", "at")}.items():
        cd = chain_data(b, g, method, mode)
        ro, R, st, rst = cd["rooted"], cd["rep"], cd["clean_stats"], cd["root_stats"]
        ev = set(cell["pool"]); tr = set(cell["train_ids"])
        created = {v: c for v, _, c, _ in all_versions(b, g, method)}
        ver = cd["version"].rsplit(":", 1)[1] if cd["version"] else None
        trans = [(x, y) for ch in ro.values() for x, y in zip(ch, ch[1:])]
        # critic verdict consistency on repaired chain steps (train, steps>=1) vs calibration-train likelihoods
        diffs = {}
        for f, l in likes.items():
            c = Counter()
            for i in tr:
                for r in ro.get(i, [])[1:]:
                    c[(int(r["Y"]), bool(r.get(f)))] += 1
            if c[(1, True)] + c[(1, False)] and c[(0, True)] + c[(0, False)]:
                p1 = (c[(1, True)] + 1) / (c[(1, True)] + c[(1, False)] + 2); p0 = (c[(0, True)] + 1) / (c[(0, True)] + c[(0, False)] + 2)
                diffs[f] = round(max(abs(p1 - l["P_pass_given_Y1"]), abs(p0 - l["P_pass_given_Y0"])), 3)
        row.update({
            f"{tag}_version": cd["version"], f"{tag}_created": created.get(ver),
            f"{tag}_n_chain_instances_clean": cd["n_chains"],
            f"{tag}_dropped_inconsistent_dup": st.get("drop_inconsistent_duplicate_steps", 0),
            f"{tag}_dedup_consistent_dup": st.get("dedup_consistent_duplicates", 0),
            f"{tag}_truncated_missing_Y_or_critic": st.get("truncated_missing_Y_or_critic", 0),
            f"{tag}_n_step0_eq_p0": len(ro), f"{tag}_n_step0_ne_p0": rst.get("step0_not_p0", 0),
            f"{tag}_n_step0_matches_only_p1_or_p2": rst.get("step0_matches_p1_or_p2_only", 0),
            f"{tag}_n_replayable_all": sum(1 for c in ro.values() if len(c) >= 2),
            f"{tag}_n_replayable_eval": sum(1 for i, c in ro.items() if i in ev and len(c) >= 2),
            f"{tag}_frac_eval_replayable": round(sum(1 for i, c in ro.items() if i in ev and len(c) >= 2) / len(ev), 3),
            f"{tag}_mean_chain_len_rooted": round(sum(len(c) for c in ro.values()) / len(ro), 3) if ro else None,
            f"{tag}_n_transitions": len(trans),
            f"{tag}_frac_transitions_unchanged": round(sum(1 for x, y in trans if ident(x) == ident(y)) / len(trans), 3) if trans else None,
            f"{tag}_frac_unchanged_with_Y_flip": round(sum(1 for x, y in trans if ident(x) == ident(y) and int(x["Y"]) != int(y["Y"])) / max(1, sum(1 for x, y in trans if ident(x) == ident(y))), 3) if trans else None,
            f"{tag}_train_kernel_P_fix": round(R["P_fix"], 4) if R and R["n_pairs"] else None,
            f"{tag}_train_kernel_P_break": round(R["P_break"], 4) if R and R["n_pairs"] else None,
            f"{tag}_train_pairs": R["n_pairs"] if R else 0, f"{tag}_train_counts": json.dumps(R["counts"]) if R else None,
            f"{tag}_train_q_same_Y1": round(R["q_same_Y1"], 3) if R and R["n_pairs"] else None,
            f"{tag}_train_q_same_Y0": round(R["q_same_Y0"], 3) if R and R["n_pairs"] else None,
            f"{tag}_train_P_fix_changed": round(R["P_fix_chg"], 3) if R and R["n_pairs"] else None,
            f"{tag}_train_P_break_changed": round(R["P_break_chg"], 3) if R and R["n_pairs"] else None,
            f"{tag}_critic_like_maxabsdiff_chain_vs_cal": json.dumps(diffs),
        })
    rows.append(row)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", len(rows))
for r in rows:
    print(f"{r['bench']:13s}{r['gen']:12s} at {r['sr_at_n_step0_eq_p0']:4d}/{r['n_cal_instances']} repl_eval {r['sr_at_frac_eval_replayable']:.2f} | mx {r['sr_mx_version'].split('records:')[1] if r['sr_mx_version'] else None} {r['sr_mx_created'][:10] if r['sr_mx_created'] else ''} {r['sr_mx_n_step0_eq_p0']:4d} repl_eval {r['sr_mx_frac_eval_replayable']:.2f} unch {r['sr_mx_frac_transitions_unchanged']} pf {r['sr_mx_train_kernel_P_fix']} | rx {r['rx_at_n_step0_eq_p0']} {r['rx_at_frac_eval_replayable']:.2f} | critdiff {r['sr_mx_critic_like_maxabsdiff_chain_vs_cal']}")
