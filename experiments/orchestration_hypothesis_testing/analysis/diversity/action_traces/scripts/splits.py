import random, math
import numpy as np
def split_variants(ids):
    """ids: list of instance ids. returns name -> (train_set, eval_set)"""
    ids = list(dict.fromkeys(ids)); n = len(ids); out = {}
    for ntn, nt in [("int", int(n*0.75)), ("round", int(round(n*0.75)))]:
        s = sorted(ids); random.Random(42).shuffle(s); out[f"pyR_sorted_{ntn}"] = s[:nt], s[nt:]
        s = list(ids); random.Random(42).shuffle(s); out[f"pyR_file_{ntn}"] = s[:nt], s[nt:]
        s = sorted(ids); o = np.random.default_rng(42).permutation(n); out[f"npG_sorted_{ntn}"] = [s[i] for i in o[:nt]], [s[i] for i in o[nt:]]
        s = sorted(ids); o = np.random.RandomState(42).permutation(n); out[f"npRS_sorted_{ntn}"] = [s[i] for i in o[:nt]], [s[i] for i in o[nt:]]
        s = list(ids); o = np.random.default_rng(42).permutation(n); out[f"npG_file_{ntn}"] = [s[i] for i in o[:nt]], [s[i] for i in o[nt:]]
        s = list(ids); o = np.random.RandomState(42).permutation(n); out[f"npRS_file_{ntn}"] = [s[i] for i in o[:nt]], [s[i] for i in o[nt:]]
        s = sorted(ids); rng = np.random.default_rng(42); s2 = list(s); rng.shuffle(s2); out[f"npGshuf_sorted_{ntn}"] = s2[:nt], s2[nt:]
    return out
