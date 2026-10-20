"""Shared helpers. Imports (never copies) the diversity_fixes engine / protocol / runner."""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DF = os.path.abspath(os.path.join(HERE, "..", "..", "diversity_fixes", "scripts"))
AT = os.path.abspath(os.path.join(HERE, "..", "..", "action_traces"))
if DF not in sys.path:
    sys.path.insert(0, DF)

import json

import numpy as np

from cells import cell_keys  # noqa: E402
from runner import _cell  # noqa: E402
from analyze_util import boot_ci  # noqa: E402,F401  (paired bootstrap, 2000, seed 42)

MATCHED = {tuple(k.split("/")): v["reproduces_paper_BG_BDP"] for k, v in json.load(open(os.path.join(AT, "versions.json")))["cells"].items()}


def gname(g):
    return g.replace("_local", "")


def all_cells():
    """[(bench, gen_key, gen_short, cell)] in cells.CAL order; cell = protocol.build_cell output (L1 not a critic)."""
    out = []
    for b, g in cell_keys():
        _, cell = _cell(b, g, False)
        out.append((b, g, gname(g), cell))
    return out


def fast_auc(s, y):
    """Mann-Whitney AUROC with average ranks (ties count 1/2). nan if one class missing."""
    from scipy.stats import rankdata
    s = np.asarray(s, float); y = np.asarray(y, int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
