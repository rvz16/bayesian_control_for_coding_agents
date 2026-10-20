"""Per-cell artifact versions selected to match the paper (prior_pass_rate.tex and
transition_kernel.tex). Each entry: (artifact_collection, version)."""
import os
from data import DL, load_records

L = "_local"
CAL = {
    ("lcb_hard", "gpt5_mini"): "v0", ("lcb_hard", "qwen3_coder"): "v0", ("lcb_hard", "haiku45"): "v0",
    ("lcb_hard", "sonnet45"): "v0", ("lcb_hard", "qwen25_32b"): "v0", ("lcb_hard", "gpt_oss_20b" + L): "v2",
    ("lcb_medium", "gpt5_mini"): "v0", ("lcb_medium", "qwen3_coder"): "v0", ("lcb_medium", "haiku45"): "v0",
    ("lcb_medium", "sonnet45"): "v0", ("lcb_medium", "qwen25_32b"): "v2", ("lcb_medium", "gpt_oss_20b" + L): "v2",
    ("lcb_easy", "gpt5_mini"): "v0", ("lcb_easy", "qwen3_coder"): "v0", ("lcb_easy", "haiku45"): "v0",
    ("lcb_easy", "sonnet45"): "v0", ("lcb_easy", "qwen25_32b"): "v0", ("lcb_easy", "gpt_oss_20b" + L): "v2",
    ("mbpp", "gpt5_mini"): "v1", ("mbpp", "qwen3_coder"): "v1", ("mbpp", "haiku45"): "v1",
    ("mbpp", "sonnet45"): "v1", ("mbpp", "qwen25_32b"): "v1", ("mbpp", "gpt_oss_20b" + L): "v0",
    ("humaneval", "gpt5_mini"): "v1", ("humaneval", "qwen3_coder"): "v1", ("humaneval", "haiku45"): "v1",
    ("humaneval", "sonnet45"): "v1", ("humaneval", "qwen25_32b"): "v1", ("humaneval", "gpt_oss_20b" + L): "v0",
    ("swe_lite", "gpt5_mini"): "v1", ("swe_lite", "qwen3_coder"): "v1", ("swe_lite", "haiku45"): "v1",
    ("swe_lite", "sonnet45"): "v1", ("swe_lite", "qwen25_32b"): "v1", ("swe_lite", "gpt_oss_20b" + L): "v0",
    ("swe_verified", "gpt5_mini"): "v1", ("swe_verified", "qwen3_coder"): "v1", ("swe_verified", "haiku45"): "v2",
    ("swe_verified", "sonnet45"): "v2", ("swe_verified", "qwen25_32b"): "v1", ("swe_verified", "gpt_oss_20b" + L): "v0",
    ("humanevalfix", "gpt5_mini"): "v0", ("humanevalfix", "qwen3_coder"): "v0", ("humanevalfix", "haiku45"): "v0",
    ("humanevalfix", "sonnet45"): "v0", ("humanevalfix", "qwen25_32b"): "v0", ("humanevalfix", "gpt_oss_20b" + L): "v0",
    ("codecontests", "gpt5_mini"): "v0", ("codecontests", "qwen3_coder"): "v0", ("codecontests", "haiku45"): "v0",
    ("codecontests", "sonnet45"): "v0", ("codecontests", "qwen25_32b"): "v0", ("codecontests", "gpt_oss_20b" + L): "v0",
}
# (selfrefine version, reflexion version); chosen where they reproduce transition_kernel.tex
ITER = {
    ("lcb_hard", "gpt5_mini"): ("v0", "v0"), ("lcb_hard", "qwen3_coder"): ("v0", "v0"), ("lcb_hard", "haiku45"): ("v1", "v1"),
    ("lcb_hard", "sonnet45"): ("v1", "v1"), ("lcb_hard", "qwen25_32b"): ("v2", "v2"), ("lcb_hard", "gpt_oss_20b" + L): ("v4", "v4"),
    ("lcb_medium", "gpt5_mini"): ("v0", "v0"), ("lcb_medium", "qwen3_coder"): ("v0", "v0"), ("lcb_medium", "haiku45"): ("v1", "v1"),
    ("lcb_medium", "sonnet45"): ("v1", "v1"), ("lcb_medium", "qwen25_32b"): ("v2", "v2"), ("lcb_medium", "gpt_oss_20b" + L): ("v4", "v4"),
    ("lcb_easy", "gpt5_mini"): ("v0", "v0"), ("lcb_easy", "qwen3_coder"): ("v0", "v0"), ("lcb_easy", "haiku45"): ("v1", "v1"),
    ("lcb_easy", "sonnet45"): ("v1", "v1"), ("lcb_easy", "qwen25_32b"): ("v2", "v2"), ("lcb_easy", "gpt_oss_20b" + L): ("v3", "v2"),
    ("mbpp", "gpt5_mini"): ("v0", "v0"), ("mbpp", "qwen3_coder"): ("v1", "v1"), ("mbpp", "haiku45"): ("v0", "v0"),
    ("mbpp", "sonnet45"): ("v0", "v0"), ("mbpp", "qwen25_32b"): ("v0", "v0"), ("mbpp", "gpt_oss_20b" + L): ("v0", "v0"),
    ("humaneval", "gpt5_mini"): ("v1", "v1"), ("humaneval", "qwen3_coder"): ("v1", "v1"), ("humaneval", "haiku45"): ("v0", "v0"),
    ("humaneval", "sonnet45"): ("v0", "v0"), ("humaneval", "qwen25_32b"): ("v0", "v0"), ("humaneval", "gpt_oss_20b" + L): ("v0", "v0"),
    ("swe_lite", "gpt5_mini"): ("v0", "v0"), ("swe_lite", "qwen3_coder"): ("v0", "v0"), ("swe_lite", "haiku45"): ("v1", "v1"),
    ("swe_lite", "sonnet45"): ("v0", "v0"), ("swe_lite", "qwen25_32b"): ("v1", "v0"), ("swe_lite", "gpt_oss_20b" + L): ("v0", "v0"),
    ("swe_verified", "gpt5_mini"): ("v0", "v0"), ("swe_verified", "qwen3_coder"): ("v0", "v0"), ("swe_verified", "haiku45"): ("v1", "v1"),
    ("swe_verified", "sonnet45"): ("v1", "v1"), ("swe_verified", "qwen25_32b"): ("v1", "v1"), ("swe_verified", "gpt_oss_20b" + L): ("v0", "v0"),
    ("humanevalfix", "gpt5_mini"): ("v0", "v0"), ("humanevalfix", "qwen3_coder"): ("v0", "v0"), ("humanevalfix", "haiku45"): ("v1", "v0"),
    ("humanevalfix", "sonnet45"): ("v1", "v0"), ("humanevalfix", "qwen25_32b"): ("v0", "v0"), ("humanevalfix", "gpt_oss_20b" + L): ("v0", "v0"),
    ("codecontests", "gpt5_mini"): ("v1", "v1"), ("codecontests", "qwen3_coder"): ("v1", "v1"), ("codecontests", "haiku45"): ("v2", "v2"),
    ("codecontests", "sonnet45"): ("v2", "v3"), ("codecontests", "qwen25_32b"): ("v0", "v0"), ("codecontests", "gpt_oss_20b" + L): ("v0", "v0"),
}


def art_dir(name, ver):
    d = f"{DL}/{name}__{ver}"
    return d if os.path.isdir(d) else None


def cell_keys():
    return list(CAL.keys())


def load_cell(bench, gen, cal_ver=None, sr_ver=None, rx_ver=None):
    cv = cal_ver or CAL[(bench, gen)]
    sv, rv = ITER[(bench, gen)]
    sv = sr_ver or sv; rv = rx_ver or rv
    cal_name = f"{bench}_{gen}_critic_results"
    sr_name = f"{bench}_{gen}_selfrefine_records"
    rx_name = f"{bench}_{gen}_reflexion_records"
    cd, sd, rd = art_dir(cal_name, cv), art_dir(sr_name, sv), art_dir(rx_name, rv)
    return {
        "cal": load_records(cd), "sr": load_records(sd) if sd else [], "rx": load_records(rd) if rd else [],
        "versions": {"calibration": f"{cal_name}:{cv}", "selfrefine": f"{sr_name}:{sv}" if sd else None,
                     "reflexion": f"{rx_name}:{rv}" if rd else None},
    }
