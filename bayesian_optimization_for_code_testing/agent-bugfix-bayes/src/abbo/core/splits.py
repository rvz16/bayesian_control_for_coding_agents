"""Bug-level train / validation / test splitting with optional repo stratification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitResult:
    """Indices and identifiers for train / val / test splits."""

    train: list[str]
    val: list[str]
    test: list[str]


def split_bugs(
    bug_ids: list[str],
    repos: list[str] | None = None,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> SplitResult:
    """Split bug IDs into train / val / test, optionally stratified by repo.

    Each bug appears in exactly one split.  When *repos* is provided the
    split is stratified so that each repository's bugs are distributed
    proportionally across the three sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    rng = np.random.default_rng(seed)

    if repos is None:
        return _random_split(bug_ids, train_ratio, val_ratio, rng)

    assert len(repos) == len(bug_ids)
    repo_to_bugs: dict[str, list[str]] = {}
    for bid, repo in zip(bug_ids, repos):
        repo_to_bugs.setdefault(repo, []).append(bid)

    train, val, test = [], [], []
    for _repo, bugs in sorted(repo_to_bugs.items()):
        s = _random_split(bugs, train_ratio, val_ratio, rng)
        train.extend(s.train)
        val.extend(s.val)
        test.extend(s.test)

    return SplitResult(train=train, val=val, test=test)


def _random_split(
    ids: list[str],
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> SplitResult:
    order = rng.permutation(len(ids)).tolist()
    n = len(ids)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n > 2 else 0
    train = [ids[i] for i in order[:n_train]]
    val = [ids[i] for i in order[n_train : n_train + n_val]]
    test = [ids[i] for i in order[n_train + n_val :]]
    return SplitResult(train=train, val=val, test=test)
