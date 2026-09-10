"""Nearest-neighbour "mismatch" controls.

The mismatch control should be a fluent, on-topic, *wrong* answer. Picking a
random other question gives an off-topic answer that every judge trivially
rejects, so instead we take the reference answer of the most lexically similar
other question in the split (word-level Jaccard over question + reference).
"""

from __future__ import annotations

import re
from typing import Any

_STOP = set(
    "the a an of in and or to is are was were for with on at by from as that this "
    "which what who whom whose how why when where be been being has have had do does "
    "did should would could can may might most likely following patient presents "
    "year old man woman male female history".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9\-]+", text.lower()) if t not in _STOP}


def nearest_neighbour_map(
    targets: list[dict[str, Any]], pool: list[dict[str, Any]]
) -> list[int]:
    """For each target row, index into ``pool`` of the most similar *other* row.

    Rows are compared by Jaccard similarity of their question+reference tokens.
    A target is 'the same row' as a pool entry when both share ``dataset_row``.
    """
    pool_tok = [_tokens(r["question"] + " " + r["reference_answer"]) for r in pool]
    out = []
    for t in targets:
        tt = _tokens(t["question"] + " " + t["reference_answer"])
        best, best_j = -1, -1.0
        for j, pt in enumerate(pool_tok):
            if pool[j].get("dataset_row") == t.get("dataset_row"):
                continue
            inter = len(tt & pt)
            if not inter:
                continue
            jac = inter / len(tt | pt)
            if jac > best_j:
                best, best_j = j, jac
        out.append(best)
    return out
