"""Killing classifier definitions shared with the Rust crate.

``label`` true means the cell is still alive. ``p_dead`` is the probability of
the absent class. Once a crop is labeled dead, later alive labels are cleared.
Death time is the end of the longest prefix whose alive fraction is at least
``CLEAN_THRESHOLD``. The kill curve counts crops whose death time is still ahead,
and drops crops whose death time is 0.
"""

from __future__ import annotations

import math
from collections import defaultdict

DEAD_LABEL_THRESHOLD = 0.5
CLEAN_THRESHOLD = 0.8

Row = dict[str, object]
CropKey = tuple[int, str, int]


def dead_probability(absent_logit: float, present_logit: float) -> float:
    """P(absent). The first logit is the absent class."""
    peak = max(absent_logit, present_logit)
    first = math.exp(absent_logit - peak)
    second = math.exp(present_logit - peak)
    return first / (first + second)


def alive_label(p_dead: float) -> bool:
    """True while ``p_dead`` is below the dead threshold."""
    return p_dead < DEAD_LABEL_THRESHOLD


def clean_predictions(rows: list[Row]) -> list[Row]:
    grouped: dict[CropKey, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(dict(row))
    cleaned: list[Row] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: int(item["t"]))
        seen_false = False
        for row in group:
            if not bool(row["label"]):
                seen_false = True
                cleaned.append(row)
            elif seen_false:
                flipped = dict(row)
                flipped["label"] = False
                cleaned.append(flipped)
            else:
                cleaned.append(row)
    cleaned.sort(
        key=lambda item: (
            int(item["pos"]),
            str(item["sample"]),
            int(item["crop"]),
            int(item["t"]),
        )
    )
    return cleaned


def death_times(rows: list[Row]) -> dict[CropKey, int]:
    grouped: dict[CropKey, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(row)
    found: dict[CropKey, int] = {}
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item["t"]))
        t_min = int(ordered[0]["t"]) if ordered else 0
        true_ts = [int(item["t"]) for item in ordered if bool(item["label"])]
        if not true_ts:
            found[key] = 0
            continue
        chosen_end: int | None = None
        for end in reversed(true_ts):
            span = [item for item in ordered if t_min <= int(item["t"]) <= end]
            n_true = sum(1 for item in span if bool(item["label"]))
            if span and n_true / len(span) >= CLEAN_THRESHOLD:
                chosen_end = end
                break
        if chosen_end is None:
            chosen_end = true_ts[0]
        span_duration = chosen_end - t_min + 1
        found[key] = 0 if span_duration == 1 else chosen_end
    return found


def kill_curve(times: dict[CropKey, int], sample: str) -> list[tuple[int, int]]:
    sample_deaths = [
        death
        for (_pos, name, _crop), death in times.items()
        if name == sample and death > 0
    ]
    if not sample_deaths:
        return []
    observed = [death for (_pos, name, _crop), death in times.items() if name == sample]
    max_t = max([*observed, *sample_deaths])
    return [
        (t, sum(1 for death in sample_deaths if death >= t)) for t in range(max_t + 1)
    ]


def _key(row: Row) -> CropKey:
    return (int(row["pos"]), str(row["sample"]), int(row["crop"]))
