"""Killing kinds this package provides.

Death reporter reads a fluorescent reporter of a death event. Label-free
judges death from brightfield morphology.
"""

from __future__ import annotations

from typing import Literal

DEATH_REPORTER = "death-reporter"
LABEL_FREE = "label-free"

KillingAssayKind = Literal["death-reporter", "label-free"]

ASSAY_KINDS: tuple[KillingAssayKind, ...] = (DEATH_REPORTER, LABEL_FREE)


def is_killing_kind(value: str) -> bool:
    return value in ASSAY_KINDS
