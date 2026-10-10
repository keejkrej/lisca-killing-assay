"""Sample mapping read from a LiSCA ``assay.json``.

Lisca owns the on-disk contract. This reader is the small slice the killing
commands need: sample names, position lists, and the shared channel roles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleChannels:
    name: str
    positions: tuple[int, ...]
    signal: tuple[int, ...]
    segmentation: int


def load_samples(workspace: Path) -> tuple[list[SampleChannels], float]:
    path = workspace / "assay.json"
    payload = json.loads(path.read_text())
    interval = payload.get("interval") or {}
    minutes = _interval_minutes(interval.get("value"), interval.get("unit"))
    channels = (payload.get("analysis") or {}).get("channels") or {}
    signal = tuple(int(channel) for channel in channels.get("signal") or [])
    segmentation = int(channels.get("segmentation", 0))
    samples: list[SampleChannels] = []
    for sample in payload.get("samples") or []:
        samples.append(
            SampleChannels(
                name=str(sample["name"]),
                positions=_parse_positions(str(sample.get("positions", ""))),
                signal=signal,
                segmentation=segmentation,
            )
        )
    if not samples:
        raise ValueError(f"{path} defines no samples")
    return samples, minutes


def _interval_minutes(amount: object, unit: object) -> float:
    if not isinstance(amount, int | float) or isinstance(amount, bool):
        raise ValueError("assay.json interval.value must be a positive number")
    if amount <= 0:
        raise ValueError("assay.json interval.value must be positive")
    factor = {"second": 1.0 / 60.0, "minute": 1.0, "hour": 60.0, None: 1.0}.get(
        unit if isinstance(unit, str) or unit is None else "invalid"
    )
    if factor is None:
        raise ValueError(f"unsupported interval unit {unit!r}")
    return float(amount) * factor


def _parse_positions(raw: str) -> tuple[int, ...]:
    found: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) == 1:
            position = int(parts[0])
        elif len(parts) in (2, 3):
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step == 0 or stop < start:
                raise ValueError(f"invalid position range: {token}")
            for position in range(start, stop + 1, step):
                if position not in seen:
                    seen.add(position)
                    found.append(position)
            continue
        else:
            raise ValueError(f"invalid position range: {token}")
        if position not in seen:
            seen.add(position)
            found.append(position)
    return tuple(found)
