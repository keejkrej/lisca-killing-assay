"""Fluorescent engagement counts.

One fluorescence scale per position: background is the median, and the
threshold sits eight robust-noise widths above it. Round spots about one
engager across are kept. A peanut of two touching spots is split into two
circles. A spot inside the crop is a possible engager, not proof it touches
the tumor; tumor touch is still recorded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_COMPONENT_PIXELS = 4
SIGNAL_SIGMA = 6.0
ENGAGER_DIAMETER_PX = 10.0
GLOBAL_NOISE_SIGMAS = 8.0
SEED_SPACING_FRACTION = 0.55
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class EngagementCounts:
    tumor_cells: int
    t_cells: int
    engagements: int


@dataclass(frozen=True)
class FluorescenceScale:
    low: float
    threshold: float


class FluorescenceHistogram:
    """Histogram of fluorescence values in ``[0, 65535]``."""

    def __init__(self) -> None:
        self._bins = np.zeros(65536, dtype=np.uint64)
        self.count = 0

    def sample(self, pixels: np.ndarray) -> None:
        values = np.asarray(pixels, dtype=np.float64).ravel()
        if values.size == 0:
            return
        chosen = values[::4]
        finite = chosen[np.isfinite(chosen) & (chosen >= 0.0)]
        if finite.size == 0:
            return
        bins = np.rint(finite).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < self._bins.size)]
        if bins.size == 0:
            return
        counts = np.bincount(bins, minlength=self._bins.size)
        self._bins += counts.astype(np.uint64)
        self.count += int(bins.size)

    def scale(self) -> FluorescenceScale:
        if self.count == 0:
            return FluorescenceScale(low=0.0, threshold=math.inf)
        low = _percentile_bin(self._bins, self.count, 0.5)
        deviations = np.zeros(self._bins.size, dtype=np.uint64)
        occupied = np.flatnonzero(self._bins)
        for bin_index in occupied:
            deviation = int(round(abs(float(bin_index) - low)))
            slot = min(deviation, deviations.size - 1)
            deviations[slot] += self._bins[bin_index]
        mad = _percentile_bin(deviations, self.count, 0.5)
        sigma = _MAD_TO_SIGMA * mad
        threshold = low if sigma == 0.0 else low + GLOBAL_NOISE_SIGMAS * sigma
        return FluorescenceScale(low=low, threshold=threshold)


def tumor_mask(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float64).ravel()
    threshold = _otsu_threshold(values)
    if threshold is None:
        return np.zeros(values.size, dtype=bool)
    above = values > threshold
    above_count = int(above.sum())
    below_count = values.size - above_count
    if above_count == 0 or below_count == 0:
        return np.zeros(values.size, dtype=bool)
    if above_count <= below_count:
        return above
    return ~above


def mask_with_scale(pixels: np.ndarray, scale: FluorescenceScale) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float64).ravel()
    return np.isfinite(values) & (values > scale.threshold)


def tcell_mask(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float64).ravel()
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    mean = float(values.mean())
    std = float(values.std())
    if std == 0.0:
        return np.zeros(values.size, dtype=bool)
    return values > mean + SIGNAL_SIGMA * std


def count_engagements(
    tumor: np.ndarray,
    tcells: np.ndarray,
    width: int,
    height: int,
) -> EngagementCounts:
    tumor_mask_values = np.asarray(tumor, dtype=bool).ravel()
    tcell_mask_values = np.asarray(tcells, dtype=bool).ravel()
    if (
        width == 0
        or height == 0
        or tumor_mask_values.size != width * height
        or tcell_mask_values.size != tumor_mask_values.size
    ):
        raise ValueError(
            f"engagement frame is {tumor_mask_values.size} tumor / "
            f"{tcell_mask_values.size} t-cell pixels, expected {width}x{height}"
        )
    tumor_labels = label_components(tumor_mask_values, width, height)
    tcell_labels = split_round_components(
        tcell_mask_values, width, height, ENGAGER_DIAMETER_PX
    )
    engaged: set[int] = set()
    tcell_ids: set[int] = set()
    for index, label in enumerate(tcell_labels.tolist()):
        if label == 0:
            continue
        tcell_ids.add(label)
        if label in engaged:
            continue
        x = index % width
        y = index // width
        if _touches_tumor(tumor_labels, width, height, x, y):
            engaged.add(label)
    return EngagementCounts(
        tumor_cells=component_count(tumor_labels),
        t_cells=len(tcell_ids),
        engagements=len(engaged),
    )


def split_round_components(
    mask: np.ndarray,
    width: int,
    height: int,
    diameter_px: float,
) -> np.ndarray:
    values = np.asarray(mask, dtype=bool).ravel()
    if values.size != width * height or width == 0 or height == 0:
        return np.zeros(values.size, dtype=np.uint32)
    distance = _distance_to_background(values, width, height)
    spacing = max(diameter_px * SEED_SPACING_FRACTION, 1.0)
    min_distance = max(diameter_px * 0.25, 1.0)
    seeds = _engager_seeds(distance, values, width, height, spacing, min_distance)
    if not seeds:
        labels = label_components(values, width, height)
    else:
        labels = _watershed(distance, values, width, height, seeds)
        labels = _drop_small_components(labels, MIN_COMPONENT_PIXELS)
    return _keep_diameter(labels, diameter_px)


def label_components(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool).ravel()
    labels = np.zeros(values.size, dtype=np.uint32)
    sizes = [0]
    next_label = 1
    for start in range(values.size):
        if not values[start] or labels[start] != 0:
            continue
        stack = [start]
        labels[start] = next_label
        size = 0
        while stack:
            index = stack.pop()
            size += 1
            x = index % width
            y = index // width
            for neighbor in _neighbors(width, height, x, y):
                if values[neighbor] and labels[neighbor] == 0:
                    labels[neighbor] = next_label
                    stack.append(neighbor)
        sizes.append(size)
        next_label += 1
    for index, label in enumerate(labels):
        if label != 0 and sizes[int(label)] < MIN_COMPONENT_PIXELS:
            labels[index] = 0
    return labels


def component_count(labels: np.ndarray) -> int:
    present = np.asarray(labels).ravel()
    present = present[present != 0]
    if present.size == 0:
        return 0
    return int(np.unique(present).size)


def full_frame_roi_stats(frame: np.ndarray) -> tuple[int, float, float, float]:
    """Area, sum, 10th-percentile background, and corrected fluorescence."""

    values = np.asarray(frame, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    area = int(finite.size) if values.size else 0
    if values.size == 0:
        return 0, 0.0, 0.0, 0.0
    intensity = float(np.nansum(values))
    if finite.size == 0:
        background = 0.0
    elif finite.size == 1:
        background = float(finite[0])
    else:
        background = float(np.quantile(finite, 0.1, method="linear"))
    corrected = intensity - area * background
    return area, intensity, background, corrected


def _percentile_bin(bins: np.ndarray, count: int, quantile: float) -> float:
    if count == 0:
        return 0.0
    target = int(round((count - 1) * quantile))
    seen = 0
    for bin_index, bin_count in enumerate(bins.tolist()):
        seen += int(bin_count)
        if seen > target:
            return float(bin_index)
    return 0.0


def _otsu_threshold(pixels: np.ndarray) -> float | None:
    if pixels.size == 0:
        return None
    finite = pixels[np.isfinite(pixels)]
    if finite.size == 0:
        return None
    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        return None
    bin_count = 256
    scale = (bin_count - 1) / (high - low)
    histogram = np.zeros(bin_count, dtype=np.int64)
    indices = np.rint((finite - low) * scale).astype(np.int64)
    indices = np.clip(indices, 0, bin_count - 1)
    histogram += np.bincount(indices, minlength=bin_count).astype(np.int64)
    total = float(finite.size)
    sum_all = float(np.dot(np.arange(bin_count), histogram))
    sum_background = 0.0
    weight_background = 0.0
    best_variance = -1.0
    best_bin = 0
    for bin_index, count in enumerate(histogram.tolist()):
        weight_background += count
        if weight_background == 0.0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0.0:
            break
        sum_background += bin_index * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        between = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if between > best_variance:
            best_variance = between
            best_bin = bin_index
    return low + best_bin / scale


def _distance_to_background(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    inf = 1.0e15
    grid = np.full(mask.size, inf, dtype=np.float64)
    grid[~mask] = 0.0
    horizontal = np.empty(mask.size, dtype=np.float64)
    for y in range(height):
        start = y * width
        row = _squared_distance_1d(grid[start : start + width])
        horizontal[start : start + width] = row
    squared = np.empty(mask.size, dtype=np.float64)
    for x in range(width):
        column = horizontal[x::width]
        transformed = _squared_distance_1d(column)
        squared[x::width] = transformed
    distance = np.sqrt(squared)
    distance[~mask] = 0.0
    return distance


def _squared_distance_1d(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    count = int(data.size)
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    envelope = np.zeros(count, dtype=np.int64)
    bounds = np.empty(count + 1, dtype=np.float64)
    k = 0
    envelope[0] = 0
    bounds[0] = -math.inf
    bounds[1] = math.inf
    for q in range(1, count):
        intersection = _parabola_intersection(data, int(envelope[k]), q)
        while intersection <= bounds[k]:
            k -= 1
            intersection = _parabola_intersection(data, int(envelope[k]), q)
        k += 1
        envelope[k] = q
        bounds[k] = intersection
        bounds[k + 1] = math.inf
    k = 0
    distance = np.empty(count, dtype=np.float64)
    for q in range(count):
        while bounds[k + 1] < q:
            k += 1
        origin = float(envelope[k])
        delta = q - origin
        distance[q] = delta * delta + data[int(envelope[k])]
    return distance


def _parabola_intersection(values: np.ndarray, left: int, right: int) -> float:
    left_f = float(left)
    right_f = float(right)
    return (
        (float(values[right]) + right_f * right_f)
        - (float(values[left]) + left_f * left_f)
    ) / (2.0 * right_f - 2.0 * left_f)


def _engager_seeds(
    distance: np.ndarray,
    mask: np.ndarray,
    width: int,
    height: int,
    spacing: float,
    min_distance: float,
) -> list[int]:
    peaks: list[int] = []
    for index, on in enumerate(mask.tolist()):
        if not on or float(distance[index]) < min_distance:
            continue
        x = index % width
        y = index // width
        is_peak = True
        for neighbor in _neighbors(width, height, x, y):
            if mask[neighbor] and float(distance[neighbor]) > float(distance[index]):
                is_peak = False
                break
        if is_peak:
            peaks.append(index)
    peaks.sort(key=lambda peak: (-float(distance[peak]), peak))
    spacing_sq = spacing * spacing
    seeds: list[int] = []
    for peak in peaks:
        x = peak % width
        y = peak // width
        crowded = False
        for seed in seeds:
            dx = x - (seed % width)
            dy = y - (seed // width)
            if dx * dx + dy * dy < spacing_sq:
                crowded = True
                break
        if not crowded:
            seeds.append(peak)
    return seeds


def _watershed(
    distance: np.ndarray,
    mask: np.ndarray,
    width: int,
    height: int,
    seeds: list[int],
) -> np.ndarray:
    import heapq

    labels = np.zeros(mask.size, dtype=np.uint32)
    heap: list[tuple[int, int, int]] = []
    for offset, seed in enumerate(seeds):
        label = offset + 1
        labels[seed] = label
        # Max-heap on distance, matching the Rust BinaryHeap order.
        heapq.heappush(heap, (-int(float(distance[seed]) * 1000.0), seed, label))
    while heap:
        _negative, index, label = heapq.heappop(heap)
        if labels[index] != 0 and int(labels[index]) != label:
            continue
        labels[index] = label
        x = index % width
        y = index // width
        for neighbor in _neighbors(width, height, x, y):
            if not mask[neighbor] or labels[neighbor] != 0:
                continue
            labels[neighbor] = label
            heapq.heappush(
                heap,
                (-int(float(distance[neighbor]) * 1000.0), neighbor, label),
            )
    return labels


def _keep_diameter(labels: np.ndarray, diameter_px: float) -> np.ndarray:
    min_diameter = diameter_px * 0.7
    max_diameter = diameter_px * 1.3
    kept = labels.copy()
    present = kept[kept != 0]
    if present.size == 0:
        return kept
    ids, areas = np.unique(present, return_counts=True)
    drop = set()
    for label, area in zip(ids.tolist(), areas.tolist(), strict=True):
        diameter = 0.0 if area == 0 else 2.0 * math.sqrt(area / math.pi)
        if diameter < min_diameter or diameter > max_diameter:
            drop.add(int(label))
    if drop:
        for index, label in enumerate(kept.tolist()):
            if int(label) in drop:
                kept[index] = 0
    return kept


def _drop_small_components(labels: np.ndarray, min_pixels: int) -> np.ndarray:
    kept = labels.copy()
    present = kept[kept != 0]
    if present.size == 0:
        return kept
    ids, areas = np.unique(present, return_counts=True)
    pairs = zip(ids.tolist(), areas.tolist(), strict=True)
    drop = {int(label) for label, area in pairs if area < min_pixels}
    if drop:
        for index, label in enumerate(kept.tolist()):
            if int(label) in drop:
                kept[index] = 0
    return kept


def _neighbors(width: int, height: int, x: int, y: int) -> list[int]:
    found: list[int] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            found.append(ny * width + nx)
    return found


def _touches_tumor(tumor: np.ndarray, width: int, height: int, x: int, y: int) -> bool:
    if int(tumor[y * width + x]) != 0:
        return True
    return any(int(tumor[index]) != 0 for index in _neighbors(width, height, x, y))
