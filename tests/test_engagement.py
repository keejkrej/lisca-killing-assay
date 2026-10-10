import numpy as np

from killing.core.engagement import (
    ENGAGER_DIAMETER_PX,
    EngagementCounts,
    FluorescenceHistogram,
    component_count,
    count_engagements,
    label_components,
    mask_with_scale,
    split_round_components,
    tcell_mask,
    tumor_mask,
)
from killing.core.fluorescence import measure_frame


def _paint_disk(
    mask: np.ndarray, width: int, cx: float, cy: float, radius: float
) -> None:
    radius_sq = radius * radius
    height = mask.size // width
    for y in range(height):
        for x in range(width):
            dx = x + 0.5 - cx
            dy = y + 0.5 - cy
            if dx * dx + dy * dy <= radius_sq:
                mask[y * width + x] = True


def _paint_value(
    pixels: np.ndarray, width: int, cx: float, cy: float, radius: float, value: float
) -> None:
    radius_sq = radius * radius
    height = pixels.size // width
    for y in range(height):
        for x in range(width):
            dx = x + 0.5 - cx
            dy = y + 0.5 - cy
            if dx * dx + dy * dy <= radius_sq:
                pixels[y * width + x] = value


def _quiet(length: int, salt: int) -> np.ndarray:
    return np.array(
        [190.0 + ((index * 17 + salt) % 31) for index in range(length)],
        dtype=np.float64,
    )


def test_a_touching_t_cell_is_one_engagement() -> None:
    width = 48
    height = 48
    tumor = np.zeros(width * height, dtype=bool)
    tcells = np.zeros(width * height, dtype=bool)
    _paint_disk(tumor, width, 16.0, 24.0, 6.0)
    _paint_disk(tcells, width, 26.0, 24.0, 5.0)
    _paint_disk(tcells, width, 40.0, 8.0, 5.0)
    counts = count_engagements(tumor, tcells, width, height)
    assert counts == EngagementCounts(tumor_cells=1, t_cells=2, engagements=1)


def test_separated_cells_are_not_engagements() -> None:
    width = 40
    tumor = np.zeros(width * width, dtype=bool)
    tcells = np.zeros(width * width, dtype=bool)
    _paint_disk(tumor, width, 10.0, 10.0, 5.0)
    _paint_disk(tcells, width, 30.0, 30.0, 5.0)
    counts = count_engagements(tumor, tcells, width, width)
    assert counts.engagements == 0
    assert counts.t_cells == 1
    assert counts.tumor_cells == 1


def test_a_three_by_three_patch_is_below_the_engager_diameter() -> None:
    pixels = np.full(32 * 32, 10.0)
    for y in range(4, 7):
        for x in range(4, 7):
            pixels[y * 32 + x] = 1000.0
    mask = tcell_mask(pixels)
    assert int(mask.sum()) == 9
    counts = count_engagements(np.zeros(pixels.size, dtype=bool), mask, 32, 32)
    assert counts.t_cells == 0
    assert not tcell_mask(np.full(32 * 32, 12.0)).any()


def test_a_peanut_splits_and_one_disk_stays_one() -> None:
    width = 40
    height = 32
    mask = np.zeros(width * height, dtype=bool)
    _paint_disk(mask, width, 16.0, 16.0, 5.0)
    _paint_disk(mask, width, 24.0, 16.0, 5.0)
    labels = split_round_components(mask, width, height, ENGAGER_DIAMETER_PX)
    assert component_count(labels) == 2

    one = np.zeros(48 * 32, dtype=bool)
    _paint_disk(one, 48, 16.0, 16.0, 5.0)
    labels = split_round_components(one, 48, 32, ENGAGER_DIAMETER_PX)
    assert component_count(labels) == 1


def test_a_shared_scale_leaves_an_empty_field_dark() -> None:
    with_engager = _quiet(96 * 96, 3)
    empty = _quiet(96 * 96, 11)
    _paint_value(with_engager, 96, 48.0, 48.0, 5.0, 4500.0)
    histogram = FluorescenceHistogram()
    histogram.sample(with_engager)
    histogram.sample(empty)
    scale = histogram.scale()
    assert scale.threshold > float(empty.max())
    assert scale.threshold < 4500.0
    assert not mask_with_scale(empty, scale).any()
    assert mask_with_scale(with_engager, scale).any()


def test_otsu_keeps_the_minority_cell_patch() -> None:
    bright = np.zeros(32 * 32)
    for y in range(2, 6):
        for x in range(2, 6):
            bright[y * 32 + x] = 200.0
    mask = tumor_mask(bright)
    assert mask[2 * 32 + 2]
    assert not mask[0]
    dark = np.full(32 * 32, 200.0)
    for y in range(2, 6):
        for x in range(2, 6):
            dark[y * 32 + x] = 0.0
    mask = tumor_mask(dark)
    assert mask[2 * 32 + 2]
    assert not mask[0]


def test_a_one_pixel_spur_does_not_split_a_circle() -> None:
    width = 32
    mask = np.zeros(width * width, dtype=bool)
    _paint_disk(mask, width, 16.0, 16.0, 5.0)
    mask[16 * width + 22] = True
    mask[16 * width + 23] = True
    labels = split_round_components(mask, width, width, ENGAGER_DIAMETER_PX)
    assert component_count(labels) == 1
    assert component_count(label_components(mask, width, width)) == 1


def test_death_reporter_background_is_the_tenth_percentile() -> None:
    stats = measure_frame(np.array([1.0, 2.0, 3.0, 40.0]))
    assert stats["area"] == 4
    assert stats["sum"] == 46.0
    assert stats["background"] == 1.3
    assert stats["corrected"] == 40.8
