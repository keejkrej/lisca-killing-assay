from killing.core.classifier import (
    alive_label,
    clean_predictions,
    dead_probability,
    death_times,
    kill_curve,
)


def _row(t: int, crop: int, label: bool) -> dict:
    return {"t": t, "crop": crop, "label": label, "pos": 1, "sample": "A"}


def test_dead_probability_and_alive_label():
    assert dead_probability(2.0, 0.0) > 0.8
    assert dead_probability(0.0, 2.0) < 0.2
    assert alive_label(0.1)
    assert alive_label(0.49)
    assert not alive_label(0.5)
    assert not alive_label(1.0)


def test_clean_enforces_monotonicity():
    cleaned = clean_predictions(
        [_row(0, 1, True), _row(1, 1, True), _row(2, 1, False), _row(3, 1, True)]
    )
    assert cleaned[2]["label"] is False
    assert cleaned[3]["label"] is False


def test_death_time_uses_clean_threshold():
    rows = clean_predictions(
        [
            _row(0, 1, True),
            _row(1, 1, True),
            _row(2, 1, True),
            _row(3, 1, False),
            _row(4, 1, False),
        ]
    )
    assert death_times(rows)[(1, "A", 1)] == 2


def _crop_traces(crops: list[tuple[int, int]]) -> list[dict]:
    rows = []
    for crop, last_alive_t in crops:
        for t in range(11):
            rows.append(_row(t, crop, t <= last_alive_t))
    return rows


def test_polarity_correct_convention_yields_decreasing_kill_curve():
    cleaned = clean_predictions(_crop_traces([(1, 10), (2, 5), (3, 3)]))
    curve = kill_curve(death_times(cleaned), "A")
    assert curve == [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 2),
        (5, 2),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (10, 1),
    ]


def test_inverted_all_alive_at_t0_yields_empty_curve():
    rows = _crop_traces([(1, 10), (2, 5), (3, 3)])
    for row in rows:
        row["label"] = not row["label"]
    cleaned = clean_predictions(rows)
    assert kill_curve(death_times(cleaned), "A") == []
