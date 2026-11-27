"""Seed outlines never share pixels (seen from above, seeds can't overlap)."""
import numpy as np

from chickpea.measure import _feret
from seedlabeler.processing import polygons_to_labels, resolve_overlaps


def _sq(x0, y0, x1, y1, status="auto"):
    return {"polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "status": status}


def _overlap_px(seeds, shape):
    count = np.zeros(shape, int)
    for s in seeds:
        count += polygons_to_labels([s["polygon"]], shape) > 0
    return int((count > 1).sum())


def test_disjoint_outlines_are_untouched():
    seeds = [_sq(0, 0, 10, 10), _sq(20, 20, 30, 30)]
    out, kept = resolve_overlaps(seeds, (50, 50))
    assert out == seeds and kept == [0, 1]


def test_overlap_split_between_auto_seeds():
    seeds = [_sq(0, 0, 20, 20), _sq(10, 0, 30, 20)]
    out, kept = resolve_overlaps(seeds, (40, 40))
    assert kept == [0, 1] and _overlap_px(out, (40, 40)) <= 25  # at most a shared edge left


def test_hand_drawn_outline_wins_over_auto():
    seeds = [_sq(0, 0, 20, 20), _sq(10, 0, 30, 20, "manual")]
    out, kept = resolve_overlaps(seeds, (40, 40))
    assert kept == [0, 1] and out[1] == seeds[1]  # the hand-drawn outline never changes
    assert np.asarray(out[0]["polygon"])[:, 0].max() <= 10  # the auto one gave way


def test_drawn_over_seed_is_removed():
    seeds = [_sq(12, 2, 16, 6), _sq(10, 0, 30, 20, "manual")]
    out, kept = resolve_overlaps(seeds, (40, 40), protect=1)
    assert kept == [1] and out == [seeds[1]]


def test_feret_survives_a_one_pixel_sliver():
    assert _feret(np.array([[0, 4], [2, 2], [4, 0], [1, 3], [3, 1]], float))[1] == 0.0
