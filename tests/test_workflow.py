"""End-to-end workflow without the GUI: project -> import -> detect -> edit -> remeasure -> dataset -> export.

Uses a real almond pair from the ChickpeaSeed repo if present (skips otherwise).
    pytest tests -s
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from seedlabeler import ml, processing
from seedlabeler.project import Project

CODE = Path(__file__).resolve().parents[2]
PAIR = CODE / "ChickpeaSeed" / "data" / "almonds02"
CALIB = CODE / "SonyRX0" / "calib.npz"


def _pair():
    lefts = sorted((PAIR / "left").glob("*.JPG"))
    rights = sorted((PAIR / "right").glob("*.JPG"))
    if not lefts or not rights or not CALIB.exists():
        pytest.skip("almond test pair / calibration not available")
    return lefts[-1], rights[-1]


def test_workflow(tmp_path):
    left, right = _pair()
    p = Project.create(tmp_path / "proj", "almond test", crop="almond", calib=str(CALIB))
    s = p.add_pair(left, right)
    assert [x.id for x in p.samples()] == [s.id] and s.status() == "new"

    ann = processing.detect(p, s)
    assert len(ann["seeds"]) == 19, len(ann["seeds"])
    assert s.status() == "detected"
    seeds = pd.read_csv(s.seeds_csv)
    assert len(seeds) == 19 and 18 < seeds.length_mm.mean() < 25

    # user deletes one seed and edits another: traits follow the polygons
    ann["seeds"] = ann["seeds"][1:]
    ann["seeds"][0]["status"] = "edited"
    s.save_annotations(ann)
    seeds, summary = processing.remeasure(p, s, ann)
    assert len(seeds) == 18 and (seeds.status == "edited").sum() == 1
    assert "length_mm" in set(summary.trait)

    # polygons <-> labels round trip keeps every seed
    layers = processing.load_layers(s)
    labels = processing.polygons_to_labels([x["polygon"] for x in ann["seeds"]], layers[0].shape[:2])
    assert len(processing.labels_to_polygons(labels)) == 18

    split = ml.add_to_dataset(p, s)
    for kind in ml.INPUTS:  # colour and colour+height datasets, as tiles with the same labels
        labs = list((ml.dataset_root(p, kind) / "labels" / split).glob(f"{s.id}__t*.txt"))
        imgs = list((ml.dataset_root(p, kind) / "images" / split).glob(f"{s.id}__t*"))
        assert labs and len(labs) == len(imgs)
        assert (ml.dataset_root(p, kind) / "data.yaml").exists()
    counts = ml.dataset_counts(p)[split]
    assert counts[0] == 1 and counts[2] >= 18  # seeds in tile overlaps count more than once
    assert s.status() == "dataset"

    paths = processing.export_project(p)
    assert len(paths) == 2 and all(x.exists() for x in paths)


def test_swapped_pair_is_detected():
    from sonyrx0.stereo.common import read_image
    left, right = _pair()
    L, R = read_image(left), read_image(right)
    assert processing.check_swapped(CALIB, L, R) is False
    assert processing.check_swapped(CALIB, R, L) is True
