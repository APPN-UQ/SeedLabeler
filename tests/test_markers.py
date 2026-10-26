"""Stage sheet, corner-marker ground plane and the colour + depth (OAK) path, on a synthetic scene.

The scene: the sheet printed at 68% (A3 shrunk onto A4) lies on a plane 450 mm
from the camera, tilted 5 degrees; three 6 mm discs ("seeds") stand on it; the
depth reads 4% short, as the OAK-D's did on the bench.
"""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from chickpea.pipeline import background_level, measure_rgbd
from chickpea.stage import Stage
from seedlabeler import processing, sheet
from seedlabeler.project import Project

K = np.array([[1600.0, 0, 1000], [0, 1600, 750], [0, 0, 1]])
SIZE = (2000, 1500)
SCALE = 0.68
SEEDS = [(150.0, 120.0), (210.0, 150.0), (260.0, 110.0)]  # disc centres, design mm on the A3 sheet
SEED_R, SEED_H = 12.0, 6.0  # design mm (x SCALE on the print), mm high
DEPTH_SHORT = 0.96


def _scene(paper="A3"):
    lay = sheet.layout(paper)
    tilt = np.radians(5)
    R = cv2.Rodrigues(np.array([tilt, 0, 0]))[0]       # sheet frame -> camera
    w, h = lay["sheet_mm"]
    t = np.array([-w * SCALE / 2, -h * SCALE / 2, 450.0])
    n_cam = R[:, 2]                                     # sheet normal in the camera frame
    u, v = np.meshgrid(np.arange(SIZE[0], dtype=float), np.arange(SIZE[1], dtype=float))
    rays = np.dstack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)])
    s = (n_cam @ t) / (rays @ n_cam)                    # ray parameter where it meets the sheet
    sheet_pts = ((rays * s[..., None]) - t) @ R          # sheet coordinates (printed mm)
    xd, yd = sheet_pts[..., 0] / SCALE, sheet_pts[..., 1] / SCALE  # design mm
    img = np.full(SIZE[::-1] + (3,), 235, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, lay["dictionary"]))
    for m in lay["markers"]:
        (x0, y0), size = m["corners_mm"][0], lay["marker_mm"]
        bits = cv2.aruco.generateImageMarker(dictionary, m["id"], 7)
        inside = (xd >= x0) & (xd < x0 + size) & (yd >= y0) & (yd < y0 + size)
        ci = np.clip(((xd - x0) / size * 7).astype(int), 0, 6)
        ri = np.clip(((yd - y0) / size * 7).astype(int), 0, 6)
        img[inside] = np.where(bits[ri[inside], ci[inside]] > 0, 235, 20)[:, None]
    depth = s * rays[..., 2]
    for cx, cy in SEEDS:
        disc = (xd - cx) ** 2 + (yd - cy) ** 2 < SEED_R ** 2
        img[disc] = (40, 90, 160)  # brown (BGR)
        # top of the disc: the plane raised SEED_H towards the camera
        s_top = (n_cam @ t - SEED_H) / (rays @ n_cam)
        depth[disc] = (s_top * rays[..., 2])[disc]
    depth_mm = np.round(depth * DEPTH_SHORT).astype(np.uint16)
    return img, depth_mm, lay, n_cam, n_cam @ t


def test_sheet_files(tmp_path):
    for paper in ("A4", "A3"):
        js = sheet.write_sheet(tmp_path / f"s_{paper}.pdf", paper)
        lay = json.loads(js.read_text())
        assert (tmp_path / f"s_{paper}.pdf").stat().st_size > 1000
        assert [m["id"] for m in lay["markers"]] == [96, 97, 98, 99]
        assert lay["sheet_mm"] == list(sheet.PAPERS[paper]["size"])
        # markers sit inside the page with room for the printer's margin
        xy = np.concatenate([m["corners_mm"] for m in lay["markers"]])
        assert xy.min() >= 8 and (lay["sheet_mm"] - xy.max(0)).min() >= 8


def test_markers_plane_depth_scale_and_heights():
    img, depth_mm, lay, n_true, d_true = _scene()
    xyz = processing.rgbd_xyz(depth_mm, K)
    seeds, ex = measure_rgbd(img, K, xyz, K, markers=(lay, SCALE), min_seed_width_mm=6.0, min_thickness_mm=1.0)
    g = ex["ground"]
    assert g["markers"] == 4 and g["marker_error_px"] < 1.0
    assert ex["stage"].d == pytest.approx(d_true, abs=1.0)
    assert abs(float(ex["stage"].n @ n_true)) > np.cos(np.radians(0.3))
    assert g["depth_scale"] == pytest.approx(1 / DEPTH_SHORT, abs=0.01)
    assert len(seeds) == len(SEEDS)  # the markers and the rest of the view are left out
    assert seeds.thickness_mm.between(SEED_H - 0.7, SEED_H + 0.7).all(), seeds.thickness_mm.tolist()
    d = 2 * SEED_R * SCALE
    assert seeds.length_mm.between(d - 1.0, d + 1.0).all(), seeds.length_mm.tolist()


def test_wrong_print_scale_is_reported():
    img, depth_mm, lay, _, _ = _scene()
    with pytest.raises(RuntimeError, match="printed scale"):
        measure_rgbd(img, K, processing.rgbd_xyz(depth_mm, K), K, markers=(lay, 1.0))  # bar not measured


def test_background_level_ignores_seeds():
    res = 0.2
    yy, xx = np.mgrid[0:1000, 0:1500]
    bend = (0.004 * (xx - 750) * res + 0.003 * (yy - 500) * res).astype(np.float32)  # tilted by a few mm
    h = bend.copy()
    for cx, cy in ((300, 300), (900, 600), (1200, 200)):
        h[(xx - cx) ** 2 + (yy - cy) ** 2 < 40 ** 2] += 6.0  # 16 mm seeds, 6 mm tall
    lvl = h - background_level(h, res)
    assert np.abs(np.median(lvl[h - bend < 1])) < 0.2
    assert np.median(lvl[(xx - 900) ** 2 + (yy - 600) ** 2 < 30 ** 2]) == pytest.approx(6.0, abs=0.5)


def test_oak_project_detect(tmp_path):
    """An OAK sample (rgb.jpg + depth.npz) with corner markers goes through detection end to end."""
    img, depth_mm, lay, _, _ = _scene()
    js = tmp_path / "sheet.json"
    js.write_text(json.dumps(lay))
    p = Project.create(tmp_path / "proj", "oak", crop="chickpea", camera="oak")
    assert p.config.settings.level_height  # on by default for the OAK
    p.config.markers_sheet, p.config.markers_bar_mm = str(js), 68.0
    p.adopt_files()
    p.save()
    assert p.config.markers_sheet == "calibration/stage_sheet.json"
    raw = p.root / "samples" / "s1" / "raw"
    raw.mkdir(parents=True)
    cv2.imwrite(str(raw / "rgb.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.savez_compressed(raw / "depth.npz", depth=depth_mm, K_rgb=K, K_depth=K)
    s = Project.open(p.root).samples()[0]
    assert s.is_rgbd and s.image == s.rgb
    ann = processing.detect(Project.open(p.root), s)
    meta = s.meta()
    assert len(ann["seeds"]) == len(SEEDS) and meta["camera"] == "oak" and meta["stage_method"] == "markers"
    assert pd.read_csv(s.seeds_csv).thickness_mm.between(SEED_H - 1, SEED_H + 1).all()
