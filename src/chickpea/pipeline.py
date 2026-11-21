"""Stereo pair -> per-seed traits and batch uniformity.

    chickpea-stage   --calib calib.npz --left stage_L.JPG --square-mm 14.9 [--offset-mm 7.0]
    chickpea-measure --calib calib.npz --stage stage.npz --left L.JPG --right R.JPG --out results/sample1

Without --stage the stage plane is fitted to the background in each pair's
depth map (needs a background with fine texture) and --min-depth/--max-depth
must be given.
"""
import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, segmentation

from sonyrx0.stereo.common import load_pairs, make_board, read_image
from sonyrx0.stereo.depth import StereoRig, colourise, compute_depth

from .measure import measure_seeds
from .ortho import make_grid
from .segment import plane_fit_mask, segment_seeds, trim_flat
from .stage import Stage, depth_scale_from_markers, fit_plane_ransac, stage_from_board, stage_from_markers
from .summary import summarise


def _intrinsics(rig, scale=1.0):
    """Rectified left camera matrix, optionally for an image resized by `scale`."""
    return np.array([[rig.f * scale, 0, rig.cx * scale], [0, rig.f * scale, rig.cy * scale], [0, 0, 1.0]])


def rectify_left_full(rig, img_l):
    m1, m2 = rig.maps[0]
    return cv2.remap(img_l, m1, m2, cv2.INTER_LINEAR)


def measure_pair(rig, img_l, img_r, stage=None, min_depth=None, max_depth=None, res=None,
                 split=True, min_height=None, min_area_mm2=4.0, min_seed_width_mm=4.0,
                 min_thickness_mm=1.0, detector=None, trim_shadows=True, markers=None, level=False):
    """Run the whole pipeline on one raw stereo pair. Returns (seeds DataFrame, extras dict).

    detector: optional callable (ortho_bgr, valid, res, height) -> (labels, info)
    replacing the classic colour segmentation, e.g. a trained model.
    markers, level: see measure_rgbd.
    """
    if stage is not None:
        min_depth = min_depth or stage.d - 40.0  # seeds are at most a few cm tall
        max_depth = max_depth or stage.d + 10.0
    if min_depth is None or max_depth is None:
        raise ValueError("Give min/max depth, or a stage file so the range can be derived")
    rect_l, _, disp, xyz = compute_depth(rig, img_l, img_r, min_depth, max_depth)
    seeds, extras = measure_rgbd(rectify_left_full(rig, img_l), _intrinsics(rig), xyz, _intrinsics(rig, rig.scale),
                                 stage, res, split, min_height, min_area_mm2, min_seed_width_mm, min_thickness_mm,
                                 detector, trim_shadows, markers, level, plane_img=rect_l)
    extras["disparity"] = disp
    return seeds, extras


def measure_rgbd(full, K_full, xyz, K_xyz, stage=None, res=None, split=True, min_height=None, min_area_mm2=4.0,
                 min_seed_width_mm=4.0, min_thickness_mm=1.0, detector=None, trim_shadows=True, markers=None,
                 level=False, plane_img=None):
    """The pipeline from an undistorted colour image and a 3D point per depth pixel.

    full: colour image with intrinsics K_full; xyz: (h, w, 3) points in mm in the
    same camera frame, for intrinsics K_xyz (NaN where unknown), e.g. a stereo
    pair's left rectified frame or an RGB-D camera's depth aligned to its colour.
    markers: optional (layout, scale) of a printed stage sheet: its corner markers
    then give the stage plane, and the depth is rescaled to agree with them.
    level: measure height against the stage sheet around each seed (a smooth
    surface fitted to the background) rather than the single plane, for depth that
    bends by a few mm across the view.
    """
    info = {}
    h, w = full.shape[:2]
    area = None
    if markers is not None:
        layout, scale = markers
        stage, found, err, area = stage_from_markers(full, K_full, layout, scale)
        k = depth_scale_from_markers(stage, found, xyz, K_xyz, full.shape)
        xyz = xyz * k
        info = {"markers": len(found), "marker_error_px": round(err, 2), "depth_scale": round(k, 4)}
    if stage is None:
        # fit the stage to the background only: seeds are excluded first, since on a
        # plain background they provide most of the stereo matches
        if plane_img is None:
            plane_img = cv2.resize(full, (xyz.shape[1], xyz.shape[0]), interpolation=cv2.INTER_AREA)
        stage, _ = fit_plane_ransac(xyz[plane_fit_mask(plane_img)], threshold=1.0)

    grid = make_grid(stage, K_full, (w, h), res)
    seed_area = np.full((h, w), 255, np.uint8)
    if area is not None:  # only the sheet between the markers: not the desk, the markers or the print
        grid = grid.cropped(K_full, area["outline"])
        seed_area[:] = 0
        cv2.fillPoly(seed_area, [np.round(area["outline"]).astype(np.int32)], 255)
        cv2.fillPoly(seed_area, [np.round(p).astype(np.int32) for p in area["holes"]], 0)
    ortho = grid.warp(full, K_full)
    valid = grid.warp(seed_area, K_full, cv2.INTER_NEAREST) > 0
    height = stage.height(xyz).astype(np.float32)
    height[(height < -30) | (height > 150)] = np.nan  # mismatches (e.g. on blank paper), not seeds
    ortho_h = grid.warp(height, K_xyz, cv2.INTER_NEAREST, border=np.nan)
    if level:
        ortho_h -= background_level(ortho_h, grid.res)

    if detector is None:
        labels, seg_info = segment_seeds(ortho, valid, grid.res, ortho_h, min_height,
                                         min_area_mm2, split, min_seed_width_mm)
    else:
        labels, seg_info = detector(ortho, valid, grid.res, ortho_h)
    # 1) Not seeds: things that don't stand up off the stage (stains, print, holes
    #    such as a board's handle slot), judged on every pixel with a height. This
    #    must come before shadow trimming, which would whittle a hole down to an
    #    unmeasurable fragment. Objects with no height at all are kept.
    rejected = _not_raised(labels, ortho_h, min_thickness_mm)
    if rejected:
        labels[np.isin(labels, rejected)] = 0
        labels, _, _ = segmentation.relabel_sequential(labels)
    # 2) Shadows: trim the flat (height ~0) parts off each seed.
    if trim_shadows:
        labels, trimmed_mm2 = trim_flat(labels, ortho_h, grid.res, ortho)
        seg_info = {**seg_info, "shadow_trimmed_mm2": round(trimmed_mm2, 1)}
    rows = measure_seeds(labels, ortho, ortho_h, grid.res, stage.d,
                         pixel_offset=(grid.a0 / grid.res, grid.b0 / grid.res))
    seg_info = {**seg_info, "rejected_not_raised": len(rejected)}
    extras = {"stage": stage, "grid": grid, "ortho": ortho, "valid": valid, "height": ortho_h, "labels": labels,
              "segmentation": seg_info, "ground": info}
    return pd.DataFrame(rows), extras


def background_level(height, res, cell_mm=20.0, window_cells=5):
    """Smooth height of the background under everything (mm), from a top-down height map.

    Median per cell, then a median over window_cells x window_cells cells, so
    seeds (which never cover most of a ~100 mm window) don't lift it. Subtract it
    to measure each seed against the sheet around it.
    """
    c = max(1, int(round(cell_mm / res)))
    rows, cols = height.shape
    gr, gc = -(-rows // c), -(-cols // c)
    pad = np.full((gr * c, gc * c), np.nan, np.float32)
    pad[:rows, :cols] = height
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells
        cells = np.nanmedian(pad.reshape(gr, c, gc, c).transpose(0, 2, 1, 3).reshape(gr, gc, -1), axis=2)
    if not np.isfinite(cells).any():
        return np.zeros_like(height)
    cells = np.where(np.isfinite(cells), cells, np.nanmedian(cells)).astype(np.float32)
    cells = ndimage.median_filter(cells, size=window_cells, mode="nearest")
    return cv2.resize(cells, (gc * c, gr * c), interpolation=cv2.INTER_LINEAR)[:rows, :cols]


def _not_raised(labels, height, min_thickness_mm, min_known_px=20):
    """Labels of objects that don't stand at least min_thickness_mm above the stage."""
    out = []
    for r in measure.regionprops(labels):
        h = height[r.slice][r.image]
        h = h[np.isfinite(h)]
        if len(h) >= min_known_px and np.median(h) < min_thickness_mm:  # median: robust to stray matches
            out.append(r.label)
    return out


def overlay(ortho, labels, seeds, max_side=3000):
    out = segmentation.mark_boundaries(ortho[..., ::-1], labels, color=(1, 1, 0), mode="thick")
    out = (out[..., ::-1] * 255).astype(np.uint8)
    for _, s in seeds.iterrows():
        ys, xs = np.nonzero(labels == s.seed_id)
        cv2.putText(out, str(int(s.seed_id)), (int(xs.mean()) - 10, int(ys.mean()) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    f = min(1.0, max_side / max(out.shape[:2]))
    return cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else out


def save_results(out_dir, seeds, extras, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds.insert(0, "sample", name)
    seeds.to_csv(out / f"{name}_seeds.csv", index=False)
    summary = summarise(seeds.drop(columns="sample"))
    summary.insert(0, "sample", name)
    summary.to_csv(out / f"{name}_summary.csv", index=False)
    cv2.imwrite(str(out / f"{name}_overlay.jpg"), overlay(extras["ortho"], extras["labels"], seeds))
    h = extras["height"]
    hv = np.where(np.isfinite(h), np.clip(h, 0, None), np.nan)
    cv2.imwrite(str(out / f"{name}_height.png"), colourise(hv))
    cv2.imwrite(str(out / f"{name}_labels.png"), extras["labels"].astype(np.uint16))
    stage, grid = extras["stage"], extras["grid"]
    meta = {"stage_normal": stage.n.tolist(), "stage_distance_mm": stage.d, "stage_method": stage.method,
            "ortho_res_mm": grid.res, "n_seeds": len(seeds), **extras["segmentation"]}
    (out / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    return summary


def main_measure():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--left", required=True, help="left image or folder")
    ap.add_argument("--right", required=True, help="right image or folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stage", help="stage.npz from chickpea-stage (recommended)")
    ap.add_argument("--min-depth", type=float)
    ap.add_argument("--max-depth", type=float)
    ap.add_argument("--scale", type=float, default=0.5, help="stereo matching scale")
    ap.add_argument("--res", type=float, help="top-down grid resolution, mm/px (default: native)")
    ap.add_argument("--no-split", action="store_true", help="don't split touching seeds")
    ap.add_argument("--min-height", type=float, help="also require seeds to stand this many mm above the stage")
    ap.add_argument("--min-area", type=float, default=4.0, help="ignore objects smaller than this (mm^2)")
    ap.add_argument("--seed-width", type=float, default=4.0, help="smallest expected seed width (mm), for splitting")
    ap.add_argument("--min-thickness", type=float, default=1.0,
                    help="objects must stand at least this high above the stage (mm) to count as seeds")
    args = ap.parse_args()

    rig = StereoRig(args.calib, args.scale)
    stage = Stage.load(args.stage) if args.stage else None
    left, right = Path(args.left), Path(args.right)
    pairs = load_pairs(left, right) if left.is_dir() else [(left, right)]
    for lp, rp in pairs:
        seeds, extras = measure_pair(rig, read_image(lp), read_image(rp), stage, args.min_depth, args.max_depth,
                                     args.res, not args.no_split, args.min_height, args.min_area, args.seed_width,
                                     args.min_thickness)
        summary = save_results(args.out, seeds, extras, lp.stem)
        s = summary.set_index("trait")
        msg = f"{lp.stem}: {len(seeds)} seeds"
        for t in ("length_mm", "width_mm", "thickness_mm", "volume_mm3"):
            if t in s.index:
                msg += f" | {t.split('_')[0]} {s.loc[t, 'mean']:.2f} (CV {s.loc[t, 'cv_pct']:.1f}%)"
        print(msg)


def main_stage():
    ap = argparse.ArgumentParser(description="Measure the stage plane from a photo of the "
                                             "calibration board lying flat on the stage.")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--left", required=True, help="left image of the board lying on the stage")
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--squares-x", type=int, default=11)
    ap.add_argument("--squares-y", type=int, default=8)
    ap.add_argument("--offset-mm", type=float, default=0.0,
                    help="height of the board surface above the stage (e.g. tablet thickness)")
    ap.add_argument("--out", default="stage.npz")
    args = ap.parse_args()

    rig = StereoRig(args.calib, 1.0)
    full = rectify_left_full(rig, read_image(args.left))
    board = make_board(args.squares_x, args.squares_y, args.square_mm)
    stage, n = stage_from_board(full, _intrinsics(rig), board, args.offset_mm)
    stage.save(args.out)
    tilt = np.degrees(np.arccos(abs(stage.n[2])))
    print(f"Stage from {n} board corners: {stage.d:.1f} mm from the camera, tilted {tilt:.1f} deg. Saved {args.out}")
