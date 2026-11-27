"""Per-seed traits: size, shape, colour and 3D thickness/volume.

All 2D traits are measured on the top-down stage grid (mm) and corrected for
perspective: the outline seen from above is the seed's widest girth, which for
a seed lying on the stage sits at about half its thickness t, where it appears
enlarged by D / (D - t/2) (D = camera-to-stage distance).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

from .segment import to_lab


def _feret(coords):
    """Max and min caliper (Feret) widths of a point set (pixels), via its convex hull."""
    from scipy.spatial import ConvexHull, QhullError
    if len(coords) < 3:
        return 0.0, 0.0
    try:
        hull = coords[ConvexHull(coords).vertices]
    except QhullError:  # all on one line (a 1-pixel sliver): no hull, but the points give the same widths
        hull = coords
    angles = np.deg2rad(np.arange(0, 180, 0.5))
    proj = hull @ np.stack([np.cos(angles), np.sin(angles)])
    widths = proj.max(axis=0) - proj.min(axis=0)
    return float(widths.max()), float(widths.min())


def fill_height(height, mask):
    """Height inside a seed with gaps (unmatched pixels) filled from the nearest valid pixel."""
    h = np.where(mask, height, np.nan)
    valid = np.isfinite(h)
    if not valid.any():
        return h, 0.0
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = h[tuple(idx)]
    return np.where(mask, filled, np.nan), float(valid[mask].mean())


def measure_seeds(labels, ortho_bgr, height, res, stage_distance, pixel_offset=(0.0, 0.0)):
    """One dict of traits per labelled seed."""
    rows = []
    for region in measure.regionprops(labels):
        sl = region.slice
        mask = region.image
        h_full, coverage = fill_height(height[sl], mask)

        # thickness from the top of the seed, ignoring the outer rim where depth is weakest
        inner = ndi.binary_erosion(mask, morphology.disk(max(1, int(0.3 / res))))
        h_inner = h_full[inner] if inner.any() else h_full[mask]
        h_inner = h_inner[np.isfinite(h_inner)]
        thickness = float(np.percentile(h_inner, 98)) if len(h_inner) else np.nan

        # perspective correction for 2D traits (lengths scale by k, areas by k^2)
        k = (stage_distance - thickness / 2) / stage_distance if np.isfinite(thickness) else 1.0
        px = res * k

        coords = np.argwhere(mask).astype(float)
        feret_max, feret_min = _feret(coords)
        length, width = (feret_max + 1) * px, (feret_min + 1) * px  # +1: pixel extent
        area = region.area * px ** 2
        perimeter = region.perimeter * px

        # volume, assuming the hidden underside mirrors the top about mid-thickness
        # (exact for ellipsoids); also the plain "under the top surface" volume
        hh = np.nan_to_num(h_full[mask])
        a_px = px ** 2
        vol_sym = float(np.sum(np.clip(2 * (hh - thickness / 2), 0, None)) * a_px)
        vol_top = float(np.sum(np.clip(hh, 0, None)) * a_px)
        vol_ell = float(np.pi / 6 * length * width * thickness)

        # colour inside the seed, avoiding the edge pixels mixed with background
        core = inner if inner.sum() > 20 else mask
        lab_crop = to_lab(ortho_bgr[sl])  # per seed: fast re-measuring
        L, A, B = (lab_crop[core][:, i] for i in range(3))
        rgb = ortho_bgr[sl][core][:, ::-1]
        chroma = np.hypot(A, B)

        cy, cx = region.centroid
        rows.append({
            "seed_id": region.label,
            "x_mm": (cx + pixel_offset[0]) * res, "y_mm": (cy + pixel_offset[1]) * res,
            "length_mm": length, "width_mm": width, "thickness_mm": thickness,
            "area_mm2": area, "perimeter_mm": perimeter,
            "major_axis_mm": region.axis_major_length * px, "minor_axis_mm": region.axis_minor_length * px,
            "equiv_diameter_mm": region.equivalent_diameter_area * px,
            "aspect_ratio": length / width if width else np.nan,
            "circularity": 4 * np.pi * area / perimeter ** 2 if perimeter else np.nan,
            "roundness": 4 * area / (np.pi * length ** 2) if length else np.nan,
            "solidity": region.solidity, "eccentricity": region.eccentricity,
            "sphericity": float(np.cbrt(width * thickness / length ** 2)) if length else np.nan,
            "flatness": thickness / width if width else np.nan,
            "volume_mm3": vol_sym, "volume_top_mm3": vol_top, "volume_ellipsoid_mm3": vol_ell,
            "L_mean": float(L.mean()), "a_mean": float(A.mean()), "b_mean": float(B.mean()),
            "L_sd": float(L.std()), "a_sd": float(A.std()), "b_sd": float(B.std()),
            "chroma": float(chroma.mean()),
            "hue_deg": float(np.degrees(np.arctan2(B.mean(), A.mean())) % 360),
            "R": float(np.median(rgb[:, 0])), "G": float(np.median(rgb[:, 1])), "B": float(np.median(rgb[:, 2])),
            "depth_coverage": coverage,
        })
    return rows
