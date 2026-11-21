"""Find seeds in the top-down stage image.

Default: colour contrast with the background (controlled lighting, plain or
finely textured background of a contrasting colour). Optionally require the
seed to stand above the stage in the height map, which rejects flat marks and
shadows. Touching seeds are split with a distance-transform watershed.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import color, measure, morphology, segmentation


def to_lab(bgr):
    """CIELAB (L 0-100, a/b ~ -128..127, D65) via OpenCV: ~10x faster than skimage."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def background_lab(lab, valid):
    """Background colour: median Lab of the valid border region of the stage view."""
    border = np.zeros(valid.shape, bool)
    k = max(5, min(valid.shape) // 20)
    border[:k], border[-k:], border[:, :k], border[:, -k:] = True, True, True, True
    # the valid area may not reach the grid border (oblique view): use its own rim
    rim = valid & ~ndi.binary_erosion(valid, iterations=k)
    sel = (border & valid) | rim
    return np.median(lab[sel], axis=0)


def find_stage_region(lab, valid, lightness_weight=0.3, k=4, work_side=1000, min_contrast=15.0):
    """Background colour and the stage region: the colour region that encloses the seeds.

    The image is grouped into k colour clusters; the stage is the cluster whose
    filled outline encloses the most seed-sized holes that contrast clearly with
    it (a board on a table, a tray filling the frame, a cloth...). The contrast
    requirement stops a textured surface (wood grain) from counting its own
    patches as holes. Anything outside it (table, bench, a shoe)
    is ignored. Falls back to the median colour at the edge of the view.
    Returns (background Lab, stage region mask, typical seed contrast or None).
    """
    f = min(1.0, work_side / max(lab.shape[:2]))
    small = lab[::int(round(1 / f)), ::int(round(1 / f))] if f < 1 else lab
    vsmall = valid[::int(round(1 / f)), ::int(round(1 / f))] if f < 1 else valid
    feat = small.copy()
    feat[..., 0] *= lightness_weight
    pts = feat[vsmall].astype(np.float32)
    if len(pts) < 100:
        return background_lab(lab, valid), valid, None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)  # deterministic clustering: same image, same result
    _, _, centres = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    d = np.linalg.norm(feat[..., None, :] - centres[None, None], axis=-1)
    clusters = np.where(vsmall, d.argmin(axis=-1), -1)

    best = None
    for c in range(k):
        # the same surface under uneven light can split into several similar
        # clusters: treat all clusters close in colour to c as one surface
        similar = np.nonzero(np.linalg.norm(centres - centres[c], axis=1) < min_contrast)[0]
        m = ndi.binary_opening(np.isin(clusters, similar), iterations=1)
        filled = ndi.binary_fill_holes(m)
        holes, n = ndi.label(filled & ~m)
        if n == 0:
            continue
        sizes = np.bincount(holes.ravel())[1:]
        idx = np.arange(1, n + 1)
        hole_feat = np.stack([ndi.mean(feat[..., j], holes, idx) for j in range(3)], axis=1)
        contrast = np.linalg.norm(hole_feat - centres[c], axis=1)
        seedlike = (sizes >= 15) & (sizes <= 0.03 * filled.sum()) & (contrast > min_contrast)
        score = int(seedlike.sum())
        if score and (best is None or score > best[0] or (score == best[0] and m.sum() > best[2].sum())):
            # stage = the filled component(s) that contain those holes
            comp, _ = ndi.label(filled)
            hole_ids = np.nonzero(seedlike)[0] + 1
            keep = np.unique(comp[np.isin(holes, hole_ids)])
            seed_contrast = float(np.average(contrast[seedlike], weights=sizes[seedlike]))
            best = (score, np.isin(comp, keep[keep > 0]), m, seed_contrast)
    if best is None:
        return background_lab(lab, valid), valid, None
    # Stages (boards, trays, cloths laid flat) are convex: the hull also takes in
    # seeds that sit near the edge or touch each other, which the plain
    # filled outline can leave open to the outside.
    region_small = morphology.convex_hull_image(best[1])
    region = cv2.resize(region_small.astype(np.uint8), (lab.shape[1], lab.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool) & valid
    stage_px = cv2.resize(best[2].astype(np.uint8), (lab.shape[1], lab.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool) & region
    return np.median(lab[stage_px], axis=0), region, best[3]


def _seed_threshold(dist, region, seed_contrast):
    """Colour-distance threshold: halfway between typical background and typical seed.

    The seed contrast comes from the objects the stage encloses. Without it,
    fall back to the Otsu/midpoint rule, which can split a strongly textured
    background into bright and dark rather than seed and background.
    """
    if seed_contrast is None:
        return _midpoint_threshold(dist[region])
    return (float(np.median(dist[region])) + seed_contrast) / 2


def plane_fit_mask(bgr, lightness_weight=0.3, dilate_px=15):
    """Pixels of an image to use for the automatic stage-plane fit: stage background only.

    Excludes the seeds (on a plain background they provide most stereo matches)
    and anything outside the stage region (e.g. the table around a board).
    """
    lab = to_lab(bgr)
    valid = np.ones(lab.shape[:2], bool)
    bg, region, seed_contrast = find_stage_region(lab, valid, lightness_weight)
    diff = lab - bg
    diff[..., 0] *= lightness_weight
    dist = ndi.gaussian_filter(np.linalg.norm(diff, axis=2), 1.0)
    thresh = _seed_threshold(dist, region, seed_contrast)
    seeds = ndi.binary_dilation((dist > thresh) & region, iterations=dilate_px)
    return ndi.binary_erosion(region, iterations=dilate_px) & ~seeds


def _midpoint_threshold(vals):
    """Otsu, then moved to the half-seed/half-background point between the two classes."""
    thresh = _otsu(vals)
    for _ in range(3):
        lo, hi = vals[vals <= thresh], vals[vals > thresh]
        if not len(lo) or not len(hi):
            break
        thresh = (np.median(lo) + np.median(hi)) / 2
    return thresh


def segment_seeds(ortho_bgr, valid, res, height=None, min_height=None, min_area_mm2=4.0,
                  split=True, min_seed_width_mm=4.0, lightness_weight=0.3, px_per_seed=50):
    """Label image of seeds (0 = background) on the ortho grid.

    Seeds are found and split on a working image with ~px_per_seed pixels across
    the smallest seed (fast), then their outlines are refined at full resolution
    with the same colour threshold, so edge precision is not lost.
    """
    work_res = max(res, min_seed_width_mm / px_per_seed)
    f = res / work_res
    if f > 0.9:
        return _segment(ortho_bgr, valid, res, height, min_height, min_area_mm2, split,
                        min_seed_width_mm, lightness_weight)
    size = (max(1, round(valid.shape[1] * f)), max(1, round(valid.shape[0] * f)))
    small = cv2.resize(ortho_bgr, size, interpolation=cv2.INTER_AREA)
    vsmall = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    hsmall = None if height is None else cv2.resize(height, size, interpolation=cv2.INTER_NEAREST)
    labels_s, info = _segment(small, vsmall, work_res, hsmall, min_height, min_area_mm2, split,
                              min_seed_width_mm, lightness_weight)
    if labels_s.max() == 0:
        return np.zeros(valid.shape, np.int32), info

    # refine: full-resolution colour mask, each pixel given the nearest seed's label
    lab = to_lab(ortho_bgr)
    diff = lab - np.asarray(info["background_lab"], np.float32)
    diff[..., 0] *= lightness_weight
    dist = ndi.gaussian_filter(np.linalg.norm(diff, axis=2), sigma=max(0.5, 0.1 / res))
    mask = (dist > info["colour_threshold"]) & valid
    up = cv2.resize(labels_s.astype(np.int32).astype(np.float32), (valid.shape[1], valid.shape[0]),
                    interpolation=cv2.INTER_NEAREST).astype(np.int32)
    labels = segmentation.expand_labels(up, distance=int(np.ceil(2 / f))) * mask
    labels = ndi.binary_opening(labels > 0, morphology.disk(max(1, int(round(0.15 / res))))) * labels
    # keep each seed's main piece only (refinement can leave crumbs at the edges)
    out = np.zeros_like(labels)
    for r in measure.regionprops(labels):
        comp, n = ndi.label(r.image)
        if n > 1:
            sizes = np.bincount(comp.ravel())[1:]
            keep = comp == (np.argmax(sizes) + 1)
        else:
            keep = r.image
        out[r.slice][keep] = r.label
    out, _, _ = segmentation.relabel_sequential(out)
    return out, {**info, "work_res_mm": work_res}


def _segment(ortho_bgr, valid, res, height=None, min_height=None, min_area_mm2=4.0,
             split=True, min_seed_width_mm=4.0, lightness_weight=0.3):
    """Label image of seeds (0 = background) on the ortho grid.

    ortho_bgr: top-down colour image; valid: pixels inside the camera's view;
    res: mm per pixel; height: optional height map (mm, NaN unknown).
    """
    lab = to_lab(ortho_bgr)
    bg, region, seed_contrast = find_stage_region(lab, valid, lightness_weight)
    valid = valid & region  # only look for seeds on the stage
    # Colour difference from the background, with lightness down-weighted: a textured
    # background (needed for stereo) and shading vary mostly in lightness, while
    # seeds differ from a contrasting background mostly in hue/chroma.
    diff = lab - bg
    diff[..., 0] *= lightness_weight
    dist = np.linalg.norm(diff, axis=2)
    # light smoothing: background texture otherwise leaves speckle that grows spurs on outlines
    dist = ndi.gaussian_filter(dist, sigma=max(0.5, 0.1 / res))
    dist[~valid] = 0
    vals = dist[valid]
    # Otsu separates seed from background, but with ~98% background it sits too
    # close to the background, so blurred edge pixels get counted as seed. The
    # true edge is where a pixel is half seed, half background: the midpoint of
    # the two classes' typical colour distances.
    thresh = _seed_threshold(dist, valid, seed_contrast)
    mask = (dist > thresh) & valid
    # Refine to the true edge: halfway between the seeds' interior colour and the
    # clear background around them (both measured, so blur and shading cancel).
    interior = ndi.binary_erosion(mask, iterations=max(1, int(0.5 / res)))
    clear_bg = valid & ~ndi.binary_dilation(mask, iterations=max(1, int(1.0 / res)))
    if interior.sum() > 50 and clear_bg.sum() > 50:
        thresh = (float(np.median(dist[clear_bg])) + float(np.median(dist[interior]))) / 2
        mask = (dist > thresh) & valid

    if height is not None and min_height is not None:
        h = np.nan_to_num(height, nan=0.0)
        tall = ndi.binary_dilation(h > min_height, iterations=max(1, int(0.5 / res)))
        mask &= tall

    r = max(1, int(round(0.3 / res)))  # removes spurs thinner than ~0.6 mm
    mask = ndi.binary_opening(mask, morphology.disk(r))
    mask = ndi.binary_fill_holes(mask)
    lab_ids, _ = ndi.label(mask)
    sizes = np.bincount(lab_ids.ravel())
    mask = (sizes >= min_area_mm2 / res ** 2)[lab_ids] & mask

    if split:
        labels = _merge_fragments(_watershed_split(mask, res, min_seed_width_mm, height))
    else:
        labels = measure.label(mask)

    # drop seeds cut off by the edge of the view
    edge = ~ndi.binary_erosion(valid, iterations=2)
    for lab_id in np.unique(labels[edge & (labels > 0)]):
        labels[labels == lab_id] = 0
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels, {"background_lab": bg.tolist(), "colour_threshold": float(thresh)}


def _otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[np.argmax(between)])


def _watershed_split(mask, res, min_seed_width_mm, height=None, split_depth_mm=1.0):
    """Split touching seeds along the valleys between them.

    Each seed is a dome: high in the middle (height above the stage) and far
    from its outline (distance transform). Touching seeds form a valley where
    they meet, deep in height even when the joint outline has only a shallow
    waist (seeds side by side). Markers are the domes' peaks that rise at least
    split_depth_mm above the valley (h-maxima), and the watershed follows the
    valleys.
    """
    dist_mm = ndi.distance_transform_edt(mask) * res
    surface = dist_mm
    if height is not None:
        h = np.where(mask, height, np.nan)
        known = np.isfinite(h)
        if known[mask].mean() > 0.3:
            idx = ndi.distance_transform_edt(~known, return_distances=False, return_indices=True)
            h = np.where(mask, h[tuple(idx)], 0.0)
            h = ndi.gaussian_filter(h, sigma=max(1.0, 0.4 / res))
            surface = np.clip(h, 0, None) + 0.5 * dist_mm
    surface = np.where(mask, surface, 0.0)
    markers = ndi.label(morphology.h_maxima(surface, split_depth_mm))[0]
    # a flat-topped seed can produce a few nearby peaks: merge those within 40% of a seed width
    markers = _merge_close_markers(markers, 0.4 * min_seed_width_mm / res)
    return segmentation.watershed(-surface, markers, mask=mask)


def _merge_close_markers(markers, min_sep):
    props = measure.regionprops(markers)
    if len(props) < 2:
        return markers
    cents = np.array([p.centroid for p in props])
    parent = list(range(len(props)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = np.linalg.norm(cents[:, None] - cents[None], axis=2)
    for i, j in zip(*np.nonzero(np.triu(d < min_sep, 1))):
        parent[find(i)] = find(j)
    lut = np.zeros(markers.max() + 1, int)
    for i, p in enumerate(props):
        lut[p.label] = find(i) + 1
    return lut[markers]


def _merge_fragments(labels, min_fraction=0.35):
    """Merge split-off pieces much smaller than a typical seed back into the neighbour they share most border with.

    Guards against a false split (e.g. noisy depth at the edge of the view):
    a genuine seed is rarely under a third of the median seed size while also
    touching another seed.
    """
    for _ in range(3):
        ids, areas = np.unique(labels[labels > 0], return_counts=True)
        if len(ids) < 2:
            return labels
        small = ids[areas < min_fraction * np.median(areas)]
        changed = False
        for i in small:
            region = labels == i
            ring = ndi.binary_dilation(region, iterations=2) & ~region
            neigh, counts = np.unique(labels[ring], return_counts=True)
            keep = neigh > 0
            if keep.any():
                labels[region] = neigh[keep][np.argmax(counts[keep])]
                changed = True
        if not changed:
            break
    return labels


def trim_flat(labels, height, res, bgr=None, frac=0.2, min_mm=0.6, min_known=0.4, chroma_frac=0.6):
    """Remove shadows from each seed: parts that are flat on the stage AND background-coloured.

    Height alone is not enough: along one side of every seed the stereo pair
    sees the stage beside it from different angles, and edge pixels often get
    the stage's height. Trimming on height alone cut notches into real seeds.
    A shadow is flat and also has the stage's dull colour, while a seed's edge
    keeps the seed's colour whatever its measured height. So a pixel is
    dropped only if its height is known and below max(min_mm, frac * seed
    thickness), and (when bgr is given) its colour saturation is below
    chroma_frac of the seed body's. Seeds with too little depth, or dull seeds
    whose own saturation can't be told from the stage's, are left untouched.
    """
    lab = to_lab(bgr) if bgr is not None else None
    out = np.zeros_like(labels)
    trimmed = 0
    for r in measure.regionprops(labels):
        sl, m = r.slice, r.image
        h = height[sl]
        known = np.isfinite(h) & m
        if known.sum() < min_known * m.sum():
            out[sl][m] = r.label
            continue
        top = np.percentile(h[known], 98)
        flat = known & (h < max(min_mm, frac * top))
        if lab is not None:
            chroma = np.hypot(lab[sl][..., 1], lab[sl][..., 2])
            body = m & ~flat
            if body.sum() < 20:
                out[sl][m] = r.label
                continue
            seed_chroma = float(np.median(chroma[body]))
            if seed_chroma < 8:  # dull seed: shadow and seed can't be told apart by colour
                out[sl][m] = r.label
                continue
            flat &= chroma < chroma_frac * seed_chroma
        keep = m & ~flat
        keep = ndi.binary_opening(keep, morphology.disk(max(1, int(round(0.2 / res)))))
        comp, n = ndi.label(keep)
        if n == 0:
            out[sl][m] = r.label
            continue
        sizes = np.bincount(comp.ravel())[1:]
        keep = ndi.binary_fill_holes(comp == (np.argmax(sizes) + 1))
        trimmed += int(m.sum() - keep.sum())
        out[sl][keep] = r.label
    return out, trimmed * res * res
