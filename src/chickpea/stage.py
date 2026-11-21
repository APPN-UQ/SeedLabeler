"""The stage: the flat surface the seeds lie on, as a plane in the left rectified camera frame.

Plane: n . X + d = 0 (mm), with n pointing towards the camera so that
height(X) = n . X + d is positive above the stage. The camera sits at the
origin, so d is its perpendicular distance to the stage.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Stage:
    n: np.ndarray  # unit normal, towards the camera
    d: float       # camera-to-stage distance (mm)
    method: str = ""

    @classmethod
    def from_point_normal(cls, point, normal, method=""):
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        d = -float(n @ point)
        if d < 0:  # orient towards the camera (origin)
            n, d = -n, -d
        return cls(n, d, method)

    def height(self, xyz):
        """Height above the stage (mm) for an (..., 3) array of points."""
        return xyz @ self.n + self.d

    def save(self, path):
        np.savez(path, n=self.n, d=self.d, method=self.method)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["n"], float(z["d"]), str(z["method"]))


def fit_plane_ransac(points, threshold=0.5, iters=500, seed=0):
    """Dominant plane in a point cloud (mm). Returns (Stage, inlier mask)."""
    rng = np.random.default_rng(seed)
    pts = points[np.isfinite(points).all(axis=1)]
    sample = pts[rng.choice(len(pts), min(len(pts), 50000), replace=False)]
    best_count, best = 0, None
    for _ in range(iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        count = int((np.abs((sample - a) @ normal) < threshold).sum())
        if count > best_count:
            best_count, best = count, (a, normal)
    a, normal = best
    inliers = np.abs((pts - a) @ normal) < threshold
    # least-squares refinement on all inliers
    q = pts[inliers]
    centre = q.mean(axis=0)
    normal = np.linalg.svd(q - centre, full_matrices=False)[2][-1]
    stage = Stage.from_point_normal(centre, normal, "ransac")
    return stage, inliers


def stage_from_board(rect_left_full, K, board, offset_mm=0.0):
    """Stage plane from a ChArUco board lying flat on it, seen in the full-res rectified left image.

    offset_mm: height of the board surface above the stage (e.g. a tablet's
    thickness when the board is shown on a tablet lying on the stage).
    """
    gray = cv2.cvtColor(rect_left_full, cv2.COLOR_BGR2GRAY) if rect_left_full.ndim == 3 else rect_left_full
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < 8:
        raise RuntimeError("Calibration board not found in the stage image")
    obj, img = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None)
    if not ok:
        raise RuntimeError("Could not estimate the board pose")
    R = cv2.Rodrigues(rvec)[0]
    stage = Stage.from_point_normal(tvec.ravel(), R[:, 2], "board")
    stage.d += offset_mm  # stage lies further from the camera than the board surface
    return stage, len(ids)


def find_markers(img, layout):
    """The sheet's corner markers in an image: {id: (4, 2) corners}, OpenCV corner order."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, layout["dictionary"])))
    corners, ids, _ = det.detectMarkers(gray)
    wanted = {m["id"] for m in layout["markers"]}
    if ids is None:
        return {}
    return {int(i): c[0].astype(np.float64) for c, i in zip(corners, ids.ravel()) if int(i) in wanted}


def stage_from_markers(img, K, layout, scale=1.0, min_markers=2):
    """Stage plane from the corner markers of a printed stage sheet (the seeds lie on the sheet).

    layout: the sheet's JSON (marker ids and corner positions in mm, see
    seedlabeler.sheet); scale: printed size / design size, e.g. 0.68 when the
    100 mm bar measures 68 mm. img must be undistorted with intrinsics K.
    Returns (Stage, found markers {id: corners}, mean reprojection error in px,
    seed area: (4, 2) image polygon through the four marker centres, with
    "holes": every marker plus its white border, as image polygons).
    """
    found = find_markers(img, layout)
    if len(found) < min_markers:
        raise RuntimeError(f"Corner markers: found {len(found)}, need at least {min_markers}. "
                           "Keep the sheet's corners in view.")
    pos = {m["id"]: np.array(m["corners_mm"], float) * scale for m in layout["markers"]}
    obj = np.concatenate([np.c_[pos[i], np.zeros(4)] for i in found])
    img_pts = np.concatenate(list(found.values()))
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("Corner markers: could not estimate the sheet's pose")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img_pts, K, None, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())
    R = cv2.Rodrigues(rvec)[0]

    def project(pts_mm):
        p, _ = cv2.projectPoints(np.c_[np.asarray(pts_mm, float) * scale, np.zeros(len(pts_mm))], rvec, tvec, K, None)
        return p.reshape(-1, 2)

    # seeds go between the markers: that leaves out the markers, the header text and the scale bar
    area = {"outline": project([m["centre_mm"] for m in layout["markers"]]), "holes": []}
    for m in layout["markers"]:
        c = np.array(m["centre_mm"])
        area["holes"].append(project(c + (np.array(m["corners_mm"]) - c) * 2.0))  # marker + white border
    return Stage.from_point_normal(tvec.ravel(), R[:, 2], "markers"), found, err, area


def depth_scale_from_markers(stage, markers, xyz, K_xyz, img_shape, max_error=0.2):
    """Factor that makes the measured depth agree with the marker plane.

    The markers' printed size fixes the plane's distance; stereo depth on the
    (well textured) markers gives the same distance up to the depth's scale
    error. Multiply xyz by the returned factor. xyz is (h, w, 3) for intrinsics
    K_xyz; markers are corners in the full image of shape img_shape.
    """
    h, w = xyz.shape[:2]
    sx, sy = w / img_shape[1], h / img_shape[0]
    ratios = []
    for c in markers.values():
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(c * (sx, sy)).astype(np.int32)], 1)
        ys, xs = np.nonzero(m)
        z = xyz[ys, xs, 2]
        ok = np.isfinite(z) & (z > 0)
        if ok.sum() < 10:
            continue
        rays = np.column_stack([(xs[ok] - K_xyz[0, 2]) / K_xyz[0, 0], (ys[ok] - K_xyz[1, 2]) / K_xyz[1, 1],
                                np.ones(ok.sum())])
        z_plane = -stage.d / (rays @ stage.n)  # where each pixel's ray meets the plane
        ratios.append(float(np.median(z_plane / z[ok])))
    if not ratios:
        raise RuntimeError("Corner markers: no depth on the markers")
    k = float(np.median(ratios))
    if abs(k - 1) > max_error:
        raise RuntimeError(f"Corner markers and depth disagree by {100 * (k - 1):+.0f}%: "
                           "check the printed scale (measured 100 mm bar) in settings")
    return k
