"""Top-down ("orthographic") view of the stage in millimetres.

Every image pixel's viewing ray is intersected with the stage plane, which is a
homography for a pinhole camera. The colour image and the height map are both
warped onto the same stage grid, so they stay aligned and all lengths are in mm
regardless of camera tilt.

Objects standing on the stage are projected along the viewing rays, so a
feature at height h appears enlarged by D / (D - h), with D the camera-to-stage
distance. measure.py corrects for this per seed.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OrthoGrid:
    origin: np.ndarray  # stage point under the optical axis (mm, camera frame)
    e1: np.ndarray      # stage x axis (unit, camera frame)
    e2: np.ndarray      # stage y axis
    a0: float           # stage x of ortho pixel column 0 (mm)
    b0: float           # stage y of ortho pixel row 0 (mm)
    res: float          # mm per ortho pixel
    shape: tuple        # (rows, cols)

    def image_from_ortho(self, K):
        """3x3 homography mapping ortho pixel (col, row, 1) to image pixel for intrinsics K."""
        H = K @ np.column_stack([self.e1, self.e2, self.origin])
        S = np.array([[self.res, 0, self.a0], [0, self.res, self.b0], [0, 0, 1.0]])
        return H @ S

    def warp(self, img, K, interpolation=cv2.INTER_LINEAR, border=0):
        M = self.image_from_ortho(K)
        rows, cols = self.shape
        return cv2.warpPerspective(img, M, (cols, rows), flags=interpolation | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    def cropped(self, K, polygon, margin_px=20):
        """The part of the grid that covers an image polygon (image pixels, intrinsics K)."""
        p = np.linalg.inv(self.image_from_ortho(K)) @ np.vstack([np.asarray(polygon, float).T, np.ones(len(polygon))])
        cols, rows = p[0] / p[2], p[1] / p[2]
        c0 = int(max(0, np.floor(cols.min()) - margin_px))
        r0 = int(max(0, np.floor(rows.min()) - margin_px))
        c1 = int(min(self.shape[1], np.ceil(cols.max()) + margin_px))
        r1 = int(min(self.shape[0], np.ceil(rows.max()) + margin_px))
        return OrthoGrid(self.origin, self.e1, self.e2, self.a0 + c0 * self.res, self.b0 + r0 * self.res, self.res,
                         (r1 - r0, c1 - c0))

    def to_image(self, K, cols, rows):
        """Map ortho pixel coordinates to image pixel coordinates."""
        pts = np.stack([cols, rows, np.ones_like(cols, dtype=float)])
        p = self.image_from_ortho(K) @ pts
        return p[0] / p[2], p[1] / p[2]


def make_grid(stage, K, image_size, res=None, max_side=6000):
    """Stage grid covering the camera's view of the stage.

    res defaults to the image's own pixel footprint on the stage at the centre.
    """
    n = stage.n
    z = np.array([0.0, 0.0, 1.0])
    origin = z * (-stage.d / (n @ z))
    e1 = np.array([1.0, 0, 0]) - n * n[0]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    if e2[1] < 0:  # keep stage y roughly along image y (rows downwards)
        e2 = -e2
    if res is None:
        res = float(np.linalg.norm(origin)) / K[0, 0]
    # stage coordinates of the image corners
    w, h = image_size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    H = K @ np.column_stack([e1, e2, origin])
    ab = np.linalg.inv(H) @ corners
    ab = ab[:2] / ab[2]
    a0, b0 = ab[0].min(), ab[1].min()
    cols = int(np.ceil((ab[0].max() - a0) / res))
    rows = int(np.ceil((ab[1].max() - b0) / res))
    if max(cols, rows) > max_side:  # very oblique views: cap the grid size
        res *= max(cols, rows) / max_side
        cols = int(np.ceil((ab[0].max() - a0) / res))
        rows = int(np.ceil((ab[1].max() - b0) / res))
    return OrthoGrid(origin, e1, e2, float(a0), float(b0), float(res), (rows, cols))
