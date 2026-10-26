"""Printable stage sheet: four small ArUco markers in the corners, a clear area for the seeds.

The sheet is optional (lab setups): when a project uses it, the corner markers
give the stage plane in every capture and correct the depth's scale (see
chickpea.stage.stage_from_markers). write_sheet() makes a vector PDF plus a JSON
layout beside it: marker ids, size and corner positions in sheet mm (origin
top-left, x right, y down). Markers come from DICT_5X5_100 like the calibration
board, but use ids 96-99, which the 11x8 ChArUco board (ids 0-43) never does.

Printers often shrink the page to their printable area, so after printing,
measure the 100 mm bar: the project stores that as the sheet's scale.
"""
import json
from pathlib import Path

import cv2

DICT = "DICT_5X5_100"
IDS = (96, 97, 98, 99)  # top-left, top-right, bottom-right, bottom-left
PAPERS = {  # landscape size, marker size, sheet edge to marker centre (mm)
    "A3": {"size": (420.0, 297.0), "marker_mm": 20.0, "inset_mm": 25.0},
    "A4": {"size": (297.0, 210.0), "marker_mm": 15.0, "inset_mm": 18.0},
}


def layout(paper="A4", marker_mm=None, inset_mm=None):
    p = PAPERS[paper]
    (w, h), marker_mm, inset_mm = p["size"], marker_mm or p["marker_mm"], inset_mm or p["inset_mm"]
    centres = [(inset_mm, inset_mm), (w - inset_mm, inset_mm), (w - inset_mm, h - inset_mm), (inset_mm, h - inset_mm)]
    r = marker_mm / 2
    markers = []
    for mid, (cx, cy) in zip(IDS, centres):
        # OpenCV's corner order: top-left, top-right, bottom-right, bottom-left
        corners = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        markers.append({"id": mid, "centre_mm": [cx, cy], "corners_mm": [list(c) for c in corners]})
    return {"paper": paper, "dictionary": DICT, "sheet_mm": [w, h], "marker_mm": marker_mm, "markers": markers}


def load_layout(path):
    return json.loads(Path(path).read_text())


def _draw_marker(ax, mid, x0, y0, size_mm, dictionary):
    from matplotlib.patches import Rectangle

    bits = cv2.aruco.generateImageMarker(dictionary, mid, 7)  # 5x5 code + 1-cell black border
    cell = size_mm / bits.shape[0]
    for r in range(bits.shape[0]):
        c = 0
        while c < bits.shape[1]:  # one rectangle per run of black cells: no hairline seams
            if bits[r, c] == 0:
                c1 = c
                while c1 < bits.shape[1] and bits[r, c1] == 0:
                    c1 += 1
                ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), (c1 - c) * cell, cell,
                                       facecolor="black", edgecolor="none", antialiased=False))
                c = c1
            else:
                c += 1


def write_sheet(out, paper="A4", marker_mm=None, inset_mm=None):
    """Write the sheet as a PDF (or PNG, by suffix) and its layout as JSON beside it. Returns the JSON path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lay = layout(paper, marker_mm, inset_mm)
    w, h = lay["sheet_mm"]
    marker_mm = lay["marker_mm"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
    fig = plt.figure(figsize=(w / 25.4, h / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    grey = "#9a9a9a"
    for m in lay["markers"]:
        (x0, y0), (cx, cy) = m["corners_mm"][0], m["centre_mm"]
        _draw_marker(ax, m["id"], x0, y0, marker_mm, dictionary)
        below = cy < h / 2
        ax.text(cx, cy + (1 if below else -1) * (marker_mm / 2 + 3), str(m["id"]), color=grey, fontsize=6,
                ha="center", va="top" if below else "bottom")
    # print-scale check along the bottom edge, outside the seed area
    sx, sy = w / 2 - 50, h - 12
    ax.plot([sx, sx + 100], [sy, sy], color="black", lw=0.6)
    for i in range(11):
        t = 2.5 if i % 5 == 0 else 1.5
        ax.plot([sx + 10 * i, sx + 10 * i], [sy - t, sy], color="black", lw=0.6)
    ax.text(w / 2, sy + 2, "100 mm", fontsize=6, ha="center", va="top")
    dx = lay["markers"][1]["centre_mm"][0] - lay["markers"][0]["centre_mm"][0]
    dy = lay["markers"][3]["centre_mm"][1] - lay["markers"][0]["centre_mm"][1]
    ax.text(w / 2, 8, f"SeedLabeler stage sheet {paper} · ArUco {DICT} ids {IDS[0]}-{IDS[-1]}, {marker_mm:g} mm · "
                      f"marker centres {dx:g} x {dy:g} mm apart · print at 100% (actual size), tape down flat, "
                      "then measure the 100 mm bar",
            color=grey, fontsize=6, ha="center", va="center")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    js = out.with_suffix(".json")
    js.write_text(json.dumps(lay, indent=2))
    return js
