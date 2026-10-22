# SeedLabeler

Desktop app for stereo seed phenotyping with two Sony RX0 cameras.

The deliverable is the **traits tables** (per seed and per sample: size, shape,
colour, thickness, volume, uniformity). On the way, every corrected sample
becomes training data, so the app can train an instance-segmentation model for
each crop and use it as the detector.

Builds on [`sonyrx0`](../SonyRX0) (cameras, calibration, depth) and
[`chickpea-seed`](../ChickpeaSeed) (stage, top-down view, segmentation, traits).

## Install

```
conda create -n seedlabeler python=3.11
conda activate seedlabeler
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs CUDA 12.8 builds
pip install -e ../SonyRX0 -e ../ChickpeaSeed -e .[train,dev]
pip uninstall -y opencv-python            # ultralytics pulls this in; it shadows opencv-contrib (aruco)
pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84
seedlabeler                               # or: python -m seedlabeler [project_dir]
```

## Workflow

1. **NEW** project → pick a folder → **SETTINGS**: crop, stereo calibration
   (`calib.npz`), optional stage plane (`stage.npz`), camera settings, seed width,
   minimum thickness.
2. **LIVE** shows the left camera (~25 fps); **DETECT** next to it outlines and
   counts seeds on the live view (a quick 2D framing aid, not a measurement).
3. **SAMPLE ID**: scan a barcode (scanners type the code + Enter) or type an ID.
   With **AUTO** on, the scan captures straight away under that ID; otherwise
   press **CAPTURE** (cameras in PC Remote mode) or **IMPORT PAIR** (left image
   first). Full detection then runs automatically (~10 s for a 15 MP pair).
   Right-click a sample (or select it and press `Delete`) to **RENAME** or
   **DELETE** it; renaming keeps its CSVs and dataset entries consistent.
4. Review on the canvas, fix what's wrong (traits update after every edit):

   | | |
   |---|---|
   | Click | select a seed (also selects its row in the table, and vice versa) |
   | Drag seed / white square | move the seed / a vertex |
   | Double-click an edge | insert a vertex |
   | Right-click a vertex | remove it |
   | `Delete` | delete selected seeds |
   | **DRAW** | click points, double-click or `Enter` to close, `Backspace` undo point, `Esc` cancel |
   | Wheel, middle-drag or `Space`+drag, `F` | zoom, pan, fit |
   | **UNDO / REDO**, **HEIGHT** | edit history, colour/height view |

   Outline colours: white = automatic, blue = edited, red = drawn by hand.
5. **EXPORT CSV**: all samples into `exports/<project>_seeds.csv` and
   `_summary.csv` (each sample's own CSVs are always in its folder).

## Models

- **ADD TO DATASET** writes the sample's corrected outlines into the dataset of
  the project's current crop (`datasets/<crop>/`, YOLO segmentation format,
  automatic ~80/20 train/val split by sample).
- **TRAIN** fine-tunes a pretrained segmentation model (or continues from an
  earlier one) on the GPU in a background process, with a live log. Runs land in
  `models/<crop>/<run>/`.
- Pick the detector (**CLASSIC** or a trained model, with its validation mask
  mAP) next to **DETECT**. Switching crop in settings switches to that crop's
  dataset and models, so each crop gets its own models.
- Every polygon keeps its status (auto / edited / manual) and the original
  automatic outlines are stored too, so detector accuracy can be measured
  against the corrections.

Ultralytics YOLO is AGPL-3.0: fine for research use, check before
distributing the app.

## Project layout

```
<project>/project.json, rig.json
<project>/samples/<id>/raw/{left,right}.jpg, ortho.png, height.npz, meta.json,
                       annotations.json, <id>_seeds.csv, <id>_summary.csv
<project>/datasets/<crop>/   <project>/models/<crop>/   <project>/exports/
```

## Tests

`pytest tests -s` runs import → detect → edit → remeasure → dataset → export
on a real almond pair, plus offscreen GUI checks (vertex dragging, sample
rename/delete). `python tools/screenshot.py <project> out.png` renders
the window offscreen.
