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

## Screens and modes

**Home**: recent projects, NEW PROJECT (name, location, crop, calibration,
optional stage, camera settings) and OPEN PROJECT.

**Hub** (after opening a project): counts (captured, to process, labelled
samples and seeds, models), the **suggested workflow** with the next step
highlighted, and the three modes. Switch modes any time from the top bar.

1. **LABEL**: capture and correct the first ~20 samples (target in SETTINGS).
2. **TRAIN**: train the crop's model on them. The new model becomes the detector,
   so the next samples you label start from its predictions. Keep going: correct
   another batch, add it to the dataset, **RETRAIN** (default: continue from the
   latest model, on the whole dataset, so it keeps what it learned). The hub
   counts samples added since the last training and suggests RETRAIN NOW once
   5 (or a quarter of the dataset) are new.
3. **CAPTURE**: collect the rest quickly, no detection while working. Locked
   until the crop has a model: train one, or **LOAD MODEL** on the hub (e.g.
   best.pt from another project's run folder; its settings come with it).
4. **PROCESS**: batch-detect everything captured, check flagged samples, export.

| Mode | For | |
|---|---|---|
| **LABEL** | building the model, deliberate | capture or import → detect + SAM → correct → ADD TO DATASET / VIEW DATASET → TRAIN |
| **CAPTURE** | a technician, throughput | live view with the depth-zone box; scan barcode → capture → next; instant checks (both images, sharpness, exposure); session counts |
| **PROCESS** | end of a session | table of samples with status, seeds, depth coverage and flags (LOW DEPTH, COUNT?, NO SEEDS, L/R FIXED); PROCESS NEW / PROCESS SELECTED in the background with progress; double-click to open in LABEL; EXPORT CSV |

The cameras stay connected for the session (`camera.py`): live view, capture
and settings share one connection, so a capture takes a few seconds and needs
no reconnect. Camera settings from the project are applied before the first
capture.

### Editing in LABEL

| | |
|---|---|
| Click | select a seed (also selects its row in the table, and vice versa) |
| Drag seed / white square | move the seed / a vertex |
| Double-click an edge | insert a vertex |
| Right-click a vertex | remove it |
| `Delete` | delete selected seeds |
| **BOX** (SAM) | drag a box around a seed, or just click a seed: SAM outlines it. Then click to include more, Shift-click / right-click to exclude |
| **SAM FIX** or `R` | redraw the selected seed with SAM (clicks then refine it) |
| **REFINE ALL** | redraw every automatic outline with SAM |
| Wheel, middle-drag or `Space`+drag, `F` | zoom, pan, fit |
| **UNDO / REDO**, **DEPTH MAP / RGB** | edit history; switch view |

Each seed is filled with its own colour; the table's status column shows
· automatic, ~ edited, + added by hand. Right-click a sample to rename or delete it.

## SAM

SAM (Segment Anything) runs on a full-resolution crop around each prompt; it
separates seeds from their shadows by itself. Pick the model in SETTINGS:
MobileSAM (~30 ms per seed) or SAM 2.1 base (~100 ms, a little sharper);
weights download once to `~/.seedlabeler/weights`. **SAM AFTER DETECT** runs
REFINE ALL automatically after every detection: the detector finds the seeds,
SAM draws them precisely. On the synthetic seeds, SAM matches the classic
outlines' accuracy (within ~0.1 mm) and halves the seed-to-seed scatter in
length.

## Models

- **ADD TO DATASET** writes the sample's corrected outlines into the datasets of
  the project's current crop, in YOLO segmentation format with an automatic
  ~80/20 train/val split by sample. Images are resampled to ~64 px across the
  crop's seed width (SETTINGS) and cut into overlapping 640 px tiles, so seeds
  look the same size whatever the camera height. Two versions are written:
  `datasets/<crop>/` (colour) and `datasets/<crop>-rgbh/` (brightness, colour
  saturation and height above the stage packed into three channels).
- **INPUT** in the training dialog picks COLOUR or COLOUR + HEIGHT. With height
  the model can tell seeds from shadows and stains of the same colour. The model
  records its input and tiling; detection rebuilds the same input automatically.
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
