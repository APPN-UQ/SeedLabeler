# SeedLabeler

Desktop app for stereo seed phenotyping with two Sony RX0 cameras, or one
Luxonis OAK-D (chosen per project: CAMERA SYSTEM).

The deliverable is the **traits tables** (per seed and per sample: size, shape,
colour, thickness, volume, uniformity). On the way, every corrected sample
becomes training data, so the app can train an instance-segmentation model for
each crop and use it as the detector.

Builds on [`sonyrx0`](../SonyRX0) (cameras, calibration, depth) and
[`chickpea-seed`](../ChickpeaSeed) (stage, top-down view, segmentation, traits).

## Install (any Windows PC)

1. Get the three repos side by side in one folder: they are already together in
   OneDrive (`uni_UQ_APPF/code/`), or clone them:
   ```
   git clone https://github.com/qmwnebrv/SonyRX0
   git clone https://github.com/qmwnebrv/ChickpeaSeed
   git clone https://github.com/qmwnebrv/SeedLabeler
   ```
2. Double-click **`SeedLabeler/setup.bat`**. It finds conda (offers to install
   Miniconda if missing), creates the `seedlabeler` environment, installs PyTorch
   for this PC's GPU (CUDA 12.8, 12.6 for older drivers, or CPU), installs the
   three packages with the tested versions (`constraints.txt`), runs a self-test
   and puts a **SeedLabeler** shortcut on the desktop. Run it again after a
   `git pull`; options: `-Cpu`, `-EnvName name`, `-NoShortcut`.
3. Plug in the cameras (PC Remote mode, data cables): no drivers or Sony software
   needed. An OAK-D goes in a USB 3 (blue) port; no drivers either.

Without an NVIDIA GPU everything works, but SAM and detection are slower and
training a model takes much longer.

### OAK-D

One camera with colour (12 MP) and stereo depth; it carries its factory
calibration, so there is no calibration session. Live view shows the distance
to the stage: keep it **40 cm or more** (the depth starts at 33 cm, and seed tops
must stay beyond that). A capture saves `raw/rgb.jpg` and `raw/depth.npz`
(the median of 10 depth frames, aligned to the colour), in about 4 s. The colour
camera's focus stays at its calibrated lens position, which keeps depth and
colour aligned. CAMERA settings: `--exposure 8000 --iso 400 --wb 5000` (µs, K;
`auto` for any). Its depth reads a few % off and bends by a few mm across the
view, so OAK projects use the corner markers and LEVEL HEIGHT (below).

### Corner markers (optional, for the lab)

SETTINGS > GROUND PLANE > **Corner markers**, then **PRINT SHEET** (A4 or A3):
a sheet with a small ArUco marker in each corner. Print at 100% ("Actual
size"), tape it down flat, measure its **100 mm bar** with a ruler and enter it
(printers often shrink the page: an A3 sheet fitted onto A4 measured 68 mm).
Seeds go between the markers. In every capture the markers then give the stage
plane, correct the depth's scale (it must agree with the markers' printed size),
and limit detection to the sheet. **LEVEL HEIGHT** measures each seed against
the sheet around it rather than one plane. Also `python tools/make_stage_sheet.py --paper A4`.

### Moving between PCs

- **Projects are self-contained**: calibration, the camera left/right assignment
  (`rig.json`) and the stage file are copied into `<project>/calibration/` and
  stored as relative paths, so a project opens wherever its folder lands. Keep
  projects in OneDrive to have them on both PCs.
- **Code**: `git pull` in each repo (or let OneDrive sync), then run `setup.bat`
  if dependencies changed.
- Don't have the same project open on two PCs at once (OneDrive would merge the
  two sets of edits badly).
- SAM weights download again on each PC on first use (~40-150 MB).

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
<project>/project.json, rig.json, calibration/{calib.npz, stage.npz, stage_sheet.json}
<project>/samples/<id>/raw/{left,right}.jpg (RX0) or raw/rgb.jpg + raw/depth.npz (OAK-D),
                       ortho.png, height.npz, meta.json, annotations.json, <id>_seeds.csv, <id>_summary.csv
<project>/datasets/<crop>/   <project>/models/<crop>/   <project>/exports/
```

## Tests

`pytest tests -s` runs import → detect → edit → remeasure → dataset → export
on a real almond pair, plus offscreen GUI checks (vertex dragging, sample
rename/delete). `tests/test_markers.py` checks the stage sheet, the corner-marker
plane, depth-scale correction and the OAK colour + depth path on a synthetic scene. `python tools/screenshot.py <project> out.png` renders
the window offscreen.
