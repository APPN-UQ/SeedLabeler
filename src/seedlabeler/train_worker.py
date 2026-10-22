"""Training process launched by the GUI: python -m seedlabeler.train_worker --data ... --project ... --name ..."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--base", default="yolo11s-seg.pt")
    ap.add_argument("--input", default="rgb", choices=["rgb", "rgbh"])
    ap.add_argument("--height-scale", type=float, default=6.0)
    ap.add_argument("--work-res", type=float, default=None)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    # Small datasets (the usual case when starting a crop): Ultralytics' auto batch
    # puts every image in one batch, i.e. one optimiser step per epoch, and its
    # learning-rate warm-up lasts at least 100 steps, so the model barely learns.
    # Use a few images per batch and skip warm-up until there are enough steps.
    root = Path(args.data).parent
    n_train = len(list((root / "images" / "train").glob("*.*")))
    batch = -1 if n_train >= 64 else max(2, min(8, n_train // 2))
    steps_per_epoch = max(1, n_train // max(batch, 1)) if batch > 0 else 8
    warmup = 3.0 if steps_per_epoch * 3 >= 100 else 0.0
    print(f"SEEDLABELER {n_train} training images, batch {batch}, warmup epochs {warmup}", flush=True)

    model = YOLO(args.base)
    model.train(data=args.data, project=args.project, name=args.name, exist_ok=True,
                epochs=args.epochs, imgsz=args.imgsz, batch=batch, patience=max(50, args.epochs // 3),
                warmup_epochs=warmup, workers=2, plots=False, verbose=True,
                # seeds have no "up": allow any rotation/flip
                degrees=180, fliplr=0.5, flipud=0.5,
                **({"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0} if args.input == "rgbh" else {}))
    run = Path(args.project) / args.name
    (run / "seedlabeler.json").write_text(json.dumps({
        "base": args.base, "data": args.data, "imgsz": args.imgsz,
        "input": args.input, "height_scale_mm": args.height_scale,
        "work_res_mm": args.work_res, "tile": args.tile, "overlap": args.overlap}, indent=2))
    print(f"SEEDLABELER_DONE {run}", flush=True)


if __name__ == "__main__":
    main()
