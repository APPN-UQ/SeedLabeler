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
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(data=args.data, project=args.project, name=args.name, exist_ok=True,
                epochs=args.epochs, imgsz=args.imgsz, batch=-1, patience=30,
                workers=2, plots=False, verbose=True,
                # seeds have no "up": allow any rotation/flip
                degrees=180, fliplr=0.5, flipud=0.5)
    run = Path(args.project) / args.name
    (run / "seedlabeler.json").write_text(json.dumps({"base": args.base, "data": args.data,
                                                      "imgsz": args.imgsz}, indent=2))
    print(f"SEEDLABELER_DONE {run}", flush=True)


if __name__ == "__main__":
    main()
