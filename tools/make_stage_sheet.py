"""Write a stage sheet with corner markers (also available in the app: SETTINGS > PRINT SHEET).

    python tools/make_stage_sheet.py [--paper A4|A3] [--out boards/stage_sheet_a4.pdf]

Print at 100% / "Actual size", landscape, then measure the 100 mm bar and enter
it in the project settings. See seedlabeler/sheet.py.
"""
import argparse
from pathlib import Path

from seedlabeler.sheet import PAPERS, write_sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", choices=list(PAPERS), default="A4")
    ap.add_argument("--out", help="default: boards/stage_sheet_<paper>.pdf")
    ap.add_argument("--marker-mm", type=float)
    ap.add_argument("--inset-mm", type=float, help="sheet edge to marker centre")
    args = ap.parse_args()
    out = args.out or Path(__file__).resolve().parent.parent / "boards" / f"stage_sheet_{args.paper.lower()}.pdf"
    js = write_sheet(out, args.paper, args.marker_mm, args.inset_mm)
    print(f"Wrote {out} and {js.name}. Print at 100% (actual size), then measure the 100 mm bar.")


if __name__ == "__main__":
    main()
