"""Produce the submission file from the competition data bundle.

    python run.py --data /path/to/data --out TEAM_NAME.csv

`--data` is the unpacked bundle (the folder that contains `tactical/`). Only the tactical
tracking is read:  tactical/BAR-ATH_tactical_atd.xml  and  tactical/BAR-ATH_tactical_atd.txt.

Every frame of the two graded halves gets H, A or D from the two-stage model in models/ (trained on
external matches only, see TECHNICAL.md); every other frame gets X. The frame ranges of the two
halves are the ones the organizers publish in DATA.md. The script prints nothing about the data
beyond a line saying the file was written and validated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from possession import epts, refine
from possession.data import Clip, LETTERS, X

N_FRAMES = 199_319                                   # frames of the tactical video = rows to write
HALVES = ((11_185, 90_651), (110_808, 190_197))      # graded video frames, inclusive (DATA.md)
PITCH = (105.3, 68.0)                                # metres (organizer_code/games.py)


def find(data: Path, suffix: str) -> Path:
    hits = sorted(data.rglob(f"*tactical{suffix}"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one '*tactical{suffix}' under {data}, found {len(hits)}")
    return hits[0]


def predict(data: Path, model_path: Path, n_frames: int, halves, pitch) -> np.ndarray:
    meta = epts.read_metadata(find(data, "_atd.xml"))
    xy, ball = epts.read_raw(find(data, "_atd.txt"), meta, pitch, n_frames)
    first, second, decode = refine.load(model_path)
    labels = np.full(n_frames, X, dtype=np.int8)
    for i, (start, last) in enumerate(halves, 1):
        clip = Clip(name=f"half{i}", fps=meta["fps"], pitch=pitch, xy=xy[start:last + 1],
                    team=meta["team"], ball=ball[start:last + 1])
        labels[start:last + 1] = refine.predict(first, second, clip, decode)
    return labels


def write(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{letter}\n" for i, letter in enumerate(LETTERS[labels]))


def validate(path: Path, n_frames: int) -> None:
    """The organizer's own submission checks (evaluation/validate_submission.py logic)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "organizer_code" / "evaluation"))
    import possession_metrics as pm
    problem = pm.validate_submission(pm.read_label_csv(path), n_frames)
    if problem:
        raise SystemExit(f"INVALID submission: {problem}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the possession submission CSV.")
    ap.add_argument("--data", type=Path, required=True, help="unpacked data bundle (contains tactical/)")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "models" / "gbm.txt")
    # The three options below exist so the pipeline can be tested on external data in the same format.
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--halves", type=int, nargs="+", default=[v for h in HALVES for v in h],
                    help="first last [first last ...] graded video frames, inclusive")
    ap.add_argument("--pitch", type=float, nargs=2, default=PITCH)
    args = ap.parse_args()
    halves = list(zip(args.halves[0::2], args.halves[1::2]))
    labels = predict(args.data, args.model, args.frames, halves, tuple(args.pitch))
    write(args.out, labels)
    validate(args.out, args.frames)
    print(f"wrote {args.out} ({args.frames} rows) -- validation passed")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
