"""Export SoccerTrack v2 provider possession labels as video-frame spells.

The raw tracker export (raw/<match>/<match>_tracker_box_data.xml) carries a per-frame
`ballStatus` (HOME / AWAY / NEUTRAL / BALLOUT) on the provider's clock. The released
videos, ball tracks and ball-action events use 1-based video frames. Per half, the two
clocks differ by a constant offset, which we recover by matching the raw provider
events (raw/<match>/<match>_12_class_events.json, provider clock) against the released
ball-action events (bas/..., video clock): the most common same-label time difference.

    video_ms    = matchTime - offset
    video_frame = round(video_ms / 40) + 1

Output: labeling/labels_soccertrack.js, a JS file (so the viewer works from file://)
defining SOCCERTRACK_LABELS[match][half] = {offset_ms, n_frames, spells: [[start, end, code], ...]}
with code in H (home), A (away), N (neutral), D (ball out); frames inclusive, 1-based.
"""
from __future__ import annotations

import argparse
import collections
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HALVES = {"FIRST_HALF": ("1", "1st"), "SECOND_HALF": ("2", "2nd")}
CODE = {"HOME": "H", "AWAY": "A", "NEUTRAL": "N", "BALLOUT": "D"}


def clock_offset(raw_events: list[dict], bas_events: list[dict], half: str) -> int:
    """Most common (provider ms - video ms) over same-label event pairs in one half."""
    raw = [(e["label"], int(e["position"])) for e in raw_events if e["gameTime"].startswith(half + " ")]
    bas = [(e["label"], int(e["position"])) for e in bas_events if e["gameTime"].startswith(half + " ")]
    by_label: dict[str, list[int]] = collections.defaultdict(list)
    for label, pos in raw:
        by_label[label].append(pos)
    counts: collections.Counter[int] = collections.Counter()
    for label, pos in bas:
        for rpos in by_label.get(label, ()):
            counts[rpos - pos] += 1
    offset, hits = counts.most_common(1)[0]
    if hits < 0.5 * len(bas):
        raise RuntimeError(f"clock offset for half {half} is ambiguous ({hits}/{len(bas)} events agree)")
    return offset


def read_tracker_status(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows: dict[str, list[tuple[int, str]]] = {h: [] for h in HALVES}
    for _, el in ET.iterparse(path):
        if el.tag == "frame":
            period = el.get("eventPeriod")
            if period in rows:
                rows[period].append((int(el.get("matchTime")), el.get("ballStatus")))
            el.clear()
    return {h: (np.array([r[0] for r in v]), np.array([r[1] for r in v])) for h, v in rows.items()}


def spells(frames: np.ndarray, codes: list[str]) -> list[list]:
    out: list[list] = []
    for f, c in zip(frames.tolist(), codes):
        if out and out[-1][2] == c and out[-1][1] == f - 1:
            out[-1][1] = f
        else:
            out.append([f, f, c])
    return out


def export_match(root: Path, match: str) -> dict:
    raw_ev = json.loads((root / "raw" / match / f"{match}_12_class_events.json").read_text("utf-8"))["annotations"]
    bas_ev = json.loads((root / "bas" / match / f"{match}_12_class_events.json").read_text("utf-8"))["actions"]
    status = read_tracker_status(root / "raw" / match / f"{match}_tracker_box_data.xml")
    result = {}
    for period, (half, tag) in HALVES.items():
        n_frames = len(np.load(root / "ball" / f"{match}_{tag}_ball.npz")["frame"])
        offset = clock_offset(raw_ev, bas_ev, half)
        t, st = status[period]
        vf = np.floor((t - offset) / 40 + 0.5).astype(int) + 1
        keep = (vf >= 1) & (vf <= n_frames)
        result[tag] = {
            "offset_ms": offset,
            "n_frames": n_frames,
            "spells": spells(vf[keep], [CODE[s] for s in st[keep]]),
        }
        print(f"{match} {tag}: offset {offset} ms, {keep.sum()} labelled frames of {n_frames}, "
              f"{len(result[tag]['spells'])} spells")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("data/soccertrack"))
    ap.add_argument("--matches", nargs="+", default=["117093", "118575"])
    ap.add_argument("--out", type=Path, default=Path("labeling/labels_soccertrack.js"))
    args = ap.parse_args()
    data = {m: export_match(args.root, m) for m in args.matches}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("window.SOCCERTRACK_LABELS = " + json.dumps(data, separators=(",", ":")) + ";\n", "utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
