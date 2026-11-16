"""SoccerTrack v2 (external data, CC BY 4.0) -> Clip, one per half.

Source: the provider's raw tracking export, raw/<match>/<match>_tracker_box_data.xml, which has
per frame the pitch position of every player, a ball position and `ballStatus`
(HOME / AWAY / NEUTRAL / BALLOUT) -- the provider's own possession annotation, used here as
external ground truth.

The raw export runs on the provider clock; the released videos / ball / event files use 1-based
video frames. Per half the two differ by a constant offset, recovered by matching the provider's
raw events against the released (video-clock) events. Clip index i is video frame i + 1.

    python -m possession.soccertrack            # builds data/cache/soccertrack/<match>_<half>.npz
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .data import AWAY, HOME, Clip, A, D, H, X

ROOT = Path("data/soccertrack")
CACHE = Path("data/cache/soccertrack")
# The eight matches whose raw export carries possession status. The other two released matches
# (132831, 132877) only say INPLAY, so they have no possession ground truth and are not used.
MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058"]
HALVES = {"FIRST_HALF": ("1", "1st"), "SECOND_HALF": ("2", "2nd")}
FPS = 25
FRAME_MS = 1000 // FPS

# Provider status codes kept verbatim in Clip.extra["status"].
S_HOME, S_AWAY, S_NEUTRAL, S_OUT, S_NONE = 0, 1, 2, 3, -1
STATUS = {"HOME": S_HOME, "AWAY": S_AWAY, "NEUTRAL": S_NEUTRAL, "BALLOUT": S_OUT}

_FRAME = re.compile(r'<frame matchTime="(-?\d+)" frameNumber="[^"]*" eventPeriod="(\w+)" ballStatus="(\w+)"')
_OBJ = re.compile(r'<(?:player|ball) playerId="([^"]+)" loc="\[([^,\]]+), ([^\]]+)\]"')


def clock_offset(raw_events: list[dict], bas_events: list[dict], half: str) -> int:
    """Most common (provider ms - video ms) over same-label event pairs in one half."""
    by_label: dict[str, list[int]] = collections.defaultdict(list)
    for e in raw_events:
        if e["gameTime"].startswith(half + " "):
            by_label[e["label"]].append(int(e["position"]))
    bas = [(e["label"], int(e["position"])) for e in bas_events if e["gameTime"].startswith(half + " ")]
    counts: collections.Counter[int] = collections.Counter()
    for label, pos in bas:
        for rpos in by_label.get(label, ()):
            counts[rpos - pos] += 1
    offset, hits = counts.most_common(1)[0]
    if hits < 0.5 * len(bas):
        raise RuntimeError(f"clock offset for half {half} is ambiguous ({hits}/{len(bas)} events agree)")
    return offset


def read_metadata(path: Path) -> tuple[list[str], np.ndarray, tuple[float, float]]:
    """(player ids in slot order, team per slot, pitch size)."""
    root = ET.parse(path).getroot()
    side = {t.get("id"): t.get("side") for t in root.iter("team")}
    players = sorted({(side[p.get("teamId")] != "home", p.get("id")) for p in root.iter("player")
                      if p.get("id") and p.get("teamId") in side})
    pitch = root.find("pitch")
    return ([pid for _, pid in players],
            np.array([AWAY if away else HOME for away, _ in players], dtype=np.int8),
            (float(pitch.get("width")), float(pitch.get("height"))))


def status_to_labels(status: np.ndarray) -> np.ndarray:
    """Provider status -> competition labels.

    BALLOUT -> D. HOME/AWAY -> H/A. NEUTRAL (ball alive, nobody in control) has no class in the
    competition, whose rule is that a loose or contested ball stays with the last team that had
    it: forward-fill from the last owner within the same alive stretch; a NEUTRAL stretch that
    opens play right after a stoppage takes the next owner instead. Unlabelled frames -> X.
    """
    labels = np.full(len(status), X, dtype=np.int8)
    labels[status == S_OUT] = D
    labels[status == S_HOME] = H
    labels[status == S_AWAY] = A
    neutral = status == S_NEUTRAL
    if neutral.any():
        idx = np.arange(len(status))
        known = ~neutral
        prev = np.maximum.accumulate(np.where(known, idx, -1))
        nxt = np.minimum.accumulate(np.where(known, idx, len(status))[::-1])[::-1]
        for i in np.flatnonzero(neutral):
            p, n = prev[i], nxt[i]
            if p >= 0 and labels[p] in (H, A):
                labels[i] = labels[p]
            elif n < len(status) and labels[n] in (H, A):
                labels[i] = labels[n]
            else:
                labels[i] = D  # alive but never owned between two stoppages
    return labels


def load_match(match: str, root: Path = ROOT) -> dict[str, Clip]:
    raw_dir = root / "raw" / match
    player_ids, team, pitch = read_metadata(raw_dir / f"{match}_tracker_box_metadata.xml")
    slot = {pid: i for i, pid in enumerate(player_ids)}
    raw_ev = json.loads((raw_dir / f"{match}_12_class_events.json").read_text("utf-8"))["annotations"]
    bas_ev = json.loads((root / "bas" / match / f"{match}_12_class_events.json").read_text("utf-8"))["actions"]

    n_frames, offset, xy, ball, status = {}, {}, {}, {}, {}
    for period, (half, tag) in HALVES.items():
        n_frames[period] = len(np.load(root / "ball" / f"{match}_{tag}_ball.npz")["frame"])
        offset[period] = clock_offset(raw_ev, bas_ev, half)
        xy[period] = np.full((n_frames[period], len(player_ids), 2), np.nan, dtype=np.float32)
        ball[period] = np.full((n_frames[period], 2), np.nan, dtype=np.float32)
        status[period] = np.full(n_frames[period], S_NONE, dtype=np.int8)

    length, width = pitch
    period, i, unknown = None, -1, collections.Counter()
    with open(raw_dir / f"{match}_tracker_box_data.xml", encoding="utf-8") as handle:
        for line in handle:
            m = _OBJ.search(line)
            if m:
                if i < 0:
                    continue
                pid, x, y = m.groups()
                try:
                    px, py = float(x) * length, float(y) * width
                except ValueError:
                    continue
                if pid == "ball":
                    ball[period][i] = (px, py)
                elif pid in slot:
                    xy[period][i, slot[pid]] = (px, py)
                else:
                    unknown[pid] += 1
                continue
            m = _FRAME.search(line)
            if m:
                t, period, st = int(m.group(1)), m.group(2), m.group(3)
                i = -1
                if period in HALVES:
                    k = int(np.floor((t - offset[period]) / FRAME_MS + 0.5))  # 0-based video frame
                    if 0 <= k < n_frames[period]:
                        i = k
                        status[period][i] = STATUS[st]
    if unknown:
        print(f"  {match}: {sum(unknown.values())} positions of {len(unknown)} ids not in metadata, skipped")

    # The export's ball is not a measurement: it is drawn in straight lines between annotated
    # events, and during a stoppage it glides to a fixed parking spot beside the pitch. That
    # glide would give the label away, so there is simply no ball while the ball is out.
    for period in HALVES:
        ball[period][status[period] == S_OUT] = np.nan

    clips = {}
    for period, (_, tag) in HALVES.items():
        clips[tag] = Clip(name=f"st_{match}_{tag}", fps=FPS, pitch=pitch, xy=xy[period], team=team,
                          ball=ball[period], labels=status_to_labels(status[period]),
                          extra={"status": status[period], "offset_ms": np.int64(offset[period])})
    return clips


def cached(match: str, tag: str, cache: Path = CACHE) -> Clip:
    return Clip.load(cache / f"{match}_{tag}.npz")


def available(root: Path = ROOT) -> list[str]:
    return [m for m in MATCHES if (root / "raw" / m / f"{m}_tracker_box_data.xml").exists()
            and (root / "ball" / f"{m}_2nd_ball.npz").exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the SoccerTrack clip cache.")
    ap.add_argument("--matches", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for match in args.matches or available():
        if not args.force and all((CACHE / f"{match}_{t}.npz").exists() for t in ("1st", "2nd")):
            continue
        for tag, clip in load_match(match).items():
            clip.save(CACHE / f"{match}_{tag}.npz")
            lab = clip.labels
            share = {n: round(float((lab == c).mean()), 3) for n, c in (("H", H), ("A", A), ("D", D), ("X", X))}
            print(f"{clip.name}: {clip.n_frames} frames, {clip.xy.shape[1]} slots, "
                  f"ball seen {np.isfinite(clip.ball[:, 0]).mean():.3f}, labels {share}")


if __name__ == "__main__":
    main()
