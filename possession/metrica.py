"""Metrica Sports sample data (external, github.com/metrica-sports/sample-data) -> Clip per half.

Three anonymised professional matches with synchronised, corrected tracking and events, from the
same provider as the competition. Games 1-2 are CSV, game 3 is the FIFA/EPTS format.

Ground truth derived here (no per-frame owner is published, so it comes from tracking + events):
  D  where the tracking has no ball -- in this data the ball is only absent while it is dead.
  H/A elsewhere: the team of the most recent on-ball event (PASS, CARRY, SHOT, RECOVERY,
     SET PIECE, BALL LOST, BALL OUT). A lost ball stays with the team that lost it until the
     other team's RECOVERY, matching the competition rule for loose balls.
  With `controlled=True` (default), a team spell made only of a RECOVERY followed by losing
     the ball again (no pass, carry, shot or restart of its own) is a touch without control and
     and shorter than 3 s is given back to the team before it -- the competition's "does not
     count" cases. With it, possessions average 11-16 s, in line with the 14.2 s the organizers
     publish for their ground truth; without it they average 8-10 s.

    python -m possession.metrica      # builds data/cache/metrica/game<N>_<half>.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import epts
from .data import AWAY, HOME, Clip, A, D, H, X

ROOT = Path("data/metrica_sample/data")
CACHE = Path("data/cache/metrica")
GAMES = [1, 2, 3]
PITCH = (105.0, 68.0)
FPS = 25
ON_BALL = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT"}
CONTROL = {"PASS", "CARRY", "SHOT", "SET PIECE"}   # events that show the team actually had it
KEEP_S = 3.0  # "a player keeps the ball for a few seconds" also counts as control (games 1-2 log no carries)


def labels_from_events(ball_seen: np.ndarray, events: list[tuple[int, int, str]],
                       controlled: bool = True) -> np.ndarray:
    """events: (0-based frame, team code, TYPE), any order. Returns H/A/D codes for one clip."""
    n = len(ball_seen)
    events = sorted((f, t, k) for f, t, k in events if k in ON_BALL and 0 <= f < n)
    owner = np.full(n, -1, dtype=np.int8)
    for f, t, _ in events:
        owner[f] = t
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(owner >= 0, idx, -1))
    nxt = np.minimum.accumulate(np.where(owner >= 0, idx, n)[::-1])[::-1]
    filled = np.where(last >= 0, owner[np.maximum(last, 0)], owner[np.minimum(nxt, n - 1)])
    labels = np.where(ball_seen, np.where(filled == HOME, H, A), D).astype(np.int8)
    if (filled < 0).all():
        labels[ball_seen] = X
        return labels
    if not controlled:
        return labels

    # Team spells within each alive stretch; drop the ones with no sign of control.
    ev_frames = np.array([f for f, _, _ in events])
    ev_ctrl = np.array([k in CONTROL for _, _, k in events])
    ev_team = np.array([t for _, t, _ in events])
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], change, [n]))
    for i in range(1, len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        code = labels[a]
        if code == D or labels[a - 1] == D:
            continue  # a spell that opens play after a stoppage always has a clear possessor
        team = HOME if code == H else AWAY
        inside = (ev_frames >= a) & (ev_frames < b) & (ev_team == team)
        if not ev_ctrl[inside].any() and (b - a) < KEEP_S * FPS:
            labels[a:b] = labels[a - 1]
    return labels


def _clip(name, xy, team, ball, events, controlled=True) -> Clip:
    seen = np.isfinite(ball[:, 0])
    return Clip(name=name, fps=FPS, pitch=PITCH, xy=xy.astype(np.float32), team=team,
                ball=ball.astype(np.float32), labels=labels_from_events(seen, events, controlled),
                extra={"labels_raw": labels_from_events(seen, events, False)})


def load_csv_game(game: int, root: Path = ROOT) -> dict[str, Clip]:
    folder = root / f"Sample_Game_{game}"
    frames = {}
    for side in ("Home", "Away"):
        frames[side] = pd.read_csv(folder / f"Sample_Game_{game}_RawTrackingData_{side}_Team.csv", skiprows=2)
    period = frames["Home"]["Period"].values
    scale = np.array(PITCH, dtype=np.float32)
    parts, team = [], []
    for code, side in ((HOME, "Home"), (AWAY, "Away")):
        values = frames[side].iloc[:, 3:-2].values.astype(np.float32)
        parts.append(values.reshape(len(values), -1, 2))
        team += [code] * parts[-1].shape[1]
    xy = np.concatenate(parts, axis=1) * scale
    ball = frames["Home"].iloc[:, -2:].values.astype(np.float32) * scale
    team = np.array(team, dtype=np.int8)

    ev = pd.read_csv(folder / f"Sample_Game_{game}_RawEventsData.csv")
    clips = {}
    for p, tag in ((1, "1st"), (2, "2nd")):
        rows = np.flatnonzero(period == p)
        first = rows[0]
        events = [(int(f) - 1 - first, HOME if t == "Home" else AWAY, str(k).upper())
                  for f, t, k, q in zip(ev["Start Frame"], ev["Team"], ev["Type"], ev["Period"]) if q == p]
        clips[tag] = _clip(f"mt_game{game}_{tag}", xy[rows], team, ball[rows], events)
    return clips


def load_epts_game(game: int = 3, root: Path = ROOT) -> dict[str, Clip]:
    folder = root / f"Sample_Game_{game}"
    meta = epts.read_metadata(folder / f"Sample_Game_{game}_metadata.xml")
    xy, ball = epts.read_raw(folder / f"Sample_Game_{game}_tracking.txt", meta, PITCH)
    data = json.loads((folder / f"Sample_Game_{game}_events.json").read_text("utf-8"))["data"]
    home_id = meta["team_ids"][0]
    p = meta["params"]
    spans = {"1st": (int(p["first_half_start"]), int(p["first_half_end"])),
             "2nd": (int(p["second_half_start"]), int(p["second_half_end"]))}
    clips = {}
    for period, tag in ((1, "1st"), (2, "2nd")):
        a, b = spans[tag][0] - 1, min(spans[tag][1], len(xy))
        events = [(int(e["start"]["frame"]) - 1 - a, HOME if e["team"]["id"] == home_id else AWAY,
                   e["type"]["name"].upper()) for e in data if e["period"] == period and e.get("team")]
        clips[tag] = _clip(f"mt_game{game}_{tag}", xy[a:b], meta["team"], ball[a:b], events)
    return clips


def load_game(game: int) -> dict[str, Clip]:
    return load_epts_game(game) if game == 3 else load_csv_game(game)


def cached(game: int | str, tag: str, cache: Path = CACHE) -> Clip:
    return Clip.load(cache / f"game{game}_{tag}.npz")


def available(cache: Path = CACHE) -> list[str]:
    return [str(g) for g in GAMES if (cache / f"game{g}_2nd.npz").exists()]


def main() -> None:
    for game in GAMES:
        for tag, clip in load_game(game).items():
            clip.save(CACHE / f"game{game}_{tag}.npz")
            for name, lab in (("controlled", clip.labels), ("raw", clip.extra["labels_raw"])):
                change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
                bounds = np.concatenate(([0], change, [len(lab)]))
                codes, dur = lab[bounds[:-1]], np.diff(bounds) / FPS
                poss = dur[(codes == H) | (codes == A)]
                share = {n: round(float((lab == c).mean()), 3) for n, c in (("H", H), ("A", A), ("D", D))}
                print(f"{clip.name} {name:<10} {clip.n_frames} frames, labels {share}, "
                      f"{len(poss)} possessions, mean {poss.mean():.1f}s median {np.median(poss):.1f}s")


if __name__ == "__main__":
    main()
