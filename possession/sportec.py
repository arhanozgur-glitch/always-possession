"""DFL / Sportec open data (external) -> Clip per half.

Bassek et al. 2025, "An integrated dataset of spatiotemporal and event data in elite soccer",
Scientific Data; files from figshare, doi 10.6084/m9.figshare.28196177, licence CC BY 4.0.
Seven matches of the German Bundesliga and 2. Bundesliga (2022/23) with the official optical
tracking at 25 fps: every player, a measured ball, and for every frame the provider's own
`BallPossession` (1 = home, 2 = guest) and `BallStatus` (1 = in play, 0 = dead).

Ground truth: D where BallStatus is 0, otherwise H / A from BallPossession. The provider hands
the ball over on every touch (possessions average 7 s); the competition does not count a touch
without control, so a spell under 3 s that merely interrupts the other team is given back
(`controlled`, the raw labels stay in `extra["labels_raw"]`). As for the other
sources, the ball position is blanked while the ball is dead (degrade.py decides what a camera
system would have seen then); the measured dead-ball track is kept in `extra["ball_dead"]`.

    python -m possession.sportec      # builds data/cache/sportec/<match>_<half>.npz
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .data import AWAY, HOME, Clip, A, D, H

ROOT = Path("data/sportec")
CACHE = Path("data/cache/sportec")
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
FPS = 25
SECTIONS = {"firstHalf": "1st", "secondHalf": "2nd"}
KEEP_S = 3.0  # same length as in metrica.py: a shorter interruption is a touch, not a possession


def _file(kind: str, match: str, root: Path) -> Path:
    hits = sorted(root.glob(f"DFL_{kind}_*_DFL-MAT-{match}.xml"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{kind} file of {match} under {root}: found {len(hits)}")
    return hits[0]


def read_info(path: Path) -> tuple[dict[str, int], tuple[float, float]]:
    """({team id: HOME/AWAY}, pitch size in metres)."""
    root = ET.parse(path).getroot()
    env = root.find(".//Environment")
    teams = {t.get("TeamId"): HOME if t.get("Role") == "home" else AWAY for t in root.iter("Team")}
    return teams, (float(env.get("PitchX")), float(env.get("PitchY")))


def read_positions(path: Path) -> list[dict]:
    """One dict per FrameSet: section, team, person, n [k], xy [k, 2] and, for the ball, poss / status."""
    sets, cur = [], None
    for event, el in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            if el.tag == "FrameSet":
                cur = dict(section=el.get("GameSection"), team=el.get("TeamId"), person=el.get("PersonId"),
                           n=[], x=[], y=[], poss=[], status=[])
                cur["ball"] = (cur["team"] or "").upper() == "BALL"
        elif el.tag == "Frame":
            a = el.attrib
            cur["n"].append(int(a["N"]))
            cur["x"].append(float(a["X"]))
            cur["y"].append(float(a["Y"]))
            if cur["ball"]:
                cur["poss"].append(int(a["BallPossession"]))
                cur["status"].append(int(a["BallStatus"]))
            el.clear()
        elif el.tag == "FrameSet":
            for key in ("n", "poss", "status"):
                cur[key] = np.array(cur[key], dtype=np.int64)
            cur["xy"] = np.stack([np.array(cur.pop("x"), np.float32), np.array(cur.pop("y"), np.float32)], axis=1)
            sets.append(cur)
            el.clear()
    return sets


def load_match(match: str, root: Path = ROOT) -> dict[str, Clip]:
    teams, pitch = read_info(_file("02_01", match, root))
    sets = read_positions(_file("04_03", match, root))
    half_pitch = np.array(pitch, dtype=np.float32) / 2
    people = sorted({(teams[s["team"]], s["person"]) for s in sets if s["team"] in teams})
    slot = {person: i for i, (_, person) in enumerate(people)}
    team = np.array([t for t, _ in people], dtype=np.int8)
    clips = {}
    for section, tag in SECTIONS.items():
        ball = next(s for s in sets if s["ball"] and s["section"] == section)
        first, n = int(ball["n"][0]), int(ball["n"][-1] - ball["n"][0] + 1)
        xy = np.full((n, len(people), 2), np.nan, dtype=np.float32)
        for s in sets:
            if s["section"] == section and s["team"] in teams:
                xy[s["n"] - first, slot[s["person"]]] = s["xy"] + half_pitch
        rows = ball["n"] - first
        track = np.full((n, 2), np.nan, dtype=np.float32)
        track[rows] = ball["xy"] + half_pitch
        labels = np.full(n, D, dtype=np.int8)             # frames without a ball record count as dead
        alive = ball["status"] == 1
        labels[rows[alive]] = np.where(ball["poss"][alive] == 1, H, A)
        dead = labels == D
        clips[tag] = Clip(name=f"sp_{match}_{tag}", fps=FPS, pitch=pitch, xy=xy, team=team,
                          ball=np.where(dead[:, None], np.nan, track).astype(np.float32),
                          labels=controlled(labels),
                          extra={"labels_raw": labels,
                                 "ball_dead": np.where(dead[:, None], track, np.nan).astype(np.float32)})
    return clips


def controlled(labels: np.ndarray, fps: float = FPS, keep_s: float = KEEP_S) -> np.ndarray:
    """Give a team spell shorter than `keep_s` that interrupts the other team (X Y X, no stoppage
    on either side) back to that team: a touch without control does not change possession."""
    labels = labels.copy()
    while True:
        change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(labels)]))
        codes, length = labels[bounds[:-1]], np.diff(bounds)
        hit = [i for i in range(1, len(codes) - 1)
               if codes[i] != D and codes[i - 1] == codes[i + 1] != D and length[i] < keep_s * fps]
        if not hit:
            return labels
        i = min(hit, key=lambda j: length[j])          # shortest first, then look again
        labels[bounds[i]:bounds[i + 1]] = codes[i - 1]


def cached(match: str, tag: str, cache: Path = CACHE) -> Clip:
    return Clip.load(cache / f"{match}_{tag}.npz")


def available(cache: Path = CACHE) -> list[str]:
    return [m for m in MATCHES if (cache / f"{m}_2nd.npz").exists()]


def describe(clip: Clip) -> str:
    lab = clip.labels
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(lab)]))
    codes, dur = lab[bounds[:-1]], np.diff(bounds) / clip.fps
    poss = dur[(codes == H) | (codes == A)]
    share = {n: round(float((lab == c).mean()), 3) for n, c in (("H", H), ("A", A), ("D", D))}
    # Does the provider's possession agree with the team of the player nearest to the ball?
    seen = np.isfinite(clip.ball[:, 0])
    dist = np.linalg.norm(clip.xy[seen] - clip.ball[seen][:, None], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    near = clip.team[np.argmin(dist, axis=1)]
    agree = float((near == (lab[seen] == A)).mean())
    on_pitch = np.isfinite(clip.xy[..., 0]).sum(axis=1)
    return (f"{clip.name} {clip.n_frames} frames, labels {share}, {len(poss)} possessions, "
            f"mean {poss.mean():.1f}s median {np.median(poss):.1f}s, nearest-team agreement {agree:.3f}, "
            f"players per frame {np.median(on_pitch):.0f}")


def main() -> None:
    for match in MATCHES:
        try:
            clips = load_match(match)
        except FileNotFoundError as error:
            print(f"skip {match}: {error}")
            continue
        for tag, clip in clips.items():
            clip.save(CACHE / f"{match}_{tag}.npz")
            print(describe(clip), flush=True)


if __name__ == "__main__":
    main()
