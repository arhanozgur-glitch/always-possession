"""Per-frame features from a Clip: what the geometry of the players and the ball says.

Everything is computed from positions and team membership only (no player identities), so it
works on tracklet slots as well as on persistent players. Features are built so that swapping
the two teams mirrors them, and the whole table is float32 [T, F].

Three families:
  ball      where the ball is relative to each team, on a gap-filled ball track, plus how long
            the ball has been / will be unseen (the ball is missing in long stretches);
  players   how fast everybody moves, team shapes, and which way the block of 22 is drifting
            relative to the direction each team attacks;
  context   the same signals averaged over short and long windows, before and after the frame.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .data import AWAY, HOME, Clip

CAP_DIST = 30.0     # metres; distances beyond this carry no extra information
CAP_GAP = 500       # frames (20 s)
MAX_PLAYER_SPEED = 12.0  # m/s; faster "movement" is a tracklet identity switch
CONTROL_M = 2.0          # a player this close to the ball can be playing it
CONTROL_MARGIN_M = 0.5   # ... if no opponent is within this much of the same distance
VERSION = "v3"           # bump when the feature set changes (invalidates cached training rows)
# Automatic tracking jitters from frame to frame, and a frame-to-frame difference turns 0.3 m of
# jitter into 10 m/s of "speed". So positions are averaged over a short window and velocities are
# taken over a longer baseline than one frame.
SMOOTH_PLAYERS, SMOOTH_BALL = 7, 5     # frames averaged (centred)
BASE_PLAYERS, BASE_BALL = 6, 3         # velocity = (x[t + k] - x[t - k]) / 2k frames


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centred mean over n frames along axis 0 that ignores NaNs; NaN stays NaN."""
    ok = np.isfinite(x)
    total = uniform_filter1d(np.where(ok, x, 0.0).astype(np.float32), n, axis=0, mode="nearest")
    count = uniform_filter1d(ok.astype(np.float32), n, axis=0, mode="nearest")
    return np.where(ok, total / np.maximum(count, 1e-6), np.nan).astype(np.float32)


def _velocity(x: np.ndarray, k: int, fps: float) -> np.ndarray:
    """Per-second displacement over a baseline of 2k frames (shorter at the two ends)."""
    idx = np.arange(len(x))
    a, b = np.clip(idx - k, 0, len(x) - 1), np.clip(idx + k, 0, len(x) - 1)
    span = (b - a).reshape((-1,) + (1,) * (x.ndim - 1))
    return (x[b] - x[a]) / np.maximum(span, 1) * fps


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    count = np.isfinite(a).sum(axis=axis)
    return np.where(count > 0, np.nansum(a, axis=axis) / np.maximum(count, 1), np.nan)


def _fill(x: np.ndarray) -> np.ndarray:
    """Forward- then backward-fill NaNs along axis 0 (1-D or 2-D)."""
    x = x.copy()
    flat = x.reshape(len(x), -1)
    idx = np.arange(len(x))
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        if not ok.any():
            flat[:, j] = 0.0
            continue
        last = np.maximum.accumulate(np.where(ok, idx, -1))
        nxt = np.minimum.accumulate(np.where(ok, idx, len(x))[::-1])[::-1]
        src = np.where(last >= 0, last, nxt)
        flat[:, j] = flat[np.clip(src, 0, len(x) - 1), j]
    return x


def clean_ball(ball: np.ndarray, fps: float, max_speed: float = 45.0) -> np.ndarray:
    """Drop isolated false detections: points that need > max_speed to reach from both neighbours."""
    ball = ball.copy()
    seen = np.flatnonzero(np.isfinite(ball[:, 0]))
    if len(seen) < 3:
        return ball
    p, t = ball[seen], seen.astype(np.float64)
    step = np.hypot(*(p[1:] - p[:-1]).T) / np.maximum(t[1:] - t[:-1], 1) * fps
    bad = np.zeros(len(seen), dtype=bool)
    bad[1:-1] = (step[:-1] > max_speed) & (step[1:] > max_speed)
    ball[seen[bad]] = np.nan
    return ball


def gap_fill(ball: np.ndarray):
    """(ball linearly interpolated over gaps, frames since seen, frames until seen, seen flag)."""
    n = len(ball)
    seen = np.isfinite(ball[:, 0])
    idx = np.arange(n)
    if not seen.any():
        return np.zeros_like(ball), np.full(n, CAP_GAP), np.full(n, CAP_GAP), seen
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    since = np.where(last >= 0, idx - last, CAP_GAP)
    until = np.where(nxt < n, nxt - idx, CAP_GAP)
    a, b = np.clip(last, 0, n - 1), np.clip(nxt, 0, n - 1)
    a = np.where(last < 0, b, a)
    b = np.where(nxt >= n, a, b)
    w = np.where(b > a, (idx - a) / np.maximum(b - a, 1), 0.0)[:, None]
    filled = ball[a] * (1 - w) + ball[b] * w
    return filled.astype(np.float32), np.minimum(since, CAP_GAP), np.minimum(until, CAP_GAP), seen


def attack_direction(xy: np.ndarray, team: np.ndarray, fps: float) -> np.ndarray:
    """+1 where the home team attacks towards +x, -1 otherwise, per frame.

    The side a team defends is where its deepest player (the goalkeeper) stands. Compared over a
    two-minute window, so it follows the change of ends at half time on a full-match clip.
    """
    hx, ax = xy[:, team == HOME, 0], xy[:, team == AWAY, 0]
    with np.errstate(all="ignore"):
        lean = (np.nanmin(ax, 1) + np.nanmax(ax, 1)) - (np.nanmin(hx, 1) + np.nanmax(hx, 1))
    lean = _fill(np.where(np.isfinite(lean), lean, np.nan))
    lean = uniform_filter1d(lean, int(120 * fps), mode="nearest")
    return np.where(lean >= 0, 1.0, -1.0).astype(np.float32)


def _roll(x: np.ndarray, half: int) -> np.ndarray:
    return uniform_filter1d(x, 2 * half + 1, mode="nearest")


def _past(x: np.ndarray, n: int) -> np.ndarray:
    """Mean over the n frames before each frame."""
    return uniform_filter1d(x, n, origin=(n - 1) // 2, mode="nearest") if n > 1 else x


def _future(x: np.ndarray, n: int) -> np.ndarray:
    return uniform_filter1d(x, n, origin=-(n // 2), mode="nearest") if n > 1 else x


def compute(clip: Clip, team: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    team = clip.team if team is None else team
    fps = clip.fps
    length, width = clip.pitch
    xy = smooth(clip.xy, SMOOTH_PLAYERS)
    T = clip.n_frames
    cols: dict[str, np.ndarray] = {}

    # ---- ball -------------------------------------------------------------------------------
    ball, since, until, seen = gap_fill(clean_ball(clip.ball, fps))
    ball = smooth(ball, SMOOTH_BALL)
    gap = np.where(seen, 0, since + until)
    dist = np.hypot(xy[:, :, 0] - ball[:, None, 0], xy[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    home, away = team == HOME, team == AWAY
    dh = np.minimum(dist[:, home].min(1), CAP_DIST)
    da = np.minimum(dist[:, away].min(1), CAP_DIST)
    cols["seen"] = seen.astype(np.float32)
    cols["since"] = np.log1p(since)
    cols["until"] = np.log1p(until)
    cols["gap"] = np.log1p(gap)
    cols["d_near"] = np.minimum(dh, da)
    cols["d_diff"] = np.clip(da - dh, -8, 8)                      # > 0: home player is closer
    cols["d_ratio"] = (da - dh) / (da + dh + 0.5)
    cols["n3_diff"] = (dist[:, home] < 3).sum(1) - (dist[:, away] < 3).sum(1)
    cols["n3_all"] = (dist < 3).sum(1)
    cols["n8_all"] = (dist < 8).sum(1)
    velocity = _velocity(ball, BASE_BALL, fps)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    cols["ball_speed"] = np.minimum(speed, 40)
    edge = np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]])
    cols["edge"] = np.clip(edge, -5, 20)
    cols["edge_goal"] = np.clip(np.minimum(ball[:, 0], length - ball[:, 0]), -5, 52)
    cols["edge_side"] = np.clip(np.minimum(ball[:, 1], width - ball[:, 1]), -5, 34)

    # ---- control: who last had / will next have the ball at his feet -------------------------
    # A frame is a "control" frame for a team when the ball is trusted (seen, or a gap of at
    # most half a second) and that team has a player on it who is clearly closer than any
    # opponent. Runs of such frames are the visible part of a possession; the time since the
    # last one and until the next one, per team, is what tells a pass from a hand-over.
    trusted = gap <= 0.5 * fps
    ctrl_h = trusted & (dh < CONTROL_M) & (dh < da - CONTROL_MARGIN_M)
    ctrl_a = trusted & (da < CONTROL_M) & (da < dh - CONTROL_MARGIN_M)
    idx = np.arange(T)

    def since_until(mask):
        last = np.maximum.accumulate(np.where(mask, idx, -CAP_GAP))
        nxt = np.minimum.accumulate(np.where(mask, idx, T + CAP_GAP)[::-1])[::-1]
        return np.minimum(idx - last, CAP_GAP), np.minimum(nxt - idx, CAP_GAP)

    since_h, until_h = since_until(ctrl_h)
    since_a, until_a = since_until(ctrl_a)
    cols["c_since_h"], cols["c_since_a"] = np.log1p(since_h), np.log1p(since_a)
    cols["c_until_h"], cols["c_until_a"] = np.log1p(until_h), np.log1p(until_a)
    last_team = np.sign(since_a - since_h).astype(np.float32)        # +1: home touched last
    next_team = np.sign(until_a - until_h).astype(np.float32)
    cols["c_last"], cols["c_next"] = last_team, next_team
    cols["c_same"] = last_team * next_team                            # +1: ball between team-mates
    cols["c_span"] = np.log1p(np.minimum(since_h, since_a) + np.minimum(until_h, until_a))
    ctrl = ctrl_h.astype(np.float32) - ctrl_a.astype(np.float32)
    for n in (50, 150, 400):
        cols[f"c_past{n}"] = _past(ctrl, n)
        cols[f"c_fut{n}"] = _future(ctrl, n)
        cols[f"c_any{n}"] = _roll(np.abs(ctrl), n // 2)
    cols["c_turn150"] = cols["c_fut150"] - cols["c_past150"]         # the ball is changing sides

    # ---- players ----------------------------------------------------------------------------
    direction = attack_direction(xy, team, fps)
    step = _velocity(xy, BASE_PLAYERS, fps)
    pspeed = np.hypot(step[:, :, 0], step[:, :, 1])
    bad = ~(pspeed <= MAX_PLAYER_SPEED)
    pspeed = np.where(bad, np.nan, pspeed)
    vx = np.where(bad, np.nan, step[:, :, 0])
    n_seen = np.isfinite(xy[:, :, 0]).sum(1)
    cols["n_players"] = n_seen
    with np.errstate(all="ignore"):
        cols["p_speed_med"] = _fill(np.nanmedian(pspeed, axis=1))
        cols["p_slow"] = _fill(_nanmean((pspeed < 1.0).astype(np.float32) + 0 * pspeed, 1))
        cols["p_fast"] = _fill(_nanmean((pspeed > 4.0).astype(np.float32) + 0 * pspeed, 1))
        drift = _fill(_nanmean(vx, 1)) * direction                 # > 0: block moves to away goal
        cx_h, cx_a = _fill(_nanmean(xy[:, home, 0], 1)), _fill(_nanmean(xy[:, away, 0], 1))
        cy_h, cy_a = _fill(_nanmean(xy[:, home, 1], 1)), _fill(_nanmean(xy[:, away, 1], 1))
        sx_h, sx_a = _fill(np.nanstd(xy[:, home, 0], 1)), _fill(np.nanstd(xy[:, away, 0], 1))
        sy_h, sy_a = _fill(np.nanstd(xy[:, home, 1], 1)), _fill(np.nanstd(xy[:, away, 1], 1))
        sp_h, sp_a = _fill(_nanmean(pspeed[:, home], 1)), _fill(_nanmean(pspeed[:, away], 1))
    cols["drift"] = drift
    cols["speed_diff"] = sp_h - sp_a
    cols["block_x"] = ((cx_h + cx_a) / 2 - length / 2) * direction  # > 0: play is in away half
    cols["cx_diff"] = (cx_h - cx_a) * direction
    cols["width_diff"] = sy_h - sy_a                                # team with the ball is wider
    cols["depth_diff"] = sx_h - sx_a
    cols["width_all"] = (sy_h + sy_a) / 2
    cols["depth_all"] = (sx_h + sx_a) / 2
    cols["ball_x"] = (ball[:, 0] - length / 2) * direction
    cols["ball_vx"] = np.clip(velocity[:, 0], -30, 30) * direction
    cols["ball_to_block"] = cols["ball_x"] - cols["block_x"]
    cols["ball_side_h"] = np.abs(ball[:, 1] - cy_h) - np.abs(ball[:, 1] - cy_a)

    # ---- context ----------------------------------------------------------------------------
    for name in ("seen", "d_diff", "d_near", "ball_speed", "edge", "p_speed_med", "p_slow",
                 "drift", "n3_diff", "width_diff", "cx_diff", "speed_diff"):
        base = cols[name].astype(np.float32)
        for half in (12, 50, 150):
            cols[f"{name}_r{half}"] = _roll(base, half)
    for name in ("seen", "d_diff", "d_near", "p_speed_med", "drift", "ball_speed", "edge"):
        base = cols[name].astype(np.float32)
        for n in (25, 100):
            cols[f"{name}_p{n}"] = _past(base, n)
            cols[f"{name}_f{n}"] = _future(base, n)
    for name in ("d_diff", "edge", "ball_speed"):
        for lag in (12, 38):
            cols[f"{name}_lag{lag}"] = np.roll(cols[name], lag)
            cols[f"{name}_lead{lag}"] = np.roll(cols[name], -lag)

    names = list(cols)
    table = np.empty((T, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        table[:, j] = cols[name]
    return np.nan_to_num(table, nan=0.0, posinf=CAP_DIST, neginf=-CAP_DIST), names


def flipped_team(team: np.ndarray) -> np.ndarray:
    return (1 - team).astype(np.int8)
