"""Make clean external tracking look like automatic (computer-vision) tracking.

The competition tracking is raw CV output: the ball is missing on ~41% of graded frames, in
stretches, and "disproportionately on the frames that matter" (occlusions, crowded boxes, balls
in the air); about 20.5 of 22 players are tracked on average. The external data is much cleaner,
so before any algorithm sees it we remove observations the same way.

How exactly the competition ball goes missing is not known to us (we do not inspect that data),
so there is not one degradation but several *scenarios* that weight the plausible causes
differently. An algorithm is only trusted if it holds up across all of them.

External clips carry no ball at all while the ball is dead. A real CV tracker does see the ball
during part of a stoppage (typically once it is placed for the restart), so a scenario can put
a still ball at the restart spot for the last seconds of a stoppage.

Everything is deterministic given (clip name, scenario name).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .data import Clip, D


@dataclass(frozen=True)
class Scenario:
    name: str
    alive_missing: float = 0.30     # share of visible in-play ball frames removed
    burst_s: float = 1.5            # time scale of the in-play ball dropouts, seconds
    w_speed: float = 1.0            # fast ball (in the air) -> more likely missing
    w_crowd: float = 1.0            # many players around the ball -> more likely missing
    dead_seen_p: float = 0.6        # chance the placed ball is seen before a restart
    dead_seen_s: float = 4.0        # mean seconds it is seen for
    dead_carry_p: float = 0.25      # chance the ball is seen being carried during a stoppage
    dead_carry_s: float = 4.0       # mean seconds it is seen for
    player_missing: float = 0.07    # share of player observations dropped (20.5 of 22 seen)
    player_burst_s: float = 4.0
    jitter_m: float = 0.15          # white position noise on players and ball
    ball_outlier: float = 0.0       # share of visible-ball frames replaced by a false detection


SCENARIOS = {s.name: s for s in (
    Scenario("mixed"),
    Scenario("random", w_speed=0, w_crowd=0),
    Scenario("air", w_speed=2.5, w_crowd=0.5),
    Scenario("crowd", w_speed=0.3, w_crowd=2.5),
    Scenario("deadblind", dead_seen_p=0.0, dead_carry_p=0.0),
    Scenario("deadseen", dead_seen_p=0.95, dead_seen_s=15.0),
    Scenario("deadbusy", dead_seen_p=0.8, dead_seen_s=6.0, dead_carry_p=0.8, dead_carry_s=8.0),
    Scenario("short", burst_s=0.5),
    Scenario("long", burst_s=4.0),
    Scenario("heavy", alive_missing=0.50, player_missing=0.12, ball_outlier=0.01),
    Scenario("light", alive_missing=0.12, player_missing=0.03),
    Scenario("noisy", jitter_m=0.4),
    # Stress tests, never used for training unless stated in model.TRAIN_SCENARIOS.
    Scenario("s_noisy", jitter_m=0.6),
    Scenario("s_outlier", ball_outlier=0.06),
    Scenario("s_sparse", player_missing=0.25, player_burst_s=8.0),
    Scenario("s_blind", alive_missing=0.65, burst_s=3.0),
)}


def _rng(clip: Clip, scenario: Scenario, salt: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{clip.name}|{scenario.name}|{salt}".encode()))


def _smooth_noise(rng: np.random.Generator, shape, sigma_frames: float) -> np.ndarray:
    """Unit-variance Gaussian noise, correlated along axis 0 with the given time scale."""
    noise = rng.standard_normal(shape).astype(np.float32)
    if sigma_frames > 0.5:
        noise = gaussian_filter1d(noise, sigma_frames, axis=0, mode="wrap")
        noise /= noise.std() + 1e-9
    return noise


def _z(x: np.ndarray, where: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    ref = x[where] if where.any() else x
    return (x - ref.mean()) / (ref.std() + 1e-9)


def place_dead_ball(clip: Clip, ball: np.ndarray, scenario: Scenario) -> None:
    """In place: show a still ball at the restart spot for the end of some stoppages."""
    if clip.labels is None:
        return
    rng = _rng(clip, scenario, "dead")
    dead = clip.labels == D
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        # A player walking the ball to the restart spot: the ball is seen moving with him.
        if rng.random() < scenario.dead_carry_p and stop - start > 2 * clip.fps:
            n = int(min(stop - start, max(clip.fps, rng.exponential(scenario.dead_carry_s) * clip.fps)))
            a = start + int(rng.integers(0, stop - start - n + 1))
            tracks = clip.xy[a:a + n]
            ok = np.flatnonzero(np.isfinite(tracks[:, :, 0]).all(axis=0))
            if len(ok):
                ball[a:a + n] = tracks[:, rng.choice(ok)] + rng.normal(0, 0.4, 2)
        if stop < len(ball) and np.isfinite(ball[stop, 0]) and rng.random() < scenario.dead_seen_p:
            n = int(min(stop - start, rng.exponential(scenario.dead_seen_s) * clip.fps))
            ball[stop - n:stop] = ball[stop]


def ball_dropout_mask(clip: Clip, ball: np.ndarray, scenario: Scenario) -> np.ndarray:
    """True where a visible ball observation is removed."""
    seen = np.isfinite(ball[:, 0])
    share = scenario.alive_missing

    filled = np.where(seen[:, None], ball, 0.0)
    smooth = gaussian_filter1d(filled, 3.0, axis=0)
    speed = np.hypot(*np.gradient(smooth, axis=0).T) * clip.fps
    dist = np.hypot(clip.xy[:, :, 0] - ball[:, None, 0], clip.xy[:, :, 1] - ball[:, None, 1])
    crowd = np.nansum(dist < 4.0, axis=1)

    score = scenario.w_speed * _z(np.log1p(speed), seen) + scenario.w_crowd * _z(crowd, seen)
    score = score + 1.5 * _smooth_noise(_rng(clip, scenario, "ball"), len(ball),
                                        scenario.burst_s * clip.fps)
    return seen & (score > np.quantile(score[seen], 1.0 - share))


def degrade(clip: Clip, scenario: Scenario | str = "mixed") -> Clip:
    """A copy of `clip` with CV-like missing data. Labels are untouched."""
    if isinstance(scenario, str):
        scenario = SCENARIOS[scenario]
    ball = clip.ball.copy()
    xy = clip.xy.copy()
    length, width = clip.pitch

    place_dead_ball(clip, ball, scenario)
    ball[ball_dropout_mask(clip, ball, scenario)] = np.nan

    rng = _rng(clip, scenario, "misc")
    if scenario.jitter_m > 0:
        ball += rng.normal(0, scenario.jitter_m, ball.shape).astype(np.float32)
        xy += rng.normal(0, scenario.jitter_m, xy.shape).astype(np.float32)
    if scenario.ball_outlier > 0:
        bad = np.isfinite(ball[:, 0]) & (rng.random(len(ball)) < scenario.ball_outlier)
        ball[bad] = np.c_[rng.uniform(0, length, bad.sum()), rng.uniform(0, width, bad.sum())]

    if scenario.player_missing > 0:
        noise = _smooth_noise(_rng(clip, scenario, "players"), xy.shape[:2],
                              scenario.player_burst_s * clip.fps)
        seen = np.isfinite(xy[:, :, 0])
        if seen.any():
            xy[noise > np.quantile(noise[seen], 1.0 - scenario.player_missing)] = np.nan

    return replace(clip, name=f"{clip.name}@{scenario.name}", xy=xy, ball=ball)
