"""Rule-based possession: hand-written scores + Viterbi. No training; the fallback system.

  team   The closer team to the (gap-filled) ball gets the evidence, in proportion to how much
         closer it is. While the ball is unseen the evidence fades with the length of the gap,
         so the decoder simply keeps the last owner -- the competition's own rule for a loose
         ball.
  dead   A stoppage is a long stretch without the ball during which the players slow down, or
         a ball outside the pitch. A ball that is seen and moving is evidence of live play.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import features
from .data import Clip
from .decode import viterbi


@dataclass(frozen=True)
class RuleParams:
    team_gain: float = 0.35      # log-odds per metre of distance advantage
    gap_fade_s: float = 2.0      # ball-less evidence halves... fades with this time scale
    dead_gap_s: float = 4.0      # a ball-less stretch this long starts to look like a stoppage
    dead_gain: float = 1.2
    slow_gain: float = 2.0       # players slowing down adds to the stoppage evidence
    seen_alive: float = 1.0      # a seen ball inside the pitch speaks against a stoppage
    cost_team: float = 12.0
    cost_dead: float = 14.0


def scores(clip: Clip, p: RuleParams = RuleParams()) -> np.ndarray:
    table, names = features.compute(clip)
    col = {n: table[:, i] for i, n in enumerate(names)}
    fps = clip.fps
    gap_s = np.expm1(col["gap"]) / fps
    fade = np.exp(-gap_s / p.gap_fade_s)
    team = p.team_gain * col["d_diff_r12"] * fade * (col["d_near"] < 6)

    slow = col["p_slow_r50"] - 0.35                      # share of players under 1 m/s
    dead = (p.dead_gain * np.clip(gap_s / p.dead_gap_s - 1.0, -1.0, 3.0)
            + p.slow_gain * np.clip(slow, -0.3, 0.5) * (gap_s > 1.0)
            - p.seen_alive * col["seen"]
            + 3.0 * (col["seen"] * (col["edge"] < -0.5)))
    out = np.zeros((clip.n_frames, 3))
    out[:, 0] = team / 2
    out[:, 1] = -team / 2
    out[:, 2] = dead
    return out


def predict(clip: Clip, p: RuleParams = RuleParams()) -> np.ndarray:
    return viterbi(scores(clip, p), p.cost_team, p.cost_dead)
