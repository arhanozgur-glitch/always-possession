"""The organizer's reference baseline, re-expressed on a Clip (same logic as starter/baseline.py).

Nearest tracked player to the ball gives the team; no ball -> D; then a centred majority vote
over +/- `window` frames (ties to home, D where no team was seen in the window).
"""
from __future__ import annotations

import numpy as np

from .data import AWAY, HOME, Clip, A, D, H

NOBODY = -1


def nearest_team(clip: Clip) -> np.ndarray:
    """Team of the tracked player closest to the ball per frame, NOBODY without a ball."""
    dist = np.hypot(clip.xy[:, :, 0] - clip.ball[:, None, 0], clip.xy[:, :, 1] - clip.ball[:, None, 1])
    has = np.isfinite(dist).any(axis=1)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    out = np.full(clip.n_frames, NOBODY, dtype=np.int8)
    out[has] = clip.team[dist[has].argmin(axis=1)]
    return out


def smooth(codes: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.where(codes == HOME, H, np.where(codes == AWAY, A, D)).astype(np.int8)
    home_c = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_c = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - window), np.minimum(len(codes), index + window + 1)
    home, away = home_c[high] - home_c[low], away_c[high] - away_c[low]
    labels = np.full(len(codes), D, dtype=np.int8)
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], H, A)
    return labels


def predict(clip: Clip, window: int = 50) -> np.ndarray:
    return smooth(nearest_team(clip), window)
