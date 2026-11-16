"""Turn per-frame class scores into a consistent H/A/D sequence.

A three-state hidden Markov model decoded with Viterbi: per-frame log-scores for H, A, D plus a
fixed price for every change of state. The price is what stops the label flickering; one price
for a change of team while the ball is alive, another for entering or leaving a stoppage.
"""
from __future__ import annotations

import numpy as np

from .data import A, D, H


def viterbi(log_scores: np.ndarray, cost_team: float, cost_dead: float) -> np.ndarray:
    """log_scores [T, 3] in the order H, A, D. Returns int8 labels [T]."""
    T = len(log_scores)
    cost = np.array([[0.0, cost_team, cost_dead],
                     [cost_team, 0.0, cost_dead],
                     [cost_dead, cost_dead, 0.0]])
    score = log_scores[0].astype(np.float64).copy()
    back = np.zeros((T, 3), dtype=np.int8)
    ls = log_scores.astype(np.float64)
    for t in range(1, T):
        cand = score[:, None] - cost            # [from, to]
        best = cand.argmax(axis=0)
        back[t] = best
        score = cand[best, (0, 1, 2)] + ls[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


try:  # the loop above is fine for one clip but slow across a grid search; compile it if we can
    from numba import njit

    @njit(cache=True)
    def _viterbi_fast(ls, cost_team, cost_dead):
        T = ls.shape[0]
        cost = np.array([[0.0, cost_team, cost_dead],
                         [cost_team, 0.0, cost_dead],
                         [cost_dead, cost_dead, 0.0]])
        score = ls[0].copy()
        new = np.empty(3)
        back = np.zeros((T, 3), dtype=np.int8)
        for t in range(1, T):
            for j in range(3):
                best, arg = -1e300, 0
                for i in range(3):
                    v = score[i] - cost[i, j]
                    if v > best:
                        best, arg = v, i
                back[t, j] = arg
                new[j] = best + ls[t, j]
            for j in range(3):
                score[j] = new[j]
        path = np.empty(T, dtype=np.int8)
        path[T - 1] = np.argmax(score)
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def viterbi(log_scores: np.ndarray, cost_team: float, cost_dead: float) -> np.ndarray:  # noqa: F811
        return _viterbi_fast(np.ascontiguousarray(log_scores, dtype=np.float64), float(cost_team), float(cost_dead))
except ImportError:  # pragma: no cover
    pass

assert (H, A, D) == (0, 1, 2)
