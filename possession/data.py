"""The one in-memory shape every loader produces and every algorithm consumes.

A `Clip` is a continuous stretch of tracking at a fixed frame rate (one half of a match, or a
whole competition video). Coordinates are metres, origin at a pitch corner, x along the length.
Missing observations are NaN. Track slots need not be persistent players: the competition data
has tracklet slots, SoccerTrack has player ids. Algorithms may only rely on `team`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Label codes, identical to the organizer scorer (possession_metrics.py).
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
HOME, AWAY = 0, 1


@dataclass
class Clip:
    name: str
    fps: float
    pitch: tuple[float, float]            # (length, width) in metres
    xy: np.ndarray                        # float32 [T, S, 2], NaN = not tracked
    team: np.ndarray                      # int8 [S], HOME or AWAY
    ball: np.ndarray                      # float32 [T, 2], NaN = not tracked
    labels: np.ndarray | None = None      # int8 [T] of H/A/D/X, ground truth if known
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.xy.shape[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(xy=self.xy, team=self.team, ball=self.ball,
                      fps=np.float32(self.fps), pitch=np.array(self.pitch, np.float32),
                      name=np.array(self.name))
        if self.labels is not None:
            arrays["labels"] = self.labels
        for key, value in self.extra.items():
            arrays["extra_" + key] = value
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Clip":
        z = np.load(path, allow_pickle=False)
        extra = {k[6:]: z[k] for k in z.files if k.startswith("extra_")}
        return cls(name=str(z["name"]), fps=float(z["fps"]), pitch=tuple(z["pitch"].tolist()),
                   xy=z["xy"], team=z["team"], ball=z["ball"],
                   labels=z["labels"] if "labels" in z.files else None, extra=extra)
