"""Registry of methods for the leaderboard. Each is a function Clip -> label codes."""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import baseline, rules
from .data import Clip

Method = Callable[[Clip], np.ndarray]

METHODS: dict[str, Method] = {
    "baseline_raw": lambda c: baseline.predict(c, 0),
    "baseline_smooth25": lambda c: baseline.predict(c, 25),
    "baseline_smooth50": lambda c: baseline.predict(c, 50),
    "baseline_smooth125": lambda c: baseline.predict(c, 125),
    "rules": rules.predict,
}


def _final(clip: Clip) -> np.ndarray:
    """The shipped model (models/gbm.txt). In-sample on every match it was trained on."""
    from . import refine
    if "loaded" not in _final.__dict__:
        _final.loaded = refine.load("models/gbm.txt")
    return refine.predict(*_final.loaded[:2], clip, _final.loaded[2])


METHODS["gbm_final"] = _final
