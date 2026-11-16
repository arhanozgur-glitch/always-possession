"""Learned per-frame scores (LightGBM) + Viterbi decoding.

The model sees only the features of features.py and is trained on external matches only
(SoccerTrack v2, Metrica sample data), each degraded under several scenarios and also with the
two teams swapped, so it cannot learn anything about a particular team or side.

    python -m possession.model train --out models/gbm.txt        # train on all external data
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import features, metrica, soccertrack, sportec
from .data import Clip, A, D, H, X
from .decode import viterbi
from .degrade import degrade

FEATURE_CACHE = Path("data/cache/features")
TRAIN_SCENARIOS = ["mixed", "deadseen", "deadbusy", "heavy", "light", "crowd", "noisy"]
STEP = 5  # train on every 5th frame; neighbouring frames are near-duplicates
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.08, num_leaves=63,
              min_data_in_leaf=300, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=5.0, max_bin=127, verbose=-1, num_threads=6, seed=0)
ROUNDS = 300
BOUNDARY_WEIGHT = 3.0    # extra weight of training rows right at a label change (0 = off)
BOUNDARY_TAU_S = 1.0     # ... fading with this time constant


@dataclass(frozen=True)
class Decode:
    cost_team: float = 12.0
    cost_dead: float = 12.0
    temper: float = 1.0      # multiplies the log-probabilities before decoding


def sources() -> dict[str, tuple]:
    """{clip key: (loader, match, half)} for every cached external half."""
    out = {}
    for m in soccertrack.available():
        for t in ("1st", "2nd"):
            out[f"st_{m}_{t}"] = (soccertrack.cached, m, t)
    for g in metrica.available():
        for t in ("1st", "2nd"):
            out[f"mt_{g}_{t}"] = (metrica.cached, g, t)
    for m in sportec.available():
        for t in ("1st", "2nd"):
            out[f"sp_{m}_{t}"] = (sportec.cached, m, t)
    return out


def match_of(key: str) -> str:
    return key.rsplit("_", 1)[0]


def flip_labels(labels: np.ndarray) -> np.ndarray:
    out = labels.copy()
    out[labels == H], out[labels == A] = A, H
    return out


def training_rows(key: str, scenario: str, flip: bool, cache: Path = FEATURE_CACHE):
    """(features [n, F], labels [n]) for one external half, subsampled, cached on disk."""
    path = cache / features.VERSION / f"{key}@{scenario}{'@flip' if flip else ''}.npz"
    if path.exists():
        z = np.load(path)
        return z["x"], z["y"]
    loader, match, tag = sources()[key]
    clip = loader(match, tag)
    noisy = degrade(clip, scenario)
    team = features.flipped_team(noisy.team) if flip else noisy.team
    table, _ = features.compute(noisy, team)
    labels = flip_labels(clip.labels) if flip else clip.labels
    offset = (hash((key, scenario)) % STEP + STEP) % STEP if False else 0
    x, y = table[offset::STEP], labels[offset::STEP]
    keep = y != X
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x[keep], y=y[keep])
    return x[keep], y[keep]


def near_change(y: np.ndarray, tau_s: float = BOUNDARY_TAU_S) -> np.ndarray:
    """exp(-distance / tau) to the nearest label change, per training row of one half."""
    change = np.flatnonzero(y[1:] != y[:-1])
    if len(change) == 0:
        return np.zeros(len(y))
    idx = np.arange(len(y))
    pos = np.searchsorted(change, idx)
    left = np.where(pos > 0, idx - change[np.maximum(pos - 1, 0)] - 1, len(y))            # change before
    right = np.where(pos < len(change), change[np.minimum(pos, len(change) - 1)] - idx, len(y))
    return np.exp(-np.minimum(left, right) * STEP / 25.0 / tau_s)


def build(keys: list[str], scenarios: list[str] = TRAIN_SCENARIOS):
    xs, ys, ws = [], [], []
    for key in keys:
        for i, scenario in enumerate(scenarios):
            x, y = training_rows(key, scenario, flip=(i + sum(map(ord, key))) % 2 == 1)
            xs.append(x)
            ys.append(y)
            ws.append(1.0 + BOUNDARY_WEIGHT * near_change(y))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def fit(keys: list[str], scenarios: list[str] = TRAIN_SCENARIOS, rounds: int = ROUNDS) -> lgb.Booster:
    x, y, near = build(keys, scenarios)
    counts = np.bincount(y, minlength=3).astype(np.float64)
    weight = (counts.sum() / (3 * counts))[y] * near      # the three classes count equally
    return lgb.train(PARAMS, lgb.Dataset(x, y, weight=weight, feature_name=feature_names(), free_raw_data=False), rounds)


def feature_names() -> list[str]:
    key = next(iter(sources()))
    loader, match, tag = sources()[key]
    clip = loader(match, tag)
    small = Clip(clip.name, clip.fps, clip.pitch, clip.xy[:500], clip.team, clip.ball[:500])
    return features.compute(small)[1]


def log_scores(model: lgb.Booster, clip: Clip) -> np.ndarray:
    table, _ = features.compute(clip)
    return np.log(np.clip(model.predict(table), 1e-6, 1.0))


def predict(model: lgb.Booster, clip: Clip, d: Decode = Decode()) -> np.ndarray:
    return viterbi(d.temper * log_scores(model, clip), d.cost_team, d.cost_dead)


def load(path: Path) -> tuple[lgb.Booster, Decode]:
    meta = json.loads(Path(str(path) + ".json").read_text())
    return lgb.Booster(model_file=str(path)), Decode(**meta["decode"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the per-frame model on all external data.")
    ap.add_argument("command", choices=["train"])
    ap.add_argument("--out", type=Path, default=Path("models/gbm.txt"))
    ap.add_argument("--cost-team", type=float, default=Decode.cost_team)
    ap.add_argument("--cost-dead", type=float, default=Decode.cost_dead)
    args = ap.parse_args()
    keys = list(sources())
    model = fit(keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.out))
    _, names = features.compute(sources()[keys[0]][0](*sources()[keys[0]][1:]))
    Path(str(args.out) + ".json").write_text(json.dumps({
        "decode": {"cost_team": args.cost_team, "cost_dead": args.cost_dead, "temper": 1.0},
        "features": names, "trained_on": keys, "scenarios": TRAIN_SCENARIOS, "params": PARAMS,
        "rounds": ROUNDS}, indent=1))
    print(f"trained on {len(keys)} halves -> {args.out}")


if __name__ == "__main__":
    main()
