"""Second stage: a model that sees the first stage's probabilities around each frame.

The first model scores every frame from the tracking features. The second one gets the same
features plus the first model's H/A/D probabilities at several offsets before and after the
frame and their averages over short and long windows, so it can learn how a real hand-over or
stoppage looks in the first model's output (and which blips are not one).

To keep the second stage honest, the probabilities it is trained on are out-of-sample: the
training matches are split in two groups, and each group is scored by a first-stage model
trained on the other group.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import features
from . import model as M
from .data import Clip
from .decode import viterbi

OFFSETS = (1, 2, 5, 10, 20)     # in units of model.STEP frames (0.2 s)
WINDOWS = (5, 15, 41)
TREND = 10


def _shift(p: np.ndarray, k: int) -> np.ndarray:
    """Row t holds p[t + k]; the ends repeat the edge value."""
    idx = np.clip(np.arange(len(p)) + k, 0, len(p) - 1)
    return p[idx]


def context(p: np.ndarray, unit: int) -> np.ndarray:
    """Context columns from probabilities p [n, 3] whose rows are `1/unit` STEPs apart."""
    cols = [p]
    for k in OFFSETS:
        cols += [_shift(p, -k * unit), _shift(p, k * unit)]
    for w in WINDOWS:
        cols.append(uniform_filter1d(p, w * unit, axis=0, mode="nearest"))
    half = TREND * unit
    mean = uniform_filter1d(p, half, axis=0, mode="nearest")
    cols.append(_shift(mean, half // 2 + 1) - _shift(mean, -(half // 2 + 1)))   # after minus before
    return np.concatenate(cols, axis=1).astype(np.float32)


def context_names() -> list[str]:
    names = [f"p0_{c}" for c in "had"]
    for k in OFFSETS:
        names += [f"p-{k}_{c}" for c in "had"] + [f"p+{k}_{c}" for c in "had"]
    for w in WINDOWS:
        names += [f"pm{w}_{c}" for c in "had"]
    return names + [f"ptrend_{c}" for c in "had"]


def files(keys: list[str], scenarios: list[str]):
    for key in keys:
        for i, scenario in enumerate(scenarios):
            yield key, scenario, (i + sum(map(ord, key))) % 2 == 1


def fit(keys: list[str], scenarios: list[str] = M.TRAIN_SCENARIOS, rounds: int = M.ROUNDS):
    """(first-stage model on all keys, second-stage model)."""
    matches = sorted({M.match_of(k) for k in keys})
    groups = [set(matches[0::2]), set(matches[1::2])]
    xs, ys = [], []
    for held in groups:
        inner = M.fit([k for k in keys if M.match_of(k) not in held], scenarios, rounds)
        for key, scenario, flip in files([k for k in keys if M.match_of(k) in held], scenarios):
            x, y = M.training_rows(key, scenario, flip)
            xs.append(np.concatenate([x, context(inner.predict(x), 1)], axis=1))
            ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    del xs
    counts = np.bincount(y, minlength=3).astype(np.float64)
    weight = (counts.sum() / (3 * counts))[y]
    names = M.feature_names() + context_names()
    second = lgb.train(M.PARAMS, lgb.Dataset(x, y, weight=weight, feature_name=names), rounds)
    return M.fit(keys, scenarios, rounds), second


def log_scores(first: lgb.Booster, second: lgb.Booster, clip: Clip) -> np.ndarray:
    table, _ = features.compute(clip)
    x = np.concatenate([table, context(first.predict(table), M.STEP)], axis=1)
    return np.log(np.clip(second.predict(x), 1e-6, 1.0))


# ---- the shipped system: models/gbm.txt (first stage), models/gbm2.txt (second), models/gbm.txt.json

def second_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + "2" + path.suffix)


def load(path) -> tuple[lgb.Booster, lgb.Booster, M.Decode]:
    meta = json.loads(Path(str(path) + ".json").read_text())
    return (lgb.Booster(model_file=str(path)), lgb.Booster(model_file=str(second_path(path))),
            M.Decode(**meta["decode"]))


def predict(first: lgb.Booster, second: lgb.Booster, clip: Clip, d: M.Decode) -> np.ndarray:
    return viterbi(d.temper * log_scores(first, second, clip), d.cost_team, d.cost_dead)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the shipped two-stage model on all external data.")
    ap.add_argument("command", choices=["train"])
    ap.add_argument("--out", type=Path, default=Path("models/gbm.txt"))
    ap.add_argument("--cost-team", type=float, default=10.0)
    ap.add_argument("--cost-dead", type=float, default=40.0)
    args = ap.parse_args()
    keys = list(M.sources())
    first, second = fit(keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    first.save_model(str(args.out))
    second.save_model(str(second_path(args.out)))
    Path(str(args.out) + ".json").write_text(json.dumps({
        "decode": {"cost_team": args.cost_team, "cost_dead": args.cost_dead, "temper": 1.0},
        "features": first.feature_name(), "second_stage_features": second.feature_name(),
        "trained_on": keys, "scenarios": M.TRAIN_SCENARIOS, "params": M.PARAMS, "rounds": M.ROUNDS,
        "boundary_weight": M.BOUNDARY_WEIGHT, "boundary_tau_s": M.BOUNDARY_TAU_S}, indent=1))
    print(f"trained on {len(keys)} halves -> {args.out}, {second_path(args.out)}")


if __name__ == "__main__":
    main()
