"""Honest test of the learned model: never score a match the model was trained on.

Three splits, each trained from scratch:
  st->mt   train on all SoccerTrack, test on Metrica sample      (different provider, level, ball)
  mt->st   train on Metrica sample, test on SoccerTrack
  folds    train on both sets minus a group of matches, test on that group

For each test half and scenario the per-frame log-scores are stored, then one grid of decoder
costs is scored on all of them, so the costs are chosen on out-of-sample predictions only.

    python -m experiments.cross --splits st->mt mt->st
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from possession import model as M  # noqa: E402
from possession.degrade import degrade  # noqa: E402
from possession.decode import viterbi  # noqa: E402
from possession.leaderboard import score_match  # noqa: E402

EVAL_SCENARIOS = ["mixed", "deadseen", "deadbusy", "heavy", "light", "air", "long", "random"]
from possession import features as F  # noqa: E402
OUT = Path("data/cache/cross") / F.VERSION
TAG = ""   # suffix of the cache files for runs with non-default training options
STACK = False
from possession import refine as R  # noqa: E402


def splits(which: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    every = list(M.sources())
    st = [k for k in every if k.startswith("st_")]
    mt = [k for k in every if k.startswith("mt_")]
    sp = [k for k in every if k.startswith("sp_")]
    keys = st + mt
    out = {}
    if "st->mt" in which:
        out["st->mt"] = (st, mt)
    if "mt->st" in which:
        out["mt->st"] = (mt, st)
    if "base->sp" in which:          # the two original sets -> DFL (a third, unseen provider)
        out["base->sp"] = (keys, sp)
    if "sp->base" in which:
        out["sp->st"] = (sp, st)
        out["sp->mt"] = (sp, mt)
    matches = sorted({M.match_of(k) for k in keys})
    for i in range(4):
        held = set(matches[i::4])
        train = [k for k in keys if M.match_of(k) not in held]
        test = [k for k in keys if M.match_of(k) in held]
        if "folds" in which:
            out[f"fold{i}"] = (train, test)
        if "folds+sp" in which:      # same held-out matches, DFL added to the training side
            out[f"fold{i}+sp"] = (train + sp, test)
    if "spfolds" in which:           # DFL matches held out in turn, everything else in training
        sp_matches = sorted({M.match_of(k) for k in sp})
        for i in range(3):
            held = set(sp_matches[i::3])
            out[f"spfold{i}"] = (keys + [k for k in sp if M.match_of(k) not in held],
                                 [k for k in sp if M.match_of(k) in held])
    return out


def collect(name: str, train: list[str], test: list[str], scenarios: list[str]) -> dict:
    path = OUT / f"{name.replace('>', '')}{TAG}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    t0 = time.time()
    if STACK:
        first, model = R.fit(train)
        score = lambda clip: R.log_scores(first, model, clip)  # noqa: E731
    else:
        model = M.fit(train)
        score = lambda clip: M.log_scores(model, clip)  # noqa: E731
    print(f"[{name}] trained on {len(train)} halves in {time.time() - t0:.0f}s", flush=True)
    src = M.sources()
    out = {}
    for key in test:
        loader, match, tag = src[key]
        clip = loader(match, tag)
        for scenario in scenarios:
            out[key, scenario] = (score(degrade(clip, scenario)).astype(np.float32), clip.labels)
        print(f"[{name}] scored {key} ({time.time() - t0:.0f}s)", flush=True)
    imp = dict(zip(model.feature_name(), model.feature_importance("gain")))
    out["__importance__"] = imp
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def evaluate(scores: dict, scenarios: list[str], cost_team: float, cost_dead: float, temper: float = 1.0):
    """{scenario: (mean spellF1, mean macroF1)} over the test matches of one split."""
    keys = sorted({k[0] for k in scores if isinstance(k, tuple)})
    matches = sorted({M.match_of(k) for k in keys})
    out = {}
    for scenario in scenarios:
        spell, macro = [], []
        for match in matches:
            halves = [k for k in keys if M.match_of(k) == match]
            gts = [scores[k, scenario][1] for k in halves]
            preds = [viterbi(temper * scores[k, scenario][0], cost_team, cost_dead) for k in halves]
            r = score_match(gts, preds)
            spell.append(r["spell_f1_mean"])
            macro.append(r["macro_f1"])
        out[scenario] = (float(np.mean(spell)), float(np.mean(macro)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["st->mt", "mt->st"])
    ap.add_argument("--scenarios", nargs="+", default=EVAL_SCENARIOS)
    ap.add_argument("--team", nargs="+", type=float, default=[6, 10, 15, 22, 30])
    ap.add_argument("--dead", nargs="+", type=float, default=[6, 10, 15, 22, 30])
    ap.add_argument("--boundary-weight", type=float, default=M.BOUNDARY_WEIGHT, help="model.BOUNDARY_WEIGHT for this run")
    ap.add_argument("--boundary-tau", type=float, default=M.BOUNDARY_TAU_S)
    ap.add_argument("--stack", action="store_true", help="two-stage model (possession/refine.py)")
    args = ap.parse_args()
    global TAG, STACK
    STACK = args.stack
    M.BOUNDARY_WEIGHT, M.BOUNDARY_TAU_S = args.boundary_weight, args.boundary_tau
    if args.boundary_weight:
        TAG = f"@bw{args.boundary_weight:g}t{args.boundary_tau:g}"
    TAG += "@stack" if STACK else ""
    all_scores ={name: collect(name, tr, te, args.scenarios) for name, (tr, te) in splits(args.splits).items()}

    grid = {}
    for ct, cd in itertools.product(args.team, args.dead):
        grid[ct, cd] = {name: evaluate(s, args.scenarios, ct, cd) for name, s in all_scores.items()}
    def overall(res):  # mean spellF1 over splits and scenarios
        return float(np.mean([v[0] for per in res.values() for v in per.values()]))
    ranked = sorted(grid, key=lambda k: -overall(grid[k]))
    print("\ncost_team cost_dead  mean spellF1   per split")
    for ct, cd in ranked[:8]:
        per = "  ".join(f"{n}:{np.mean([v[0] for v in r.values()]):.4f}" for n, r in grid[ct, cd].items())
        print(f"{ct:>9.0f} {cd:>9.0f}  {overall(grid[ct, cd]):.4f}        {per}")
    best = ranked[0]
    print(f"\nbest costs {best}; spellF1 / macroF1 per scenario:")
    for name, res in grid[best].items():
        print(f"  {name:<8}" + "  ".join(f"{s}:{v[0]:.3f}/{v[1]:.3f}" for s, v in res.items()))
    for name, s in all_scores.items():
        imp = sorted(s["__importance__"].items(), key=lambda kv: -kv[1])[:12]
        total = sum(s["__importance__"].values())
        print(f"  top features [{name}]: " + ", ".join(f"{k} {v / total:.0%}" for k, v in imp))


if __name__ == "__main__":
    main()
