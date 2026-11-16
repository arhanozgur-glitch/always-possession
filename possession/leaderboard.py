"""Local leaderboard on external data, scored with the organizer's own metric code.

    python -m possession.leaderboard                       # all methods, default scenarios
    python -m possession.leaderboard --methods smooth50 --scenarios mixed air

A method is a function Clip -> int8 label array (H/A/D/X codes). It is run on every half of
every external match under each degradation scenario; the two halves of a match are scored
together (as the competition scores its two halves) and results are averaged over matches.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import baseline, metrica, soccertrack, sportec
from .data import Clip, X
from .degrade import SCENARIOS, degrade

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "organizer_code" / "evaluation"))
import possession_metrics as pm  # noqa: E402  (organizer scorer, unmodified)

from .methods import METHODS, Method  # noqa: E402

DEFAULT_SCENARIOS = ["mixed", "random", "air", "crowd", "deadblind", "deadseen", "short", "long", "heavy", "light"]


def score_match(gts: list[np.ndarray], preds: list[np.ndarray]) -> dict:
    """Organizer metrics for one match: halves joined with an ungraded gap, as in the competition."""
    gap = np.full(250, X, dtype=np.int8)
    gt = np.concatenate([part for g in gts for part in (g, gap)])
    pred = np.concatenate([part for p in preds for part in (p, gap)])
    return pm.evaluate(gt.astype(np.int8), pred.astype(np.int8))


def run(methods: dict[str, Method], scenario: str, matches: list[str], clips: dict | None = None) -> dict:
    """{method: {match: organizer result dict}} under one scenario (each clip degraded once)."""
    out = {name: {} for name in methods}
    for match in matches:
        gts, preds = [], {name: [] for name in methods}
        for tag in ("1st", "2nd"):
            clip = clips[match, tag] if clips else soccertrack.cached(match, tag)
            noisy = clip if scenario == "clean" else degrade(clip, scenario)
            gts.append(clip.labels)
            for name, method in methods.items():
                preds[name].append(method(noisy))
        for name in methods:
            out[name][match] = score_match(gts, preds[name])
    return out


def summarize(results: dict) -> dict:
    keys = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_gt_alive")
    row = {k: float(np.mean([r[k] for r in results.values()])) for k in keys}
    row["spell_min"] = float(min(r["spell_f1_mean"] for r in results.values()))
    row["n_pred/n_gt"] = float(np.mean([r["n_pred_spells"] / max(1, r["n_gt_spells"]) for r in results.values()]))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Local leaderboard on external data.")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--matches", nargs="+", default=None)
    ap.add_argument("--data", nargs="+", default=["st", "mt"], choices=["st", "mt", "sp"])
    args = ap.parse_args()

    names = args.methods or list(METHODS)
    for source in args.data:
        matches, clips = load_source(source, args.matches)
        print(f"== {SOURCES[source]}: {len(matches)} matches ==")
        table(names, args.scenarios, matches, clips)


SOURCES = {"st": "SoccerTrack v2", "mt": "Metrica sample", "sp": "DFL / Sportec open data"}


def load_source(source: str, only: list[str] | None = None):
    module = {"st": soccertrack, "mt": metrica, "sp": sportec}[source]
    matches = [m for m in module.available() if not only or m in only]
    return matches, {(m, t): module.cached(m, t) for m in matches for t in ("1st", "2nd")}


def table(names: list[str], scenarios: list[str], matches: list[str], clips: dict) -> dict:
    header = f"{'method':<22}{'scenario':<9}{'spellF1':>8}{'(min)':>7}{'macroF1':>9}{'aliveF1':>9}{'teamAcc':>9}{'spells x':>9}"
    t0 = time.time()
    rows = {name: [] for name in names}
    for scenario in scenarios:
        results = run({n: METHODS[n] for n in names}, scenario, matches, clips)
        for name in names:
            rows[name].append(summarize(results[name]))
        print(f"  scored {scenario} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    print(header)
    args = argparse.Namespace(scenarios=scenarios)
    for name in names:
        for scenario, row in zip(args.scenarios, rows[name]):
            print(f"{name:<22}{scenario:<10}{row['spell_f1_mean']:>7.4f}{row['spell_min']:>7.3f}{row['macro_f1']:>9.4f}"
                  f"{row['alive_dead_f1']:>9.4f}{row['team_acc_gt_alive']:>9.4f}{row['n_pred/n_gt']:>9.2f}")
        if len(rows[name]) > 1:
            mean = {k: float(np.mean([r[k] for r in rows[name]])) for k in rows[name][0]}
            worst = min(r["spell_f1_mean"] for r in rows[name])
            print(f"{name:<22}{'MEAN':<10}{mean['spell_f1_mean']:>7.4f}{worst:>7.3f}{mean['macro_f1']:>9.4f}"
                  f"{mean['alive_dead_f1']:>9.4f}{mean['team_acc_gt_alive']:>9.4f}{mean['n_pred/n_gt']:>9.2f}")
        print(flush=True)
    return rows


if __name__ == "__main__":
    main()
