"""Where does the decoded output still lose possessions? Uses the cached out-of-sample scores.

    python -m experiments.diagnose --team 30 --dead 30
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from possession import model as M  # noqa: E402
from possession.decode import viterbi  # noqa: E402
from possession.leaderboard import score_match  # noqa: E402

from possession import features as F  # noqa: E402
OUT = Path("data/cache/cross") / F.VERSION


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=float, default=30)
    ap.add_argument("--dead", type=float, default=30)
    ap.add_argument("--temper", type=float, default=1.0)
    ap.add_argument("--scenarios", nargs="+", default=["mixed", "heavy"])
    args = ap.parse_args()
    for path in sorted(OUT.glob("*.pkl")):
        scores = pickle.loads(path.read_bytes())
        keys = sorted({k[0] for k in scores if isinstance(k, tuple)})
        matches = sorted({M.match_of(k) for k in keys})
        for scenario in args.scenarios:
            rows = []
            for match in matches:
                halves = [k for k in keys if M.match_of(k) == match]
                gts = [scores[k, scenario][1] for k in halves]
                preds = [viterbi(args.temper * scores[k, scenario][0], args.team, args.dead) for k in halves]
                r = score_match(gts, preds)
                oracle = score_match(gts, [np.where(g == 2, 2, p) for g, p in zip(gts, preds)])       # perfect D
                frame_arg = score_match(gts, [scores[k, scenario][0].argmax(1).astype(np.int8) for k in halves])
                rows.append([r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"] / r["n_gt_spells"],
                             r["spell_f1"][0.3]["f1"], r["spell_f1"][0.5]["f1"], r["spell_f1"][0.7]["f1"],
                             r["spell_f1"][0.5]["precision"], r["spell_f1"][0.5]["recall"],
                             r["per_class"]["H"]["f1"], r["per_class"]["A"]["f1"], r["per_class"]["D"]["f1"],
                             r["transitions"][12]["f1"] if "f1" in r["transitions"][12] else np.nan,
                             oracle["spell_f1_mean"], frame_arg["macro_f1"]])
            m = np.nanmean(np.array(rows, dtype=float), axis=0)
            print(f"{path.stem:<6} {scenario:<8} spellF1 {m[0]:.3f}  macroF1 {m[1]:.3f}  spells x{m[2]:.2f}  "
                  f"F1@.3/.5/.7 {m[3]:.2f}/{m[4]:.2f}/{m[5]:.2f}  P/R@.5 {m[6]:.2f}/{m[7]:.2f}  "
                  f"classF1 H {m[8]:.2f} A {m[9]:.2f} D {m[10]:.2f}  trans@12 {m[11]:.2f}  "
                  f"| with perfect D: {m[12]:.3f} | per-frame argmax macroF1 {m[13]:.3f}")


if __name__ == "__main__":
    main()
