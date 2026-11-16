"""End-to-end test of run.py without the competition data.

Metrica Sample Game 3 is in the competition's tracking format, so it is laid out like the
competition bundle (tactical/BAR-ATH_tactical_atd.xml / .txt) in a temporary folder, run.py is
run on it exactly as it will be run on the real data, and the CSV is scored against the labels
derived for that game. The score itself is in-sample for the final model (game 3 is training
data); what this proves is the plumbing: file discovery, frame indexing, halves, CSV format.

    python -m experiments.test_run
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from possession import epts, metrica  # noqa: E402
from possession.data import LETTERS, X  # noqa: E402
from possession.leaderboard import pm  # noqa: E402


def main() -> None:
    game = ROOT / "data/metrica_sample/data/Sample_Game_3"
    meta = epts.read_metadata(game / "Sample_Game_3_metadata.xml")
    p = meta["params"]
    spans = [(int(p["first_half_start"]) - 1, int(p["first_half_end"]) - 1),
             (int(p["second_half_start"]) - 1, int(p["second_half_end"]) - 1)]
    n = spans[1][1] + 1
    with tempfile.TemporaryDirectory() as tmp:
        tactical = Path(tmp) / "tactical"
        tactical.mkdir()
        shutil.copy(game / "Sample_Game_3_metadata.xml", tactical / "BAR-ATH_tactical_atd.xml")
        shutil.copy(game / "Sample_Game_3_tracking.txt", tactical / "BAR-ATH_tactical_atd.txt")
        out = Path(tmp) / "test.csv"
        cmd = [sys.executable, str(ROOT / "run.py"), "--data", tmp, "--out", str(out), "--frames", str(n),
               "--pitch", "105", "68", "--halves", *[str(v) for s in spans for v in s]]
        print(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip())
        pred = pm.label_codes(pd.read_csv(out))
    gt = np.full(n, X, dtype=np.int8)
    for (a, b), tag in zip(spans, ("1st", "2nd")):
        lab = metrica.cached(3, tag).labels
        gt[a:a + len(lab)] = lab
    r = pm.evaluate(gt, pred)
    print(f"rows {len(pred)}, labels {dict(zip(*np.unique(LETTERS[pred], return_counts=True)))}")
    print(f"spellF1 {r['spell_f1_mean']:.3f}  macroF1 {r['macro_f1']:.3f}  (in-sample; a plumbing check)")
    shifted = pm.evaluate(gt, np.roll(pred, 25))
    print(f"same output shifted by 1 s: spellF1 {shifted['spell_f1_mean']:.3f}  -> frame indexing is right "
          f"if this is clearly lower")


if __name__ == "__main__":
    main()
