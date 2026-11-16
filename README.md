# Team Always — possession estimation, More Than A Hack 2027

Per-frame ball possession (`H` home, `A` away, `D` dead ball, `X` not graded) for the BAR–ATH
tactical video, from the automatic tactical tracking only.

The method is a two-stage gradient-boosted model followed by a three-state Viterbi decoder. It was
trained and tuned **only on external public matches** (SoccerTrack v2, Metrica sample data, DFL open
data). Nothing was trained, tuned or calibrated on the provided match, its video or its ground
truth. Details: [TECHNICAL.md](TECHNICAL.md).

## Reproduce the submission

Requirements: Python 3.12 (3.13, 3.14 and 3.15 produce a byte-identical file in our tests),
about 2 GB of free memory. No GPU.

```bash
git clone https://github.com/arhanozgur-glitch/always-possession
cd always-possession
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
python run.py --data /path/to/data --out Always.csv
```

`--data` is the unpacked data bundle, i.e. the folder that contains `tactical/`. Only two files
are read:

```
tactical/BAR-ATH_tactical_atd.xml
tactical/BAR-ATH_tactical_atd.txt
```

The script writes the 199,319-row CSV, runs the organizers' submission checks on it and prints one
line:

```
wrote Always.csv (199319 rows) -- validation passed
```

It takes a few minutes on a laptop. The trained model is part of the repository (`models/`), so
this command needs no external data and no training.

The match data is not in this repository and must not be added to it.

## Repository layout

| path | what |
|---|---|
| `run.py` | the command above: tracking in, submission CSV out |
| `models/gbm.txt`, `models/gbm2.txt`, `models/gbm.txt.json` | the trained first and second stage, decoder settings and training record |
| `possession/epts.py` | reader for the FIFA/EPTS tracking format (honours `DataFormatSpecification` blocks) |
| `possession/features.py` | per-frame features from player and ball positions |
| `possession/model.py`, `possession/refine.py` | first-stage and second-stage models, training |
| `possession/decode.py` | Viterbi decoder over H / A / D |
| `possession/degrade.py` | makes clean external tracking look like automatic tracking |
| `possession/soccertrack.py`, `metrica.py`, `sportec.py` | loaders for the three external data sets |
| `possession/baseline.py`, `rules.py`, `leaderboard.py`, `methods.py` | organizer baseline and a rule-based system on external data, local scoreboard |
| `experiments/` | held-out evaluation (`cross.py`), diagnostics, end-to-end test of `run.py` |
| `organizer_code/` | the organizers' public scorer and documentation (no data), used unmodified for all local scoring |
| `labeling/`, `tools/` | a viewer for checking external provider labels against external video (not used for training) |

## Optional: retrain from the external data

Not needed to reproduce the submission. The files in `models/` are the ones that produced it;
retraining gives an equivalent model but is not guaranteed to be bit-identical on another machine.

1. Get the external data:
   - SoccerTrack v2 (Hugging Face `atomscott/soccertrack-v2`): folders `raw/`, `bas/`, `ball/` of
     matches 117092, 117093, 118575, 118576, 118577, 118578, 128057, 128058 into `data/soccertrack/`.
   - Metrica sample data: `git clone https://github.com/metrica-sports/sample-data data/metrica_sample`.
   - DFL open data (figshare, doi 10.6084/m9.figshare.28196177): all XML files into `data/sportec/`.
2. Build the caches and train:

```bash
python -m possession.soccertrack
python -m possession.metrica
python -m possession.sportec
python -m possession.refine train
```

3. Evaluate on held-out matches and test the pipeline end to end:

```bash
python -m experiments.cross --splits folds --stack
python -m experiments.test_run
```

## Data sources and acknowledgements

- **SoccerTrack v2** — A. Scott, I. Uchida, K. Kuroda, Y. Kim, K. Fujii, "SoccerTrack v2: A Full-Pitch
  Panoramic Video Dataset for Game State Reconstruction and Ball Action Spotting", arXiv:2508.01802.
  CC BY 4.0.
- **Metrica Sports sample data** — https://github.com/metrica-sports/sample-data. We thank Metrica
  Sports for making it available.
- **DFL open data** — M. Bassek et al., "An integrated dataset of spatiotemporal and event data in
  elite soccer", Scientific Data, 2025. doi 10.6084/m9.figshare.28196177. CC BY 4.0.
- The organizers' evaluation code in `organizer_code/` belongs to the challenge organizers.
