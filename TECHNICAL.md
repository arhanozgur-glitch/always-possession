# Team Always — technical document

More Than A Hack 2027, Initial Challenge: per-frame ball possession for the BAR–ATH tactical video.

## 1. Approach

**Input.** Only the automatic tactical tracking (`BAR-ATH_tactical_atd.xml` / `.txt`): player
positions with a team per track slot, and the ball position where it was detected.

**Pipeline.**

1. **Read the tracking** (`possession/epts.py`). The reader follows the `DataFormatSpecification`
   blocks of the metadata, because the order of the players in a line changes over the match.
   Coordinates are converted to metres. Each graded half is processed separately.
2. **Per-frame features** (`possession/features.py`, about 125 columns). They are relative and
   team-symmetric, never absolute identities. Positions are first averaged over a few frames, and
   speeds are measured over a baseline of several frames, so that tracking jitter does not turn
   into false movement:
   - ball: seen or not, time since it was last seen and until it is seen again, gap length,
     speed, distance to the touchlines and goal lines;
   - proximity: distance from the ball to the nearest player of each team, their difference and
     ratio, number of players of each team within 3 m;
   - control: a player "controls" the ball when it is within 2 m and clearly nearer than any
     opponent; from this, time since / until each team last / next controlled it, which team did
     so last and next, and each team's share of control in the past and future 2, 6 and 16 s;
   - team shape: median player speed, share of slow and fast players, width and depth of each
     team, movement of the two blocks (teams spread when they have the ball and move as a block
     towards the goal they attack; players slow down and stand when play is stopped);
   - the same quantities averaged over short and long windows and taken at lags and leads.
   The whole video is available, so features look both backwards and forwards in time.
3. **First-stage model** (`possession/model.py`): LightGBM, three classes (home, away, dead), on
   those features. Training rows near a label change get more weight.
4. **Second-stage model** (`possession/refine.py`): LightGBM on the same features plus the
   first stage's probabilities at offsets from 0.2 s to 4 s before and after the frame, their
   averages over 1, 3 and 8 s, and their trend. It is trained on out-of-sample first-stage
   probabilities (the training matches are split in two groups; each is scored by a first stage
   trained on the other).
5. **Decoder** (`possession/decode.py`): Viterbi over the three states with a cost for a change of
   team (10) and a higher cost for entering or leaving dead ball (40), in log-probability units.
   This removes flicker and places each change at the most likely frame.
6. Frames outside the two graded halves are written as `X`.

**Why this rather than something else.**

- The organizer baseline (team of the nearest player, majority-smoothed) fails for known reasons:
  the ball is missing on about 41% of graded frames, nearest player is not possession, and there is
  no dead-ball detector. We reproduced it on external data (spellF1 0.23–0.30, close to the
  published 0.24 on the real match) and used it as the reference.
- We first built a rule-based system with the same decoder (`possession/rules.py`). On unseen
  external matches it scored 0.35–0.40 spellF1; the learned model scored far higher, so the
  learned model was kept.
- No video model and no deep learning: with four days, no permission to train on the provided
  video, and tracking already available, tabular features with gradient boosting gave a system we
  could validate properly on matches it had never seen.
- The ranking metric rewards whole possession spells placed at the right time, so the decoder and
  the second stage (which mostly decide where spells start and end) mattered more than per-frame
  accuracy.

**How we validated without the provided match.** All scoring used the organizers' unmodified
`possession_metrics.py`. Clean external tracking was first degraded to resemble automatic
tracking (section 3). Models were always scored on matches they had not been trained on:

| test (spellF1, mean over degradation scenarios) | score |
|---|---|
| organizer baseline, smooth 50, on our degraded external matches | 0.23–0.30 |
| rule-based system | 0.35–0.40 |
| single-stage model, trained on one data set, tested on another | 0.53–0.57 |
| single-stage model, 4 folds of held-out matches (SoccerTrack + Metrica) | 0.654 |
| + weight near label changes | 0.664 |
| + second stage | 0.692 (macroF1 0.91) |
| + jitter-robust features and a noisy training scenario (**the submitted system**) | **0.738** (macroF1 0.93) |
| submitted system on three stress scenarios it was not trained on: position jitter 0.6 m / 6% false ball detections / ball missing on 65% of in-play frames | 0.72 / 0.74 / 0.66 |
| submitted method trained on SoccerTrack + Metrica only, tested on all 7 DFL matches (a provider it never saw) | 0.672 (macroF1 0.91); jitter stress 0.65 |
| earlier single-stage model, same test | 0.667 |
| earlier two-stage model, DFL matches held out in 3 folds (other DFL matches in training) | 0.707 |

The jitter step came from a stress test: with 0.6 m of frame-to-frame position noise the earlier
model lost a third of its score, because speeds were taken from one frame to the next. Positions
are now averaged over about a quarter of a second and velocities are taken over a longer
baseline. This also improved the ordinary scenarios.

Decoder costs and all other settings were chosen on these held-out external scores only.

## 2. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| tactical tracking (`_atd.xml`, `_atd.txt`) | **yes** | the only input |
| tactical and broadcast video | no | we did not download them; no video model |
| broadcast tracking | no | covers only what the broadcast camera shows; the tactical tracking is complete |
| homography | no | the tracking is already in pitch coordinates |
| Smart Tagging (automatic phases) | no | we have no equivalent on external data, so we could not measure whether it helps without tuning on the provided match |
| public 5-minute ground truth | not for development | only for the pipeline check the organizers describe; no parameter was changed afterwards |

## 3. Training data

All labelled data is external. None of it comes from the provided game. No hand labels were used
for training; the only manual work is the label check on an external match described below.

| data set | matches | tracking | possession labels: source |
|---|---|---|---|
| SoccerTrack v2 (Scott et al., arXiv:2508.01802, CC BY 4.0) | 8 | provider tracking of university matches; ball position is interpolated between events | the provider's per-frame `ballStatus` (HOME / AWAY / NEUTRAL / BALLOUT) in the raw tracking files |
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | 3 | corrected professional tracking, measured ball | derived automatically by our code from the provider's event data: dead where the ball is absent; otherwise the team of the latest on-ball event |
| DFL open data (Bassek et al., Scientific Data 2025, CC BY 4.0) | 7 | official optical tracking of Bundesliga / 2. Bundesliga, measured ball | the provider's per-frame `BallPossession` and `BallStatus` |

**Label check against video.** A team member watched the first half of SoccerTrack match 117093
with the provider's labels shown frame by frame (`labeling/viewer.html`) and noted every moment
that looked wrong. Seven remarks in 45 minutes: four were loose balls, clearances and a restart
where the provider keeps possession with the last team or the restarting team, which is exactly
the competition's definition; two were stoppages that start about a second too early or too late;
one was a loose ball given to a team for about 4 s. We found no systematic error and changed
nothing. These notes are on an external match and were not used for training.

Total: 18 matches, 36 halves, about 2.5 million labelled frames (about 28 hours). Two further
SoccerTrack matches were dropped because their files carry no possession status.

**Label harmonisation** (automatic, same rules for every match of a set): SoccerTrack `NEUTRAL`
(about 5% of frames) goes to the team that had the ball before; for Metrica and DFL, a spell
shorter than 3 s that only interrupts the other team and shows no controlled action is given back
to that team. After this, possessions average 9–16 s in all three sets (the organizers report
14.2 s for their ground truth).

**Degradation** (`possession/degrade.py`). The external tracking is much cleaner than automatic
tracking, so before training and before every evaluation we remove and perturb observations:
in-play ball dropouts in bursts (12–50% of frames, more likely for fast balls and in crowds),
missing players (3–12%), position noise, occasional false ball detections, and, during stoppages,
a ball that is either unseen, seen being carried, or seen placed before the restart. We do not
know how the provided tracking actually fails, so there are sixteen scenarios with different mixes;
the model is trained on seven of them and also tested on scenarios it was not trained on,
including four deliberately harsh ones (strong jitter, many false ball detections, a quarter of
the players missing, the ball missing on 65% of in-play frames). Every
training half is also used with the two teams swapped, so the model cannot learn a side or a team.

The simulator's parameters were set from the figures in the organizers' documentation (share of
frames without a ball, average number of tracked players) and from general knowledge of tracking
systems — not measured on, or fitted to, the provided data.

## 4. Possession rules implemented

The rules are not hand-coded thresholds; they are what the training labels and the label
harmonisation above express, plus the decoder:

- **Loose and contested balls** stay with the team that last had possession. A touch, deflection
  or challenge without control does not change possession; in the training labels an interruption
  of under 3 s without a controlled action is not a possession. In the output, the decoder's
  team-change cost has the same effect: a few frames in which an opponent is nearest do not flip
  the label.
- **Change of possession** happens when the other team shows control (the ball stays with its
  players), and is dated at the frame where the evidence switches, looking both backwards and
  forwards in time.
- **Ball not detected** is not treated as dead ball. While the ball is missing, the features
  describe who controlled it before and after the gap and how the teams move during it.
- **Dead ball** (out of play, fouls, stoppages before restarts) is its own class, learned from
  provider ball-status labels: mainly the ball leaving the pitch or stopping, players slowing down
  and standing, and the restart that follows. It ends at the restart. Entering or leaving dead
  ball costs more in the decoder than a change of team, because stoppages are rarer and longer.
- **After a restart** the team that takes it has possession from the first frame of live play.

## 5. Limitations

- **The real tracking was never seen during development.** Our scores are measured on external
  matches degraded by a simulator. If the provided tracking fails in ways the simulator does not
  cover (for example swapped team identities or systematic false ball detections), the score on
  the real match will be lower than the 0.67–0.74 measured on external data.
- **Short possessions are missed.** On external matches the output has about 10–20% fewer possessions than the truth;
  recall is lower than precision. Hand-overs are the main error source; dead-ball detection is
  comparatively good.
- **Timing of changes** is still off by more than half a second in many cases, which the
  IoU-based metric punishes.
- **Label style differs between providers** (when exactly a possession starts, what counts as
  dead). We harmonised with simple rules, but the organizers' definition may differ in detail.
- **Track slots are taken at face value**: the team given in the metadata is trusted, and
  goalkeepers and referees are not treated specially.
- **The gain from the later steps is smaller on an unseen provider.** With other matches of the
  same provider in training the method scores 0.71–0.74 on held-out matches; on a provider it
  never saw (DFL) it scores 0.67, about the same as the earlier single-stage model. Part of what
  the model learns is each provider's own way of labelling possession, and the provided match is,
  for us, an unseen provider. We therefore expect at most about 0.67 on it, and less if its
  tracking is worse than our simulation.

With more time: use the video around ambiguous moments (who actually plays the ball), model
individual ball touches explicitly, and validate Smart Tagging on external data produced the
same way.

## 6. Reproduction

```bash
pip install -r requirements.txt          # Python 3.12
python run.py --data /path/to/data --out Always.csv
```

`--data` is the folder that contains `tactical/`. The trained model is in `models/`; the command
needs no other data and prints `wrote Always.csv (199319 rows) -- validation passed`.

Checked before submission: a team member installed the repository on a second computer (a
different Python version, the data downloaded separately) and obtained a file with the same
SHA-256 as the submitted one. On an external test case the output is byte-identical under
Python 3.12, 3.13, 3.14 and 3.15.
Retraining from the external data and the evaluation commands are described in the README.

## 7. Compliance notes

- No training, fitting or calibration on the provided match, its video or its ground truth.
- No hand labels or manual notes on the provided game.
- An AI coding assistant helped write the code. It had no access to the provided data: the data
  was kept in a folder it could not read, and the code was run on the provided data only by a team
  member. No data or output file from the provided match was given to it, and nothing observed
  on the provided match was used to change the system.
