# Counterfactual responsibility with CAT's DenseTNT

> Note (`careful`, branch `td3-without-metadrive`): the MetaDrive scripts this reference describes for training and policy rollouts (`cat_RLtrain.py`, `cat_advgen.py`, `collect_rollouts.py`, `run_rl.sh`, `run_eval.sh`, `verify_adv_export.py`) are removed on this branch; training is `scripts/rl/train_td3.py` (README.md, docs/RUNBOOK.md step 7).

Measures whether a vehicle — by default the self-driving car — drives
aggressively, using the counterfactual responsibility metrics of Hsu et al.,
["Interpretable Trajectory Prediction for Autonomous Vehicles via
Counterfactual Responsibility"](https://saferobotics.princeton.edu/research/responsibility)
(IROS 2023), with CAT's pretrained DenseTNT as the model of what drivers do.
It is a port of the SMART/WOMD implementation in the `catk` repository; the
geometry, CVaR and HMM code are shared verbatim.

At every context step k of a scene (every 0.5–1 s):

| quantity | question | definition |
|---|---|---|
| **safety responsibility** β_s (m) | Did the agent give up safety margin compared with what it could otherwise have done? | CVaR_α over N DenseTNT samples of the agent's motion of D_g(sample, b) − D_g(logged, b), max over interacting neighbours b. D_g = closest approach over the next 2 s, saturated at 10 m |
| **courtesy responsibility** β_c (nats), the default (`--courtesy-estimator boltzmann`) | Did its actual motion block others' plausible options? | Sec. V-B / Eq. 7 of the paper (catk's default): b's 20 samples with and 20 without the agent, pooled, each scored R = −10 c_coll − c_dev against the others' logged futures (2 s), with the agent and without; KL between the two Boltzmann distributions; 0 when the two stay more than 20 m apart; max over vehicle neighbours b |
| **courtesy as the goal KL** β_c (nats), `--courtesy-estimator kl` (the default until 2026-10-18) | Did its presence change what others intend? | Eq. 4 as written: KL(π_b(goal \| scene) ‖ π_b(goal \| scene without the agent)), exact over b's goal grid, max over vehicle neighbours b. Needs the model to react to the agent, which DenseTNT hardly does (CAT attackers 0.52, see below) |
| **courtesy as demand** β_d (m/s²), `--courtesy-estimator demand` | How hard did its actual motion make others brake? | CVaR_α over b's motion set *without* the agent of the constant deceleration that, after a 1 s reaction, keeps each plan's footprint clear of the agent's actual future over 5 s (10 = no braking up to 9 m/s² clears it); max over vehicle neighbours b; pairs sharing a lane are not measured |

β_s > 0: most of the agent's own alternatives would have kept more distance to
someone than it did. β_c large: the agent's actual motion blocked most of what
some vehicle could plausibly have done in the next 2 s. Both are open-loop, like the paper: DenseTNT only sees
the 1.1 s of history up to k, never anyone's future.

## Setup

The code runs in CAT's environment (Python 3.9; `advgen/utils_cython*.so` is
built for it) plus `pytest`. Without conda, a CPU environment with
[uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.9
uv venv --python 3.9 --seed ~/venvs/cat39 && source ~/venvs/cat39/bin/activate
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
uv pip install "tensorflow-cpu==2.12.0" "numpy<1.24" pyyaml matplotlib tqdm pytest scipy     "opencv-python-headless==4.7.0.72"   # MP4 videos; without it the visualisation writes GIF only
```

MetaDrive is not needed: scenes are read straight from `raw_scenes_500/*.pkl`.
`advgen/pretrained/densetnt.bin` must be in place (see the main readme).
Run everything from the repository root.

```bash
python -m pytest tests/responsibility -q                       # no model or data needed
python -m scripts.responsibility.verify_densetnt --n 5         # against CAT on real scenes
```

`verify_densetnt` checks that the wrapper rebuilds CAT's inputs field by field
and reproduces its 32 adversary trajectories (to ~1e-12 m), that goal sampling
matches the goal distribution, that repeated passes agree exactly, that
removing an agent drops exactly its polyline and moves a nearby vehicle's
distribution more than a far one, that DenseTNT's partner slot is
immaterial (~1e-7 nats), and that a replayed rollout (below) gives the logged
values once the objects MetaDrive does not spawn are left out.

### More scenes: a whole WOMD split

CAT's 500 scenes are a convenience sample. CAT converted 14 of the 150
validation_interactive shards (`scripts/convert_WOMD_to_MD.py`) and kept
the 91-step scenarios whose self-driving car is one of the two objects of
interest (`scripts/select_cases.py`). There was no selection by
interaction or criticality, and 3 scenes are duplicates (497 ids).

`convert_womd_split` converts a whole split with CAT's own conversion
(`convert_scenario`), so the scenes have the format of `raw_scenes_500`:

- it keeps every scenario whose two objects of interest are vehicles;
- `index.csv` beside the scene folder marks `sdc_in_ooi` (the scenes CAT's
  closed loop can use) and `in_cat`;
- one process per shard; the run resumes;
- it runs in cat39, with the WOMD protos borrowed from the catk environment
  (same protobuf version).

```bash
python -m scripts.responsibility.convert_womd_split --tfrecords ~/womd_v1_2_1/validation_interactive \
    --out-dir ~/womd_v1_2_1/cat_format/validation_interactive --workers 8   # --shards 1 --limit 20: a test
```

Converting CAT's 497 scenarios from WOMD v1.2.1 this way (54 s, one
process) gives the same tracks as CAT's v1.1 files: the same agents and
ids, positions equal to the bit. The maps differ: v1.2.1 reprocessed them,
the light count differs in 222 scenes, and driveways were added. The
right-of-way check moves from 87.3% to 86.5% on them.

v1.2 adds DRIVEWAY features, and MetaDrive writes UNKNOWN for unknown lines
and edges. DenseTNT's input leaves both out, both here (`scene.py`) and in
CAT's `advgen`, since it was trained on v1.1 maps without them and asserts
types below 20. CAT's 500 scenes contain neither, so their inputs are
unchanged. MetaDrive builds its map from lanes, lines and edges only, so it
ignores them anyway.

DenseTNT also asserts map feature ids below 1000 for the points it uses.
Ids reach 1000 in 8.1% of the 3166 scenes of the local shards, but the
input keeps at most 20000 map points, in id order. That cap cuts the map
in 237 of CAT's 500 scenes (CAT's behaviour, kept). No id of 1000 or more
reached DenseTNT's view (80 m around a point 30 m ahead) at any step in
CAT's scenes, their v1.2.1 conversions, or 444 new scenes.

So the new scenes work in the closed loop too. `cat_advgen.py` on three
new scenes with driveways ran MetaDrive and CAT's generation
(`SDL_VIDEODRIVER=dummy` on a headless machine): 2 of 3 attacks succeeded,
0.5 s per generation. The closed loop needs a folder of `sdc_in_ooi`
scenes numbered `0.pkl` … in MetaDrive's order. `cat_RLtrain.py` takes
scenes 0–399 for training and the rest for testing (`swap.scene_split`).

## Command reference

Run from the repository root with the environment active. The sections below
explain each step; this is the whole pipeline in one place, for the
simplified evaluation of one pair per scene (`--use-ooi`: the self-driving
car and CAT's adversary) with the NMS motion set, the most accurate 40-trajectory set
measured (see "Valid counterfactuals" under Design decisions).

**Compute.** `run_h200.sh` runs every agent in shards, then the summaries and
the levels, into `OUT/<agent>/`:

```bash
OUT=logs/responsibility/ooi_nms AGENTS="sdc adv" SHARDS=8 RECORDS=1 \
EXTRA="--use-ooi --motion-set nms --valid-counterfactuals" \
bash scripts/responsibility/run_h200.sh
```

The same for one agent by hand (`--agent adv` swaps the roles of the pair):

```bash
python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 \
    --out-dir logs/responsibility/ooi_nms/sdc --agent sdc \
    --use-ooi --motion-set nms --valid-counterfactuals --save-records --device cuda
```

- Shards: `--num-shards N --shard-index i` for i = 0 … N−1, all into the
  same `--out-dir`. Each shard writes `windows.shard-<i>-of-<N>.csv`, and
  every reader takes both forms.
- A subset: `--n 30 --first 0`. Leaving out β_c is 3–10× faster:
  `--no-courtesy`.
- `--valid-counterfactuals` is `--lane-route`, `--drivable-half-width 3`,
  `--kinematics`, `--collision-filter` and `--courtesy-valid-goals`.
- On a map built by perception (no lane topology), use
  `--intent --drivable-edges --kinematics --collision-filter` instead.
- To take β_c only over the neighbour's own logged drive mode, add
  `--courtesy-same-mode lanes` (HD map) or `--courtesy-same-mode path`
  (no map needed; 6 m of its logged path). This overrides
  `--courtesy-valid-goals`.
- Motion sets: `sampled` (default), `weighted` (the reference; a median of
  944 trajectories per window), `topk`, `nms`.
- MTR instead of DenseTNT: `--model mtr --checkpoint <ckpt>`.
- A driving policy instead of the log:
  `--rollouts rollouts/<policy>/<adversary>`.
- An output directory resumes. Its `config.json` pins the settings, and a
  run with other settings is refused.

**Summarise and fit levels** (`run_h200.sh` does both):

```bash
python -m scripts.responsibility.summarize_responsibility --run logs/responsibility/ooi_nms/sdc
python -m scripts.responsibility.fit_levels \
    --runs logs/responsibility/ooi_nms/sdc logs/responsibility/ooi_nms/adv \
    --out-dir logs/responsibility/ooi_nms/levels
```

**Visualise.**

- Offline, from records, with no model needed:

  ```bash
  python -m scripts.responsibility.visualize_responsibility \
      --record logs/responsibility/ooi_nms/sdc/records/{17,23,42}.pkl \
      --levels logs/responsibility/ooi_nms/levels/levels.csv \
      --out-dir logs/responsibility/ooi_nms/videos_sdc
  ```

- Live, with the model loaded once for every scene and both roles:

  ```bash
  python -m scripts.responsibility.visualize_responsibility --scene 17 23 42 \
      --agent sdc adv --use-ooi --motion-set nms --device cuda \
      --out-dir logs/responsibility/videos_ooi
  ```

- Live, with a run's settings, so the values equal its `windows.csv`:

  ```bash
  python -m scripts.responsibility.visualize_responsibility --scene 17 23 \
      --run logs/responsibility/ooi_nms/sdc --agent sdc --use-ooi \
      --levels logs/responsibility/ooi_nms/levels/levels.csv \
      --out-dir logs/responsibility/video_17_23
  ```

One video goes to `--out-dir`. Several go to `--out-dir/<scene>`, or to
`--out-dir/<scene>_<agent>` with several agents. With `--run`, the motion
set, the filters and `use_ooi` come from the run's `config.json`, and
`--use-ooi` is refused for a run made without it. A DenseTNT process takes a
few hundred MB of GPU memory.

## Is the ego aggressive?

```bash
# 1. responsibility of the self-driving car in every window of 50 scenes
python -m scripts.responsibility.compute_responsibility --out-dir logs/responsibility/sdc --n 50
# 2. flag windows and scenes
python -m scripts.responsibility.summarize_responsibility --run logs/responsibility/sdc
```

Step 1 writes `windows.csv` (one row per scene and context step: β_s, β_c,
speed, and the neighbour each maximum came from) and per-neighbour details in
`obs/<scene>.pkl`; it is resumable. Useful flags: `--agent adv` (CAT's
adversary) or a track id, `--stride`, `--n-samples`, `--d-sat`,
`--no-courtesy` (3–10× faster), `--use-ooi` (below).

Step 2 flags a window when β_s or β_c exceeds a threshold and a scene when it
has a flagged window, and writes `summary/scenes.csv`,
`summary/windows_flagged.csv` and `summary/responsibility.png` (β_s vs β_c).
Thresholds are either given (`--safety-threshold`, `--courtesy-threshold`) or
the `--quantile` (0.9) of the positive values of a **reference** run
(`--reference`, default the run itself). To judge a driving policy, calibrate
on logged driving and judge the policy's run against it:

```bash
python -m scripts.responsibility.summarize_responsibility --run logs/responsibility/policy \
    --reference logs/responsibility/sdc
```

Windows below `--min-speed` (1 m/s) are not judged.

To evaluate a trajectory produced in simulation instead of the log (e.g. the
ego of a CAT-trained policy), replace the agent's track before computing:
`Scene.with_track(agent, positions, headings, velocities)` over the 91 steps.

### Forced yields: assertive without being unsafe

β_s and the right of way's yield duty both read the unsafe end. A driver
who takes the right of way from someone still far enough away to stop
registers in neither. `responsibility/forced_yield.py` measures it, open
loop, for a pair (the agent and another road user):

- the conflict point, the conflict zones and the holder of the right of way
  come from `right_of_way.py`;
- at each step the agent enters its zone, it computes the braking demanded
  of the other, `a_req = v² / (2 (d − v ρ))`. Here d is the other's distance
  to its own zone, v its speed and ρ = 1 s;
- the result is `forced`, the worst entry's a_req, counted only when the
  other held the right of way.

Above 4 m/s² (RSS's brake_min) the yield duty is violated. Below that,
a_req grades how hard the agent made a holder brake that still could.

The ground truth (`scripts/responsibility/make_go_first.py`) rewrites logged
pairs in which the yielder let the holder pass first, so that the yielder
goes first: it leaves its zone a post-encroachment time (PET) before the
holder enters. The pairs come from `validate_right_of_way`'s `pairs.csv`.
Two rewrites are made:

- `delay` shifts the holder's whole track later. The yielder keeps its
  lawful, logged motion, so its kinematics alone cannot tell the rewrite
  apart. Only rules whose priority does not depend on time are rewritten:
  stop-sign, driveway, through-road and left-turn, with no traffic light.
- `retime` re-times the yielder along its own path, at ≤ 3 m/s² and
  ≤ 15 m/s.

The holder enters the conflict about 4 s into the 9.1 s clip, so `retime`
rarely fits and PET 4 s hardly ever does. Logged pairs in which the yielder
went first anyway (`log_yielder_first`) carry no label. They are the
reference for the threshold: the braking that drivers who take a gap leave
to the holder.

Open loop the whole tracks are known, so the conflict point is the first
crossing of the two tracks (`track_conflict`), judged at the step the later
of the two reaches it. Before, P* came from the paths up to the closest
approach, extended 15 m straight on, as for a collision; P* then moved
with the timing, and a rewrite was judged at a different point than its
log. Tracks that do not cross are judged at their closest approach. A
rewrite keeps the paths, so it keeps P*.

`evaluate_forced_yield` judges a rewritten scene with its log's right of
way, moved onto the rewritten tracks (`carry_priority`). The rewrite keeps
the map and, for the rules it is made for, the priority, so the ground
truth tests the braking demand alone; `validate_right_of_way` checks the
right of way. The rewritten scene's own reading agrees in about 80% of
pairs. It differs where the delayed holder's straight run-up, which the
right of way reads as its approach, crosses other lanes, or where a car
leaving a driveway is matched to the road at P*. `--own-priority` judges
each scene by its own reading.

```bash
python -m scripts.responsibility.make_go_first --pairs logs/responsibility/right_of_way/validation_interactive/pairs.csv \
    --out-dir logs/responsibility/forced_yield/scenes --pet 1 2 4 --workers 4
S=logs/responsibility/forced_yield/scenes
python -m scripts.responsibility.evaluate_forced_yield --out-dir logs/responsibility/forced_yield/eval \
    --sets log=$S/log delay_pet2=$S/delay_pet2 delay_pet1=$S/delay_pet1 retime_pet1=$S/retime_pet1 \
           natural=$S/log_yielder_first --reference natural --workers 4
python -m scripts.responsibility.visualize_forced_yield --scenes $S --sets delay_pet2 delay_pet1 log_yielder_first \
    --out-dir logs/responsibility/forced_yield/figures      # index.md: a gallery sorted by the demand; --gif animates
```

The figures put the log next to the rewrite. Each map shows both tracks,
with a dot every second, P*, the conflict zones, and the footprints at the
agent's worst entry. A time–distance panel shows each agent's distance to
P* against time.

Held-out WOMD (the 14 local shards above, 2026-10-10) gives 242 lawful
pairs. The threshold is the 0.9 quantile of the 42 logged go-first pairs,
1.53 m/s²:

| set | pairs | other holds, agent entered | forced > 1 m/s² | > 2 | > 4 (duty violated) | > 1.53 |
|---|---|---|---|---|---|---|
| logged, holder first | 242 | 209 | 3% | 3% | 3% | 3% |
| logged, yielder first (reference) | 42 | 31 | 35% | 3% | 0% | 10% |
| delay, PET 4 s | 2 | 2 | 100% | 0% | 0% | 50% |
| delay, PET 2 s | 18 | 18 | 89% | 56% | 6% | 83% |
| delay, PET 1 s | 47 | 47 | 98% | 79% | 28% | 87% |
| retime, PET 1 s | 4 | 4 | 75% | 75% | 25% | 75% |

At PET 2 s the yield duty flags 1 of the 18 rewritten pairs. That pair is
a left turn whose measured PET is 1.3 s: after a merge the two paths
coincide, so the zone ends where it was estimated only roughly. a_req flags
83% of the rewritten pairs, against 3% of the lawful ones and 10% of the
logged gap-takers. Few pairs survive the rewrite: of 242 lawful pairs, 48
are under time-dependent rules or at lights, 46 have a yielder that never
leaves its zone in the clip, and 41 have no holder at the first track
crossing. At PET 2 s, 85 more would put the holder's entry past the clip.

## Visualisation

```bash
python -m scripts.responsibility.visualize_responsibility --scene 17 \
    --run logs/responsibility/sdc --levels logs/responsibility/levels/levels.csv \
    --out-dir logs/responsibility/video_17
```

`--scene` and `--record` take several values (the model is loaded once),
and `--agent sdc adv` renders each scene for both objects of interest; see
the command reference above. One frame per context step t_k (`frames/t_XXX.png`), stitched into
`responsibility.gif` and `responsibility.mp4` (the MP4 needs OpenCV,
`opencv-python-headless`, which `setup_env.sh` installs). Each frame shows:

- **scene:** the map, every agent at t_k with 1 s of history, the agent's
  motion set exactly as β_s scored it (DenseTNT goal samples, or with
  `--motion-set weighted|topk|nms` the weighted set; the first 2 s solid,
  coloured by goal probability or weight), its logged future, and the neighbours it was compared with,
  with their logged futures. The neighbour behind β_s is outlined red, the one
  behind β_c blue. `--ego-heatmap` adds the agent's own goal distribution.
- **courtesy:** that neighbour's goal distribution with the agent in the scene
  and without it, i.e. the two sides of the KL. With
  `--courtesy-estimator demand`: the neighbour's plans without the agent,
  each coloured by the braking it would need to clear the agent's actual
  future (black: no braking clears it), and the probability of the plans per
  braking bin, with β_d (the CVaR), the mean, the share in conflict and the
  share that cannot clear marked. A neighbour in the agent's lane is named
  instead (not measured).
- **timeline:** β_s and β_c over the clip (β_c in m/s² with demand) with the current step marked; with
  `--levels`, every window's level as background, aggressive levels hatched.

To compare courtesy estimators on one scene, pass several:
`--courtesy-estimator kl boltzmann demand` computes the scene once per
estimator (β_s and the agent's motion set are identical in each, since
courtesy draws from its own random stream) and renders one video: the
scene, one row of courtesy panels per estimator, and β_s and each β_c on
their own axes. Offline, `--record a.pkl b.pkl --compare` does the same.
On logged scene 31 (the SDC waits at 0.4 m/s to turn left across an
oncoming car that has the right of way, and lets it pass) the goal KL
reads up to 0.25 nats, the Boltzmann estimate and β_d stay 0; on CAT's
scene 158 (the attacker pulls across the ego's lane, overlap at 3.3 s) the
KL peaks at 0.30 nats, as high as for the lawful turn, while the Boltzmann
estimate reaches 3.68 nats (ln 40, every sample but one blocked) 1.3 s
before the overlap and β_d is 10 from 1.5 s on.

It runs the same code as `compute_responsibility`, and `--run` takes that
run's settings, so the numbers equal the run's `windows.csv`. Without `--run`,
live mode takes `--motion-set`, `--n-samples` and `--use-ooi` itself; with
`--run`, `--use-ooi` is refused unless the run used it.

### Records: inspect and visualise a run offline

`compute_responsibility --save-records` also writes
`OUT/records/<scene>.pkl` (0.5–1.5 MB per scene), and a live visualisation
saves its `record.pkl`. A record holds the scene itself, plus, for every
context step t_k:

- the agent's motion set (samples [N, 80, 2] and their goal log-probabilities,
  or for a weighted set the log of each trajectory's weight);
- its goal distribution;
- each vehicle neighbour's goal distributions with and without the agent;
- all values, per neighbour included.

Goal distributions keep the most probable goals covering `--top-mass` (0.99)
of the probability, in scene coordinates. Values are identical with or
without records. Records render without DenseTNT, TensorFlow or the scene
files:

```bash
python -m scripts.responsibility.visualize_responsibility \
    --record logs/responsibility/sdc/records/17.pkl --levels logs/responsibility/levels/levels.csv \
    --out-dir logs/responsibility/video_17
```

In Python: `responsibility.records.load_record(path)` returns a dict with
`scene` (`scene_of(record)` rebuilds the `Scene`), `agent`, `agent_id`,
`config` and `frames`. Each frame has `step`, `observation`, `samples`,
`sample_log_prob`, `goals` and `courtesy` {neighbour id: {with, without,
kl}}. In windows without neighbours the metric draws no samples, so the
record keeps a display-only motion set (`metric_samples` False), drawn off
the metric's random stream.

## Responsibility levels

Thresholds say "more than X"; levels say what kinds of behaviour logged
driving contains. `fit_levels` fits a Gaussian HMM (paper Sec. IV-A; catk's
implementation) over the (β_s, β_c) sequences of one or more runs, chooses
the number of levels by BIC, orders them from calmest (0) to most aggressive,
and labels every window with the causal Bayes filter (Eq. 6):

```bash
python -m scripts.responsibility.fit_levels --runs logs/responsibility/sdc logs/responsibility/adv \
    --out-dir logs/responsibility/levels
# judge a policy against levels learned from logged driving
python -m scripts.responsibility.fit_levels --runs logs/responsibility/sdc logs/responsibility/policy \
    --fit-runs logs/responsibility/sdc --out-dir logs/responsibility/levels_policy
```

On all 500 scenes (the self-driving car and CAT's adversary as logged, 0.5 s
windows, fitted together) BIC chooses 7 levels:

| level | β_s (m) | β_c (nats) | share sdc / adv | elevated in |
|---|---|---|---|---|
| 0 | 0.00 | 0.01 | 36% / 38% | – (calm) |
| 1 | 0.00 | 0.02 | 1% / 2% | – |
| 2 | 0.01 | 0.09 | 18% / 18% | – |
| 3 | 0.01 | 0.27 | 16% / 14% | courtesy |
| 4 | 0.44 | 0.08 | 15% / 19% | safety: margin given up |
| 5 | 0.01 | 0.82 | 8% / 5% | courtesy: others' plans changed |
| 6 | 0.39 | 0.59 | 7% / 5% | both |

The levels separate *kinds* of aggressiveness rather than forming one
ranking, so every level that stands out from the calmest one by more than
half a spread (the feature's standard deviation over the fitted windows) in
safety or in courtesy counts as aggressive (here levels 3 to 6;
`--aggressive-levels K` takes the top K instead), and a scene counts when at
least `--scene-share` (50%) of its windows are. Because levels are relative to the fitted population, the
informative output is the comparison between runs: each run's share of
windows per level (printed, and in `scenes_levels.csv`). `--log-courtesy`
fits on log(1 + β_c), which tames courtesy's heavy tail. Outputs: `hmm.pkl`,
`bic.json`, `levels.csv` (every window with its level and posterior),
`scenes_levels.csv`, `levels.png` (scatter by level, BIC curve).

## Can it judge aggression?

The test is whether the framework above, given an agent and the neighbour
it interacts with, tells after the fact whether the agent's logged driving
was aggressive. CAT's adversaries collide with the ego by construction, so
they are labelled aggressive without annotation. In every condition exactly
one agent's trajectory differs from the log and everything else stays
logged, so a change in its score can only come from that agent. All
adversaries are DenseTNT's own candidates, chosen by different rules
(`responsibility/aggression_test.py` holds the table):

| condition | trajectory | expected |
|---|---|---|
| logged (adv, sdc) | the log | reference |
| mode | the most probable non-colliding candidate (`near` 0 ± 1000 m) | as the log: negative control |
| near 1 m < fair-inf < CAT | a 1 m miss; an avoidable collision (avoidability ≥ 0.5); CAT's rule | aggressive, graded |
| max-gap | the candidate with the largest gap (`near` 30 m) | timid |
| victim | the logged SDC in CAT's scene, unchanged | not aggressive: tests attribution |
| SDC × 0.5 / 0.7 / 1.3 | the logged path re-timed (`retime_agent.py`) | dose-response |

Scenes 0–99 of CAT's 500. Each condition is scored by AUROC against the
same agent's logged drive (95% intervals from 2000 scene bootstraps).

```bash
CAT_SCENES=adv_scenes/cat bash scripts/responsibility/run_aggression_test.sh   # scenes, runs, offline tables
python -m scripts.responsibility.evaluate_aggression --config paper --cat-scenes adv_scenes/cat \
    --json logs/responsibility/validity/eval_paper.json                       # the paper's pipeline
python -m scripts.responsibility.analyze_aggression_offline --config nms_ooi   # H, alpha, d_sat; two-sided score
python -m scripts.responsibility.aggression_hmm_schemes                        # HMM vs thresholds
python -m scripts.responsibility.probe_inertia --cat-scenes adv_scenes/cat     # mechanism probes
python -m scripts.responsibility.probe_courtesy --cat-scenes adv_scenes/cat --out logs/responsibility/validity/courtesy_probe.csv
python -m scripts.responsibility.plot_aggression --out-dir logs/responsibility/validity/figures ...
```

One scene under every condition, with its paths, its gap to the ego and
β_s and the three courtesy estimates per window (from the runs, no model):
`python -m scripts.responsibility.plot_conditions --scene 4 --cat-scenes adv_scenes/cat`.
In scene 4 the adversary turns into the ego's road: logged it stays 9.5 m
behind; the near miss passes 1.2 m ahead (β_s 1.25 m, β_d 3.5 m/s², the
Boltzmann estimate 0.74 nats for one window); the avoidable collision and
CAT's overlap at 6.1 s (Boltzmann 3.5–3.7 nats from 4.5 s and 5 s, β_d 10
from 4 s and 5 s; β_s 0.34 m for the avoidable one, 1.36 m for CAT's);
the largest gap reads 0 throughout; the goal KL stays below 0.05 nats
except one window after the near miss's closest approach (0.51).

**Hand-labelled aggressive drivers** (2026-10-19). 31 logged scenes whose
adversary and 16 whose SDC were judged aggressive by hand, measured with the
defaults (40 i.i.d. samples, Boltzmann courtesy) every 0.2 s (`--stride 2`)
against the other object of interest only (`--use-ooi`), and scored by
`score_against_reference.py` against the same settings on the 91 unlabelled
scenes of 0–99, both agents (182 drives; scene q90 of max β_s 0.644 m, of max
β_c 0.044 nats):

| labelled | flagged by max β_s | by max β_c | by either | median percentile β_s / β_c |
|---|---|---|---|---|
| adversary (31) | 23% | 19% | 39% | 60 / 40 |
| SDC (16) | 12% | 19% | 19% | 21 / 40 |
| reference (182) | 10% | 10% | 19% | — |

The adversaries are flagged about twice as often as logged drives; the
labelled SDCs not at all more often. In 8 of 16 SDC scenes and 8 of 31
adversary scenes the pair never comes within 5 m, and in many the labelled
driver comes within 0.1–1.5 m of a third vehicle instead (SDC 197: 0.1 m,
436: 0.3 m): the behaviour judged aggressive is often toward a vehicle
`--use-ooi` does not look at. β_s over all interacting neighbours (a run
stopped part-way, same settings otherwise) moves e.g. adversary 75 from
0.03 to 3.48 m and SDC 197 from 0 to 1.34 m, but against the matching
reference (q90 1.68 m) it flags 29% of the adversaries and 0% of the SDCs.
With all neighbours the Boltzmann estimate is unbounded where the
neighbour stands among others: b's samples all overlap them in the circle
cover, both distributions become one-hot on different samples, and the KL
is their reward gap (134 nats for a standing car 6.5 m from the agent in
scene 58), so that mode needs a rule for standing neighbours first.
Videos: `logs/responsibility/labelled/videos/{adv,sdc}/<scene>/`.

Without the limits (`--d-sat 0 --courtesy-skip-distance 0`, the same runs
otherwise; videos in `videos_nolimit/`): no window has β_s exactly 0 any
more (52% of the labelled adversaries' windows and 72% of the SDCs' had),
but the reference's scene q90 rises from 0.644 to 1.538 m, because
alternatives of agents far apart gain distance trivially, and the flags drop:
max β_s flags 13% of the adversaries (was 23%) and 0% of the SDCs (was 12%);
median percentile 48 (was 60) and 56 (was 21). The scenes that rise are far
pairs (adversary 12: 0 → 4.31 m with the pair never within 8 m; 384:
0 → 1.94 m), while close calls fall below the higher q90 (55, 210, 304, 319,
430). The Boltzmann estimate is unchanged in every window (largest
difference 0): its zeros (97% of the windows) are not the skip distance but
b's 40 samples not meeting the agent's actual future within the 2 s
horizon.

**Who of the pair was aggressive** (`pair_aggression.py`; the same 47 labelled
scenes, both directions of every pair, no reference needed). Per measure,
the share of the scenes showing it where the labelled driver scores more
aggressive than its partner:

| measure | scenes showing it | aggressor scores higher |
|---|---|---|
| cut-in: the other's headway after the driver moved ahead of it in its lane | 26 | 0.85 |
| took the right of way (forced yield on a holder) | 7 | 0.57 |
| β_s toward the other | 29 | 0.45 |
| courtesy, goal KL | 46 | 0.63 |
| courtesy, Boltzmann (the default) | 18 | 0.17 |
| courtesy, braking demand on DenseTNT's / constant-velocity plans | 41 / 31 | 0.34 / 0.16 |

The courtesy estimates that look at conflicts point at the partner, and as
much with plans from no model at all (constant velocity, or the speed b
had 1.5 s earlier: 0.18), so neither DenseTNT nor b's reaction in its
history causes it. It is the definition: courtesy charges a for what its
actual future does to b's plans, and in a cut-in or a pull-out the victim
is already on its way, so it is the victim's actual motion that blocks the
aggressor's plans (the aggressor's braking demand is 10, "cannot clear",
in most cut-ins), while the aggressor's actual motion blocks nothing of the
victim's (it merges ahead with a gap, and once in the victim's lane the
pair is not measured). Read the other way, the other's actual future
against the agent's own plans, the same numbers name the aggressor: 0.83
(Boltzmann), 0.66 (braking demand, DenseTNT), 0.84 (constant velocity). A
lawful yielder's own plans are blocked by the holder too, so that reading
needs "and went ahead anyway" before it can stand for aggression; it is
not validated on lawful yields yet.

**Labels for both objects of interest** (2026-10-22; `responsibility/labels/`).
All 500 scenes, labelled in both directions: the SDC toward its OOI partner,
and the OOI toward the SDC. Third vehicles do not count. X is aggressive
toward Y when its own action shifted risk or effort onto Y beyond what the
situation required; holding the right of way does not excuse a compressed
margin. There are two passes. The lane pass (cut-in, tailgating, close
pass) is rule-checkable. The right-of-way pass is the one that needs
judgment: for the 257 pairs whose swept boxes overlap at a crossing or merge,
annotators recorded who held the right of way (California Vehicle Code), whether
the first car broke its own control, and why the second car slowed;
`derive_row_labels.py` draws the verdict from those facts and the measured
reaction (PET below 1.5 s, braking of 3.4 m/s², after the literature in
`labels/README.md`). `label_ooi_aggression.py` supplies the model-free
evidence and renders a figure and a video per scene (the light or stop sign
each car faced, the PET band, the labels).

The SDC is aggressive in 8 scenes, all cut-ins. The OOI is aggressive in
63: 50 cut-ins, 11 tailgates, 1 close pass and 1 right turn on red without
stopping (128). Only scene 482 has both. Logged driving rarely takes the
right of way: of the 257 conflicts the first car lacked it in 26, and only 1
verdict is aggressive. A person reviewed the first version's right-of-way
positives: 7 of 8 were not aggressive, and the rules were refined until they
agreed (`labels/README.md`).

Agreement on 48 conflicts annotated twice: κ 0.76 for the holder, 0.78 for
whether the first car broke its control, 0.19 for why the second car slowed
(it settles a verdict only where the reaction is hard, 4 cases). The
verdict's κ (1.0) rests on one shared positive and means little.

`right_of_way.py` named a holder in 140 of the 257 conflicts: the
annotators agreed in 128, named the other car in 8 and no holder in 4. In
100 conflicts the rule named none and the annotators could. The first,
lane-only version of these labels found only cut-ins, which a rule can
find; the right-of-way pass is what adds judgment.

These labels replace the hand labels above for pair-level work. Those were
made without the right of way, and many concerned third vehicles. Of the
16 SDCs judged aggressive then, 2 are aggressive toward their OOI here
(342, 482). Of the 31 adversaries, 4 are (75, 86, 210, 430).

SMART runs the same conditions through `run_smart.sh` (one OUT per scene
set) and `import_catk_obs.py`, which splits catk's observations into
`runs/smart/<condition>/obs`; then `evaluate_aggression --config smart
--frame-s 0.5`. `probe_smart.py` is the SMART version of the probes.
`smart_records.py` (in catk's environment) measures one agent per scene with
SMART through catk's own functions and keeps the rollouts it drew, writing a
run of this repository (`windows.csv`, `obs/`, `records/`, `config.json`), so
that `visualize_responsibility.py --record` draws SMART's motion set and
courtesy samples as it draws DenseTNT's. Its defaults match the labelled
DenseTNT runs (40 safety and 20 courtesy samples, a window every 0.2 s).

**The paper's pipeline with DenseTNT (every 0.1 s, T_f = 2 s, 40 samples,
CVaR α 0.1, max over neighbours, HMM with BIC over 3–11 levels, Eq. 6),
sequence level:**

| condition | max β_s, target pair | max β_s, paper | HMM verdict (stay) | max β_c |
|---|---|---|---|---|
| mode | 0.51 | 0.49 | 0.52 | 0.51 |
| near 1 m | 0.78 | 0.66 | 0.63 | 0.55 |
| fair-inf | 0.85 | 0.74 | 0.69 | 0.57 |
| CAT | 0.93 [0.90, 0.96] | 0.85 | 0.72 [0.65, 0.78] | 0.56 |
| victim | 0.88 | 0.81 | 0.67 | 0.56 |
| max-gap (timid) | 0.36 | 0.58 | 0.53 | 0.48 |

Against the criteria fixed before the test, the negative control and the
grading pass, the rest fails. At 10% false alarms on logged drives the HMM
verdict detects 15% of CAT's attackers. Attacker against victim is 0.59
[0.51, 0.66] (criterion ≥ 0.8), and the timid side does not show (0.49 in
the 61 scenes where the logged pair comes within 10 m). The signal is in
β_s toward the partner and is lost downstream: in the maximum over
unrelated neighbours, and in the HMM, whose Gaussian emissions fit the half
of the frames with β_s exactly 0 badly, whose BIC picks 9 or 11 levels
depending on the fold, and whose courtesy levels persist (stay probability
0.77–0.93) longer than a 2–3 s attack.

Why, measured:

- **β_s is symmetric.** D_g is a shared distance, and CAT aims at the
  victim's logged path, so any alternative of the victim widens the gap
  too. β_s says the interaction was avoidable, not who caused it.
- **Horizon and inertia.** CAT commits at step 10 and the median first
  overlap is 4.3 s later, outside T_f = 2 s. Once the attack is in the
  1.1 s of history, DenseTNT continues it: 0–1 s before the overlap, 61% of
  the attacker's own alternatives also come within 3 m of the ego, and on
  average they keep only 0.40 m more distance (`probe_inertia`).
- **No reaction.** With the attacker 0.5 s from impact, DenseTNT's
  prediction of the ego is unchanged by it (median progress change +0.07 m,
  courtesy KL 0.061 against 0.042 nats in the log; higher in only 57–62% of
  paired frames), so β_c carries no signal (`probe_courtesy`).
- **Few braking alternatives** (`probe_diversity`, 40 i.i.d. samples,
  windows at ≥ 3 m/s): members slowing to r2 < 0.75 (2 s progress over
  constant speed) carry 1.9% (SDC logged), 3.1% (adversary logged) and
  6.1% (CAT attacker) of the probability, and 92% / 91% / 78% of the
  windows have none. DenseTNT follows braking already in the history
  (r2 0.82 predicted vs 0.80 driven, 233 windows) but misses its onset
  after k (0.96 vs 0.86, 27 windows): an accurate predictor, not a
  generator of alternatives.
- **Not the motion set.** Sampled and NMS sets give the same AUROC (0.94),
  and the valid-counterfactual filters change the AUROC of every condition
  by at most 0.012 (near 1 m 0.770 → 0.782).

What helps, from the offline tables (`analyze_aggression_offline`,
`aggression_hmm_schemes`; NMS set, target pair, 0.5 s windows):

| change | effect |
|---|---|
| target pair instead of max over neighbours | CAT 0.80 → 0.94; logged false alarms 10% → 1.9% |
| two-sided score rel = β_s(a→b) − β_s(b→a) | attacker vs victim 0.59 → 0.64 at 2 s; 0.77 → 0.89 at 5 s |
| horizon 2 → 5 s | the attacker's alternatives keep 0.34 → 2.2 m more |
| threshold at the logged q90 of rel (5 s) instead of the HMM | scenes flagged: logged 7–16%, CAT 59%, victim 10% |
| three-level HMM on signed-log rel | attacker 0.90, but 39–50% of logged windows land in the aggressive level |

The dose-response check agrees: re-timing only the SDC toward its partner
raises its own maximum β_s by 1.50 m and the partner's by 0.47 m (2 s).

**SMART (CAT-K, `clsft_E9`) instead of DenseTNT**, every 0.5 s, the same
settings and evaluation (the adversaries still come from DenseTNT):

| sequence AUROC vs logged | DenseTNT | SMART |
|---|---|---|
| CAT attacker, max β_s (paper) | 0.84 | 0.94 [0.90, 0.97] |
| CAT attacker, max β_c | 0.55 | 0.81 [0.75, 0.86] |
| CAT attacker, HMM verdict | 0.70 | 0.84 |
| detected at 10% false alarms (max β_s) | 51% | 86% |
| victim, max β_s | 0.81 | 0.84 |
| attacker vs victim, max β_s | 0.59 | 0.69 [0.61, 0.76] |
| mode (control), max β_s | 0.49 | 0.56 |

SMART reacts to the attacker: it predicts the ego to hold back by 0.72 m
(median) 0.5 s before the overlap, against +0.07 m for DenseTNT
(`probe_smart`). β_c becomes informative and partly attributes (attacker
0.81, victim 0.61). The price is about five times the compute (median 64 s
per scene on an L4, 10 h for 500 scenes, against 2 h), catk's separate
environment, thresholds to re-calibrate (SMART's β of CAT's adversaries is
about a third of DenseTNT's), and its unexplained courtesy bias for the SDC
(median 0.54 nats against 0.06 for adversaries), which inverts the HMM's
attacker-vs-victim result (0.35) because the HMM is fitted on both agents.
SMART draws all scenes' samples from one seeded random stream, so a run
over a subset of the scenes gives other per-frame values (in
`probe_smart --limit 3`, the progress change of one frame moved by 1.9 m);
a run over the same scenes repeats exactly, and the medians over the 100
scenes are what is reported. The fair-inf and max-gap conditions have not been run with
SMART.

**Courtesy as demand instead of the KL** (`responsibility/demand.py`;
`compute_responsibility --courtesy-estimator demand`, or
`compute_demand.py` + `evaluate_demand.py` for the table). The KL needs the
model to react to the agent, which DenseTNT does not. β_d asks DenseTNT only
for b's plans without the agent and lets the agent's actual future decide how
hard b would have had to brake (definition in the table at the top). Same
100 scenes, DenseTNT, NMS set of 40, 0.5 s windows before the first
overlap, sequence AUROC against the same agent's logged drive:

| condition | max β_d | max β_c (KL) | max β_s target |
|---|---|---|---|
| mode (control) | 0.50 | 0.46 | 0.50 |
| near 1 m | 0.63 | 0.48 | 0.77 |
| fair-inf | 0.79 | 0.49 | 0.85 |
| CAT attacker | 0.88 [0.83, 0.93] | 0.52 | 0.94 |
| CAT victim | 0.89 | 0.55 | 0.89 |
| max-gap (timid) | 0.26 | 0.43 | 0.36 |
| ego × 0.5 | 0.45 | 0.44 | 0.56 |

- β_d carries the signal the KL lacks: it grows with the grade (0.63 →
  0.79 → 0.88), stays at 0.50 for the control and falls below it for the
  timid adversary.
- It does not attribute: attacker vs victim 0.53 [0.46, 0.59] (β_s target
  0.59). A CAT collision is geometrically symmetric: each side's plan
  without the other runs into the other's actual future. Discounting
  demands on a b that owed the agent the right of way does not change this
  (0.52), nor does the track-based forced yield (0.50).
- It saturates close to the event: 82% of CAT's attackers have a window
  where most plans cannot clear (10 m/s²), against 5–7% of logged drives.
- 14–30% of the windows are not measured because the pair shares a lane.
- β_s toward the target is the better detector of the two (0.94).
- **The paper's own courtesy estimate** (`--courtesy-estimator boltzmann`,
  run `runs/nms_ooi_boltz`, 8 conditions): the paper defines β_c by the KL
  of Eq. 4 but computes it by Eq. 7, which this port had not implemented.
  CAT's attackers 0.99 [0.97, 1.00], fair-inf 0.86, near 1 m 0.61, timid
  0.46, control 0.53; attacker vs victim 0.47. It rises only within its
  2 s horizon, once the agent's actual future overlaps b's samples, and
  saturates near ln 40 = 3.69 nats when P1 falls on the one sample that
  stays clear: close to "a conflict within 2 s". Spearman with β_s over
  1832 windows: 0.33 (β_d 0.26, KL −0.03).
- **The HMM on the Boltzmann estimate** (`evaluate_aggression --config
  nms_ooi_boltz --frame-s 0.5` against `--config nms_ooi_court`, the same
  windows with the goal KL; cross-fitted on the logged drives of even and odd
  scenes). BIC now picks 3 levels in both folds (KL: 5 and 7), one of them
  aggressive, elevated in courtesy. Longest stay in it, sequence AUROC: CAT's
  attackers 0.98 [0.95, 0.99] (KL 0.63), fair-inf 0.83 (0.60), near 1 m
  0.58 (0.56), control 0.53, timid 0.49; 97% of CAT's attackers flagged at
  4% of logged drives (KL: 16% at 8%). Attacker vs victim 0.58 (KL 0.48). The
  HMM adds little to the maximum of the Boltzmann estimate itself (0.99), and
  its aggressive level is reached only within 2 s of the overlap (posterior
  0.86 at 1–2 s, 0.08 at 2–3 s): the estimate's own horizon.
- What DenseTNT adds (`compute_demand --plans cv|logged`, no model): with
  b's constant-velocity continuation as its only plan, CAT's attackers
  score 0.87 (DenseTNT's set 0.88), fair-inf 0.73 (0.79), near 1 m 0.56
  (0.63) and the timid adversary 0.46 (0.26). Most of the CAT signal comes
  from the agent's actual future and the kinematics; the model's spread of
  plans adds the graded cases and the timid side. b's logged future as the
  plan only detects the collision itself (CAT 0.94, near 1 m 0.52): the
  logged b already avoided the agent, so it shows no forced adaptation.

Limits: 100 development scenes; the logged drives include some aggressive
drivers, so the false-alarm rates are upper bounds; the DenseTNT results
score adversaries that DenseTNT itself proposed.

## Duty, cost and attribution: an aggression judge without a motion model

The results above show what the counterfactual scores do not do on their
own: β_s says an interaction was avoidable, not who caused it, and neither
score knows what one driver owed the other. A rule-based review of CAT's 500
scenes (the driving-aggression-judge skill) found the converse: the law
settles who owed what, but hand-written geometry rules ("who moved over")
break on lane branches and rebuilt maps, and hand-set thresholds drift.
`../aggression_generalisation_plan.md` combines the two. An agent is
aggressive toward another when

- it **owed** the other something: to yield (right_of_way.py), to move over
  only with reasonable safety, not to follow too closely (CVC 21658(a),
  22107, 21703, as recalled);
- it **caused** the conflict (`attribution.py`, A1);
- and the **braking that asked of the other** is above what logged drivers
  in the same situation ask: the 0.9 quantile of a natural reference on the
  same data (`calibrate_aggression.py`). There is no hand-set threshold.

The pieces:

- `attribution.py` (A1, no model).
  - A crossing is caused by the yielder that entered first; with no holder,
    the cause is unknown.
  - At a join (one car comes into the other's path ahead of it), the cause
    is the car that moved sideways against the road. The road's direction is
    the median of the lane centrelines and road edges within 6 m of each car,
    not the car's own lane, so a car following a lane that branches across
    the road is seen to move.
  - When following, the follower is the cause.
- `forced_merge.py` (no model): the same-direction counterpart of
  forced_yield. a_req is the deceleration the car behind needs if the car
  ahead brakes at 4 m/s² and the car behind responds after 1 s. For a cut-in
  it is taken over the 1 s after the join; for tailgating, as the level kept
  for 1 s at 5 m/s or more, excusing the 3 s after the other moved in.
- `aggression_judge.py`: the verdict (aggressive, not aggressive or
  abstain), with every demand's quantile in its reference.
- `catk_cache.py`: catk's cached WOMD as scenes, so every tool here also
  runs on catk's rebuilt maps.

```bash
python -m scripts.responsibility.calibrate_aggression --scenes raw_scenes_500 --out-dir logs/responsibility/aggression/cat500 --workers 4
python -m scripts.responsibility.judge_aggression --scenes raw_scenes_500 \
    --calibration logs/responsibility/aggression/cat500/calibration.json --out-dir logs/responsibility/aggression/judge_cat500 \
    --violations ../ooi_labels/new_labels/screen_law_us_v4.csv --labels ../aggression_agent/runs/cat500/labels.csv
python -m scripts.responsibility.make_cut_in --scenes raw_scenes_500 --out-dir logs/responsibility/aggression/cut_in_scenes --workers 4
S=logs/responsibility/aggression/cut_in_scenes
python -m scripts.responsibility.evaluate_aggression_sets --out-dir logs/responsibility/aggression/eval_cut_in \
    --calibration logs/responsibility/aggression/cat500/calibration.json \
    --sets log=$S/log h0.6=$S/advance_h0.6 h0.4=$S/advance_h0.4 h0.25=$S/advance_h0.25
python -m scripts.responsibility.sensitivity_cut_in --scenes raw_scenes_500 --sets log=$S/log h0.6=$S/advance_h0.6 \
    h0.4=$S/advance_h0.4 h0.25=$S/advance_h0.25 --labels ../aggression_agent/runs/cat500/labels.csv \
    --out logs/responsibility/aggression/sensitivity_cut_in.md
# E4, left-hand traffic: mirror the scenes, then rerun every step above on them with --traffic-side left
python -m scripts.responsibility.mirror_scenes --scenes raw_scenes_500 --out-dir logs/responsibility/aggression/mirror/raw_scenes_500
bash scripts/responsibility/run_attribution_e1.sh     # E1 on the server: A1 next to rel for attacker and victim
```

**Measured on CAT's 500 scenes (2026-10-29, CPU, no model).**

*A1 against what is known.*

- 9 scenes whose actor was clear in the skill's review (237, 293, 342 and
  483: the SDC cut in; 156, 219, 250, 337 and 496: the OOI did) are all
  attributed right. They include 293, 337 and 483, where the car follows a
  lane branching across the road (the skill's screen missed them).
- Against the 40 cut-in and tailgate verdicts of the reviewing agents
  (`aggression_agent/runs/cat500`), A1 names the same actor in 37.
  - 290 and 437 (a car half moving in behind the SDC): A1 reads both or
    neither as having moved.
  - 482: the agents call the OOI a tailgater, but A1 has the SDC cutting in
    ahead of it, which excuses the short gap. The agents themselves excused
    it in 342, the same recording cut later.

*Natural references (calibration).*

| kind | references | q0.9 threshold |
|---|---|---|
| crossing | 17 logged yielders that went first | 2.54 m/s² |
| cut-in | 225 logged joins | 5.32 m/s² |
| tailgate | 2549 logged following episodes at 5 m/s or more | 4.68 m/s² |

*The judge on the SDC–OOI pair, both ways (1000 judgements).*

| verdict | SDC | OOI |
|---|---|---|
| aggressive, crossing | 0 | 2 |
| aggressive, cut-in | 5 | 8 |
| aggressive, tailgate | 2 | 11 |
| abstain | 132 | 133 |
| not aggressive | 361 | 346 |

- **Abstentions** are crossings without a holder (right_of_way.py names one
  for few CAT pairs; see Limitations) and joins with no road direction.
- **Against the reviewing agents' labels:**
  - both call 16 aggressive; the agents alone 24 (21 of them cut-ins, the
    quantile question below); the judge alone 12;
  - 8 of the judge's 12 are tailgates the agents passed, at 0.6 to 1.3 s of
    headway, closing on a slower car. Two of them (50, 231) the skill's first
    labeller had flagged.
  - 2 are cut-ins: 1 and 449, identical numbers (0.44 s, 0.5 m/s faster), so
    probably one recording cut twice.
  - 2 are crossings: 488, where the OOI pulled out of a stop sign in front of
    the SDC (demand 6.3 m/s²), and 283, a low-speed all-way stop (see
    Limitations).
- `../ooi_labels/adjudication_E5.csv` lists the 36 disagreements, the 16
  agreed positives and samples of the agreed negatives (30) and the
  abstentions (15) for a person who knows the traffic code to decide (E5).

*E2, crossings* (make_go_first on CAT's scenes; threshold the 0.9 quantile of
the 11 logged yielders that went first, 2.38 m/s²).

| set | pairs | A1 names the agent | cost above | both (flagged) |
|---|---|---|---|---|
| logged, holder first | 88 | 5% | 8% | 1% |
| delay, PET 2 s | 5 | 100% | 20% | 20% |
| delay, PET 1 s | 10 | 100% | 50% | 50% |
| logged, yielder first (reference) | 11 | 100% | 9% | 9% |

- The attribution gate cuts the cost-only false alarms on lawful logged
  pairs from 8% to 1%.
- Detection is lower than on held-out WOMD in "Forced yields" above (83% at
  PET 2 s). There the threshold came from 42 logged yielders (1.53 m/s²);
  here only 11 give it.
- None of the crossing screen's 16 right-of-way violations that cost the
  other nothing is judged aggressive, and A1 names the violator in all 16.

*E3, cut-ins* (make_cut_in: 42 logged lane changes into the next lane with
1–3 s of headway, the car behind kept to its lane; the car behind advanced to
the target headway).

| set | pairs | A1 names the merging car | flagged |
|---|---|---|---|
| logged (1–3 s) | 42 | 100% | 5% |
| headway 0.6 s | 25 | 88% | 40% |
| headway 0.4 s | 25 | 88% | 60% |
| headway 0.25 s | 24 | 92% | 67% |

The misses are mostly merging cars that are faster than the car behind: a_req
forgives them unless the gap is almost gone. The front car's assumed braking
is the free number (`sensitivity_cut_in.py`). Each threshold is recalibrated
on the 225 logged joins.

| assumed braking | q0.9 threshold | logged flagged | 0.6 s | 0.4 s | 0.25 s | agents' 34 cut-ins also flagged |
|---|---|---|---|---|---|---|
| 4 m/s² | 5.32 | 2% | 40% | 60% | 67% | 11 |
| 6 m/s² | 9.82 | 0% | 40% | 72% | 75% | 11 |
| 8 m/s² | 17.31 | 0% | 48% | 72% | 79% | 11 |

The braking matters little. What decides is the reference quantile: a merge
at 0.6 s is ordinary in these logs (the skill's review found merging with
about a second of headway ordinary in dense urban traffic). The agents' rubric
(an absolute 0.8 s, 0.5 s floor) flags three times as many cut-ins as the 0.9
quantile does. Which is wanted is a normative choice, not a fit. Decided
2026-10-29: the judge keeps the relative q0.9 for cut-ins and tailgating, so
that every dataset and region sets its own threshold. A cut-in counts as
aggressive only when it is unusual for the traffic it happens in. The E3 target
(at least 80% at 0.5 s) was written for an absolute standard and does not
apply.

*E4, catk's rebuilt maps (306 validation scenes), recalibrated, not
retuned.* The same commands run on `catk/data/womd_cache/validation`
(`catk_cache.py`).

- **Thresholds.** Tailgating's threshold barely moves (4.60 against
  4.68 m/s²). The cut-in threshold rises from 5.32 to 6.20 m/s²: 111 joins,
  with a heavier tail (p95 10.3 against 5.9). Crossings give 1 reference,
  which is unusable: few catk scenes mark an SDC–OOI pair (201 of 306 mark
  none).
- **A1 on the rebuilt map.** It found a road direction for every join
  (median coverage 1.0) and named a cause in all 19 joins of the marked
  pairs.
- **Cut-in ground truth** (26 logged lane changes):

| set | pairs | A1 names the merging car | flagged, catk's threshold | flagged, CAT's threshold |
|---|---|---|---|---|
| logged (1–3 s) | 26 | 100% | 12% | 12% |
| headway 0.6 s | 17 | 94% | 29% | 35% |
| headway 0.4 s | 16 | 94% | 38% | 50% |
| headway 0.25 s | 15 | 100% | 53% | 67% |

Recalibrating made the cut-in decision stricter on catk, not better. With a
few dozen pairs these shares move by a pair or two (one pair is 4–7 points),
so E4 neither confirms nor refutes the plan's tolerance of 10 points. The 12%
of logged lane changes over the threshold (3 of 26) were not looked at.

*E4, left-hand traffic by mirroring.* `mirror_scenes.py` mirrors CAT's 500
scenes left to right (y → −y; a lane's left and right neighbours swap). The
mirrored scenes are judged with `RightOfWayParams(traffic_side="left")`
(`--traffic-side left` in every script): the turn across oncoming traffic is
the right turn, and a driver yields to the vehicle on the left. Nothing else
changes. The run tests whether the judge depends only on the duty table. It
cannot test how well that table fits a real left-hand country (it is the same
drivers).

| check | original, right-hand table | mirrored, left-hand table |
|---|---|---|
| right-of-way case and holder (validate_right_of_way) | 500 pairs | same in 497; all 144 holders kept |
| calibration (n, q0.9): crossing / cut-in / tailgate | 17, 2.54 / 225, 5.32 / 2549, 4.68 | identical |
| E2 flagged: log / PET 2 s / PET 1 s | 1% / 20% / 50% | identical |
| E3 flagged: log / 0.6 / 0.4 / 0.25 s | 5% / 40% / 60% / 67% | identical |
| judge, SDC–OOI both ways | 1000 judgements | same verdict and kind in all 1000 |

- **The three changed cases are pairs without a holder.** 271 and 347 swap
  between oncoming and uncontrolled, and 5 between no-conflict and
  same-direction, so the approach angles probably sit on a threshold.
- **Control.** The right-hand table applied to the mirrored scenes changes
  21 of the 144 holders (13 uncontrolled pairs become left turns, 9 left
  turns become uncontrolled, and 6 all-way-stop ties go the other way). So
  the table, not luck, carries the result.

*More crossing references (`--crossing-pairs all`).* Taking every pair of
moving vehicles whose tracks cross, not only the marked SDC–OOI pair, adds
few references:

| scenes | references, marked pair only | all pairs | q0.9 threshold, all pairs |
|---|---|---|---|
| CAT 500 | 17 | 27 (7 without the SDC) | 2.54 m/s² (unchanged) |
| catk 306 | 1 | 17 | 1.24 m/s² (p95 infinite) |

With the CAT threshold from all pairs, E2 flags 40% at PET 1 s (50% before:
one pair of 10) and the same at PET 2 s and on the log. Few pairs are both
given a holder by right_of_way.py and passed by the yielder first, so the
number of scenes limits the references, not the choice of pairs. Held-out
WOMD on the server is the way to get more.

*E1, locally.* Only the logged and re-timed sets run here (DenseTNT's
adversaries need the server). Re-timing the SDC changes its path's timing,
not its course, so A1 rarely changes:

- the SDC is named the cause in 13% of logged scenes;
- in 11% at speed × 1.3, 4% at × 0.7 and 2% at × 0.5.

`run_attribution_e1.sh` computes A1 for every adversarial set and
`analyze_aggression_offline --report attribution` sets it against rel.

Limits:

- **Small, development-set references.** CAT's 500 scenes give 17 crossing
  references, and 27 from all pairs. Held-out WOMD is on the server.
- **Low-speed crossings.** At an all-way stop a holder creeping at 2–3 m/s
  gets an infinite a_req: its 1 s response time covers the distance to its
  zone (scene 283).
- **Rewrites are open loop.** The car behind does not react.
- **The checks against the agents are not checks against ground truth.**

## Responsibility-constrained adversarial generation

CAT picks, among DenseTNT's 32 trajectories for the adversary, the one
maximising P(OV)·P(AV)·P(collision): whichever likely trajectory hits the ego.
Often that is the adversary driving into the ego, a crash the adversary is to
blame for and the ego could not have prevented.
`responsibility/adversarial.py` also scores each candidate j by the
adversary's **safety responsibility toward the ego**: the CVaR, over the
adversary's own DenseTNT motion set, of how much more distance to the ego its
alternatives would have kept than candidate j, over the full 8 s CAT checks,
averaged over the ego trajectories CAT keeps. Low β means the adversary would
plausibly drive this way anyway, so a collision is one the ego has to handle.

| `--adv_selection` | rule |
|---|---|
| `cat` (default) | CAT's original generator, unchanged |
| `constrained` | the highest collision score among candidates with β ≤ `--resp_threshold` (m); if none of them collides, the closest approach among them; if none qualifies, the least responsible candidate |
| `penalized` | argmax collision score · exp(−max(β, 0) / `--resp_penalty`) |
| `fair` | like `constrained`, but a candidate must also be avoidable for the ego: avoidability ≥ `--resp_avoid` (ρ, default 0.3); if none qualifies, the most avoidable candidate within the threshold |

**Ego avoidability** of a candidate is the share of the ego's own DenseTNT
motion set at the generation step that never touches the adversary driving
it (footprints covered by three circles, over the full 8 s). That motion set
is what drivers with the ego's history would do. An avoidability near 0
means a crash whatever a human in the ego's place does: nothing a policy
could learn to avoid. The ego's alternatives do not react to the adversary,
so this underestimates what a reacting driver could avoid.

`cat_advgen.py` and `cat_RLtrain.py` take these options (e.g.
`python cat_RLtrain.py --mode cat --adv_selection fair --resp_threshold 1.0 --resp_avoid 0.3`,
a run named `cat_fair1_0.3`);
the generator is a drop-in subclass of CAT's `AdvGenerator`, and
`cat_advgen.py` prints the collision rate and mean adversary responsibility of
the chosen trajectories at the end. Courtesy responsibility is not used here:
DenseTNT conditions on history only, so the adversary's influence on the ego's
goals is the same for every candidate.

Offline comparison (no MetaDrive), open-loop against the logged ego:

```bash
python -m scripts.responsibility.benchmark_advgen --n 50 --thresholds 0.5 1 2 --avoid 0.1 0.3 0.5 \
    --out logs/responsibility/advgen_benchmark.csv --plot logs/responsibility/advgen_tradeoff.png
```

It reports, per rule:
- how often the chosen trajectory is predicted to collide;
- the chosen adversaries' responsibility and ego avoidability;
- the share of scenes with an *unavoidable* collision (avoidability < 0.1).

It also reports the logged adversaries' own responsibility and avoidability,
to calibrate `--resp_threshold` (τ) and `--resp_avoid` (ρ) on. On all 500
scenes (τ = ∞ keeps every candidate, so `fair` at τ = ∞ constrains
avoidability alone: the ablation of β):

| rule | predicted collision | adversary β mean / median | ego avoidability | unavoidable collisions |
|---|---|---|---|---|
| `cat` | 95% | +5.00 / +5.14 m | 0.30 | 40% |
| `penalized` (1 m) | 95% | +3.98 / +3.96 m | 0.39 | 26% |
| `constrained` 2 m | 33% | +1.01 / +1.18 m | 0.65 | 14% |
| `constrained` 1 m | 24% | +0.32 / +0.29 m | 0.71 | 10% |
| `constrained` 0.5 m | 18% | +0.01 / +0.00 m | 0.77 | 6% |
| `fair` 2 m, ρ 0.1 | 25% | +0.87 / +1.05 m | 0.73 | 0.4% |
| `fair` 2 m, ρ 0.3 | 21% | +0.78 / +0.96 m | 0.79 | 0.4% |
| `fair` 1 m, ρ 0.3 | 15% | +0.17 / +0.12 m | 0.82 | 0.4% |
| `fair` ∞, ρ 0.3 (avoidability only) | 63% | +3.86 / +3.55 m | 0.62 | 0.4% |
| `fair` ∞, ρ 0.5 (avoidability only) | 49% | +3.39 / +3.03 m | 0.75 | 0.4% |
| logged adversaries | – | median −0.05, q90 +0.23 m | median 0.97, q10 0.52 | 2% (avoidability < 0.1) |

CAT's adversaries give up metres of margin their own alternatives would have
kept: a mean of 5 m, against a q90 of 0.23 m for the logged adversaries.
In 40% of the scenes they produce a crash that no plausible ego motion
escapes, against 2% of the logged adversaries. Bounding the adversary's
responsibility removes most of those crashes, and `fair` all but 2 scenes
(0.4%: no candidate there is avoidable enough). Each constraint costs attack
success; the collisions that remain are the ones the ego can and has to
handle.

`--plot tradeoff.png` draws this trade-off: predicted collision rate
against the chosen adversaries' mean β, and against the share of
unavoidable collisions, one curve over τ for `constrained` and for `fair` at
each ρ. The summary ends with the `fair` settings recommended for RL. A
setting must keep a predicted collision rate of `--min-collision` (30%);
among those, the ones with the fewest unavoidable collisions come first,
then the ones whose adversary is least responsible.

On 500 scenes no finite τ up to 2 m reaches 30%. `fair` peaks at 25% (τ 2 m,
ρ 0.1), and at the logged adversaries' q90 (τ ≈ 0.25 m) every rule stays at
8–16%. A fair adversary that is as responsible as logged drivers rarely
produces a collision against the logged ego. Avoidability alone (τ = ∞)
keeps 49–78% collisions with almost no unavoidable ones, so the
recommendation falls on it, but those adversaries are about as
irresponsible as CAT's (β +3.4 to +4.4 m). Avoidability makes a collision
solvable; β makes the adversary drive like people do. Which of the two
training needs is what the RL experiments have to show.
It also checks that the `cat` rule reproduces `AdvGenerator.generate` exactly
(CAT's own code run on the same candidates), and it runs the drop-in
generator the way CAT's scripts call it. Candidates are the adversary
predicted on its own; CAT batches it with the ego, which moves the prediction
slightly (see Design decisions), so every rule chooses among the same
candidates.

### Adversarial scenes as files

`scripts/responsibility/export_adv_scenes.py` writes the adversarial
counterpart of every scene. It runs CAT's generator against the logged ego,
as `cat_advgen.py`'s second round does, but without MetaDrive and keeping
the result as data.

- **What changes.** Each file is a copy of the original scene in which only
  the adversary's track from step 11 on is replaced by the generated plan:
  position, heading, velocity and validity. Steps 0–10 are the history CAT
  plans from.
- **Where it goes.** The files are `OUT/<rule>/<scene>.pkl`, named like
  `raw_scenes_500`, so `Scene.load`, `compute_responsibility.py`,
  `visualize_responsibility.py --scene` and MetaDrive read them as they are.
- **What is recorded.** `metadata.adversary` and `OUT/<rule>.index.json`
  (next to the folder, which holds only scenes) hold the rule, the chosen
  candidate, its predicted collision score, the adversary's β, the first
  step at which the plan overlaps the logged ego and the smallest gap to
  it. MetaDrive asserts that every file in a scene folder is a `.pkl`
  file, so an index that an older export left inside the folder is moved
  out when the export resumes into it, and `verify_adv_export` reads either
  place.

With `--rule cat` (500 scenes, `adv_scenes/cat`, about 1 s a scene on the
GPU), the plan overlaps the logged ego in 476 scenes (95%). The first overlap
comes at a median step of 53, and the adversary's median β is +5.14 m, as
in the benchmark above.

```bash
python -m scripts.responsibility.export_adv_scenes --out-dir adv_scenes                     # CAT's rule
python -m scripts.responsibility.export_adv_scenes --rule fair --resp_threshold 2 --resp_avoid 0.1 --out-dir adv_scenes
python -m scripts.responsibility.visualize_responsibility --scene adv_scenes/cat/0.pkl --agent adv \
    --out-dir adv_scenes/videos/cat/0_adv
```

**Offline and in MetaDrive.** The export imports no MetaDrive module. It
does differ from CAT's generation in MetaDrive in one input, the vehicle
sizes CAT uses to decide which candidates hit the ego:

- **Offline:** the sizes logged at step 10.
- **In MetaDrive:** the ego is always the default vehicle (4.51×1.85 m). An
  adversary 4–5.5 m long gets one of three vehicle models in turn, through a
  counter over the whole session (`even_sample_vehicle_class`), so its size
  depends on what was spawned before it.

`scripts/responsibility/verify_adv_export.py` runs `cat_advgen.py`'s two
rounds in MetaDrive and compares its plans with the exported ones. The check
covered scenes 0–2 and every third test scene from 400 to 487, 33 in all:

| sizes used offline | same plan as MetaDrive |
|---|---|
| the logged sizes (what the export uses) | 27 of 33 |
| MetaDrive's ego, the logged adversary | 29 of 33 |
| the sizes MetaDrive used, for both | 32 of 33 |

The sizes are therefore the only real difference. The remaining scene (451)
matched with the logged sizes, which suggests two candidates scored almost
equally. MetaDrive's adversary size cannot be known from the scene alone, so
the export keeps the logged sizes.

In these scenes the ego is the logged one and does not react to an
adversary that was not there. Its own β_s toward the adversary is therefore
often positive as well: in scene 0 it reaches 1.2 m at 3.5 s, because some
of its alternatives would have kept more distance.

### Swapping the ego and the adversary

CAT takes the ego from `metadata.sdc_id` and the adversary from the other
object of interest. MetaDrive spawns its ego vehicle, its route and the
replay policy from the same track, and steers only replayed traffic as an
adversary. So the roles are swapped in the scene data, not in advgen.
`swap_roles` writes a copy of the scenes in which the other object of
interest is the self-driving car (`sdc_id`, `sdc_track_index`) and the
logged self-driving car is the adversary (`responsibility/swap.py`):

```bash
python -m scripts.responsibility.swap_roles --scenes raw_scenes_500 --out-dir raw_scenes_500_swapped
python cat_advgen.py --scenes_dir raw_scenes_500_swapped                        # CAT's benchmark, roles swapped
python cat_RLtrain.py --mode cat --scenes_dir raw_scenes_500_swapped --seed 0  # run name cat_swapped_...
python -m scripts.responsibility.export_adv_scenes --scenes raw_scenes_500_swapped --out-dir adv_scenes_swapped
python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500_swapped --use-ooi \
    --out-dir logs/responsibility/swapped/sdc
```

- **Eligibility.** A scene is swapped only if the other object of interest
  is a vehicle valid at every step, because MetaDrive spawns the ego at
  step 0 and drives its whole logged route. 459 of the 500 scenes qualify.
  In 24 the other object is missing at step 0, and in 17 its track has
  gaps.
- **Layout.** The swapped scenes keep their file names. The folder holds
  only scenes, because MetaDrive asserts that every file in it is one; the
  index of swapped and skipped scenes is
  `raw_scenes_500_swapped.index.json`, next to it.
- **Train/test split.** `cat_RLtrain.py --scenes_dir` counts CAT's split
  over the files present: scenes 0–399 train, 400 on test. That is 369 / 90
  for the swapped folder. The evaluation runs once over the test scenes.
- **Run names.** A swapped folder adds `_swapped` to the run name.
- **Other scripts.** `cat_advgen.py --scenes_dir` runs over every scene in
  the folder. The other scripts read any scene folder with `--scenes`.

**Checked in MetaDrive** on this branch. On swapped scenes 0–2, the ego
follows the new self-driving car's log exactly (0.00 m at step 20), and the
original one is replayed traffic. In CAT's two-round generation on scenes
0–5, every adversary is the original self-driving car. The attack hits
4 of 6 swapped scenes and 6 of 6 original ones, too few scenes to compare
rates. A 300-step `cat_RLtrain.py` run on the swapped folder ran as
`cat_swapped_MDWaymo-seed99`.

### Near misses instead of collisions

`--adv_selection near` keeps CAT's generation, the same 32 DenseTNT
candidates for the adversary, but chooses a near miss instead of a
collision. For each candidate, gap_j is the smallest gap between the
adversary's and the ego's footprints over the 8 s CAT plans. Each footprint
is three circles, checked every 0.1 s, and gap_j is averaged over the ego
trajectories by P(AV_i). The rule then works as follows:

1. Drop every candidate that touches the ego, meaning CAT's own test
   predicts a collision, or the gap to any ego trajectory is ≤ 0.
2. Of the rest, take the most probable candidate whose gap is within
   `--near_gap ± --near_tol` (m). Defaults: 1.0 ± 0.5.
3. If no candidate is in that band, take the one closest to `--near_gap`.
4. If every candidate collides, take the one with the largest gap.

The circles are slightly larger than the car (0.28 m at the side of a
4.8 × 2 m car), so gap_j understates the true gap a little.

```bash
python cat_advgen.py --adv_selection near --near_gap 1.0 --near_tol 0.5
python cat_RLtrain.py --mode cat --adv_selection near --near_gap 1.0 --seed 0     # run name cat_near1_0.5_...
python -m scripts.responsibility.export_adv_scenes --rule near --near_gap 1.0 --out-dir adv_scenes
```

**Measured on the first 30 scenes.** The adversary is planned against the
logged ego, as in `export_adv_scenes`. Its closest approach is then
measured against the logged ego, and the scenes are replayed in MetaDrive
(logged ego, generated adversary):

| rule | overlap | closest approach P10 / median / P90 | within target ± 0.5 m | ego collisions in MetaDrive | adversary β_s (median) |
|---|---|---|---|---|---|
| cat | 100% | −2.64 / −1.42 / −0.73 m | – | 23 / 30 | 3.40 m |
| near 0.5 m | 0% | 0.18 / 0.61 / 1.45 m | 70% | 0 / 30 | 0.86 m |
| near 1.0 m | 0% | 0.57 / 1.20 / 1.62 m | 77% | 0 / 30 | 0.34 m |
| near 2.0 m | 0% | 0.83 / 1.84 / 2.27 m | 73% | 0 / 30 | −0.01 m |

- **Outside the target band.** In 23–30% of the scenes no candidate falls
  within the band. The closest one is taken, which accounts for the tails
  of the gap distribution.
- **Responsibility.** The farther the miss, the less responsible the
  adversary: its safety responsibility goes from 3.40 m for CAT's
  collisions to about 0 at 2 m.
- **In training.** The ego is the policy, not the log, and it reacts. The
  realised gap is that of the policy's own trajectory, against which the
  candidates were not scored: CAT scores them against the ego's past
  rollouts.

## Driving policies (rollouts)

The same measurements apply to a policy driving in MetaDrive. A **rollout**
(`responsibility/rollouts.py`; recorded by `collect_rollouts.py` on the
server) holds what happened in one scene: the ego's simulated states per
0.1 s step, the adversary's if CAT placed one, which logged objects the
simulator spawned, and how the episode ended. `--rollouts DIR` plays each
rollout back into its scene and queries the simulated ego:

```bash
python -m scripts.responsibility.compute_responsibility --rollouts rollouts/td3_cat/none \
    --out-dir logs/responsibility/policies/td3_cat/none
```

In the rebuilt scene the ego (and the adversary) follow the simulation and
end with the episode; objects the simulator never spawned are removed
(CAT runs MetaDrive with `no_static_vehicles`, which drops every vehicle
whose logged positions spread less than 3 m); everything else keeps its
logged states, since CAT's traffic is not reactive. Windows run up to a
collision (the window must see it), or stop a full metric horizon before
any other end of the episode (what the ego would have done afterwards is
unknown). Outputs are those of a logged run, so summaries, levels and
records work unchanged.

Removing parked vehicles changes the values (on scene 1, with 70 of them,
safety moves by 0.4 m on average), so compare a policy against **the
replayed log** (`--policy replay` rollouts), not against the plain logged
run: that is the logged driving in the scene the policy actually saw.

### Recording rollouts (on the server, needs MetaDrive)

```bash
# the reference: the logged ego replayed, without and with CAT's adversary
python -m scripts.responsibility.collect_rollouts --policy replay --adversary --out_dir rollouts
# a TD3 policy saved by cat_RLtrain.py --save_model (models/<mode>_s<seed>)
python -m scripts.responsibility.collect_rollouts --policy models/cat_s0 --policy_name td3_cat_s0 \
    --adversary --adv_selection cat --out_dir rollouts
```

Each scene (default: CAT's test split, 400–499) is played the way CAT's
`eval_policy` does it: a normal episode, then, with `--adversary`, an
adversary generated against the ego's own trajectory from that episode
(`--adv_selection` as for `cat_RLtrain.py`) and a second episode with it.
Rollouts go to `rollouts/<policy_name>/none/` and `.../<mode>/`. Two checks
are printed: with `--policy replay`, the recorded ego against the logged one
(should stay below 0.1 m, which also confirms that state i is the log's step
i), and in adversarial episodes how far the adversary left its logged track
and whether it follows its plan with a lag of 0 or 1 step. The recording
logic is tested against a stand-in environment (`tests/responsibility/test_collect.py`).

**Collisions of a replayed ego.** MetaDrive's agent manager moves a
replay policy's vehicle to its next logged pose after the collision check,
and the vehicle's `before_step` then clears that step's flags. So
`crash_vehicle` never reaches `info`, and `crash_vehicle_done` never ends
the episode. `collect_rollouts` therefore reads collisions from
`ego_crash_flag`, which is set by the same check and never cleared; CAT's
`cat_advgen.py` counts attacks by it for the same reason. The episode then
ends at the collision, as it would for a driven ego. Before this fix the
replay reference had no collision at all, although ego and adversary boxes
overlapped in 86 of CAT's 100 test episodes.

**A fix to CAT's adversary replay.** Once CAT's traffic manager has spawned
the adversary, it applies the plan one row per step from its first row.
That only lines up for an adversary present from step 0, which gets row
k − 1 at step k. In 24 of the 500 scenes (8 of the test split) the
adversary appears later in the log, at steps 1–10. The plan's first rows
are zero padding for the steps before it existed, so the adversary sat at
the origin for as many steps and then drove its plan as many steps late. A
gap in its logged history did the same for one step. Both generators now
hand over a `StepAlignedPlan` (`advgen/adv_generator.py`), which applies
the row of the current step and holds the adversary at its nearest logged
state where the log has none. `AdvGenerator.before_episode` also clears
the manager's adversary: a plan left over from an earlier episode used to
drive any later vehicle of the same name, for example in
`eval_policy`'s normal episodes. For an adversary present from step 0
nothing changes. With the fix, every adversary follows its plan exactly one
step late (lag-1 error 0.00 m).

`before_episode` runs only after `env.reset()`, though, and the reset already
steps the traffic manager once with the previous episode's adversary. A
`StepAlignedPlan` is never emptied, unlike CAT's popped list, so in that
first step it moved the next scenario's object of the same name. A same-named
cyclist, which is not in the manager's `v_map`, raised a KeyError that
stopped `cat_share_s1` at 483k steps on 2026-10-07. The plan now has length 0
once the env has moved to another scenario, so the manager no longer applies
it there.

Measured over training scenes 0–399, the adversary's name exists in another
scene 2.9% of the time, which is about 240 one-frame moves per training run.
Only 0.001% of these put the old adversary within 5 m of the new ego's start
(about 0.05 per run). One run crashed on a cyclist. The other 16 runs are
kept.

### Comparing policies: aggressive and timid, and whose fault the collisions were

Responsibility has two directions. β_s > 0 is aggressive: the agent kept
less margin than its own alternatives. β_s < 0 is the opposite: it kept more
margin to everyone than its alternatives would have. A little of that is
ordinary caution; far more than logged drivers ever keep is **timid**, the
failure mode of a policy trained to avoid collisions at any cost.
`summarize_responsibility` now flags both: aggressive above the
`--quantile` (0.9) of the reference's positive values, timid below the
`--timid-quantile` (0.1) of its negative safety values.

Every vehicle collision of a rollout is also **attributed**
(`responsibility/blame.py`, written to `crashes.csv` by
`compute_responsibility --rollouts`). In the 2 s window that ends with the
collision it measures both sides: β_ego, the ego's safety responsibility
toward the other, and β_other, the other's toward the ego. Each comes from
that agent's own DenseTNT motion set. A car that was rear-ended did what its
alternatives do (β ≈ 0), while the follower closed in far more than its
alternatives (β ≫ 0). The collision counts as the ego's fault when its
positive part exceeds the other's by more than 0.1 m, the other's in the
reverse case, and shared otherwise; the ego's share is
w = β_ego⁺ / (β_ego⁺ + β_other⁺).

As a baseline, every attribution also carries the verdict of the **rear-end
rule**, the traffic-law reading of the most common collision: the follower
is at fault. It applies when both travel the same way (headings within 30°)
and the other lies off the ego's front or back rather than its side, and
says nothing otherwise (`crashes.csv`, column `rule`).

The formal rule-based baseline is **RSS** (Responsibility-Sensitive Safety,
Shalev-Shwartz et al. 2017; `responsibility/rss.py`, columns `rss` and
`rss_case`). It finds the danger threshold: the step from which the two
cars stayed closer than RSS's safe distance, both longitudinally and
laterally. From there, both owe the proper response of the axis that became
unsafe last:

- longitudinal: the rear car brakes at least 4 m/s² after a 1 s response
  time, and the front car brakes at most 8 m/s²;
- lateral: both stop closing in laterally.

A car whose speed left that envelope (beyond 0.5 m/s longitudinally,
0.2 m/s laterally) is responsible. A brake-check is therefore the front
car's fault, where the rear-end rule would blame the follower. The other
parameters are ad-rss-lib's defaults.

Lane directions come from the map's nearest lane centreline. RSS covers
same-direction collisions (rear-end, cut-in, side-swipe). Oncoming and
crossing ones need right-of-way rules from the lane graph, so they get
`n/a`, and `compare_policies` reports RSS's coverage.

The **right of way** (`responsibility/right_of_way.py`, columns
`right_of_way`, `right_of_way_case`, `priority`) is the traffic-law
baseline for those: collisions between different paths (crossing, turning
across, entering a road). Same-direction conflicts (cut-ins, lane changes,
merges into one lane) are left to RSS. WOMD was recorded in US cities, so the
rules are the California Vehicle Code's. Everything is read from the
`Scene` (trajectories, lane graph, traffic-light states, stop signs), so
logged scenes and rollouts are judged alike, without MetaDrive.

1. **Conflict.** The two paths, each extended 15 m along its last heading,
   cross (or merge) at P*. Each agent's conflict zone is the stretch of its
   path where its footprint overlaps the other's swept strip.
2. **Priority.** Each agent is followed through the lane graph to P*. Its
   control is the light of the intersection lane it entered, in the state
   it showed at the stop line (WOMD lights and stop signs control the lane
   that starts at the stop line), or a stop sign. The first rule that
   applies names the holder: red-light (21453), driveway (21804),
   protected-arrow, left-turn (21801), first-in at a green light or by
   arrival at an all-way stop, stop-sign (21802), through-road at a T
   junction (21800(c)). With no rule the case says why: signal-unknown,
   uncontrolled, oncoming, same-direction, following. Same-direction pairs
   (approach headings within 45°) get neither a holder nor a verdict; the
   lane-change (21658) and merge rules behind `same_direction_rules` failed
   the held-out check (below).
3. **Duties**, as Signal Temporal Logic over the steps both are seen
   (`responsibility/stl.py`, robustness in metres):
   - `yield(y, p) = G(enters(y) → cleared(p) ∨ can_stop(p))`: the yielder
     enters the zone only when the holder has passed it or can still stop
     before it, after a 1 s response and braking at 4 m/s² (RSS's ρ and
     brake_min);
   - `avoid(x, z) = G(in(z) ∧ can_stop(x) ∧ ¬in(x) → ¬in(x) W ¬in(z))`:
     nobody drives into a zone the other occupies while it can still stop,
     right of way or not.

   A violated duty puts the collision on that agent (both: shared).

The STL is a small numpy module rather than `md-stl-monitor`: that package
evaluates per-agent rules on a world built from ScenarioNet dictionaries,
while these duties are pairwise and need the conflict zone, which lives
here.

Checked on CAT's 500 logged scenes: for the self-driving car and the other
object of interest at their closest approach, logged drivers mostly keep
the right of way, so the holder should usually pass P* first. Of 220 pairs
whose paths cross or merge and where one passed at least 0.5 s before the
other, 113 get a holder, and it passed first in 89% (101; 95% CI 82–94%):

| rule | pairs | holder passed first |
|---|---|---|
| stop-sign | 49 | 94% |
| through-road | 19 | 68% |
| all-way-stop/first-in | 16 | 100% |
| driveway | 10 | 100% |
| red-light | 8 | 88% |
| left-turn | 7 | 100% |

Most disagreements are a yielder that went first lawfully while the holder
was still far away. The other 107 pairs get no holder: signal-unknown 38
(no state for that light), uncontrolled 23, same-direction 20,
all-way-stop/order-unknown 17 (both waiting when the clip began), oncoming
9. The
CVC's order for uncontrolled intersections (first-in, then yield to the
right, 21800(a)/(b)) is off by default (`uncontrolled_order`): the map shows
no control there, but in reality such junctions mostly have signs or lights
WOMD lacks, and the holder it names passed first in only 40% (15 pairs) and
25% (8 pairs). On the logged pairs the yield duty fails for 2 of 112
yielders (2%), the duties' false-alarm rate on lawful driving.

These 500 scenes were the development set: the driveway, through-road,
oncoming and all-way-stop rules were added after looking at their
disagreements. The held-out check runs over the rest of WOMD's
validation_interactive split (converted by `convert_womd_split`, below), with the two
objects of interest as the pair. In the same pass it also evaluates the
rules that are off by default: the CVC's uncontrolled order on the pairs
left "uncontrolled", and the same-direction rules on the pairs left
"same-direction":

```bash
python -m scripts.responsibility.validate_right_of_way --scenes ~/womd_v1_2_1/cat_format/validation_interactive/scenes \
    --exclude responsibility/predictors/unitraj_configs/cat_scenario_ids.txt --workers 32 \
    --out-dir logs/responsibility/right_of_way/validation_interactive
```

It writes `pairs.csv` (one row per scene; a re-run resumes) and
`summary.md` (per rule: pairs, holder-first rate, 95% Wilson interval, and
how often logged yielders kept the yield duty). On CAT's scenes
(`--scenes raw_scenes_500 --pair sdc`) it reproduces the numbers above in
48 s on 8 processes.

Held out (2026-10-10): the 14 of 150 shards of validation_interactive on
the L4 machine, 3166 scenes, 47 of them CAT's and excluded. Of 1091 pairs
whose paths cross or merge with a passing order, 330 get a holder, and it
passed first in **83%** (275; 95% CI 79–87%), against 89% on the
development set. Logged yielders kept the yield duty in 316 of 323 (2%
false alarms).

| rule | pairs with a holder and an order | holder passed first | yield duty kept |
|---|---|---|---|
| stop-sign | 130 | 92% | 123 of 127 |
| driveway | 62 | 74% | 61 of 61 |
| through-road | 40 | 85% | 38 of 39 |
| all-way-stop/first-in | 35 | 91% | 35 of 35 |
| left-turn | 28 | 64% | 26 of 28 |
| red-light | 23 | 87% | 23 of 23 |
| all-way-stop/yield-right | 11 | 36% | 9 of 9 |
| *same-direction rules:* lane-change | 85 | 38% | 40 of 91 |
| *same-direction rules:* merge | 68 | 97% | 29 of 61 |

The same-direction rules fail there. A lane changer usually goes first,
lawfully, into a gap it judged safe, so "the one not changing lanes goes
first" does not hold (38%); and the conflict zone of the duties assumes
crossing paths, so the yield duty fails for 56% of logged lane-change and
52% of merging yielders. They are off by default (`same_direction_rules`);
RSS's lateral and longitudinal checks cover those conflicts. The CVC's
uncontrolled order stays off as well (first-in 59% of 51, yield-right 32%
of 22). The pairs without a holder are same-direction 352, signal-unknown
235, uncontrolled 73, all-way-stop/order-unknown 67, oncoming 32, plus 612
following one another.

`compute_responsibility --rollouts` writes these columns with every
collision. To add them to the crash files of a finished run, which a re-run
skips, recompute the rule-based verdicts (RSS and right of way) in place.
This needs only the scenes and rollouts, not the model:

```bash
python -m scripts.responsibility.attribute_rules --runs logs/responsibility/policies/*/*
```

```bash
python -m scripts.responsibility.fit_levels --runs P/replay/none P/td3_cat_s0/none P/td3_cat_s0/cat \
    --fit-runs P/replay/none --out-dir P/levels            # P=logs/responsibility/policies
python -m scripts.responsibility.compare_policies --runs P/replay/none P/td3_cat_s0/none P/td3_cat_s0/cat \
    --hmm P/levels/hmm.pkl --out-dir P/compare
```

`compare_policies` writes one row per run (policy × adversary mode) to
`comparison.csv`/`.md`, with thresholds from the first run (or
`--reference`), the replayed log. Columns:

- from the rollouts: crash rate, route completion, arrival and out-of-road rates;
- the ego-fault and other-fault shares of the attributed collisions, and
  their agreement with the rear-end rule, with RSS and with the right of
  way where both decide (coverage, each baseline's own ego-fault share and
  `baseline_coverage`, the share RSS or the right of way decides, in
  `comparison.csv`);
- the share of windows the ego was stopped (below `--min-speed`, not judged);
- the aggressive and timid shares of the judged windows, and each relative
  to the reference ("× ref");
- the median β_s and the p90 of β_c;
- with `--hmm`, the share of windows in each responsibility level.

`comparison.png` shows β_s per run and the level (or aggressive/timid) shares.

Runs of one training setting with different seeds (policy names ending in
`_s<seed>`, as `cat_RLtrain.py` names its models) are averaged in
`comparison_seeds.csv`/`.md`: mean ± standard deviation of every column per
training setting and test adversary. The markdown adds matrices of training
setting × test adversary for the crash rate, the ego-fault share (and
RSS's), route completion and the timid and aggressive shares: the cross evaluation of
policies trained against different adversaries (and penalties), each tested
without an adversary, with CAT's and with the fair one.

#### The reference on the test split

These are the logged egos replayed in MetaDrive on CAT's 100 test scenes,
against each test adversary. The thresholds are calibrated on the run
without an adversary.

| test adversary | collisions (closed loop / offline prediction) | counterfactual ego / other / shared | rear-end rule decides, agrees | RSS decides, agrees | route completion |
|---|---|---|---|---|---|
| none | 0 | – | – | – | 95.8% |
| CAT | 83 / 95% | 22 / 55 / 6 | 25 of 83, 16 of 23 | 42 of 83, 21 of 29 | 67.0% |
| fair (2 m, ρ 0.1) | 16 / 25% | 9 / 6 / 1 | 0 of 16 | 6 of 16, 0 of 1 | 90.2% |
| avoidability only (∞, ρ 0.5) | 38 / 49% | 12 / 21 / 5 | 11 of 38, 6 of 11 | 21 of 38, 6 of 12 | 85.3% |

The "agrees" counts include only collisions where both the
counterfactual verdict and the rule or RSS name a side (ego or other).

The replayed ego cannot react, so a collision with CAT's adversary is the
adversary's doing. The counterfactual verdict says so in two thirds of
them: the adversary's median β is +1.02 m, the ego's +0.21 m. Where the
rear-end rule or RSS decide, they agree with it about 70% of the time, but
they decide only 30% and 51% of these collisions.

The fair adversary's collisions come out balanced: median β +0.22 m for the
adversary and +0.26 m for the ego. That is what it was built for: it
drives no less acceptably than the ego.

An unchanged ego trajectory scores 1.9× as many aggressive windows next to
CAT's adversary as without one (19.0% against 10.1%), because β_s is
measured against what the others actually do. Policies must therefore be
compared under the same test adversary, which is what `comparison_seeds.md`'s
matrices do.

#### Trained policies (all 17 models, 2026-10-08)

These are the results of `run_eval.sh` with `MODELS=models_eval`, on the 100
test scenes with every model under four test adversaries
(`logs/responsibility/eval/compare`). Each value is a mean ± std over seeds:
three seeds each, except one for the two ablations.

| training | crash, CAT adversary | of which ego / other fault | route completion, no adversary | route completion, CAT adversary | arrival, no adversary |
|---|---|---|---|---|---|
| TD3 replay | 42.0 ± 7.5% | 17.3 / 19.0% | 70.3 ± 4.7% | 63.5 ± 7.6% | 48.0% |
| cat | 42.3 ± 5.0% | 15.0 / 22.0% | 66.5 ± 1.6% | 57.2 ± 3.3% | 46.3% |
| cat_share | 39.3 ± 1.5% | 19.7 / 13.7% | 72.3 ± 7.0% | 64.7 ± 6.6% | 50.3% |
| cat_rss | 35.7 ± 2.5% | 13.3 / 21.0% | 68.2 ± 3.5% | 61.6 ± 4.8% | 48.3% |
| cat_fair2_0.1 | 34.0 ± 4.4% | 16.8 / 12.8% | **76.5 ± 1.6%** | **71.9 ± 1.4%** | **56.3%** |
| cat_fairinf_0.5 (1 seed) | 35.0% | 15.0 / 16.0% | 77.2% | 69.6% | 55.0% |
| cat_fair2_0.1_share (1 seed) | 59.0% | 18.3 / 36.6% | 71.3% | 57.8% | 49.0% |

The ego- and other-fault columns are the crash rate times the share of each
verdict; shared verdicts make up the rest.

- **The fair adversary is the one clear effect.** Against `cat` (Welch
  t-test, 3 vs 3 seeds), route completion rises by 10.0 pp without an
  adversary (p = 0.002) and by 14.7 pp against CAT's adversary (p = 0.008).
  Collisions with CAT's adversary fall by 8.3 pp (p = 0.10). The fall comes
  from collisions that are the other's fault (12.8 vs 22.0%); the ego-fault
  collisions stay at 15–17%.
- **CAT's training does not beat replay training.** Against CAT's adversary
  the crash rates are 42.3 vs 42.0% (p = 0.95), and route completion is
  lower (57.2 vs 63.5%).
- **The weighted collision penalties show no significant effect.**
  Against `cat`, `cat_share` is −3.0 pp on crashes (p = 0.41) and +7.5 pp
  on route completion against CAT's adversary (p = 0.18), and `cat_rss` is
  −6.7 pp on crashes (p = 0.13). The collision penalty they weight is small: −1 a step, about 4
  for a collision, against −10 and the end of the episode for leaving the
  road.
- **Behaviour profiles do not tell the settings apart.** Every TD3 policy has
  25–34% aggressive windows (2.5–3.4× the log) and 4–8% timid ones (4–7×).
- **The policies are weak overall.** Without an adversary they still collide
  in 15–21% of episodes (with non-reacting logged traffic) and leave the road
  in 28–37%.
- **Single-seed ablations.** `cat_fairinf_0.5` matches the fair adversary,
  which suggests avoidability matters more than β. `cat_fair2_0.1_share`
  looks like an outlier (59% crashes). Both need more seeds.

#### Reproducing CAT's Table 2 (2026-10-09)

CAT (Sec. 4.3) trains TD3 on the same split as here: scenes 0-399 of these
500 WOMD scenes for training, 400-499 for testing. Training runs for about
1M steps: Fig. 5 ends near 9 x 10^5 interactions and the code defaults to
`max_timesteps=1e6`; Table 3's "10E6" looks like a typo. The table below
uses CAT's own `eval_policy`, which `cat_RLtrain.py` logs every 25k steps
(`logs/<run>_MDWaymo-seed<s>-0/logger.csv`). Each value is the final
evaluation, averaged over 3 seeds:

| | crash, log replay | crash, CAT adversary | route completion, CAT adversary |
|---|---|---|---|
| paper: replay | 19.9% | 43.3% | 63.5% |
| here: replay | 21.3% | 41.7% | 63.2% |
| paper: CAT | 13.4% | **28.2%** | 67.6% |
| here: cat | 15.7% | **39.7%** | 56.8% |

- **The replay baseline and the log-replay drop reproduce.** The replay
  numbers match the paper within 3 pp, and CAT training lowers the
  log-replay crash rate by 5.6 pp (paper: 6.5 pp).
- **The gain against CAT's adversary does not.** The paper's crash rate
  falls by 15 pp; here it falls by 2 pp. The cat runs' curves stay near
  0.40 for the whole run, and their best checkpoints reach 0.32-0.36.

The training path is CAT's code, apart from two changes:

- **What is identical:** `--adv_selection cat` returns CAT's own
  `AdvGenerator`. The hyperparameters are CAT's defaults (M = 32,
  N = 5, alpha = 0.99, min_prob 0.1), and so are the training loop and
  `eval_policy`.
- **What changed:**
  - `StepAlignedPlan`. It only matters for adversaries that appear after
    the first step: 16 of the 400 training scenes and 8 of the 100 test
    scenes, 1-10 steps late.
  - Clearing a leftover adversary plan when an episode starts.
  - `--no_store_map`, which affects memory only.
- **The environment:** MetaDrive is CAT's package, unmodified. The
  dependency versions differ (torch 2.4.1 instead of 1.12), and DenseTNT's
  candidates match CAT's to 3.6e-12 m.

The adversary does act in training. In the second half, adversarial
episodes end in a crash or off the road in 0.68 of cases, normal ones in
0.50.

**A pristine run of CAT's code rules out the two changes.** CAT's code was
checked out unmodified (commit 9de53da, `/home/Cesar/catspond-upstream`,
with the same scenes, MetaDrive and DenseTNT weights). It trained with
README's command `python cat_RLtrain.py --mode cat --seed 0` (1M steps,
11.5 h). Its own evaluations against CAT's adversary:

| crash, CAT adversary | final | last 3 | per 200k steps |
|---|---|---|---|
| upstream cat, seed 0 | 0.35 | 0.33 | 0.48, 0.43, 0.43, 0.36, 0.32 |
| here: cat, seeds 0 / 1 / 2 | 0.44 / 0.38 / 0.37 | 0.40 / 0.37 / 0.40 | roughly flat; seed 1 falls to 0.37 |
| here: replay, seeds 0 / 1 / 2 | 0.33 / 0.45 / 0.47 | 0.34 / 0.45 / 0.46 | flat |

The upstream curve falls as the paper's Fig. 5 does, but it ends where
replay seed 0 also ends. A cross evaluation separates the code from the
models: each final model went through both versions' `eval_policy` on the
100 test scenes.

| crash, CAT adversary | CAT's code | this repository's code |
|---|---|---|
| upstream model | 0.38 | 0.37 |
| cat_s0 | 0.42 | 0.45 |

Swapping the code changes a model's result by 1-3 pp, in no consistent
direction. The same model and code also vary by about 3 pp between
evaluations: the upstream model logged 0.35 during training and 0.38 when
evaluated again. So the changes here do not weaken or strengthen CAT's
adversary.

The differences between runs come from the seeds. The paper's ± (1.1-2.0
pp) is far below the spread seen here between seeds (replay 0.34-0.46).
The paper does not state how many seeds it used. Over 3 seeds of the public
code, CAT training lowers the log-replay crash rate as reported, but not
the crash rate against CAT's adversary (39.7 vs 41.7% for replay).

### Training with a responsibility-weighted collision penalty

In CAT's training environment a collision does not end the episode: at every
step the ego touches another vehicle, MetaDrive replaces that step's driving
reward with −`crash_vehicle_penalty` (1). An adversary that drives into the
ego costs it the same as a collision the ego caused, which teaches it to
avoid driving on rather than to drive well. With `--blame_weighting share`,
`cat_RLtrain.py` keeps only the ego's share of each penalty
(`responsibility/blame_reward.py`):

- a **collision** is a run of consecutive steps in contact with the same
  vehicle, attributed once at its first step, in the scene rebuilt from the
  training episode up to then (the same attribution as above);
- the **weight** is the ego's share w when the verdict is "ego" or "other".
  Otherwise the full penalty is kept (w = 1): when the sides differ by less
  than `--blame_margin` (0.1 m), after the end of the log, when the partner
  is not a predicted vehicle, or when the attribution fails. Doubt keeps the
  penalty, so noise in the attribution does not become noise in the reward;
- at every penalised step of the collision the reward becomes
  w · (−penalty) + (1 − w) · (the driving reward the penalty replaced). w = 1
  is MetaDrive's reward, w = 0 the step as if nothing had been hit.

An episode's transitions go into the replay buffer when it ends, already
weighted. Every attributed collision is logged to
`logs/blame/<run>_s<seed>.csv` (β of both sides, verdict, rule, weight, time),
and the running mean weight is printed at every evaluation. It works with any
adversary (`--mode replay` too). Runs are named with a `_share` suffix
(`cat_share`, `cat_fair1_0.3_share`). An attribution costs two DenseTNT
passes, once per collision; the constrained and fair generators share
their DenseTNT with it.

```bash
python cat_RLtrain.py --mode cat --blame_weighting share --seed 0 --save_model          # cat_share_s0
python cat_RLtrain.py --mode cat --adv_selection fair --resp_threshold 1 --resp_avoid 0.3     --blame_weighting share --seed 0 --save_model                                         # cat_fair1_0.3_share_s0
```

```bash
python -m scripts.responsibility.summarize_blame --logs logs/blame/*.csv --out logs/blame/summary.md
```

summarises those logs per run. It reports the number of collisions, the
share of each verdict and the mean weight. It also shows how often the
full penalty was kept and how often the other was mostly at fault
(w < 0.5), plus the agreement with the rear-end rule, the time per
attribution, and the mean weight over spans of training.

`--blame_weighting rss` is the rule-based baseline (`cat_rss`, …). It
removes the whole penalty when RSS puts the collision on the other car
alone, and keeps all of it otherwise. It needs no DenseTNT. Every
collision's RSS and right-of-way verdicts are logged in both modes, and
`summarize_blame` reports each one's verdicts and its agreement with the
counterfactual one.

The weighting is tested against a stand-in for the training env
(`tests/responsibility/test_blame_reward.py`: a rear-ended ego keeps none of
the penalty; doubt and failed attributions keep all of it).

## Does the verdict depend on the motion model? (SMART)

To run these SMART experiments on another server, see [CATK_H200.md](CATK_H200.md) (in Chinese):
what to pack (`pack_catk_h200.sh`), the two environments, and every experiment's commands.

`scripts/responsibility/run_smart.sh` measures CAT's scenes with CAT-K's SMART
(the `clsft_E9` checkpoint) through catk's own pipeline (`responsibility/SMART_PLAN.md`):
`export_catk.py` writes catk's cache, catk's `compute_responsibility` runs with
`--query interest --neighbor-future hidden --courtesy-estimator exact`, and
`import_catk.py` turns the observations into `sdc` and `adv` runs of this
repository's format, which `compare_models.py` compares with DenseTNT's.
`cache_womd_for_cat.py` builds the same scenes from WOMD v1.2.1 instead
(from `extract_womd_scenarios.py`'s extract of the 497 scenario ids).

```bash
bash scripts/responsibility/run_smart.sh                          # about 10 h on an L4 for the 500 scenes
~/venvs/catk/bin/python -m scripts.responsibility.import_catk --run logs/catk/scenes/run --out-dir logs/catk/scenes/windows
python -m scripts.responsibility.compare_models --runs logs/responsibility/sdc logs/catk/scenes/windows/sdc \
    --labels DenseTNT SMART --out-dir logs/catk/compare/densetnt_vs_smart_sdc
```

On the 500 logged scenes (6,500 windows per agent, flags at each model's own q90):

| comparison | agent | Spearman β_s | Spearman β_c | aggressive flag: agree / κ | scene verdict: agree / κ |
|---|---|---|---|---|---|
| SMART on WOMD v1.1 vs v1.2.1 | SDC | 0.92 | 0.99 | 0.97 / 0.86 | 0.95 / 0.90 |
| | adversary | 0.94 | 0.99 | 0.98 / 0.89 | 0.94 / 0.88 |
| DenseTNT vs SMART (v1.1) | SDC | 0.60 | 0.47 | 0.82 / 0.17 | 0.62 / 0.24 |
| | adversary | 0.63 | 0.52 | 0.84 / 0.24 | 0.69 / 0.36 |

- The data version does not matter (κ 0.86–0.90): the driveways and
  re-processed maps of v1.2.1 leave SMART's verdicts as they were.
- The motion model does. The two models rank windows moderately alike, but
  their top-decile aggressive flags agree only slightly (κ 0.17–0.24). β_s has
  the same scale in both (positive median 0.05–0.09 m, q90 0.5–0.8 m). Scene
  averages agree far better: Spearman 0.77 (SDC) and 0.82 (adversary) for the
  mean β_s.
- SMART's β_c of the SDC is unexplained so far: median 0.54 nats, against 0.06
  for the adversary; DenseTNT gives 0.05–0.07 for both. Compare SMART's β_c
  by rank only.

## GPU server (e.g. Ubuntu 24.04 + H200)

The full sequence of experiments on the server, step by step, is in
[RUNBOOK_H200.md](RUNBOOK_H200.md) (in Chinese).

| component | version | why |
|---|---|---|
| Python | **3.9** | `advgen/utils_cython.cpython-39-*.so` is prebuilt for it (rebuilding needs Cython and a compiler); TF 2.12 supports it |
| torch | **2.4.1 + CUDA 12.1** (not CAT's 1.12.0+cu116) | 1.12/cu116 has no Hopper (sm_90) kernels; 2.4.1 still has Python 3.9 wheels and loads `densetnt.bin` unchanged |
| torchvision | 0.19.1 (matches torch 2.4.1) | DenseTNT's raster CNN |
| tensorflow-cpu | 2.12.0 | only builds DenseTNT's input tensors; the CPU build stays off the GPU |
| numpy | < 1.24 | required by TF 2.12 |
| opencv-python-headless | 4.7.0.72 (CAT's opencv version, without GUI) | MP4 output of the visualisation; without it only the GIF is written |

```bash
bash scripts/responsibility/setup_env.sh          # uv venv at ~/venvs/cat39 (BACKEND=conda also works)
source ~/venvs/cat39/bin/activate
python -m pytest tests/responsibility -q
python -m scripts.responsibility.verify_densetnt --n 3 --device cuda   # must end with ALL CHECKS PASSED
bash scripts/responsibility/run_h200.sh           # step 2: sdc + adv over 500 scenes, summaries, levels
```

To run on a newer Python instead, rebuild advgen's Cython extension for it
(needs a C compiler; it overwrites `advgen/utils_cython*.so` for that Python
only) and re-run `verify_densetnt`. A rebuild from `utils_cython.pyx` with
Cython 0.29 passes every check:

```bash
pip install "cython>=0.29.34,<3" && python scripts/responsibility/build_cython.py
```

For the RL experiments, `scripts/responsibility/run_rl.sh` trains the
seven settings × three seeds, and `scripts/responsibility/run_eval.sh`
then evaluates every model and the replayed log against the four test
adversaries and writes the comparison tables. The runbook's steps 8b and
10 describe both.

`run_h200.sh` starts `SHARDS` (16) processes per agent, spread over the
visible GPUs, each taking every SHARDS-th scene (`--num-shards/--shard-index`)
and writing its own `windows.shard-<i>-of-<n>.csv`; logs are in
`OUT/<agent>/logs/`. Building DenseTNT's inputs is CPU work, so many
processes share one GPU and CPU threads are divided between them. Re-running
the same command resumes. On a laptop CPU a scene takes ~1 min per agent at
1 s windows; check the first lines of a shard log for the rate on the server
and raise `SHARDS` if the GPU and CPUs are not busy. Variables: `OUT`,
`SCENES`, `N`, `AGENTS`, `SHARDS`, `STRIDE` (0.5 s), `SAMPLES`.

## Design decisions

- **Motion set = goal samples, not NMS modes.** DenseTNT scores a 224×224
  grid of goals with a softmax; CAT keeps the 32 goals left by non-maximum
  suppression, a mode-seeking selection. Responsibility needs draws from the
  model's distribution, so goals are sampled i.i.d. from the softmax and
  completed with DenseTNT's own completion head.
- **Courtesy defaults to the paper's own estimate.** The paper defines β_c
  by the KL of Eq. 4 but computes it by Eq. 7 (Sec. V-B), as catk does by
  default; `boltzmann` is that estimate and the default since 2026-10-18.
  `kl` is Eq. 4 as written, kept for comparison: the goal grid depends only
  on the predicted agent's pose, so b's distributions with and without the
  agent share their support and the KL is exact over the grid; it measures
  intent, needs the model to react to the agent, and DenseTNT hardly does.
  `demand` is the braking variant below. Runs made before the change used
  `kl`: they resume with `--courtesy-estimator kl` (compute_responsibility
  names the differing setting), and the aggression test's `paper` and
  `nms_ooi_court` configurations pin it; `nms_ooi_boltz` is the default's.
  Runs with `--no-courtesy` resume whatever the estimator.
- **Courtesy as demand does not ask the model how b reacts.** Comparing b's
  predicted trajectories with and without the agent (instead of its goals)
  would still measure DenseTNT's reaction, which is nil (+0.07 m of progress
  0.5 s before CAT's overlap, `probe_courtesy`), and b's history up to k
  already holds whatever b did about the agent. β_d takes from the model
  only b's plans without the agent, then computes the adaptation the
  agent's actual future demands of them: forced adaptation rather than
  predicted adaptation. Its samples come from a generator of their own, so
  β_s's random stream is the same with either estimator.
- **One instance per forward pass.** In a batch, advgen pads all instances to
  the longest and its attention mask (`attention_mask[i][:n][:n]`) limits only
  the rows, so shorter instances attend to padding: CAT's own ego+adversary
  batch moves the adversary's prediction by 1–2 cm in most scenes and by 32 m
  in one (a different NMS pick). Encoding alone removes this.
- **Removal holds everything else fixed.** DenseTNT's pair model needs a second
  object of interest; it is chosen the same in both passes and never the
  removed agent. Agents are listed nearest first, since the input holds 128
  (in a 156-agent scene the scene order had dropped the ego from its
  neighbour's input).
  Measured (`probe_removal`, 100 logged scenes, the SDC as b at speed
  ≥ 4 m/s): DenseTNT's goal scoring has no slot features, so the order is
  immaterial: the other object of interest in the partner slot or in the
  rest gives a median KL of 1e-8 nats over 274 windows; the 8 above 1e-4
  (up to 0.125) are in scenes with more than 128 agents seen, where the cap
  cuts differently. Removal is effective but the model barely uses it:
  removing a vehicle that leads b in its lane moves b's goals (median KL
  0.08–0.12 nats, b's expected 8 s progress −0.4 to −0.8 m), the other
  object of interest within 10 m 0.021 nats, any other vehicle within 10 m
  0.004; Spearman(distance, KL) −0.14. A stationary vehicle placed on b's
  own path 12.9 m ahead (median) leaves no goal mass short of it (0.000 with
  and without) and cuts b's expected progress by 2.3 m of 68 m: DenseTNT
  extrapolates b's history through it.
- **Neighbours by interaction evidence** (footprint gap ≤ 10 m, post-encroachment
  time ≤ 2 s, constant-velocity TTC ≤ 4 s over the metric horizon, within
  50 m), as in catk, not a plain radius. For a simplified evaluation,
  `--use-ooi` (in `compute_responsibility` and in live visualisation)
  measures the agent against the scenario's other objects of interest only,
  in every window and whatever the evidence. Every one of CAT's 500 scenes
  has two objects of interest, the self-driving car and the adversary, so
  each scene is measured for that one pair. `--agent` must then be `sdc` or
  `adv`. The interaction scores are still written per neighbour.
- **Courtesy toward vehicles only.** CAT's DenseTNT predicts vehicles
  (`agent_type='vehicle'`); safety is measured toward every neighbour.
- **D_g saturates at 10 m** and CVaR uses the upper-tail convention with
  α = 0.1 (close to the mean), as in catk (`responsibility/risk.py`).
- **Valid counterfactuals (optional, `--valid-counterfactuals`).** β_s can be
  taken over the valid part of the motion set only
  (`responsibility/motion_filter.py`, `responsibility/lanes.py`). The
  trajectories stay the motion model's; the filters only remove some of them.
  - **Same intent, judged by lanes (`--lane-route`).** The route is a set of
    lanes, built in four steps:
    1. Each logged position from step k on is matched to one lane: the closest
       centreline within 2.5 m that runs within 45° of the agent's heading.
    2. A lane counts once the agent has driven 5 m along it. Where a turn lane
       branches off, it overlaps the straight lane for a metre or two, and a
       car going straight passes it there. The lane the agent is on at k
       always counts.
    3. The left and right neighbours of these lanes are added, since a lane
       change keeps the intent.
    4. The chain of exit lanes after the last one is added for 100 m, taking
       every branch, because where the agent goes after the log ends is
       unknown.

    DenseTNT's goal distribution is restricted to the goals within 2 m of
    these lanes **before sampling**, and renormalised, so all 40 samples are
    valid. 99% of its goal mass lies within 2 m of some lane centreline. A
    model without goals has its trajectories' end points checked instead. An
    agent the map does not cover is not restricted: on no lane at k or at its
    last logged position, or on a lane for less than half of its logged
    positions. Parking lots and driveways are the usual cases, and CAT's WOMD
    v1.1 maps have no driveways.
  - **Drivable area:** every point (every 0.5 s) is within 3 m of a vehicle
    lane centreline.
  - **Kinematics:** longitudinal acceleration stays within −9 to +5 m/s² and
    lateral acceleration below 7 m/s², from 0.5 s-averaged speeds.
  - **Collision:** the trajectory does not drive through a third agent's
    logged future. b itself does not count, because its distance is what β_s
    measures.
  - **Courtesy (`--courtesy-valid-goals`):** the KL is taken over b's goals on
    the lanes b can reach (its lanes, then their exit chain for 200 m, each
    with neighbours), with both distributions renormalised there. There is no
    restriction where less than half of b's goal mass lies on those lanes,
    because then the map misses where b is going.

    **Without lane topology, drop `--courtesy-valid-goals`** rather than
    approximate it (for b's own drive mode, see `--courtesy-same-mode`
    below). The restriction needs
    the exit chain of the lane graph. Two map-free stand-ins were measured
    against it with DenseTNT on the first 30 scenes: every SDC window and
    vehicle neighbour, 937 pairs, 730 of them with the HD support defined.
    The stand-ins were (A) b's goals within D m of b's logged path, and (B)
    b's goals within a forward cone of its heading. Both moved β_c further
    from the HD values than no restriction does:

    | β_c support | MAE | MAE on the top 10% | Spearman |
    |---|---|---|---|
    | none | 0.0039 nats | 0.018 nats | 0.996 |
    | A, logged path ±8 m | 0.017 nats | 0.122 nats | 0.983 |
    | A, logged path ±20 m | 0.0075 nats | 0.047 nats | 0.992 |
    | B, forward cone 90° | 0.0067 nats | 0.030 nats | 0.982 |
    | B, forward cone 120° | 0.0058 nats | 0.026 nats | 0.987 |

    The HD support leaves out little: a median of 1.7% of b's goal mass, and
    16.5% at the 90th percentile. A support around b's logged path does more
    harm. It cuts away the goals of another intent, and those are exactly
    where a's presence moves b's mass, so it hides the change that β_c
    measures.
  - **β_c within b's own drive mode (`--courtesy-same-mode`).**
    `--courtesy-valid-goals` keeps every goal b can reach, so a change of
    mode (straight → turn) still counts. `--courtesy-same-mode` keeps only
    the goals of the mode b drove in the log, so β_c measures how a changes
    b's plan within that mode. It has two versions:
    - `lanes`: goals within 2 m of b's lane route, as `--lane-route` builds
      it for the queried agent.
    - `path`: goals within `--courtesy-path-lateral` (6 m) of b's logged
      path, extended 100 m along its last heading. No map is needed.

    There is no restriction where b's mode holds less than 0.1% of its goal
    mass, since there is nothing to renormalise. The mass inside the
    mode is written per neighbour as `courtesy_goal_mass`.

    **Calibration of `path` against `lanes`**, on the 761 pairs above with b
    on an HD lane route. DenseTNT puts a median 97.6% of b's mass on b's
    own route (10th percentile 72.6%). The two are close in the median
    (0.015 vs 0.016 nats unrestricted), but on the 10% of pairs with the
    largest β_c the restriction changes β_c by 0.126 nats: these are the
    pairs where a moves b's mass between modes.

    | β_c support, against `lanes` | MAE | MAE on the top 10% | Spearman |
    |---|---|---|---|
    | none | 0.0165 nats | 0.126 nats | 0.986 |
    | reachable (`--courtesy-valid-goals`) | 0.0121 nats | 0.101 nats | 0.992 |
    | `path`, 3 m | 0.0170 nats | 0.099 nats | 0.950 |
    | `path`, 6 m (default) | 0.0102 nats | 0.055 nats | 0.965 |
    | `path`, 8 m | 0.0099 nats | 0.056 nats | 0.969 |

    Inside 6 m, `path` keeps 86% of the mass on b's route and 47% of the
    mass off it, which is goals on parallel lanes and just past the lanes'
    reach. Beyond 6 m it stops improving.
  - **A path-based alternative to the lane route.** `--route-tolerance` keeps
    the trajectories that stay within that many metres of the logged path,
    which is extended 100 m along the last heading.
  - **Same intent without lane topology (`--intent`, `responsibility/intent.py`).**
    A map built by perception has lane lines and road edges but almost no
    topology in intersections, so the lane route has nothing to work with
    there. `--intent` judges the manoeuvre from the trajectories:
    1. *Heading:* at the last logged step T* (at most 8 s ahead), the
       alternative heads within 45° (`--intent-heading`) of the logged path's
       direction **where the alternative is**, not of the log at the same
       time. A braking alternative still in the middle of the logged turn
       is kept; with the same-time comparison it was not. At 45° the
       same-time comparison drops 4.7% of the mass that ends on a route
       lane, and this comparison drops 2.6%. An agent whose logged path is shorter
       than 5 m shows no intent and is not restricted.
    2. *Lateral:* it stays within 8 m (`--intent-lateral`) of the logged path.
       This drops a turn that has only begun by T*, which still heads within
       45° of the path.
    3. *Road edges (`--intent-edges`, off by default):* every second, the
       segment from the alternative to the nearest point of the logged path
       crosses no `ROAD_EDGE_BOUNDARY` or `ROAD_EDGE_MEDIAN`. A missing edge
       removes nothing, so an incomplete map only loosens the test.

    **Calibration against the lane route on the first 30 scenes.** The
    reference is β_s over the weighted motion sets of
    `logs/filter_trial/weighted_filtered`, restricted by the HD lane route
    (end points within 2 m of a route lane), on 930 SDC–neighbour pairs. The
    sets are taken offline from the records, with no other filter.

    | restriction | MAE β_s | MAE on the 105 pairs > 0.05 m | largest error |
    |---|---|---|---|
    | none | 0.0041 m | 0.0094 m | 0.62 m |
    | heading 45° | 0.0026 m | 0.0102 m | 0.30 m |
    | heading 45° + lateral 8 m (default) | 0.0027 m | 0.0126 m | 0.26 m |
    | + road edges (HD) | 0.0030 m | 0.0158 m | 0.26 m |
    | + road edges, 30% dropped, 0.3 m noise | 0.0030 m | 0.0154 m | 0.26 m |
    | + road edges, 60% dropped, 0.5 m noise | 0.0028 m | 0.0137 m | 0.26 m |

    The heading test drops all the mass that ends on a non-route lane more
    than 15 m from the logged path. What it keeps of the non-route mass lies
    within 15 m and heads the same way: two lanes over, beyond the lane
    route's one-hop neighbours, or a turn that has only begun (scene 15,
    steps 20–35, which the lateral test drops). Road edges move β_s
    **away** from the lane route, and degrading them moves it back, because
    edges at intersection corners and islands cut off alternatives the lane
    route keeps. That is why they are off by default. At a 2 s horizon the
    choice matters little in any case: even no restriction is within 0.01 m
    of the lane route on the pairs that matter.
  - **Drivable area without centrelines (`--drivable-edges`,
    `responsibility/edges.py`).** The drivable test above needs lane
    centrelines, and a perceived map has none inside intersections. Here the
    path is checked every 0.5 s from the agent's position, and the
    trajectory must cross no `ROAD_EDGE_BOUNDARY` or `ROAD_EDGE_MEDIAN`. An
    edge that the agent's own logged path crosses does not count (map noise,
    or a driveway, which WOMD v1.1 lacks), and a gap in the edges removes
    nothing.

    **Measured on the first 30 scenes.** The reference is the centreline
    test (3 m) on the HD map, over the weighted sets of
    `logs/filter_trial/weighted_filtered` (345 windows, 930 SDC–neighbour
    pairs, no other filter). A perception-like map is made from the HD one:
    lane topology removed, and the centrelines of the 32% of lanes that cross
    another lane at more than 30° (intersection connectors) removed. The
    road edges are then cut into 10 m pieces, a share of them dropped, and
    the rest jittered.

    | drivable test | wrongly dropped | wrongly kept | MAE β_s | MAE, pairs > 0.05 m | largest error |
    |---|---|---|---|---|---|
    | centrelines, perceived map | 15.7% | 16.7% | 0.033 m | 0.087 m | 4.71 m |
    | road edges (as in the HD map) | 1.3% | 36% | 0.0042 m | 0.026 m | 0.72 m |
    | road edges, 30% dropped, 0.3 m noise | 1.2% | 51% | 0.0059 m | 0.035 m | 0.79 m |
    | road edges, 60% dropped, 0.5 m noise | 1.3% | 66% | 0.0080 m | 0.037 m | 1.19 m |

    "Wrongly dropped" is the share of the mass the reference keeps that the
    test removes. "Wrongly kept" is the share of the mass the reference
    removes that the test keeps. On a perceived map the centreline test
    removes the alternatives that cross an intersection. The edge test
    almost never removes too much, even with gaps and noise. It keeps road
    surface that is more than 3 m from a lane centre (parking lanes,
    shoulders), and whatever leaves through a gap. Combining the two did not
    help. Keeping points within 3 m of a centreline, or farther than C from
    every centreline, and also checking the edges, wrongly dropped 19–22%
    of the mass for C = 4.5, 6 and 8 m: points just past the end of a
    removed intersection lane fall between 3 m and C.

  `--motion-set weighted` uses DenseTNT's whole goal grid (0.999 of the mass),
  probability-weighted, instead of 40 samples. That set holds a median of 944
  trajectories per window (P10 402, P90 2742, max 15568 over the 345 windows
  with neighbours of `logs/filter_trial/weighted_filtered`), each completed
  and filtered. `--motion-set topk` keeps the `--n-samples` most probable
  goals instead, probability-weighted and renormalised over them. The top 40
  hold a median 0.69 of the mass (P10 0.39). Offline, from the weighted run's
  records, with the same filters re-applied to the top k, against the full
  weighted β_s on 961 neighbour pairs:

  | motion set | MAE β_s, all pairs | MAE on the 110 pairs with β_s > 0.05 m | largest error |
  |---|---|---|---|
  | top 40 | 0.014 m | 0.060 m | 1.00 m |
  | top 100 | 0.008 m | 0.036 m | 0.91 m |
  | top 200 | 0.004 m | 0.021 m | 0.65 m |
  | 40 samples (`logs/filter_trial/filtered`) | 0.010 m | 0.035 m | 0.89 m |
  | 40 by NMS (`--motion-set nms`, run) | 0.005 m | 0.015 m | 0.53 m |

  The top 40 are deterministic but less accurate than 40 samples where β_s
  matters: truncation drops the low-probability, slower executions that
  `weighted` was added for. `--motion-set nms` (`modes.py`) instead spreads
  its `--n-samples` goals over the distribution with CAT's goal NMS (7.2 m
  times CAT's speed scale factor, 0.5–1.0), fills up with the next most
  probable goals when fewer survive, and weights each goal by the probability
  of the grid goals nearest to it, so the weights sum to 1. It was run with
  DenseTNT on the same 30 scenes and filters (on CPU, 10–19 s per scene). It
  is the most accurate of the 40-trajectory sets, more accurate than the top
  200: correlation 0.996 with the weighted β_s (40 samples: 0.981), the
  β_s > 0.05 m flag agrees on 99.9% of the pairs, and the mean difference is
  −0.001 m. MTR's adapter applies the same rule to its 64 intention
  endpoints.

  **Results on the first 30 scenes.** The runs cover 390 SDC windows with
  DenseTNT (`logs/filter_trial`) and are compared with the unfiltered run of
  the same seed.

  | | β_s | β_c |
  |---|---|---|
  | Spearman with the unfiltered values | 0.96 | 0.99 |
  | top-decile flags: agreement / κ | 0.995 / 0.94 | 0.985 / 0.91 |
  | mean change | −0.002 m | −0.007 nats |

  - **Lane route.** The goal mass on the route lanes has a median of 0.98 and
    is below 0.9 in 21% of the windows; that is where the lane route filters
    something out. The windows that changed most are scene 29 at step 10
    (β_s +0.57 → +0.01 m) and scene 16 at step 35 (+1.27 → +0.93 m).
  - **Each filter alone, as a share of probability mass.** These were measured
    with the path-based route:
    - route 23%
    - drivable area 12%
    - collision 1%
    - kinematics 0%
  - **Why little changes.** Within the 2 s metric horizon (the paper's T_f, 20
    steps of 0.1 s on nuScenes), alternatives with another intent have not
    yet left the route: their median distance from the logged path over 2 s
    is 0.36 m. The filters make the set valid, but they cannot add reactions
    the motion model does not predict. The paper names this limit itself
    (Sec. VI-A): the motion set is open-loop, and "controls do not change even
    after observing other agents' future states". The next item tests
    reactions directly.
  - **Reactions (SMART, first 30 scenes, `logs/catk/trial30`).** catk's
    `--neighbor-future logged` lets SMART's alternatives see the neighbours'
    logged motion step by step, never ahead of time. It needs no retraining,
    since SMART is autoregressive with 0.5 s tokens. Compared with the
    open-loop `hidden` mode:

    | | SDC | adversary |
    |---|---|---|
    | Spearman β_s | 0.93 | 0.93 |
    | aggressive-flag κ | 0.65 | 0.58 |
    | mean β_s | +0.074 m in both modes | +0.123 vs +0.120 m |

    Where the logged SDC slowed by more than 2 m/s within the 2 s horizon
    (20 windows), the alternatives that slowed by at least half as much were:

    | | DenseTNT | SMART `hidden` | SMART `logged` |
    |---|---|---|---|
    | all 20 windows | 92% | 93% | 93% |
    | the one sudden onset (scene 1, step 50) | 5% | 35% | 38% |

    In the other windows the deceleration had already begun and shows in the
    history, so the alternatives simply continue it. Within 2 s, the
    neighbours' motion hardly changes what the alternatives do.

## Limitations

- At 1–2 s horizons the closest approach is often at the first step, where
  every sample starts where the log does; slow or stationary agents therefore
  get β_s ≈ 0, which says "no alternative did better", not "safe".
- Courtesy is intent-level and DenseTNT only sees 1.1 s of history, so an
  agent's influence enters only through its recent past and current state.
- Thresholds are relative to a reference population; the verdict is "unusual
  compared with that driving", not an absolute safety judgement.
- The right of way names a holder for only 30% of the held-out pairs whose
  paths cross or merge (330 of 1091): same-direction pairs are RSS's, and
  lights without a recorded state, junctions the map shows no control for
  and all-way stops both entered before the clip began get none. Forced
  yields (above) are measured only where it names one, and its reading of
  the rule can change with the step the conflict is judged at.
- Over the 2 s metric horizon, the alternatives stay close to the logged
  future: DenseTNT's median ADE is 0.26 m. They continue a deceleration that
  has begun, but rarely anticipate one that has not (see "Reactions" above).
  An earlier version of this note said that only 4% brake where the driver
  braked. That number counted alternatives braking harder than the driver,
  not alternatives braking at all.
- What the RL settings use holds up under SMART (`SMART_PLAN.md`, S4 on the
  replay reference rollouts):
  - **Adversary β.** The ordering of the adversaries' β toward the ego is the
    same under both models: logged +0.03, fair +0.77, avoidability only
    +1.27, CAT +1.98 m (SMART, per-scene median). The scale is not: SMART puts
    CAT's adversaries at about a third of DenseTNT's value, so τ = 2 m is a
    DenseTNT-calibrated threshold.
  - **Collision verdicts.** The two models agree on 62–81% of the
    collisions, about as often as the counterfactual verdict agrees with RSS
    and the rear-end rule.
- Window-level flags depend on the motion model: DenseTNT and SMART agree with
  κ 0.17–0.24 on the aggressive windows of the same scenes. Scene-level and
  aggregate results are more stable (Spearman 0.77–0.83 of the scene mean and
  maximum β_s).
