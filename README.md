# Counterfactual responsibility for driving, with any motion predictor

Counterfactual responsibility (Hsu, Leung, Chen, Fisac & Pavone, ["Interpretable
Trajectory Prediction for Autonomous Vehicles via Counterfactual
Responsibility"](https://saferobotics.princeton.edu/research/responsibility), IROS 2023)
asks two questions of every agent at every context step k:

| quantity | question | unit |
|---|---|---|
| safety responsibility β_s | Did the agent give up safety margin compared with what it could otherwise have done? CVaR over the predictor's motion set of D_g(sample, b) − D_g(logged, b), max over interacting neighbours b | m |
| courtesy responsibility β_c | Did its actual motion block what others could plausibly have done? KL between Boltzmann distributions over b's pooled samples with and without the agent (Sec. V-B, Eq. 7); or the exact KL of Eq. 4; or braking demand | nats (m/s² for demand) |

On top of these: HMM responsibility levels, counterfactual collision attribution
(plus RSS and right-of-way baselines), forced yields, a responsibility-constrained
("fair") CAT adversary and RL training with blame-weighted rewards.

This repository merges two implementations that differed only in the motion predictor:

| source | predictor | what it contributed |
|---|---|---|
| [CesarLiu/catspond](https://github.com/CesarLiu/catspond) `gpudrive-ppo` @ `0865ed8` | CAT's DenseTNT, UniTraj MTR | `responsibility/` (metrics, attribution, rules, records), CAT TD3 in MetaDrive (its GPUDrive PPO port is not carried over on this branch) |
| [CesarLiu/catk4test](https://github.com/CesarLiu/catk4test) `main` @ `1bc7e22` | CAT-K's SMART | the batched SMART counterfactual pipeline, exact token courtesy, reward networks, SMART reward fine-tuning, scenario tags, the aggression-judging skill |

The responsibility code now lives in one package, and every motion predictor
plugs into it through a single adapter interface.

> **中文概要**：本仓库把 catspond（DenseTNT / MTR，含 CAT 的 TD3 训练）和 catk4test（CAT-K 的 SMART）两份 counterfactual responsibility 实现合并。责任度量（`responsibility/`）与运动预测器解耦：预测器只需实现 `distribution()` 和 `sample()` 两个方法（`responsibility/predictors/base.py`），即可用 `--model densetnt|mtr|unitraj|smart|constant_velocity` 或 `--model 包.模块:工厂函数` 切换。UniTraj 的任一方法（MTR、Wayformer、AutoBot、EMP…）都可以通过 `--model unitraj --unitraj-method <方法>` 使用。全部代码只用 CAT-K 的 Python 3.11 环境运行；CAT 的 TD3 训练改在纯 torch 的日志回放仿真器上进行（`rl/`、`scripts/rl/`），不再依赖 MetaDrive 或 GPUDrive；SMART 奖励微调保留，并且 fair 对手的约束和碰撞归因也可以换用任意预测器（`--resp_model`、`--blame_model`）。

## Layout

```
responsibility/                 predictor-agnostic metrics and tools (operate on scene.Scene)
  scene.py                      one 10 Hz clip as arrays; ScenarioNet/MetaDrive description <-> Scene
  metrics.py                    beta_s, beta_c (boltzmann | kl | demand) at each context step
  interaction.py motion_filter.py lanes.py intent.py edges.py modes.py demand.py
  blame.py rss.py right_of_way.py stl.py      collision attribution and its rule baselines
  forced_yield.py forced_merge.py attribution.py aggression_*.py levels.py hmm.py
  records.py results.py rollouts.py recording.py swap.py
  adversarial.py blame_reward.py              hooks into CAT's adversary and RL reward
  catk_export.py catk_cache.py gpudrive_export.py   format converters
  models.py                     the --model command-line options
  predictors/                   motion predictors behind one interface
    base.py                     the interface and capability checks
    __init__.py                 the registry (register_predictor, load_predictor)
    constant_velocity.py        dependency-free kinematic reference / template
    densetnt.py                 CAT's DenseTNT (TensorFlow + advgen/)
    unitraj.py, unitraj_configs/  any UniTraj method: MTR (all 64 intentions), Wayformer, AutoBot, EMP, ...
    smart/                      CAT-K's SMART
      adapter.py                SmartPredictor: the interface over SMART rollouts
      scene_data.py             Scene -> CAT-K's PyG batch, in memory
      catk.py, configs/         finding the CAT-K checkout; Hydra config over its configs/
      native/                   CAT-K's own batched pipeline (from catk4test src/responsibility)
rl/                             MetaDrive-free CAT training: log-replay simulator, batched adversary, TD3, blame weighting
scripts/rl/                     build the scene bank, precompute CAT's candidates, check the simulator, train TD3
advgen/                         CAT's DenseTNT and adversary generator (upstream code)
saferl_algo/ saferl_plotter/ cat_RLtrain.py cat_advgen.py    CAT's MetaDrive TD3 (upstream, kept for reference; needs Python 3.9)
scripts/responsibility/         CLI for the predictor-agnostic pipeline (any --model)
scripts/smart/                  CLI for the native SMART pipeline (CAT-K environment)
tests/responsibility/  tests/smart/  tests/rl_training/
docs/                           CAT's readme, the SMART method notes and runbook
skills/judge-driving-aggression/    Claude skill: rule-based aggression judging with jurisdictions
```

`docs/RUNBOOK.md` (Chinese) is the server checklist for this repository: the
regression of old DenseTNT runs (`scripts/responsibility/regress_run.py`), the
SMART adapter on a real checkpoint, and the adapter-vs-native comparison.
`responsibility/README.md` is the detailed reference: every metric, every script,
the design decisions and the measured results. The Chinese plans and runbooks
(`responsibility/*PLAN*.md`, `RUNBOOK_H200.md`, `CATK_H200.md`)
are kept as they were. `docs/smart/responsibility.md` documents the SMART method.

## Motion predictors

A predictor is any object with two methods (`responsibility/predictors/base.py`):

```python
distribution(scene, step, agent, excluded=(), avoid_partner=()) -> dist | None
sample(dist, n, generator=None) -> (index [n] | None, log_prob [n], trajectories [n, T, 2])  # scene frame
```

`distribution` conditions on the scene up to `step`, with the agents in `excluded`
removed (the counterfactual x^{-a}); it returns `None` where the agent is not
predicted. Optional methods unlock options: `motion_set` (`--motion-set weighted|topk`),
`nms_motion_set` (`--motion-set nms`) and `with_and_without` (`--courtesy-estimator kl`).
`capabilities(model)` lists what a predictor offers and `require(...)` refuses an
unsupported option before a run starts.

| `--model` | predictor | needs | motion set | exact KL (Eq. 4) |
|---|---|---|---|---|
| `densetnt` (default) | CAT's pretrained DenseTNT, goal grid + completion | TensorFlow, `advgen/utils_cython` built for the Python in use (`scripts/setup_catk_env.sh`), `advgen/pretrained/densetnt.bin` | sampled, weighted, top-k, NMS | yes (shared goal grid) |
| `mtr` | UniTraj MTR, all 64 intention queries | UniTraj checkout (`$UNITRAJ_ROOT`), its CUDA ops, `--checkpoint` | sampled, weighted, top-k, NMS | yes (fixed intention points) |
| `unitraj` | any UniTraj method through its own forward pass (`--unitraj-method wayformer\|autobot\|EMP\|forecast\|SMART`) | UniTraj checkout, `--checkpoint` | sampled, weighted, top-k, NMS | formally; modes are not fixed points, so prefer `boltzmann` |
| `smart` | CAT-K SMART rollouts, the others teacher-forced and hidden after k (`--smart-neighbor-future`) | CAT-K checkout (`$CATK_ROOT`), Python 3.11, PyG 2.6.1, torch_cluster, `--checkpoint` | sampled | no: `boltzmann` or `demand`, or the native pipeline's `exact` |
| `constant_velocity` | kinematic reference, ignores map and agents | nothing | sampled, weighted, top-k, NMS | yes (is 0) |
| `pkg.module:factory` | your own: `factory(args, device)` returns a predictor | yours | what it implements | if it implements `with_and_without` |

Adding one:

```python
from responsibility.predictors import register_predictor

class MyPredictor:
    def distribution(self, scene, step, agent, excluded=(), avoid_partner=()):
        ...  # build inputs from scene (history up to step), leave out `excluded`
    def sample(self, dist, n, generator=None):
        ...  # return None, log_prob [n], trajectories [n, T, 2] in the scene frame

register_predictor("mine", lambda args, device: MyPredictor())
```

or, with no registration, `--model mypackage.mymodule:make_predictor`.
`constant_velocity.py` is a complete, short example.

### Same metrics, any predictor

```bash
# kinematic reference: runs anywhere, a smoke test of the whole pipeline
python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 --out-dir logs/resp/cv \
    --model constant_velocity --n 20 --save-records
# DenseTNT (CAT's environment)
python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 --out-dir logs/resp/densetnt \
    --use-ooi --motion-set nms --valid-counterfactuals --save-records --device cuda
# UniTraj: MTR, or any other trained method
UNITRAJ_ROOT=~/UniTraj python -m scripts.responsibility.compute_responsibility --out-dir logs/resp/mtr \
    --model mtr --checkpoint ckpts/mtr.ckpt --device cuda
UNITRAJ_ROOT=~/UniTraj python -m scripts.responsibility.compute_responsibility --out-dir logs/resp/wayformer \
    --model unitraj --unitraj-method wayformer --checkpoint ckpts/wayformer.ckpt --device cuda
# SMART through the same code path (CAT-K's environment)
CATK_ROOT=~/catk python -m scripts.responsibility.compute_responsibility --out-dir logs/resp/smart \
    --model smart --checkpoint ~/catk/ckpts/clsft_E9.ckpt --device cuda
# compare two predictors' windows
python -m scripts.responsibility.compare_models --help
```

Every run writes `windows.csv`, `obs/`, `records/` and `config.json`; the
predictor's settings are part of `config.json`, so a run never resumes with
another predictor. Summaries, levels, visualisation and comparisons read any
run the same way.

### SMART's native pipeline

`responsibility/predictors/smart/native/` is catk4test's `src/responsibility`
(`geometry`, `risk` and `hmm` are the shared ones). It packs many
counterfactual jobs into one SMART call and implements what only a token model
can do: the exact per-token courtesy KL, per-level reward networks,
responsibility-aware candidate selection and reward fine-tuning. It reads CAT-K's
WOMD cache directly:

```bash
export CATK_ROOT=~/catk     # any CAT-K checkout with src/smart and configs/
python -m scripts.smart.compute_responsibility --ckpt $CATK_ROOT/ckpts/clsft_E9.ckpt \
    --data-root $CATK_ROOT/data/womd_cache/validation --out-dir logs/smart/val --courtesy-estimator exact
python -m scripts.smart.verify_counterfactuals --help
python -m scripts.smart.verify_adapter --ckpt $CATK_ROOT/ckpts/clsft_E9.ckpt --scenes raw_scenes_500 --n 3
python -m scripts.smart.compare_adapter_native --ckpt $CATK_ROOT/ckpts/clsft_E9.ckpt --scenes raw_scenes_500 \
    --n 20 --agent sdc adv --out-dir logs/smart/compare_adapter_native   # adapter vs native, within seed noise?
```

`docs/smart/RUNBOOK.md` is catk4test's runbook (stages 1–5, with paths now under
`scripts/smart/`). CAT scenes go to CAT-K's format with
`scripts/responsibility/export_catk.py`, and `smart_records.py` writes SMART runs
in this repository's run format.

## RL training

Training runs in CAT-K's Python 3.11 environment, with no MetaDrive (which
does not support Python 3.11) and no GPUDrive:

| loop | simulator, algorithm | entry point | responsibility hooks |
|---|---|---|---|
| CAT | `rl/sim.py`: a batched log-replay simulator in plain torch over CAT's scenes; TD3 (`rl/td3.py`, CAT's networks and hyper-parameters) | `scripts/rl/train_td3.py --mode replay\|cat` | `--adv-rule cat\|fair` (CAT's adversary or the fair one, `--resp-threshold`, `--resp-avoid`); `--blame-weighting share\|rss` with `--blame-model` for the attribution (`rl/blame.py`) |
| SMART fine-tuning | CAT-K SMART, policy-gradient fine-tuning toward a responsibility level | `scripts/smart/finetune_reward.py` | reward networks from `scripts/smart/train_reward.py` |

`rl/sim.py` keeps CAT's setting and replaces only the simulator:
- **Kept:** CAT's 500 scenes (0–399 train, 400–499 test), the other traffic
  replayed from the log, CAT's adversary driving its chosen plan a step late,
  and the schedule of adversarial episodes.
- **Kept:** MetaDrive's reward (progress along the route, +10 arrival, −10 out
  of road, −1 per step in contact) and its termination rules (out of road at
  10 m off the route, a road edge or a solid yellow line; collisions end only
  evaluation episodes).
- **Replaced:** the ego is a kinematic bicycle (acceleration −6…4 m/s²,
  curvature ±0.3 1/m).
- **Replaced:** the observation is a vector in the ego's frame: ego state, the
  route ahead, rays to road edges and yellow lines, and the 16 nearest agents.
  MetaDrive gave a lidar.

So results are not comparable with MetaDrive's. CAT's adversary selection
(`rl/adversary.py`) is batched in torch from candidates precomputed once
per scene with DenseTNT (`scripts/rl/precompute_candidates.py`).

```bash
python -m scripts.rl.build_scenes --scenes raw_scenes_500 --out logs/rl/scenes.npz
python -m scripts.rl.precompute_candidates --scenes raw_scenes_500 --out logs/rl/candidates.npz --device cuda
python -m scripts.rl.check_sim --bank logs/rl/scenes.npz --candidates logs/rl/candidates.npz --device cuda
# CAT's adversary, collision penalty weighted by the counterfactual blame (here with SMART)
python -m scripts.rl.train_td3 --bank logs/rl/scenes.npz --candidates logs/rl/candidates.npz --mode cat \
    --blame-weighting share --blame-model smart --blame-checkpoint $CATK_ROOT/ckpts/clsft_E9.ckpt \
    --out logs/rl/runs/cat_share_s0 --seed 0 --device cuda --save-model
```

The fair adversary's *candidates* are always CAT's 32 DenseTNT NMS modes;
`precompute_candidates.py --resp_model` takes the motion sets behind β (the
adversary's alternatives) and avoidability (the ego's) from another
predictor, which must give 8 s of future. `cat_RLtrain.py` and
`cat_advgen.py`, CAT's MetaDrive scripts, are kept unchanged for reference;
they need MetaDrive and Python 3.9 and are not part of the CAT-K environment.

## Environments

Everything runs in CAT-K's Python 3.11 environment (CAT-K's
`install/setup_server.sh`, then this repository's
`scripts/setup_catk_env.sh`):

```bash
source ~/venvs/catk/bin/activate
bash scripts/setup_catk_env.sh
```

The setup script adds `cython<3`, then builds `advgen/utils_cython` for
Python 3.11 (the repository ships only the 3.9 build). It also installs
UniTraj's extra packages (h5py, einops, pytorch_lightning) and runs the tests.

| predictor or loop | needs in the CAT-K environment |
|---|---|
| DenseTNT | the 3.11 build of `advgen/utils_cython`, TensorFlow (CAT-K has 2.12), `advgen/pretrained/densetnt.bin` |
| SMART | `CATK_ROOT` (a CAT-K checkout) and a checkpoint |
| UniTraj (Wayformer, AutoBot, ...) | `UNITRAJ_ROOT`, the UniTraj extras; MTR also its CUDA ops, built in this environment |
| TD3 training (`rl/`) | nothing more: plain torch |

Not in git: `raw_scenes_500/`, `advgen/pretrained/densetnt.bin` (see
`docs/CAT_README.md`), checkpoints, WOMD and its caches.

## Tests

```bash
python -m pytest tests -q          # no model, data or GPU needed
```

`tests/responsibility` covers the metrics and tools, now including the predictor
registry and a whole `compute_responsibility` run with the constant-velocity
predictor (`test_predictors.py`). `tests/smart` covers the SMART pipeline on
CAT-K's stub SMART and the adapter (`test_adapter.py`). `tests/rl_training`
covers the simulator on synthetic scenes with known answers (the bicycle
inverse replays a straight and a curved log to 1 mm, arrival pays 85 + 10,
collisions, road edges, yellow lines, the adversary a step late on its
plan, CAT's history deque), the batched selection against CAT's numpy
code, TD3, the blame weighting both ways under RSS, and a whole training
run. Tests that need
TensorFlow, a UniTraj checkout or CAT-K skip or fail to collect without them.

Verified while merging, in one Python 3.11 environment (torch 2.4.1 CPU, PyG 2.6.1, torch_cluster 1.6.3, TensorFlow 2.15):

- `pytest tests`: 451 passed, 3 skipped (they need a UniTraj checkout, WOMD protos and MetaDrive); `test_unitraj.py` alone, with `UNITRAJ_ROOT` set, passes all 15 tests;
- `scripts/smart/verify_adapter.py` on a randomly initialised CAT-K SMART (`smart_nano_1M`), a synthetic scene with WOMD's 0.5 m point spacing: all 7 checks pass. SMART sees the map (80 tokens), the first sampled position is within 1.16 m of the logged one, samples are reproducible from the generator, removing a neighbour changes the free agent's token log-probabilities, and rollouts stop where the log does (80 steps from k = 10, 70 from k = 20);
- `scripts/smart/compare_adapter_native.py` on the same model and 3 synthetic scenes (78 window-neighbour pairs, 2 seeds per path): the adapter and CAT-K's native pipeline agree within seed noise (safety Spearman 0.960 against 0.951–0.970 between seeds, courtesy 0.875 against 0.893–0.904);
- UniTraj Wayformer and AutoBot (random weights) through `unitraj_predictor` and the metrics: 6 modes of 60 steps each, finite β_s and β_c;
- CAT's fair adversary with a randomly initialised DenseTNT, with `--resp_model densetnt` and `constant_velocity`;
- the TD3 pipeline on 8 synthetic two-lane scenes, without MetaDrive:
  - `build_scenes` built the bank;
  - `precompute_candidates` ran on a randomly initialised DenseTNT;
  - `check_sim`: the bicycle inverse replays every SDC log with 0.000 m error; 8/8 logged drives arrive; the adversary sits exactly on its plan a step late; 1,183 world steps/s with 8 worlds on 4 CPU cores. The collision check flagged 3 of the 8 scenes, all collisions in the random synthetic tracks themselves;
  - `train_td3 --mode cat`: in 30k transitions (4.3 min on CPU), arrival in training episodes rose from 0.24 to 0.93;
  - `--blame-weighting rss|share`: attributed 173–176 collisions in 6k transitions.

There were no trained checkpoints, no WOMD data and no GPU, so nothing here
measures a trained model. The measured results in `responsibility/README.md`,
`docs/smart/` and the plans come from the source repositories.

## Licences

The code from catspond and CAT is MIT-licensed (`LICENSE`). The files moved from
catk4test (`responsibility/predictors/smart/native/`, `scripts/smart/`,
`tests/smart/`, `docs/smart/`, `skills/`) came from a repository under the
NVIDIA Source Code License for CAT-K, a copy of which is in `LICENSE.CATK`
(non-commercial use; redistribution under the same licence). CAT-K's SMART
model code itself is not included: it is imported from a CAT-K checkout at run
time.
