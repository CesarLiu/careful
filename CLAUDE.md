# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Counterfactual responsibility (Hsu et al., IROS 2023) for driving, independent of the motion predictor. It merges two implementations that differed only in the predictor: `CesarLiu/catspond` (branch `gpudrive-ppo`: CAT's DenseTNT and UniTraj's MTR, with CAT's TD3 training in MetaDrive and PPO in GPUDrive) and `CesarLiu/catk4test` (CAT-K's SMART). `README.md` describes the layout, the predictor interface and the three RL training loops. The metrics are used for three things:

- measuring how aggressively or timidly a driver behaves;
- constraining CAT's adversary to "fair" (responsible and avoidable) trajectories;
- weighting the RL collision penalty by the ego's share of the blame.

`responsibility/README.md` is the user-facing reference (metrics, every script, the design decisions, measured results). Read it before changing behaviour. The research plan, the runbook and the UniTraj plan are in Chinese:

- `responsibility/RESEARCH_PLAN.md`: milestones M1–M3, each with a **状态** (status) block;
- `responsibility/RUNBOOK_H200.md`: the exact server commands, steps 0–11;
- `responsibility/UNITRAJ_PLAN.md`: the UniTraj MTR integration, milestones U0–U5.
- `responsibility/CATK_H200.md`: running the CAT-K / SMART experiments on another server (what to pack, environments, commands).

When you finish a milestone, update its status block and the README in the same commit, including any measured numbers.

## Environment and commands

Python **3.9** is required: `advgen/utils_cython.cpython-39-x86_64-linux-gnu.so` is prebuilt for it. To use another Python version, rebuild it with `python scripts/responsibility/build_cython.py` (needs `cython>=0.29.34,<3`). There is no lint or format config.

```bash
bash scripts/responsibility/setup_env.sh && source ~/venvs/cat39/bin/activate   # uv venv; BACKEND=conda also works
# minimal CPU env for the tests: torch 2.4.1 (cpu), tensorflow-cpu==2.12.0, "numpy<1.24", scipy, matplotlib, pyyaml, tqdm, pytest

python -m pytest tests -q                                                 # all tests; no model, data or GPU needed
python -m pytest tests/responsibility/test_blame.py -q                    # one file
python -m pytest tests/responsibility/test_blame.py -q -k rear            # one test by name
python -m scripts.responsibility.verify_densetnt --n 3 --device cuda      # needs data + checkpoint; must end with ALL CHECKS PASSED
```

- `test_adversarial.py` and `test_benchmark.py` import TensorFlow. Without it they skip or fail to collect.
- `test_unitraj.py` skips unless `UNITRAJ_ROOT` points at a UniTraj checkout (its dataset code also needs `metadrive/type.py`: CAT's `metadrive/` at the root or an installed MetaDrive).
- `tests/smart` stubs `src.smart.modules.agent_decoder` (conftest), so it runs without CAT-K; `scripts/smart/verify_adapter.py` checks `SmartPredictor` against a real (or randomly initialised) SMART in CAT-K's environment.
- Always run from the repository root, and run scripts as modules (`python -m scripts.responsibility.<name>`). Scripts insert the repo root into `sys.path`.

Several things are not in git and are usually **absent on a dev machine**:

- downloaded separately (see `docs/CAT_README.md`): the modified `metadrive/` package, `raw_scenes_500/*.pkl` (the scenes) and `advgen/pretrained/densetnt.bin`;
- more scenes: `scripts/responsibility/convert_womd_split.py` converts a whole WOMD split (the L4 server has v1.2.1 shards in `~/womd_v1_2_1/`) into CAT's scene format with CAT's own converter, plus an `index.csv` (`sdc_in_ooi`, `in_cat`). The open-loop tools read these scenes. The closed loop (MetaDrive, `advgen`) also runs them, given a folder of `sdc_in_ooi` scenes numbered `0.pkl` …;
- produced on the server: `logs/` (gitignored) and `rollouts/` (not ignored, so don't commit it).

Anything that needs MetaDrive (`cat_RLtrain.py`, `cat_advgen.py`, `collect_rollouts.py`; the plans mark these 🖥) can only run on the GPU server. Develop it against the stand-ins in the tests, and leave the server steps to the runbook.

## Architecture (responsibility/)

The pipeline is: scene → motion model → metrics → per-run outputs → summaries.

- **`scene.Scene`**: arrays for one 91-step WOMD/ScenarioNet clip at 0.1 s per step.
  - `from_description` / `to_description` convert to and from ScenarioNet format.
  - `with_track` replaces an agent's track with a simulated one; `drop` removes agents.
  - Every consumer works on a `Scene`, so logged driving and policy rollouts go through the same code.
- **Motion predictors live in `responsibility/predictors/`, behind one interface (`base.py`).** The registry (`predictors/__init__.py`) and `responsibility/models.py` (the CLI options) select one with `--model densetnt|mtr|unitraj|smart|constant_velocity` or `--model package.module:factory`, and import it lazily: DenseTNT pulls in TensorFlow, UniTraj its CUDA ops, SMART CAT-K and PyG. Never import a predictor module from the predictor-agnostic code. A predictor provides:
  - `distribution(scene, step, agent, excluded=(), avoid_partner=())`, which returns `None` when the agent is not predicted;
  - `sample(dist, n, generator)` -> (index or None, log_prob [n], trajectories [n, T, 2] in the scene frame);
  - optionally `with_and_without(scene, step, b, a)` over a shared discrete support (`--courtesy-estimator kl`); `base.require` refuses options a predictor lacks;
  - optionally `motion_set(dist, top_k=None)`, the exact weighted path (`--motion-set weighted`), or its `top_k` most probable members (`--motion-set topk`);
  - optionally `nms_motion_set(dist, n)`: n goals spread by CAT's goal NMS, each weighted by the probability nearest to it (`--motion-set nms`, `modes.py`).
  - `densetnt.py` wraps CAT's DenseTNT and reproduces its inputs exactly. It runs **one instance per forward pass**, because advgen's batching leaks padding into the attention.
  - `unitraj.py` adapts any UniTraj method (`mtr_predictor` keeps MTR's 64 intentions; `unitraj_predictor` calls any other method's forward) without patching UniTraj (it crops windows and injects a `scenarionet` stand-in). Removing an agent is done with masks, not by deleting the track.
  - `smart/adapter.py` (`SmartPredictor`) samples SMART rollouts with CAT-K's machinery (`smart/native/motion_set.py`): the free agent sampled, the others teacher-forced and, by default, hidden after k. A distribution is a recipe, not probabilities, so SMART supports `sampled` motion sets and the `boltzmann`/`demand` courtesy only. Rollouts are cut where the clip's log ends (`max_horizon`). `smart/native/` is catk4test's batched pipeline (exact token courtesy, reward networks, fine-tuning), driven by `scripts/smart/*`; CAT-K's `src.smart` is imported from `$CATK_ROOT`, never vendored.
  - `constant_velocity.py` is the dependency-free reference and the template for new adapters.
  - `tests/responsibility/conftest.py` has `FakeModel` and `make_scene`/`track` for synthetic tests.
- **`metrics.py`** computes the metrics at each context step k.
  - `responsibility_at` / `scene_responsibility` compute β_s and β_c against neighbours chosen by interaction evidence (`interaction.py`), or with `--use-ooi` against the scenario's other object of interest only (one pair per scene). β_c can be restricted to the goals b can reach (`--courtesy-valid-goals`) or to b's own logged drive mode (`--courtesy-same-mode lanes|path`).
  - β_s is a CVaR over the motion set of the closest-approach gap, saturated at 10 m. β_c is by default the paper's own Sec. V-B estimate (`--courtesy-estimator boltzmann`, as in catk): the KL between two Boltzmann distributions over a neighbour's pooled samples, scored with and without the agent's collision cost. `kl` (the exact goal KL of Eq. 4, the default before 2026-10-18; older runs resume with it) and `demand` (braking demand, `demand.py`) are the alternatives.
  - `geometry.py`, `risk.py` and `hmm.py` are shared by the predictor-agnostic metrics and the SMART native pipeline (they were identical copies in both source repos). A change there changes both.
- **Run outputs** are read by `results.py`, `records.py` and `fit_levels`/`summarize`/`compare`:
  - `compute_responsibility` writes `OUT/windows.csv`, `obs/<scene>.pkl`, an optional `records/<scene>.pkl` and `config.json`.
  - Runs are resumable, and `config.json` pins their settings.
  - With `--num-shards/--shard-index`, a run writes `windows.shard-<i>-of-<n>.csv` instead, and every reader accepts both forms.
  - Records hold everything needed to visualise a run offline, without the model or the scene files.
- **Valid counterfactuals** (`motion_filter.py`) filter the motion set before β_s. Same intent is judged by HD lane topology (`--lane-route`, `lanes.py`) or, for maps built by perception that have no topology, from the trajectories' heading and lateral offset, optionally plus road edges (`--intent`, `intent.py`). On such maps the drivable area is judged by road edges (`--drivable-edges`, `edges.py`) instead of lane centrelines.
- **Collision attribution**: `blame.py` is the counterfactual verdict. It has three baselines: the rear-end rule (in `blame.py`), RSS (`rss.py`, geometry only, same-direction collisions) and the right of way (`right_of_way.py`: California Vehicle Code rules from the lane graph, lights and stop signs; duties written in the numpy STL of `stl.py`; crossing and turning collisions, same-direction ones are RSS's unless `same_direction_rules`). All four verdicts go into `crashes.csv` and the training logs; `scripts/responsibility/attribute_rules.py` recomputes the rule-based ones in an existing run.
- **Forced yields** (`forced_yield.py`): open loop, the braking an agent's entry into a conflict demands of a holder of the right of way (assertive but not unsafe). Its ground truth is `make_go_first.py` (logged lawful yields rewritten so that the yielder goes first); `evaluate_forced_yield.py` scores it.
- **Policies**: `recording.py` turns a MetaDrive episode into a rollout. It touches the env only through the attributes it reads, so the tests use stand-ins. `rollouts.scene_from_rollout` rebuilds the scene from it. `compute_responsibility --rollouts` measures the ego in that scene and attributes its crashes (`blame.py` → `crashes.csv`). Compare policies against `--policy replay` rollouts, not the plain logged run, because MetaDrive drops static vehicles.
- **Hooks into CAT**:
  - `adversarial.make_adv_generator(parser)` adds `--adv_selection cat|constrained|penalized|fair|near` and returns a drop-in subclass of `advgen.AdvGenerator`. The `cat` rule must stay identical to CAT's. `near` picks a near miss (`--near_gap`, `--near_tol`) instead of a collision. `--resp_model` takes the motion sets behind beta and avoidability from another predictor; the candidates are always CAT's DenseTNT modes, and with the default `densetnt` the code path is the original one.
  - `blame_reward.BlameWeighting` is used by `cat_RLtrain.py --blame_weighting share` (`--blame_model` chooses the predictor).
  - Run and model names encode the settings (e.g. `cat_fair1_0.3_share_s0`), and `compare_policies` averages seeds by parsing the `_s<seed>` suffix.
  - Swapping ego and adversary is a relabelling of the scenes (`swap.py`, `scripts/responsibility/swap_roles.py` → `raw_scenes_500_swapped`), read by `cat_advgen.py`/`cat_RLtrain.py --scenes_dir` (not `--data_dir`: advgen owns that name). A scene folder must hold only scene `.pkl` files: MetaDrive asserts it.

Upstream CAT code is `advgen/`, `saferl_algo/`, `saferl_plotter/`, `cat_*.py` and the numbered `scripts/*.py`. Keep changes there minimal. Code from catk4test (`responsibility/predictors/smart/native/`, `scripts/smart/`, `tests/smart/`, `docs/smart/`, `skills/`) is under the NVIDIA Source Code License for CAT-K (`LICENSE.CATK`). `cat_RLtrain.py` is indented with tabs; match that when you edit it.

## Conventions

- Write comments, docstrings and docs in plain, precise prose that states units and measured numbers. Each module starts with a docstring explaining what it computes and why.
- Commit messages have an imperative, outcome-focused subject (e.g. "Add the fair adversary: responsible and avoidable"). The body says what changed and what it measured.
