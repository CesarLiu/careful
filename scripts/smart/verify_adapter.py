"""Checks SmartPredictor (responsibility/predictors/smart/adapter.py) against
a real CAT-K SMART: the Scene -> PyG conversion, the rollouts and the
predictor-agnostic metrics on top of them. Prints PASS/FAIL per check and
ends with ALL CHECKS PASSED.

  0. SMART sees the map (CAT-K's preprocessing drops polylines spaced over 3 m)
  1. every agent observed at k is in the batch, under its own track id
  2. samples start where the agent is: the first sampled position is within
     --start-tol m of its logged position at k + 1
  3. the same generator gives the same samples; another gives other ones
  4. removing a neighbour changes the free agent's token log-probabilities
     (the neighbour is in the attention graph at k), and removing a far
     agent that is not in the batch changes nothing
  5. rollouts stop where the log does (8 s from 1.0 s, 7 s from 2.0 s)
  6. responsibility_at runs and gives finite beta_s, beta_c

Runs in CAT-K's environment (Python 3.11, PyG, torch_cluster, TensorFlow
for CAT-K's metrics module), from this repository's root:

    CATK_ROOT=~/catk python -m scripts.smart.verify_adapter --ckpt ~/catk/ckpts/clsft_E9.ckpt \\
        --scenes raw_scenes_500 --n 3
    CATK_ROOT=~/catk python -m scripts.smart.verify_adapter   # random weights, synthetic scene

Without --ckpt the weights are random (checks 1-5 still hold; the values of
6 mean nothing). Without --scenes a synthetic two-lane scene is used.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from responsibility.metrics import ResponsibilityConfig, interacting_neighbours, responsibility_at  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def synthetic_scene() -> Scene:
    t = (np.arange(91) - 10) * 0.1

    def track(x0, y, v):
        pos = np.zeros((91, 3), np.float32)
        pos[:, 0], pos[:, 1] = x0 + v * t, y
        return {"type": "VEHICLE", "metadata": {}, "state": {
            "position": pos, "heading": np.zeros(91, np.float32),
            "velocity": np.tile(np.array([v, 0.0], np.float32), (91, 1)),
            "length": np.full(91, 4.8, np.float32), "width": np.full(91, 2.0, np.float32),
            "height": np.full(91, 1.5, np.float32), "valid": np.ones(91, bool)}}

    lanes = {str(100 + i): {"type": "LANE_SURFACE_STREET",
                            "polyline": np.stack([np.linspace(-50, 150, 401), np.full(401, 3.5 * i),
                                                  np.zeros(401)], 1)}  # WOMD's 0.5 m spacing
             for i in range(2)}
    return Scene.from_description({
        "id": "synthetic", "tracks": {"0": track(0.0, 0.0, 11.0), "1": track(12.0, 0.0, 8.0),
                                      "2": track(-20.0, 3.5, 9.0)},
        "map_features": lanes, "dynamic_map_states": {},
        "metadata": {"sdc_id": "0", "objects_of_interest": ["0", "1"], "scenario_id": "synthetic"}})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", default=None, help="SMART checkpoint (default: random weights).")
    p.add_argument("--model-config", default="smart")
    p.add_argument("--catk-root", default=os.environ.get("CATK_ROOT"))
    p.add_argument("--scenes", default=None, help="CAT scene files (default: a synthetic scene).")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--start-tol", type=float, default=3.0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from responsibility.catk_export import numeric_id
    from responsibility.predictors.smart import load_smart_predictor
    from responsibility.predictors.smart.catk import require_catk
    from responsibility.predictors.smart.scene_data import check_map

    root = require_catk(args.catk_root)
    predictor = load_smart_predictor(args.ckpt or "random", root, args.model_config, args.device,
                                     batch_size=args.samples)
    if not args.ckpt:
        print("random weights: checks 1-5 apply; the values of check 6 mean nothing")

    scenes = [synthetic_scene()] if not args.scenes else [Scene.load(f) for f in scene_files(args.scenes)[:args.n]]
    k = args.step
    for scene in scenes:
        print(f"--- scene {scene.scenario_id}: {scene.n_agents} agents, step {k}")
        a = scene.sdc
        ctx = predictor._context(scene, k)
        check("0 SMART sees the map", check_map(scene, ctx.data), f"{int(ctx.data['map_save']['traj_pos'].shape[0])} map tokens")
        observed = {numeric_id(scene.track_ids[i]) for i in range(scene.n_agents) if scene.valid[i, k]}
        check("1 batch holds the agents observed at k", set(ctx.rows) == observed,
              f"{len(ctx.rows)} rows, {len(observed)} observed")

        dist = predictor.distribution(scene, k, a)
        _, lp, trajs = predictor.sample(dist, args.samples, torch.Generator().manual_seed(0))
        start = np.linalg.norm(trajs[:, 0] - scene.position[a, k + 1, :2], axis=-1)
        check("2 samples start at the agent", float(start.max()) <= args.start_tol, f"max {start.max():.2f} m")

        _, lp2, trajs2 = predictor.sample(dist, args.samples, torch.Generator().manual_seed(0))
        _, _, trajs3 = predictor.sample(dist, args.samples, torch.Generator().manual_seed(1))
        check("3 reproducible from the generator", np.array_equal(trajs, trajs2) and not np.array_equal(trajs, trajs3))

        horizon = 20
        nbrs = interacting_neighbours(scene, a, k, horizon, ResponsibilityConfig().interaction)
        b = next(iter(nbrs), None)
        if b is None:
            others = [i for i in range(scene.n_agents) if i != a and scene.valid[i, k]]
            b = min(others, key=lambda i: np.linalg.norm(scene.position[i, k, :2] - scene.position[a, k, :2]))
        _, lp_wo, _ = predictor.sample(predictor.distribution(scene, k, a, excluded=(b,)), args.samples,
                                       torch.Generator().manual_seed(0))
        check("4 removing a neighbour changes the free agent's odds", not torch.allclose(lp, lp_wo),
              f"mean log p {float(lp.mean()):.3f} -> {float(lp_wo.mean()):.3f}")
        absent = [i for i in range(scene.n_agents) if not scene.valid[i, k]]
        if absent:
            _, lp_abs, _ = predictor.sample(predictor.distribution(scene, k, a, excluded=(absent[0],)),
                                            args.samples, torch.Generator().manual_seed(0))
            check("4 removing an absent agent changes nothing", torch.equal(lp, lp_abs))

        lens = {kk: predictor.sample(predictor.distribution(scene, kk, a), 1)[2].shape[1]
                for kk in (10, 20) if scene.valid[a, kk]}
        check("5 rollouts stop where the log does", all(v == predictor.max_horizon(scene.n_steps, kk)
                                                         for kk, v in lens.items()), str(lens))

        obs = responsibility_at(predictor, scene, a, k, ResponsibilityConfig(), torch.Generator().manual_seed(0))
        ok = obs is not None and np.isfinite(obs.safety) and np.isfinite(obs.courtesy)
        check("6 metrics on SMART", ok, "" if obs is None else
              f"beta_s {obs.safety:+.2f} m, beta_c {obs.courtesy:.3f} nats, {len(obs.per_neighbour)} neighbours")

    print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {sorted(set(FAILED))}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
