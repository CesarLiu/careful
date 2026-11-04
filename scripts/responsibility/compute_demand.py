"""Courtesy as demand (responsibility/demand.py) of one agent toward its
vehicle neighbours, at every context step: how hard the agent's actual
future would have made each neighbour brake, had the neighbour driven its
plan without the agent (DenseTNT's motion set of the neighbour, the agent
left out of the scene).

Writes OUT/obs/<scene>.pkl (one entry per window and neighbour, with each
member's deceleration and weight) and, once every scene is done,
OUT/demand.csv:

  scene, scenario_id, agent_id, neighbour_id, step, time
  relation      "leads" / "follows" when the two share a lane at the step
                (demand not taken), else empty
  demand        beta_d = CVaR_alpha of the members' deceleration (m/s^2;
                DemandConfig.unavoidable for a member no braking clears)
  mean, p_conflict, p_unavoidable, horizon      (demand.Demand)
  holder, row_case   right_of_way.py's holder of the pair over the whole
                clip (forced_yield.pair_priority): agent, other or none
  fy_demand, fy_forced   the track-based measure on the neighbour's logged
                track (forced_yield.py): the braking a's worst entry
                demanded of it, and that value only where it held the
                right of way

--plans takes b's plans from elsewhere, to tell what the motion model adds:
"cv" one plan, b's constant-velocity continuation from k; "logged" one
plan, b's logged future (as far as it is seen without a gap). Neither needs
the model.

Runs resume: scenes with an obs file are skipped; OUT/config.json pins the
settings.

    python -m scripts.responsibility.compute_demand --scenes raw_scenes_500 --agent adv --use-ooi --n 100 \\
        --out-dir logs/responsibility/validity/demand/adv_logged
"""

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from responsibility.demand import DemandConfig, demand_at, lane_relation  # noqa: E402
from responsibility.forced_yield import forced_yield  # noqa: E402
from responsibility.interaction import InteractionConfig, interacting_neighbours, ooi_neighbours  # noqa: E402
from responsibility.metrics import MOTION_SETS, ResponsibilityConfig, effective_horizon, motion_set, \
    window_steps  # noqa: E402
from responsibility.models import add_model_arguments, load_model, model_settings  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402
from scripts.responsibility.compute_responsibility import pick_agent  # noqa: E402

PLANS = ("densetnt", "cv", "cv_prior", "logged")
PRIOR_LAG = 15  # steps (1.5 s): "cv_prior" takes b's speed from this far back
FIELDS = ["scene", "scenario_id", "agent_id", "neighbour_id", "step", "time", "relation", "demand", "mean",
          "p_conflict", "p_unavoidable", "horizon", "holder", "row_case", "fy_demand", "fy_forced"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--agent", default="sdc", help="sdc, adv (the other object of interest) or a track id.")
    p.add_argument("--use-ooi", action="store_true", help="Neighbours: the other objects of interest only.")
    d, c = ResponsibilityConfig(), DemandConfig()
    p.add_argument("--plans", default="densetnt", choices=PLANS,
                   help="b's plans: the motion model's set without the agent, or one constant-velocity or "
                        "logged plan (no model).")
    p.add_argument("--motion-set", default="nms", choices=MOTION_SETS)
    p.add_argument("--n-samples", type=int, default=d.n_safety_samples)
    p.add_argument("--stride", type=int, default=d.window_stride, help="Steps between context steps.")
    p.add_argument("--window-horizon", type=int, default=d.metric_horizon,
                   help="Steps of log a window needs after it (as compute_responsibility's --horizon, so "
                        "the windows match its runs).")
    p.add_argument("--horizon", type=int, default=c.horizon, help="Steps of the agent's future judged.")
    p.add_argument("--rho", type=float, default=c.rho, help="s before the neighbour brakes.")
    p.add_argument("--margin", type=float, default=c.margin, help="m: footprint gap counted as a conflict.")
    p.add_argument("--cvar-alpha", type=float, default=c.cvar_alpha)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_model_arguments(p)
    return p.parse_args(argv)


def _num(x):
    return "" if x is None else float(x)


def pair_baseline(scene: Scene, a: int, b: int) -> dict:
    """right_of_way.py's holder and forced_yield.py's track-based demand of the pair."""
    try:
        fy = forced_yield(scene, a, b)
    except Exception as e:  # a map the lane matching cannot read: leave the baseline out
        return {"holder": "", "row_case": f"error: {type(e).__name__}", "fy_demand": "", "fy_forced": ""}
    return {"holder": fy.holder, "row_case": fy.case, "fy_demand": _num(fy.demand), "fy_forced": _num(fy.forced)}


def single_plan(scene: Scene, b: int, step: int, kind: str, horizon: int) -> np.ndarray:
    """b's one plan [1, T, 2] from step + 1: its constant-velocity
    continuation ("cv"); the same along its heading at step but at the
    largest speed it had over the PRIOR_LAG steps before ("cv_prior": what b
    would have driven had it not slowed down meanwhile, e.g. for the agent);
    or its logged future up to its first unseen step ("logged"; T may then be
    shorter than ``horizon``, or 0)."""
    t = np.arange(1, horizon + 1)
    if kind == "cv":
        return (scene.position[b, step, :2] + t[:, None] * 0.1 * scene.velocity[b, step, :2])[None].astype(float)
    if kind == "cv_prior":
        past = [s for s in range(max(0, step - PRIOR_LAG), step + 1) if scene.valid[b, s]]
        speed = max(float(np.linalg.norm(scene.velocity[b, s])) for s in past)
        h = float(scene.heading[b, step])
        v = speed * np.array([np.cos(h), np.sin(h)])
        return (scene.position[b, step, :2] + t[:, None] * 0.1 * v)[None].astype(float)
    fut = np.arange(step + 1, min(step + 1 + horizon, scene.n_steps))
    seen = scene.valid[b, fut]
    n = len(fut) if seen.all() else int(np.argmin(seen))
    return scene.position[b, fut[:n], :2][None].astype(float)


def scene_demand(model, scene: Scene, agent: int, rcfg: ResponsibilityConfig, dcfg: DemandConfig,
                 use_ooi: bool, generator, plans: str = "densetnt") -> list:
    rows = []
    select = ooi_neighbours if use_ooi else interacting_neighbours
    baselines = {}
    for step in window_steps(scene, agent, rcfg):
        horizon = effective_horizon(scene, step, rcfg)
        for b in select(scene, agent, step, horizon, rcfg.interaction):
            if scene.types[b] != "VEHICLE" or not scene.valid[b, step]:
                continue
            if b not in baselines:
                baselines[b] = pair_baseline(scene, agent, b)
            row = {"scenario_id": scene.scenario_id, "agent_id": scene.track_ids[agent],
                   "neighbour_id": scene.track_ids[b], "step": step, "time": round(step * 0.1, 1),
                   "relation": lane_relation(scene, agent, b, step) or "", **baselines[b]}
            if not row["relation"]:
                if plans == "densetnt":
                    dist = model.distribution(scene, step, b, excluded=(agent,))
                    if dist is None:
                        continue
                    trajs, _, weights = motion_set(model, dist, rcfg, generator)
                else:
                    trajs, weights = single_plan(scene, b, step, plans, dcfg.horizon), None
                    if trajs.shape[1] == 0:
                        continue
                d = demand_at(np.asarray(trajs), weights, scene, agent, b, step, dcfg)
                if d is not None:
                    row.update(demand=d.demand, mean=d.mean, p_conflict=d.p_conflict,
                               p_unavoidable=d.p_unavoidable, horizon=d.horizon,
                               per_member=d.per_member.astype(np.float32),
                               weights=None if weights is None else np.asarray(weights, np.float32))
            rows.append(row)
    return rows


def collect(out: Path) -> pd.DataFrame:
    frames = []
    for p in sorted((out / "obs").glob("*.pkl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        with open(p, "rb") as f:
            rows = pickle.load(f)
        frames.append(pd.DataFrame([{k: r.get(k, "") for k in FIELDS if k != "scene"} | {"scene": p.stem}
                                    for r in rows], columns=FIELDS))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FIELDS)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    (out / "obs").mkdir(parents=True, exist_ok=True)
    rcfg = ResponsibilityConfig(n_safety_samples=args.n_samples, metric_horizon=args.window_horizon,
                                window_stride=args.stride, motion_set=args.motion_set, seed=args.seed,
                                use_ooi=args.use_ooi, courtesy=False, interaction=InteractionConfig())
    dcfg = DemandConfig(horizon=args.horizon, rho=args.rho, margin=args.margin, cvar_alpha=args.cvar_alpha)
    settings = {"responsibility": asdict(rcfg), "demand": asdict(dcfg), "plans": args.plans, "agent": args.agent,
                "scenes": str(Path(args.scenes).resolve()), "model": model_settings(args)}
    config_path = out / "config.json"
    if config_path.exists():
        stored = {"plans": "densetnt", **json.loads(config_path.read_text())}  # runs before --plans used the model
        if stored != json.loads(json.dumps(settings)):
            raise SystemExit(f"{out} holds results computed with other settings; use a new --out-dir")
    else:
        config_path.write_text(json.dumps(settings, indent=2))

    files = scene_files(args.scenes)[args.first:]
    if args.n is not None:
        files = files[: args.n]
    todo = [f for f in files if not (out / "obs" / f"{f.stem}.pkl").exists()]
    print(f"{len(files)} scenes, {len(files) - len(todo)} done, {len(todo)} to go", flush=True)
    if todo:
        model = load_model(args, args.device) if args.plans == "densetnt" else None
        generator = torch.Generator().manual_seed(args.seed)
        for n, path in enumerate(todo, 1):
            t0 = time.time()
            scene = Scene.load(path)
            try:
                agent = pick_agent(scene, args.agent, args.use_ooi)
            except ValueError as e:
                print(f"{path.name}: skipped ({e})", flush=True)
                rows = []
            else:
                rows = scene_demand(model, scene, agent, rcfg, dcfg, args.use_ooi, generator, args.plans)
            tmp = out / "obs" / f"{path.stem}.pkl.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump(rows, f)
            os.replace(tmp, out / "obs" / f"{path.stem}.pkl")
            dem = [r["demand"] for r in rows if "demand" in r]
            print(f"[{n}/{len(todo)}] {path.name}: {len(rows)} windows, max demand "
                  f"{max(dem, default=0):.2f} m/s^2 ({time.time() - t0:.0f} s)", flush=True)
    table = collect(out)
    table.to_csv(out / "demand.csv", index=False)
    print(f"wrote {out / 'demand.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
