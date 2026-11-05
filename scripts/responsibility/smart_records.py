"""Responsibility of one agent per scene with CAT-K's SMART, computed by catk's
own functions, written as a run of this repository: windows.csv, obs/<scene>.pkl,
config.json and records/<scene>.pkl. visualize_responsibility.py --record then
renders SMART exactly as it renders DenseTNT, and score_against_reference.py
and fit_levels.py read the run as any other.

catk's run (run_smart.sh, compute_responsibility) keeps only the values, not
the motion sets, so it cannot be drawn. Here every context window goes
through catk's compute_scenario_responsibility with ``agents=[agent]``, and
responsibility_for_agents is wrapped so that the rollouts it drew are kept:

  samples      the agent's motion set, the first n_safety_samples (40) of its
               full-scene job, over the metric horizon (2 s), scene frame;
               sample_log_prob is SMART's mean per-token log-probability of
               each sample (reward_model.mean_token_log_prob)
  courtesy     per courtesy target b, b's pooled samples (20 with the agent, 20
               without) and the two Boltzmann distributions over them,
               recomputed from those rollouts by catk's courtesy_from_rollouts
               steps; the record's value is catk's own

The values are catk's (safety_from_rollouts, courtesy_from_rollouts); the
record only adds what was drawn. Defaults match the DenseTNT runs of
logs/responsibility/labelled: 40 safety and 20 courtesy samples, CVaR alpha
0.1, D_g saturated at 10 m, 2 s horizon, a window every 0.2 s (--stride 2),
interaction neighbours (gap 10 m, PET 2 s, TTC 4 s, at most 8), Boltzmann
courtesy, and the neighbours' futures hidden (the open-loop reading DenseTNT
has). The random stream is seeded per scene (--seed + scene number).

Runs in catk's environment, from this repository's root:

    python -m scripts.responsibility.smart_records --ckpt ckpts/clsft_E9.ckpt \\
        --scenes logs/responsibility/labelled/scenes_adv --agent adv --out-dir logs/catk/labelled/adv
    python -m scripts.responsibility.visualize_responsibility \\
        --record logs/catk/labelled/adv/records/*.pkl --out-dir logs/catk/labelled/videos_adv

The scenes are first exported into catk's format (export_catk.py) under
OUT/cache. Re-running skips scenes that have both obs/ and records/.
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from responsibility.catk_export import numeric_id  # noqa: E402
from responsibility.metrics import Observation  # noqa: E402
from responsibility.records import VERSION, save_record  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402
from scripts.responsibility import export_catk  # noqa: E402

ROW_FIELDS = ["scene", "scenario_id", "agent_id", "step", "time", "speed", "safety", "courtesy",
              "n_neighbours", "safety_against", "courtesy_toward"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True, help="CAT scene files (<n>.pkl).")
    p.add_argument("--agent", default="adv", help="sdc, adv (the other object of interest) or a track id.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt", default="ckpts/clsft_E9.ckpt", help="SMART checkpoint (CAT-K's .ckpt).")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=32, help="Scene copies per model call.")
    p.add_argument("--stride", type=int, default=2, help="Context steps between windows (0.1 s each).")
    p.add_argument("--neighbor-future", default="hidden", choices=["hidden", "logged", "simulated"])
    p.add_argument("--courtesy-estimator", default="boltzmann", choices=["boltzmann"],
                   help="Only the Boltzmann estimate is drawn (the exact KL has no pooled samples).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def pick_agent(scene: Scene, which: str) -> int:
    if which == "sdc":
        return scene.sdc
    if which == "adv":
        others = [i for i in scene.objects_of_interest if i != scene.sdc]
        if not others:
            raise ValueError("no second object of interest")
        return others[0]
    return scene.index(which)


class Capture:
    """Wraps catk's responsibility_for_agents: calls it unchanged and keeps,
    for the window being evaluated, the rollouts it drew and what the
    record needs from them."""

    def __init__(self, original, metrics):
        self.original = original
        self.m = metrics
        self.windows = []

    def __call__(self, synthesizer, data, tokenized_map, tokenized_agent, queried, neighbors_of, all_idx, cfg,
                 map_points=None, sampling_scheme=None):
        kept = {}
        run_jobs = synthesizer.run_jobs

        def capturing(*args, **kwargs):
            out = run_jobs(*args, **kwargs)
            kept["results"] = out
            return out

        synthesizer.run_jobs = capturing
        try:
            results, plan = self.original(synthesizer, data, tokenized_map, tokenized_agent, queried, neighbors_of,
                                          all_idx, cfg, map_points=map_points, sampling_scheme=sampling_scheme)
        finally:
            synthesizer.run_jobs = run_jobs
        self.windows.append(self.frame(data, kept.get("results", []), plan, queried, all_idx, cfg, map_points))
        return results, plan

    def frame(self, data, rollouts, plan, queried, all_idx, cfg, map_points):
        m = self.m
        from responsibility.predictors.smart.native.motion_set import take_samples
        from responsibility.predictors.smart.native.reward_model import mean_token_log_prob

        a = queried[0]
        horizon = m.effective_horizon(data, cfg)
        out = {"samples": None, "sample_log_prob": None, "courtesy": {}}
        if a in plan.full_job:
            r = take_samples(rollouts[plan.full_job[a]], cfg.n_safety_samples)
            out["samples"] = r["pred_traj_10hz"][:, 0, :horizon].float().cpu().numpy()
            out["sample_log_prob"] = mean_token_log_prob(r["token_log_prob"][:, 0],
                                                         r["next_token_valid"][:, 0]).float().cpu().numpy()
        ids = data["agent"]["id"]
        for b in plan.courtesy_targets.get(a, []):
            with_b = take_samples(rollouts[plan.full_job[b]], cfg.n_courtesy_samples)
            without_b = rollouts[plan.without_job[(a, b)]]
            # courtesy_from_rollouts, step by step, to keep the pooled plans and both distributions
            h = m.effective_horizon(data, cfg, with_b["pred_traj_10hz"].shape[2])
            other_with = [i for i in all_idx if i != b]
            if h == 0 or not other_with:
                continue
            other_without = [i for i in other_with if i != a]
            pooled = {k: torch.cat([with_b[k], without_b[k]], dim=0) for k in ("pred_traj_10hz", "pred_head_10hz")}
            ego = m.rollout_boxes(pooled, 0, data["agent"]["shape"][b], h)
            nb_with = m.ground_truth_boxes(data, other_with, cfg.num_historical_steps, h)
            nb_without = (m.ground_truth_boxes(data, other_without, cfg.num_historical_steps, h) if other_without
                          else m._empty_boxes(ego.pos, h))
            log_p1 = m._boltzmann_log_prob(m._reward_against_neighbors(ego, nb_with, map_points, cfg))
            log_p2 = m._boltzmann_log_prob(m._reward_against_neighbors(ego, nb_without, map_points, cfg))
            kl = float((log_p1.exp() * (log_p1 - log_p2)).sum().clamp(min=0.0))
            out["courtesy"][int(ids[b])] = {
                "kind": "boltzmann", "plans": ego.pos.float().cpu().numpy(),
                "p_with": log_p1.exp().float().cpu().numpy(), "p_without": log_p2.exp().float().cpu().numpy(),
                "n_with_a": cfg.n_courtesy_samples, "value": kl}
        return out


def to_frame(scene: Scene, agent: int, obs, captured, tid_of) -> dict:
    """One record frame in responsibility/records.py's layout from a catk
    observation and the rollouts captured in its window."""
    k = int(obs.k)
    per_neighbour = {}
    for b, (s, c) in (obs.per_neighbour or {}).items():
        tid = tid_of[int(b)]
        ev = (obs.interaction or {}).get(int(b), {})
        per_neighbour[tid] = {"safety": float(s), "courtesy": float(c), "type": scene.types[scene.index(tid)],
                              **{key: float(ev[key]) for key in ("min_gap", "pet", "ttc") if key in ev}}
    beta = [float(x) for x in obs.beta]
    observation = Observation(scenario_id=scene.scenario_id, agent=agent, agent_id=scene.track_ids[agent], step=k,
                              safety=beta[0], courtesy=beta[1],
                              speed=float(np.linalg.norm(scene.velocity[agent, k, :2])), per_neighbour=per_neighbour)
    samples, log_prob = captured["samples"], captured["sample_log_prob"]
    if samples is None:  # no neighbours: no motion set was drawn; show the agent standing
        samples = np.repeat(scene.position[agent, k, :2][None, None], 2, axis=1).astype(np.float32)
        log_prob = np.zeros(1, np.float32)
    courtesy = {tid_of[b]: v for b, v in captured["courtesy"].items()}
    for tid, v in courtesy.items():  # catk's value, not the recomputation (they agree to float precision)
        v["value"] = per_neighbour.get(tid, {}).get("courtesy", v["value"])
    return {"step": k, "observation": asdict(observation), "samples": samples, "sample_log_prob": log_prob,
            "sample_score": "SMART mean log p(token) of sample", "metric_samples": captured["samples"] is not None,
            "goals": None, "courtesy": courtesy, "kept": {}}, observation


def main(argv=None):
    args = parse_args(argv)
    ckpt = Path(args.ckpt).resolve()
    scenes_dir = Path(args.scenes).resolve()
    out = Path(args.out_dir).resolve()
    for sub in ("obs", "records"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    files = scene_files(scenes_dir)
    todo = [f for f in files if not ((out / "obs" / f"{f.stem}.pkl").exists()
                                     and (out / "records" / f"{f.stem}.pkl").exists())]
    print(f"{len(files)} scenes, {len(files) - len(todo)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    export_catk.main(["--scenes", str(scenes_dir), "--out-dir", str(out)])

    import responsibility.predictors.smart.native.dataset_builder as builder
    import responsibility.predictors.smart.native.metrics as m
    from responsibility.predictors.smart.native.interaction import InteractionConfig
    from responsibility.predictors.smart.native.motion_set import MotionSetSynthesizer
    from responsibility.predictors.smart.native.pipeline import (compose_responsibility_config, default_sampling_scheme,
                                             iter_scenarios, load_smart_model, resolve_device)
    from responsibility.predictors.smart.native.results_io import scenario_files

    hydra_cfg = compose_responsibility_config(overrides=["model=smart"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, str(ckpt), device=device)
    cfg = m.ResponsibilityConfig(
        num_historical_steps=hydra_cfg.model.model_config.decoder.num_historical_steps,
        window_stride=args.stride, neighbor_future=args.neighbor_future,
        courtesy_estimator=args.courtesy_estimator, interaction=InteractionConfig())
    synthesizer = MotionSetSynthesizer(model, default_sampling_scheme(), rollout_batch_size=args.batch)
    capture = Capture(builder.responsibility_for_agents, m)
    builder.responsibility_for_agents = capture

    settings = {
        # the keys score_against_reference.py and fit_levels.py compare across runs
        "responsibility": {"n_safety_samples": cfg.n_safety_samples, "cvar_alpha": cfg.cvar_alpha,
                           "d_sat": cfg.d_sat, "metric_horizon": cfg.metric_horizon,
                           "window_stride": cfg.window_stride, "motion_set": "sampled", "use_ooi": False,
                           "courtesy_estimator": cfg.courtesy_estimator, "n_courtesy_samples": cfg.n_courtesy_samples,
                           "courtesy_skip_distance": cfg.courtesy_skip_distance, "seed": args.seed},
        "catk": asdict(cfg),
        "agent": args.agent, "scenes": str(scenes_dir),
        "model": {"name": "smart", "ckpt": str(ckpt)},
    }
    config_path = out / "config.json"
    if config_path.exists():
        stored = json.loads(config_path.read_text())
        if stored != json.loads(json.dumps(settings)):
            raise SystemExit(f"{out} holds results computed with other settings; use a new --out-dir")
    else:
        config_path.write_text(json.dumps(settings, indent=2))

    cache = [f.stem for f in scenario_files(str(out / "cache"))]
    wanted = {f.stem: f for f in todo}
    indices = [i for i, stem in enumerate(cache) if stem in wanted]
    rows_path = out / "windows.csv"
    new_file = not rows_path.exists()
    with open(rows_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        if new_file:
            writer.writeheader()
        for n, (index, _, data) in enumerate(iter_scenarios(str(out / "cache"), device, indices=indices), 1):
            t0 = time.time()
            stem = cache[index]
            scene = Scene.load(wanted[stem])
            try:
                agent = pick_agent(scene, args.agent)
            except ValueError as e:
                print(f"{stem}: skipped ({e})", flush=True)
                continue
            tid_of = {numeric_id(t): t for t in scene.track_ids}
            ids = data["agent"]["id"].tolist()
            a = ids.index(numeric_id(scene.track_ids[agent]))
            torch.manual_seed(args.seed * 100003 + (int(stem) if stem.isdigit() else 0))
            capture.windows = []
            steps = builder.window_steps(data, cfg, synthesizer, cfg.window_stride)
            frames, observations = [], []
            for k in steps:
                capture.windows = []
                with builder.shifted_context(data, k, cfg.num_historical_steps):
                    found = builder._responsibility_at_current_step(synthesizer, data, cfg, stem, k, None, None,
                                                                   agents=[a])
                if not found:  # the agent is not observed at k
                    continue
                frame, obs = to_frame(scene, agent, found[0], capture.windows[-1], tid_of)
                frames.append(frame)
                observations.append(obs)
            record = {"version": VERSION, "scene": dict(scene.__dict__), "agent": agent,
                      "agent_id": scene.track_ids[agent], "config": {**settings["responsibility"], "model": "smart"},
                      "top_mass": None, "frames": frames, "scene_file": stem, "model": "smart"}
            save_record(record, out / "records" / f"{stem}.pkl")
            for obs in observations:
                writer.writerow({"scene": stem, **obs.as_row()})
            handle.flush()
            tmp = out / "obs" / f"{stem}.pkl.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump([asdict(o) for o in observations], f)
            os.replace(tmp, out / "obs" / f"{stem}.pkl")
            s = [o.safety for o in observations]
            c = [o.courtesy for o in observations]
            print(f"[{n}/{len(indices)}] {stem}: {len(observations)} windows, max safety {max(s, default=0):+.2f} m, "
                  f"max courtesy {max(c, default=0):.3f} nats ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
