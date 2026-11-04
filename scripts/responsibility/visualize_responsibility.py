"""A video of an agent's counterfactual responsibility through one scene: one
frame per context step t_k, stitched into a GIF and an MP4 (the MP4 needs
OpenCV).

Each frame shows
  scene     the map, every agent at t_k with its last second of history, the
            queried agent's motion set (DenseTNT goal samples, the first 2 s
            solid and coloured by goal probability, the rest faint), its logged
            future, and the neighbours the metrics compared it with; the
            neighbour behind beta_s is outlined in red, the one behind beta_c
            in blue
  courtesy  that neighbour's goal distribution with the agent in the scene and
            without it (the two sides of the KL)
  timeline  beta_s (m) and beta_c (nats) over the clip, the current step marked,
            and, given a levels.csv from fit_levels.py, the level of every window
            as background colour (aggressive levels hatched)

Two ways to run it:

  live      --scene N [N ...]: computes the frames with DenseTNT, or --model mtr (the same code as
            compute_responsibility.py; --run takes a run's settings so the
            values equal its windows.csv) and saves them as record.pkl. The
            model is loaded once for all scenes, and --agent sdc adv renders
            each scene for both objects of interest (with --use-ooi: the
            same pair, roles swapped)
  offline   --record path/to/<scene>.pkl [...]: renders records written by
            compute_responsibility.py --save-records (or by a live run) --
            no DenseTNT, TensorFlow or scene files needed

One video goes to --out-dir; several go to --out-dir/<scene> (live, with
several agents: --out-dir/<scene>_<agent>).

Comparing courtesy estimators: --courtesy-estimator kl boltzmann (or any
two or three of kl, boltzmann, demand) computes the scene once per
estimator -- beta_s and the agent's motion set are the same in each -- and
renders them in one video: the scene on the left, one row of courtesy
panels per estimator, and beta_s and each beta_c on their own axes below.
Offline, --record a.pkl b.pkl --compare does the same for records of one
scene and agent made with different estimators.

Examples (from the repository root):
    python -m scripts.responsibility.visualize_responsibility --scene 17 \\
        --run logs/responsibility/sdc --levels logs/responsibility/levels/levels.csv \\
        --out-dir logs/responsibility/video_17
    python -m scripts.responsibility.visualize_responsibility \\
        --record logs/responsibility/sdc/records/17.pkl --out-dir logs/responsibility/video_17
    python -m scripts.responsibility.visualize_responsibility --scene 17 23 42 \\
        --agent sdc adv --use-ooi --out-dir logs/responsibility/videos_ooi
    python -m scripts.responsibility.visualize_responsibility --scene 160 --agent sdc --use-ooi \\
        --motion-set nms --courtesy-estimator kl boltzmann --out-dir logs/responsibility/compare_160
"""

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from responsibility.demand import DemandConfig  # noqa: E402
from responsibility.interaction import InteractionConfig  # noqa: E402
from responsibility.metrics import COURTESY_ESTIMATORS, MOTION_SETS, ResponsibilityConfig  # noqa: E402
from responsibility.motion_filter import MotionFilterConfig  # noqa: E402
from responsibility.records import load_record, save_record, scene_of  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402

AGENT_COLOR = "#d62728"
NEIGHBOUR_COLOR = "#1f77b4"
OTHER_COLOR = "#9a9a9a"
SAFETY_COLOR = "#d62728"
COURTESY_COLOR = "#1f77b4"
MAP_STYLE = {  # type prefix -> (colour, width)
    "ROAD_EDGE": ("#505050", 1.0),
    "ROAD_LINE": ("#b0b0b0", 0.6),
    "LANE": ("#dcdcdc", 0.8),
    "CROSSWALK": ("#c9b458", 0.8),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", nargs="+",
                     help="Live: scene file indices (e.g. 17 23) or paths to .pkl scenes; the model is loaded once.")
    src.add_argument("--record", nargs="+", help="Offline: records from compute_responsibility.py --save-records.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--levels", default=None, help="levels.csv from fit_levels.py, for the timeline.")
    p.add_argument("--view-radius", type=float, default=None, help="m; default: fitted to the scene (30-90 m).")
    p.add_argument("--ego-heatmap", action="store_true", help="Also draw the agent's own goal distribution.")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--compare", action="store_true",
                   help="Offline: render the --record files (one scene and agent, different courtesy estimators) "
                        "side by side in one video.")
    live = p.add_argument_group("live mode")
    live.add_argument("--scenes", default="raw_scenes_500")
    live.add_argument("--agent", nargs="+", default=["sdc"],
                      help="sdc, adv or a track id; several (e.g. sdc adv) render each scene for each, which with "
                           "--use-ooi swaps the roles of the two objects of interest.")
    live.add_argument("--run", default=None, help="compute_responsibility.py output whose settings to use.")
    d = ResponsibilityConfig()
    live.add_argument("--stride", type=int, default=d.window_stride)
    live.add_argument("--n-samples", type=int, default=d.n_safety_samples)
    live.add_argument("--use-ooi", action="store_true",
                      help="Neighbours: the scenario's other objects of interest only (as compute_responsibility.py's).")
    live.add_argument("--courtesy-same-mode", choices=["lanes", "path"], default=None,
                      help="beta_c over the neighbour's own drive mode only (as compute_responsibility.py's).")
    live.add_argument("--courtesy-estimator", nargs="+", default=[d.courtesy_estimator], choices=COURTESY_ESTIMATORS,
                      help="beta_c as the goal KL (kl, nats), the paper's Sec. V-B estimate (boltzmann, nats) or the "
                           "braking demand (demand, m/s^2), as compute_responsibility.py's; several render one "
                           "comparison video.")
    live.add_argument("--demand-horizon", type=int, default=d.demand.horizon, help="10 Hz steps of the agent's future.")
    live.add_argument("--demand-rho", type=float, default=d.demand.rho, help="s before the neighbour brakes.")
    live.add_argument("--motion-set", default=d.motion_set, choices=MOTION_SETS,
                      help="As compute_responsibility.py's; the video draws exactly the scored set.")
    live.add_argument("--horizon", type=int, default=d.metric_horizon)
    live.add_argument("--seed", type=int, default=d.seed)
    live.add_argument("--device", default=None, help="Default: cuda if available.")
    from responsibility.models import add_model_arguments

    add_model_arguments(p)  # --model densetnt|mtr --checkpoint ... (as for the run being shown)
    return p.parse_args()


# ----------------------------------------------------------------------------
# live capture
# ----------------------------------------------------------------------------


def estimators(args) -> List[str]:
    """The courtesy estimators asked for (a list; one name is accepted too)."""
    value = getattr(args, "courtesy_estimator", None) or ["kl"]
    return [value] if isinstance(value, str) else list(value)


def config_from(args, estimator=None) -> ResponsibilityConfig:
    """The run's settings; ``estimator`` overrides the courtesy estimator
    (as one row of a comparison)."""
    cfg = _config_from(args)
    return cfg if estimator is None else replace(cfg, courtesy_estimator=estimator)


def _config_from(args) -> ResponsibilityConfig:
    if args.run is None:
        return ResponsibilityConfig(n_safety_samples=args.n_samples, metric_horizon=args.horizon,
                                    window_stride=args.stride, seed=args.seed, motion_set=args.motion_set,
                                    use_ooi=args.use_ooi, courtesy_same_mode=args.courtesy_same_mode,
                                    courtesy_estimator=estimators(args)[0],
                                    demand=DemandConfig(horizon=args.demand_horizon, rho=args.demand_rho))
    saved = json.loads((Path(args.run) / "config.json").read_text())["responsibility"]
    saved["interaction"] = InteractionConfig(**saved["interaction"])
    if "filter" in saved:
        saved["filter"] = MotionFilterConfig(**saved["filter"])
    if "demand" in saved:
        saved["demand"] = DemandConfig(**saved["demand"])
    if args.use_ooi and not saved.get("use_ooi", False):
        raise SystemExit(f"{args.run} was computed without --use-ooi: drop --use-ooi or --run")
    return ResponsibilityConfig(**saved)


def pick_agent(scene: Scene, which: str, use_ooi: bool = False) -> int:
    if which == "sdc":
        agent = scene.sdc
    elif which == "adv":
        others = [i for i in scene.objects_of_interest if i != scene.sdc]
        if not others:
            raise ValueError("no second object of interest")
        agent = others[0]
    else:
        agent = scene.index(which)
    if use_ooi and agent not in scene.objects_of_interest:
        raise ValueError(f"--use-ooi: agent {scene.track_ids[agent]} is not an object of interest")
    return agent


def scene_paths(args) -> List[Path]:
    files = {p.stem: p for p in scene_files(args.scenes)} if any(s.isdigit() for s in args.scene) else {}
    return [files[s] if s.isdigit() else Path(s) for s in args.scene]


def live_records(args) -> Iterator[Tuple[str, List[Dict]]]:
    """(name, records) for every scene and agent asked for: one record per
    courtesy estimator asked for, the model loaded once; a scene without
    the agent is skipped."""
    import torch

    from responsibility.models import load_model
    from responsibility.records import run_scene

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfgs = [config_from(args, est) for est in estimators(args)]
    cfg = cfgs[0]
    model = None
    for path in scene_paths(args):
        scene = Scene.load(path)
        for which in args.agent:
            name = path.stem if len(args.agent) == 1 else f"{path.stem}_{which}"
            try:
                agent = pick_agent(scene, which, cfg.use_ooi)
            except ValueError as e:
                print(f"{name}: skipped ({e})", flush=True)
                continue
            if model is None:
                model = load_model(args, device)
            print(f"{name}: agent {scene.track_ids[agent]}", flush=True)
            records = []
            for c in cfgs:
                observations, record = run_scene(model, scene, agent, c)
                record["scene_file"] = path.stem
                for obs in observations:
                    print(f"  [{c.courtesy_estimator}] t={obs.step / 10:.1f}s: safety {obs.safety:+.3f} m, "
                          f"courtesy {obs.courtesy:.4f} {unit_of(c)}, {len(obs.per_neighbour)} neighbour(s)",
                          flush=True)
                records.append(record)
            yield name, records


# ----------------------------------------------------------------------------
# rendering (from records only)
# ----------------------------------------------------------------------------


def map_segments(scene: Scene):
    """(colour, width) -> polylines, per map feature type group."""
    groups = {}
    for feature in scene.map_features.values():
        kind = str(feature.get("type", ""))
        style = next((v for k, v in MAP_STYLE.items() if kind.startswith(k)), None)
        pts = feature.get("polyline", feature.get("polygon"))
        if style is None or pts is None:
            continue
        pts = np.asarray(pts)[:, :2]
        if kind.startswith("CROSSWALK") and len(pts) > 2:
            pts = np.concatenate([pts, pts[:1]])
        if len(pts) > 1:
            groups.setdefault(style, []).append(pts)
    return groups


def box(center, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    dx = np.array([1, 1, -1, -1]) * length / 2
    dy = np.array([1, -1, -1, 1]) * width / 2
    return np.stack([center[0] + c * dx - s * dy, center[1] + s * dx + c * dy], -1)


def unit_of(cfg) -> str:
    """The unit of beta_c for a run's config (a ResponsibilityConfig or its saved dict)."""
    kind = cfg["courtesy_estimator"] if isinstance(cfg, dict) and "courtesy_estimator" in cfg else \
        getattr(cfg, "courtesy_estimator", "kl")
    return "m/s^2" if kind == "demand" else "nats"


def worst(obs, key):
    items = [(k, v[key]) for k, v in obs["per_neighbour"].items() if v.get(key) is not None]
    return max(items, key=lambda kv: kv[1], default=(None, None))


def fit_radius(scene, agent, frames, horizon):
    need = 25.0
    for f in frames:
        centre = scene.position[agent, f["step"], :2]
        pts = [f["samples"][:, :horizon].reshape(-1, 2)]
        pts += [scene.position[scene.index(b), f["step"], :2][None] for b in f["observation"]["per_neighbour"]]
        need = max(need, float(np.abs(np.concatenate(pts) - centre).max()) + 8.0)
    return float(np.clip(need, 30.0, 90.0))


def draw_scene(ax, scene, agent, frame, map_groups, radius, horizon, ego_heatmap, weighted=False):
    step, obs = frame["step"], frame["observation"]
    centre = scene.position[agent, step, :2]
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    for (colour, width), lines in map_groups.items():
        ax.add_collection(LineCollection(lines, colors=colour, linewidths=width, zorder=1))
    if ego_heatmap:
        g = frame["goals"]
        ax.scatter(g["points"][:, 0], g["points"][:, 1], c=np.log(g["prob"]), cmap="Greys", s=4, alpha=0.5,
                   zorder=2, linewidths=0)

    neighbours = {scene.index(b) for b in obs["per_neighbour"]}
    s_id, _ = worst(obs, "safety")
    c_id, _ = worst(obs, "courtesy")
    fut = slice(step + 1, step + 1 + horizon)
    for i in range(scene.n_agents):
        if not scene.valid[i, step]:
            continue
        colour = AGENT_COLOR if i == agent else NEIGHBOUR_COLOR if i in neighbours else OTHER_COLOR
        hist = slice(max(0, step - 10), step + 1)
        h = scene.position[i, hist, :2][scene.valid[i, hist]]
        if len(h) > 1:
            ax.plot(*h.T, color=colour, lw=0.8, alpha=0.5, zorder=3)
        length, width = scene.shape_at(i, step)
        tid = scene.track_ids[i]
        edge, lw = "black", 0.5
        if tid == s_id:
            edge, lw = SAFETY_COLOR, 2.2
        if tid == c_id:
            edge, lw = (COURTESY_COLOR, 2.2) if tid != s_id else ("#7b2d8e", 2.6)
        ax.add_patch(Polygon(box(scene.position[i, step, :2], scene.heading[i, step], length, width), closed=True,
                             facecolor=colour, edgecolor=edge, lw=lw,
                             alpha=0.9 if i == agent or i in neighbours else 0.45, zorder=6))
        if i in neighbours:
            f = scene.position[i, fut, :2][scene.valid[i, fut]]
            if len(f) > 1:
                ax.plot(*f.T, color=NEIGHBOUR_COLOR, lw=1.3, ls="--", zorder=5)
            ax.annotate(tid, scene.position[i, step, :2], fontsize=6, color=NEIGHBOUR_COLOR,
                        xytext=(3, 3), textcoords="offset points", zorder=7)

    samples = frame["samples"]
    lp = frame["sample_log_prob"] if frame["sample_log_prob"] is not None else np.zeros(len(samples))
    norm = plt.Normalize(lp.min(), lp.max() if lp.max() > lp.min() else lp.min() + 1e-6)
    cmap = plt.get_cmap("plasma")
    for j in np.argsort(lp):
        ax.plot(*samples[j, horizon - 1:].T, color=cmap(norm(lp[j])), lw=0.6, alpha=0.25, zorder=4)
        ax.plot(*samples[j, :horizon].T, color=cmap(norm(lp[j])), lw=1.4, alpha=0.8, zorder=4)
    logged = scene.position[agent, fut, :2][scene.valid[agent, fut]]
    if len(logged) > 1:
        ax.plot(*logged.T, color="black", lw=1.6, ls="--", zorder=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    weighted = weighted and frame.get("metric_samples", True)  # display-only frames are always samples
    ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.01).set_label(
        "log weight of trajectory" if weighted else frame.get("sample_score", "log p(goal) of sample"), fontsize=7)
    label = "motion set (2 s solid)" if frame.get("metric_samples", True) else "motion set (display only)"
    handles = [
        plt.Line2D([], [], color=cmap(0.8), lw=1.5, label=label),
        plt.Line2D([], [], color="black", ls="--", lw=1.5, label="agent, logged"),
        plt.Line2D([], [], color=NEIGHBOUR_COLOR, ls="--", lw=1.3, label="neighbours, logged"),
        plt.Line2D([], [], color=SAFETY_COLOR, lw=2.2, label="beta_s from"),
        plt.Line2D([], [], color=COURTESY_COLOR, lw=2.2, label="beta_c toward"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_courtesy(axes, scene, agent, frame, map_groups, radius, horizon, demand_cfg=None):
    obs, step = frame["observation"], frame["step"]
    c_id, value = worst(obs, "courtesy")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if c_id is None or c_id not in frame["courtesy"]:
        pn = obs["per_neighbour"]
        shared = [k for k, v in pn.items() if v.get("courtesy_relation")]
        unpredicted = [k for k, v in pn.items() if v.get("type") == "VEHICLE" and v.get("courtesy") is None
                       and not v.get("courtesy_relation")]
        if shared:
            text = (f"vehicle {shared[0]} is in the agent's lane\n(the agent {pn[shared[0]]['courtesy_relation']} it):"
                    "\nbraking demand is not measured\n(rear-end territory)")
        elif c_id is not None:  # measured, nothing recorded: the Boltzmann estimate's skip distance
            text = f"courtesy toward vehicle {c_id}: {value:.3f}\n(the two stay too far apart\nover the horizon to interact)"
        elif unpredicted:
            text = (f"vehicle {unpredicted[0]}: no prediction\nfrom the motion model at this step\n"
                    "(e.g. too little history):\ncourtesy not computed")
        elif pn:
            text = "no vehicle neighbour\n(courtesy is measured toward vehicles only)"
        else:
            text = "no neighbour\n(no courtesy)"
        axes[0].text(0.5, 0.5, text, ha="center", va="center", transform=axes[0].transAxes, fontsize=8)
        axes[1].axis("off")
        return
    if frame["courtesy"][c_id].get("kind") == "boltzmann":
        draw_boltzmann(axes, scene, agent, frame, c_id, map_groups, radius)
        return
    if frame["courtesy"][c_id].get("kind") == "demand":
        draw_demand(axes, scene, agent, frame, c_id, map_groups, radius, demand_cfg or DemandConfig())
        return
    b = scene.index(c_id)
    centre = scene.position[b, step, :2]
    r = min(radius, 60.0)
    fut = slice(step + 1, step + 1 + horizon)
    pair = frame["courtesy"][c_id]
    for ax, goals, title, present in ((axes[0], pair["with"], "with the agent", True),
                                      (axes[1], pair["without"], "without the agent", False)):
        for (colour, width), lines in map_groups.items():
            ax.add_collection(LineCollection(lines, colors=colour, linewidths=width * 0.8, zorder=1))
        # each panel on its own scale: a spread-out distribution would vanish next to a peaked one
        p = goals["prob"]
        keep = p > p.max() * 1e-3
        inside = goals.get("in_support", np.ones(len(p), dtype=bool))  # the goals the KL was taken over
        out = keep & ~inside
        if out.any():  # left out of the KL (another drive mode, or unreachable): grey
            ax.scatter(*goals["points"][out].T, c=np.log10(p[out]), cmap="Greys", vmin=np.log10(p.max()) - 4,
                       vmax=np.log10(p.max()) + 1, s=5, linewidths=0, zorder=2)
        keep &= inside
        pts = goals["points"][keep]
        ax.scatter(pts[:, 0], pts[:, 1], c=np.log10(p[keep]), cmap="Blues", s=5, linewidths=0, zorder=2)
        if "support_mass" in goals:
            ax.text(0.02, 0.02, f"in the KL: {goals['support_mass']:.0%} of the mass", transform=ax.transAxes,
                    fontsize=6.5, va="bottom", bbox=dict(facecolor="white", alpha=0.8, lw=0))
        length, width = scene.shape_at(b, step)
        ax.add_patch(Polygon(box(centre, scene.heading[b, step], length, width), closed=True,
                             facecolor=NEIGHBOUR_COLOR, edgecolor="black", lw=0.5, zorder=4))
        f = scene.position[b, fut, :2][scene.valid[b, fut]]
        if len(f) > 1:
            ax.plot(*f.T, color=NEIGHBOUR_COLOR, lw=1.2, ls="--", zorder=3)
        a_len, a_wid = scene.shape_at(agent, step)
        ax.add_patch(Polygon(box(scene.position[agent, step, :2], scene.heading[agent, step], a_len, a_wid),
                             closed=True, facecolor=AGENT_COLOR if present else "none",
                             edgecolor="black" if present else AGENT_COLOR, ls="-" if present else "--",
                             lw=0.8, zorder=4))
        ax.set_xlim(centre[0] - r, centre[0] + r)
        ax.set_ylim(centre[1] - r, centre[1] + r)
        ax.set_aspect("equal")
        ax.set_facecolor("#fbfbfb")
        ax.set_title(title, fontsize=8)
    restricted = "in_support" in pair["with"]
    axes[0].text(1.0, 1.13, f"courtesy: goals of vehicle {c_id} (blue"
                 + (", grey: left out of the KL" if restricted else "") + f"), KL = {value:.3f} nats",
                 transform=axes[0].transAxes, ha="center", fontsize=8.5)


def draw_boltzmann(axes, scene, agent, frame, c_id, map_groups, radius):
    """The paper's Sec. V-B courtesy: b's pooled samples, each coloured by how
    much the agent's actual future changes its Boltzmann probability
    (log P1 / P2: red loses, blue gains); the two distributions beside."""
    step = frame["step"]
    pair = frame["courtesy"][c_id]
    plans, p1, p2 = pair["plans"], pair["p_with"].astype(float), pair["p_without"].astype(float)
    H = plans.shape[1]
    b = scene.index(c_id)
    fut = slice(step + 1, step + 1 + H)
    shown = np.concatenate([plans.reshape(-1, 2), scene.position[agent, fut, :2][scene.valid[agent, fut]],
                            scene.position[[b, agent], step, :2]])
    lo, hi = shown.min(0), shown.max(0)
    centre = (lo + hi) / 2
    r = float(np.clip((hi - lo).max() / 2 + 6.0, 15.0, min(radius, 60.0)))
    ax = axes[0]
    for (colour, width), lines in map_groups.items():
        ax.add_collection(LineCollection(lines, colors=colour, linewidths=width * 0.8, zorder=1))
    shift = np.log(np.maximum(p1, 1e-12) / np.maximum(p2, 1e-12))
    cmap = plt.get_cmap("RdBu")
    norm = plt.Normalize(-3.0, 3.0)
    for j in np.argsort(-np.abs(shift)):
        ax.plot(*plans[j].T, color=cmap(norm(shift[j])), lw=1.3, alpha=0.85, zorder=3)
    a_fut = scene.position[agent, fut, :2][scene.valid[agent, fut]]
    if len(a_fut) > 1:
        ax.plot(*a_fut.T, color=AGENT_COLOR, lw=2.0, ls="--", zorder=5)
    a_len, a_wid = scene.shape_at(agent, step)
    ax.add_patch(Polygon(box(scene.position[agent, step, :2], scene.heading[agent, step], a_len, a_wid), closed=True,
                         facecolor=AGENT_COLOR, edgecolor="black", lw=0.8, zorder=6))
    length, width = scene.shape_at(b, step)
    ax.add_patch(Polygon(box(scene.position[b, step, :2], scene.heading[b, step], length, width), closed=True,
                         facecolor=NEIGHBOUR_COLOR, edgecolor="black", lw=0.5, zorder=6))
    ax.set_xlim(centre[0] - r, centre[0] + r)
    ax.set_ylim(centre[1] - r, centre[1] + r)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    ax.set_title(f"courtesy, paper Sec. V-B: vehicle {c_id}'s\n{len(plans)} samples ({H / 10:.0f} s)", fontsize=8)
    cax = ax.inset_axes([0.05, 0.05, 0.4, 0.025])
    cb = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("log P1/P2 (agent's future costs it: red)", fontsize=6)
    cax.tick_params(labelsize=6)
    bar = axes[1]
    order = np.argsort(-p2)
    x = np.arange(len(order))
    bar.bar(x - 0.2, p2[order], width=0.4, color="#9a9a9a", label="P2: without the agent")
    bar.bar(x + 0.2, p1[order], width=0.4, color=COURTESY_COLOR, label="P1: with the agent")
    bar.set_xticks([])
    bar.set_xlabel("b's samples, by P2", fontsize=7)
    bar.set_ylabel("Boltzmann probability", fontsize=7)
    bar.legend(fontsize=6.5, loc="upper right")
    bar.set_title(f"KL(P1 || P2) = {pair['value']:.3f} nats", fontsize=8)
    bar.grid(axis="y", alpha=0.3)


def draw_demand(axes, scene, agent, frame, c_id, map_groups, radius, dcfg):
    """Courtesy as demand (responsibility/demand.py): b's plans without the
    agent coloured by the braking each would need to clear the agent's
    actual future; the distribution of that braking beside it."""
    step = frame["step"]
    pair = frame["courtesy"][c_id]
    plans, w, delta = pair["plans"], pair["weights"], pair["delta"]
    H = pair["horizon"]
    b = scene.index(c_id)
    fut = slice(step + 1, step + 1 + H)
    shown = np.concatenate([plans.reshape(-1, 2), scene.position[agent, fut, :2][scene.valid[agent, fut]],
                            scene.position[[b, agent], step, :2]])  # frame the plans and the agent's future
    lo, hi = shown.min(0), shown.max(0)
    centre = (lo + hi) / 2
    r = float(np.clip((hi - lo).max() / 2 + 6.0, 15.0, min(radius, 60.0)))
    ax = axes[0]
    for (colour, width), lines in map_groups.items():
        ax.add_collection(LineCollection(lines, colors=colour, linewidths=width * 0.8, zorder=1))
    cmap = plt.get_cmap("RdYlGn_r")
    norm = plt.Normalize(0.0, dcfg.brake_max)
    cannot = delta > dcfg.brake_max  # no braking up to brake_max clears the agent
    alpha = 0.5 + 0.5 * w / w.max()
    for j in np.argsort(delta):
        ax.plot(*plans[j].T, color="black" if cannot[j] else cmap(norm(delta[j])), lw=1.5 if delta[j] > 0 else 0.9,
                alpha=float(alpha[j]), zorder=3)
    a_fut = scene.position[agent, fut, :2][scene.valid[agent, fut]]
    if len(a_fut) > 1:
        ax.plot(*a_fut.T, color=AGENT_COLOR, lw=2.0, ls="--", zorder=5)
    a_len, a_wid = scene.shape_at(agent, step)
    ax.add_patch(Polygon(box(scene.position[agent, step, :2], scene.heading[agent, step], a_len, a_wid), closed=True,
                         facecolor=AGENT_COLOR, edgecolor="black", lw=0.8, zorder=6))
    length, width = scene.shape_at(b, step)
    ax.add_patch(Polygon(box(scene.position[b, step, :2], scene.heading[b, step], length, width), closed=True,
                         facecolor=NEIGHBOUR_COLOR, edgecolor="black", lw=0.5, zorder=6))
    b_fut = scene.position[b, fut, :2][scene.valid[b, fut]]
    if len(b_fut) > 1:
        ax.plot(*b_fut.T, color=NEIGHBOUR_COLOR, lw=1.2, ls="--", zorder=4)
    ax.set_xlim(centre[0] - r, centre[0] + r)
    ax.set_ylim(centre[1] - r, centre[1] + r)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    ax.set_title(f"courtesy as demand: vehicle {c_id}'s plans\nwithout the agent ({H / 10:.0f} s)", fontsize=8)
    ax.legend(handles=[plt.Line2D([], [], color=AGENT_COLOR, ls="--", lw=2, label="agent, actual future"),
                       plt.Line2D([], [], color=NEIGHBOUR_COLOR, ls="--", lw=1.2, label="neighbour, logged"),
                       plt.Line2D([], [], color="black", lw=1.5, label="no braking clears it")],
              loc="upper left", fontsize=6, framealpha=0.85)
    cax = ax.inset_axes([0.05, 0.05, 0.4, 0.025])  # inside the panel: a side colour bar would squeeze it
    cb = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("braking needed (m/s$^2$)", fontsize=6.5)
    cax.tick_params(labelsize=6)

    bar = axes[1]
    edges = np.arange(0.0, dcfg.brake_max + 1e-9, 1.0)
    clear = delta <= 0
    hist = np.zeros(len(edges) + 1)
    hist[0] = w[clear].sum()  # clear as it is
    mid = ~clear & ~cannot
    hist[1:-1] = np.histogram(delta[mid], bins=edges, weights=w[mid])[0]
    hist[-1] = w[cannot].sum()
    labels = ["clear"] + [f"{lo:.0f}-{lo + 1:.0f}" for lo in edges[:-1]] + ["cannot"]
    colours = ["#2ca02c"] + [cmap(norm(lo + 0.5)) for lo in edges[:-1]] + ["black"]
    bar.bar(range(len(hist)), hist, color=colours, edgecolor="white")
    bar.set_xticks(range(len(hist)))
    bar.set_xticklabels(labels, fontsize=6, rotation=60)
    bar.set_ylabel("probability of the plans", fontsize=7)
    bar.set_ylim(0, max(0.3, hist.max() * 1.6))
    bar.set_title("braking each plan would need (m/s$^2$)", fontsize=8)
    bar.text(0.98, 0.95, f"$\\beta_d$ (CVaR) = {pair['value']:.2f} m/s$^2$\nmean {pair['mean']:.2f}\n"
             f"in conflict {pair['p_conflict']:.0%}\ncannot clear {pair['p_unavoidable']:.0%}",
             transform=bar.transAxes, ha="right", va="top", fontsize=7, bbox=dict(facecolor="white", alpha=0.85, lw=0))
    bar.grid(axis="y", alpha=0.3)


def load_levels(path, scene_stem, agent_id):
    if path is None:
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["scene"] == scene_stem and r["agent_id"] == agent_id:
                out[int(r["step"])] = (int(r["level"]), int(r["aggressive"]))
    return out


def draw_timeline(ax, frames, n, levels, unit="nats"):
    t = np.array([f["step"] / 10 for f in frames])
    s = np.array([f["observation"]["safety"] for f in frames])
    c = np.array([f["observation"]["courtesy"] for f in frames])
    title = "responsibility over the clip"
    if levels:
        n_levels = max(z for z, _ in levels.values()) + 1
        cmap = plt.get_cmap("viridis", max(n_levels, 2))
        half = (t[1] - t[0]) / 2 if len(t) > 1 else 0.25
        for f in frames:
            if f["step"] in levels:
                z, aggressive = levels[f["step"]]
                ax.axvspan(f["step"] / 10 - half, f["step"] / 10 + half, color=cmap(z), alpha=0.18, lw=0,
                           hatch="///" if aggressive else None)
        seen = sorted({z: a for z, a in levels.values()}.items())
        ax.legend(handles=[Patch(facecolor=cmap(z), alpha=0.35, hatch="///" if a else None, label=f"level {z}")
                           for z, a in seen], fontsize=6.5, loc="upper right", ncol=len(seen))
        title += " (background: level; hatched: aggressive)"
    ax.axhline(0, color="#999999", lw=0.7)
    ax.plot(t, s, color=SAFETY_COLOR, marker="o", ms=3, lw=1.3)
    ax.plot(t[n], s[n], "o", color="black", ms=6)
    ax.set_ylabel(r"$\beta_s$ [m]", color=SAFETY_COLOR)
    ax.set_xlabel("context time t_k [s]")
    ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
    tx = ax.twinx()
    tx.plot(t, c, color=COURTESY_COLOR, marker="s", ms=3, lw=1.3)
    tx.plot(t[n], c[n], "s", color="black", ms=6)
    tx.set_ylabel(r"$\beta_c$ [%s]" % ("m/s$^2$" if unit == "m/s^2" else unit), color=COURTESY_COLOR)
    tx.set_ylim(0, max(1.0 if unit == "m/s^2" else 0.05, c.max() * 1.15))
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.3)


def render(scene, agent, frame, n, frames, map_groups, radius, horizon, levels, ego_heatmap, weighted=False,
           demand_cfg=None, unit="nats"):
    fig = plt.figure(figsize=(12, 8.2), dpi=100)
    grid = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.2, 1.2], height_ratios=[3, 1.2], hspace=0.28, wspace=0.3)
    ax_scene = fig.add_subplot(grid[0, 0])
    ax_with, ax_without = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])
    ax_time = fig.add_subplot(grid[1, :])
    draw_scene(ax_scene, scene, agent, frame, map_groups, radius, horizon, ego_heatmap, weighted)
    draw_courtesy((ax_with, ax_without), scene, agent, frame, map_groups, radius, horizon, demand_cfg)
    draw_timeline(ax_time, frames, n, levels, unit)
    obs = frame["observation"]
    s_id, _ = worst(obs, "safety")
    c_id, _ = worst(obs, "courtesy")
    level = levels.get(frame["step"])
    fig.suptitle(
        f"scene {scene.scenario_id}  agent {scene.track_ids[agent]}  t_k = {frame['step'] / 10:.1f} s  "
        f"speed {obs['speed']:.1f} m/s\n"
        f"safety {obs['safety']:+.3f} m" + (f" (vs {s_id})" if s_id else "")
        + (f"    courtesy {obs['courtesy']:.2f} m/s^2" if unit == "m/s^2" else f"    courtesy {obs['courtesy']:.4f} nats")
        + (f" (toward {c_id})" if c_id else "")
        + (f"    level {level[0]}{' - aggressive' if level[1] else ''}" if level else ""),
        fontsize=10)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def render_record(record, out_dir: Path, levels_csv=None, view_radius=None, ego_heatmap=False, fps=2.0):
    """Renders every frame of a record and writes the frames, a GIF and (with
    OpenCV) an MP4 to ``out_dir``. Returns the written paths."""
    scene = scene_of(record)
    agent = record["agent"]
    frames = record["frames"]
    if not frames:
        raise ValueError("the record has no frames")
    horizon = record["config"]["metric_horizon"]
    weighted = record["config"].get("motion_set", "sampled") != "sampled"  # colours are weights, not goal log p
    unit = unit_of(record["config"])
    demand_cfg = DemandConfig(**record["config"]["demand"]) if unit == "m/s^2" else None
    radius = view_radius or fit_radius(scene, agent, frames, horizon)
    levels = load_levels(levels_csv, record.get("scene_file", ""), record["agent_id"])
    groups = map_segments(scene)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    images = []
    for n, frame in enumerate(frames):
        image = render(scene, agent, frame, n, frames, groups, radius, horizon, levels, ego_heatmap, weighted,
                       demand_cfg, unit)
        plt.imsave(out_dir / "frames" / f"t_{frame['step']:03d}.png", image)
        images.append(image)
    return write_video(images, out_dir, fps), radius


ESTIMATOR_LABEL = {"kl": "goal KL (Eq. 4)", "boltzmann": "Boltzmann (Sec. V-B)", "demand": "braking demand"}
ESTIMATOR_COLOR = {"kl": "#eb6834", "boltzmann": "#1baf7a", "demand": "#eda100"}


def estimator_of(record) -> str:
    return record["config"].get("courtesy_estimator", "kl")


def fmt_courtesy(value, unit) -> str:
    return f"{value:.2f} m/s^2" if unit == "m/s^2" else f"{value:.3f} nats"


def draw_series(ax, frames, n, key, colour, ylabel, floor):
    t = np.array([f["step"] / 10 for f in frames])
    v = np.array([f["observation"][key] for f in frames])
    ax.axhline(0, color="#999999", lw=0.7)
    ax.plot(t, v, color=colour, marker="o", ms=3, lw=1.4)
    ax.plot(t[n], v[n], "o", color="black", ms=6)
    ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
    ax.set_ylim(min(0.0, v.min() * 1.15), max(floor, v.max() * 1.15))
    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("context time t_k [s]", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def render_comparison(records, n, step, scene, agent, map_groups, radius, horizon, weighted):
    """One frame comparing the courtesy estimators of ``records`` (same scene,
    agent and beta_s) at ``step``: the scene, one row of courtesy panels per
    estimator, and beta_s and each beta_c over the clip."""
    rows = len(records)
    frames = [{f["step"]: f for f in r["frames"]} for r in records]
    fig = plt.figure(figsize=(13, 3.9 * rows + 2.6), dpi=100)
    grid = fig.add_gridspec(rows + 1, 3, width_ratios=[2.0, 1.2, 1.2], height_ratios=[3.0] * rows + [1.5],
                            hspace=0.55, wspace=0.3)
    ax_scene = fig.add_subplot(grid[:rows, 0])
    draw_scene(ax_scene, scene, agent, frames[0][step], map_groups, radius, horizon, False, weighted)
    lines = []
    for i, (record, by_step) in enumerate(zip(records, frames)):
        est = estimator_of(record)
        unit = unit_of(record["config"])
        dcfg = DemandConfig(**record["config"]["demand"]) if est == "demand" else None
        axes = (fig.add_subplot(grid[i, 1]), fig.add_subplot(grid[i, 2]))
        draw_courtesy(axes, scene, agent, by_step[step], map_groups, radius, horizon, dcfg)
        axes[0].text(-0.16, 0.5, ESTIMATOR_LABEL.get(est, est), rotation=90, va="center", ha="center",
                     transform=axes[0].transAxes, fontsize=9, color=ESTIMATOR_COLOR.get(est, "black"),
                     fontweight="bold")
        obs = by_step[step]["observation"]
        lines.append(f"{ESTIMATOR_LABEL.get(est, est)} {fmt_courtesy(obs['courtesy'], unit)}")
    bottom = grid[rows, :].subgridspec(1, rows + 1, wspace=0.28)
    shared = [f for f in records[0]["frames"] if all(f["step"] in fs for fs in frames)]
    k = [f["step"] for f in shared].index(step)
    draw_series(fig.add_subplot(bottom[0, 0]), shared, k, "safety", SAFETY_COLOR, r"$\beta_s$ [m]", 0.05)
    for i, (record, by_step) in enumerate(zip(records, frames)):
        est = estimator_of(record)
        unit = unit_of(record["config"])
        series = [by_step[f["step"]] for f in shared]
        label = r"$\beta_c$, " + ESTIMATOR_LABEL.get(est, est) + (" [m/s$^2$]" if unit == "m/s^2" else " [nats]")
        draw_series(fig.add_subplot(bottom[0, i + 1]), series, k, "courtesy", ESTIMATOR_COLOR.get(est, "black"),
                    label, 1.0 if unit == "m/s^2" else 0.05)
    obs = frames[0][step]["observation"]
    fig.suptitle(f"scene {scene.scenario_id}  agent {scene.track_ids[agent]}  t_k = {step / 10:.1f} s  "
                 f"speed {obs['speed']:.1f} m/s    safety {obs['safety']:+.3f} m\ncourtesy: " + "    ".join(lines),
                 fontsize=10)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def render_comparison_records(records, out_dir: Path, view_radius=None, fps=2.0):
    """Renders a comparison of the records' courtesy estimators frame by frame
    (the steps all records share) to ``out_dir``. Returns the written paths."""
    first = records[0]
    if any(r["agent_id"] != first["agent_id"] or r["scene"]["scenario_id"] != first["scene"]["scenario_id"]
           for r in records):
        raise ValueError("--compare needs records of one scene and agent")
    scene = scene_of(first)
    agent = first["agent"]
    steps = [f["step"] for f in first["frames"] if all(any(g["step"] == f["step"] for g in r["frames"])
                                                        for r in records)]
    if not steps:
        raise ValueError("the records share no frame")
    horizon = first["config"]["metric_horizon"]
    weighted = first["config"].get("motion_set", "sampled") != "sampled"
    radius = view_radius or fit_radius(scene, agent, first["frames"], horizon)
    groups = map_segments(scene)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    images = []
    for n, step in enumerate(steps):
        image = render_comparison(records, n, step, scene, agent, groups, radius, horizon, weighted)
        plt.imsave(out_dir / "frames" / f"t_{step:03d}.png", image)
        images.append(image)
    return write_video(images, out_dir, fps), radius


def write_video(images, out_dir: Path, fps: float):
    from PIL import Image

    gif = out_dir / "responsibility.gif"
    pil = [Image.fromarray(im) for im in images]
    pil[0].save(gif, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0)
    try:
        import cv2
    except ImportError:
        return gif, None
    h, w = images[0].shape[:2]
    mp4 = out_dir / "responsibility.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return gif, None
    for im in images:
        writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    writer.release()
    return gif, mp4


def main():
    args = parse_args()
    root = Path(args.out_dir)
    if args.record and args.compare:
        jobs = iter([("compare", [load_record(r) for r in args.record])])
        several = False
    elif args.record:
        jobs = ((Path(r).stem, [load_record(r)]) for r in args.record)
        several = len(args.record) > 1
    else:
        jobs = live_records(args)
        several = len(args.scene) * len(args.agent) > 1
    for name, records in jobs:  # one video in OUT, or several in OUT/<scene>[_<agent>]
        out = root / name if several else root
        out.mkdir(parents=True, exist_ok=True)
        saved = []
        if not args.record:
            for record in records:
                path = out / ("record.pkl" if len(records) == 1 else f"record_{estimator_of(record)}.pkl")
                save_record(record, path)
                saved.append(str(path))
        if len(records) == 1:
            (gif, mp4), radius = render_record(records[0], out, args.levels, args.view_radius, args.ego_heatmap,
                                               args.fps)
        else:
            (gif, mp4), radius = render_comparison_records(records, out, args.view_radius, args.fps)
        print(f"wrote {len(list((out / 'frames').glob('t_*.png')))} frames to {out / 'frames'} "
              f"(view half-width {radius:.0f} m), {gif}"
              + (f", {mp4}" if mp4 else " (no OpenCV: GIF only)")
              + (f"; records: {', '.join(saved)}" if saved else ""), flush=True)


if __name__ == "__main__":
    main()
