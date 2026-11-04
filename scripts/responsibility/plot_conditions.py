"""One scene of the aggression test under every adversary condition: where
the adversary drives in each (the logged drive, a near miss at about 1 m,
an avoidable collision, the largest gap, CAT's collision), how far its
footprint stays from the logged ego, and what beta_s and the three
courtesy estimates (goal KL, the paper's Boltzmann estimate, braking
demand) read toward the ego at every window.

The trajectories come from the condition folders (export_adv_scenes.py);
the values from the runs already made (runs/nms_ooi_court for beta_s and
the KL, runs/nms_ooi_boltz for the Boltzmann estimate, demand/ for beta_d),
so no model is needed. Windows at and after the first overlap are drawn
faint: the evaluation leaves them out. beta_d breaks off where the two
share a lane (not measured).

    python -m scripts.responsibility.plot_conditions --scene 4 --cat-scenes /home/Cesar/catspond/adv_scenes/cat
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import CONDITIONS, scene_dir  # noqa: E402
from responsibility.scene import Scene  # noqa: E402
from scripts.responsibility.export_adv_scenes import footprint_gaps  # noqa: E402
from scripts.responsibility.visualize_responsibility import box, map_segments  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# fixed order (dataviz reference palette); the logged drive is the neutral reference
SHOWN = [
    ("adv_logged", "logged (reference)", INK2, "--"),
    ("adv_near1", "near miss (~1 m)", "#2a78d6", "-"),
    ("adv_fairinf", "avoidable collision", "#eb6834", "-"),
    ("adv_maxgap", "largest gap (timid)", "#1baf7a", "-"),
    ("adv_cat", "CAT collision", "#6250d6", "-"),
]
METRICS = [("bs", r"$\beta_s$ toward the ego (m)", "nms_ooi_court", "safety"),
           ("kl", r"$\beta_c$, goal KL (nats)", "nms_ooi_court", "courtesy"),
           ("boltz", r"$\beta_c$, Boltzmann (nats)", "nms_ooi_boltz", "courtesy"),
           ("dem", r"$\beta_d$, braking demand (m/s$^2$)", None, None)]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def values(val: Path, cond: str, scene: int, sc: Scene) -> pd.DataFrame:
    """Per window: beta_s, KL, Boltzmann toward the ego, and beta_d."""
    target = sc.track_ids[sc.sdc]
    rows = {}
    for key, _, run, field in METRICS[:3]:
        p = val / "runs" / run / cond / "obs" / f"{scene}.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as f:
            for o in pickle.load(f):
                v = o["per_neighbour"].get(target, {}).get(field)
                rows.setdefault(o["step"], {"step": o["step"]})[key] = 0.0 if v is None else v
    d = pd.DataFrame(rows.values())
    p = val / "demand" / cond / "demand.csv"
    if p.exists():
        dem = pd.read_csv(p)
        dem = dem[dem.scene == scene][["step", "demand"]].rename(columns={"demand": "dem"})
        dem["dem"] = pd.to_numeric(dem.dem, errors="coerce")  # NaN where the two share a lane: not measured
        d = d.merge(dem, on="step", how="outer") if len(d) else dem
    return d.sort_values("step")


def first_overlap(sc: Scene, adv: int):
    t, gap = footprint_gaps(sc, adv, sc.sdc)
    hit = np.flatnonzero(gap <= 0)
    return (int(t[hit[0]]) if len(hit) else None), t, gap


def figure(scene: int, val: Path, raw: Path, cat: Path, out: Path, view: float = None):
    scenes = {c: Scene.load(scene_dir(CONDITIONS[c].scene_set, val, raw, cat) / f"{scene}.pkl") for c, *_ in SHOWN}
    base = scenes["adv_logged"]
    ego = base.sdc
    adv = [i for i in base.objects_of_interest if i != ego][0]
    fig = plt.figure(figsize=(14, 9.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.55, 1], hspace=0.32, wspace=0.32)
    ax_map = fig.add_subplot(grid[0, :2])
    ax_gap = fig.add_subplot(grid[0, 2:])
    ax_m = [fig.add_subplot(grid[1, i]) for i in range(4)]

    # map: the ego's logged path and the adversary in every condition
    for (colour, width), lines in map_segments(base).items():
        ax_map.add_collection(LineCollection(lines, colors=colour, linewidths=width, zorder=1))
    pts = []
    k0 = 10
    ego_path = base.position[ego, k0:, :2][base.valid[ego, k0:]]
    ax_map.plot(*ego_path.T, color=INK, lw=2.4, zorder=4, label="ego, logged")
    pts.append(ego_path)
    length, width = base.shape_at(ego, k0)
    ax_map.add_patch(Polygon(box(base.position[ego, k0, :2], base.heading[ego, k0], length, width), closed=True,
                             facecolor="#d62728", edgecolor=INK, lw=0.8, zorder=6))
    summary = []
    for cond, label, colour, ls in SHOWN:
        sc = scenes[cond]
        a = [i for i in sc.objects_of_interest if i != sc.sdc][0]
        ov, t, gap = first_overlap(sc, a)
        valid = sc.valid[a, k0:]
        path = sc.position[a, k0:, :2][valid]
        steps = np.arange(k0, sc.n_steps)[valid]
        pts.append(path)
        ax_map.plot(*path.T, color=colour, lw=2.0, ls=ls, zorder=5, label=label)
        whole = steps % 10 == 0  # one dot per second
        ax_map.plot(*path[whole].T, "o", color=colour, ms=3, zorder=5)
        if ov is not None:
            x = sc.position[a, ov, :2]
            ax_map.plot(*x, "X", color=colour, ms=11, mec=INK, mew=0.8, zorder=7)
        else:
            j = int(np.argmin(gap)) if len(gap) else None
            if j is not None:
                ax_map.plot(*sc.position[a, t[j], :2], "o", color=colour, ms=8, mec=INK, mew=0.8, zorder=7)
        ax_gap.plot(t / 10, np.maximum(gap, -3.0), color=colour, ls=ls, lw=2, label=label)
        summary.append({"condition": label, "first_overlap_s": None if ov is None else ov / 10,
                        "min_gap_m": round(float(gap.min()), 2) if len(gap) else None})
    length, width = base.shape_at(adv, k0)
    ax_map.add_patch(Polygon(box(base.position[adv, k0, :2], base.heading[adv, k0], length, width), closed=True,
                             facecolor="#9a9a9a", edgecolor=INK, lw=0.8, zorder=6))
    allp = np.concatenate(pts)
    lo, hi = allp.min(0), allp.max(0)
    c = (lo + hi) / 2
    r = view or float((hi - lo).max() / 2 + 8)
    ax_map.set_xlim(c[0] - r, c[0] + r)
    ax_map.set_ylim(c[1] - r, c[1] + r)
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_facecolor(SURFACE)
    ax_map.legend(fontsize=7.5, loc="upper left", framealpha=0.9, labelcolor=INK)
    ax_map.set_title("paths from t = 1 s (dots every 1 s; X: first overlap with the ego, o: closest approach)",
                     fontsize=8.5, color=INK)

    style(ax_gap)
    ax_gap.axhline(0, color=INK, lw=1)
    ax_gap.set_xlabel("time (s)", fontsize=8, color=INK2)
    ax_gap.set_ylabel("footprint gap to the logged ego (m, < 0: overlap)", fontsize=8, color=INK)
    ax_gap.set_title("how close the adversary comes", fontsize=8.5, color=INK)
    ax_gap.set_ylim(-3.2, None)

    for ax, (key, title, *_rest) in zip(ax_m, METRICS):
        style(ax)
        for cond, label, colour, ls in SHOWN:
            d = values(val, cond, scene, scenes[cond])
            if key not in d:
                continue
            a = [i for i in scenes[cond].objects_of_interest if i != scenes[cond].sdc][0]
            ov, _, _ = first_overlap(scenes[cond], a)
            before = d if ov is None else d[d.step < ov]
            after = d.iloc[0:0] if ov is None else d[d.step >= ov]
            ax.plot(before.step / 10, before[key], color=colour, ls=ls, lw=1.8, marker="o", ms=3)
            if len(after):
                joined = pd.concat([before.tail(1), after])
                ax.plot(joined.step / 10, joined[key], color=colour, ls=ls, lw=1.0, alpha=0.3)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("context time t_k (s)", fontsize=7.5, color=INK2)
    fig.suptitle(f"scene {scene} ({base.scenario_id}): the adversary under each condition; ego logged throughout",
                 fontsize=10.5, color=INK)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"conditions_{scene}.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scene", type=int, nargs="+", required=True)
    p.add_argument("--val-dir", default="logs/responsibility/validity")
    p.add_argument("--raw-scenes", default="raw_scenes_500")
    p.add_argument("--cat-scenes", default="adv_scenes/cat")
    p.add_argument("--view-radius", type=float, default=None, help="m; default: fitted to the paths")
    p.add_argument("--out-dir", default="logs/responsibility/validity/report/conditions")
    a = p.parse_args(argv)
    for s in a.scene:
        path, summary = figure(s, Path(a.val_dir), Path(a.raw_scenes), Path(a.cat_scenes), Path(a.out_dir),
                               a.view_radius)
        print(path)
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
