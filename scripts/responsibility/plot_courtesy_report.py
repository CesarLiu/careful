"""Figures for the courtesy reproduction report: how the three courtesy
estimators (goal KL, the paper's Boltzmann estimate, braking demand) and
beta_s compare on the aggression test, how they move as CAT's attack
approaches, how diverse DenseTNT's motion set is in speed, and how much
DenseTNT and SMART react to the attacker.

Reads (all written by earlier steps):
  VAL/demand/eval.json, VAL/demand_cv/eval.json, VAL/demand/eval_boltz.json
                                 evaluate_demand.py (default, --demand-dir
                                 demand_cv, --kl-config nms_ooi_boltz)
  VAL/runs/nms_ooi_court, VAL/runs/nms_ooi_boltz, VAL/demand   per-window values
  VAL/diversity_sampled_ooi.csv  probe_diversity.py
  VAL/courtesy_probe.csv, VAL/smart_probe.csv   probe_courtesy.py, probe_smart.py

Writes PNGs and a JSON of every plotted number to --out-dir.

    python -m scripts.responsibility.plot_courtesy_report --cat-scenes /home/Cesar/catspond/adv_scenes/cat
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import CONDITIONS, event_steps, scene_dir  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {  # fixed categorical order (dataviz reference palette, light)
    "bs": ("#2a78d6", r"$\beta_s$ toward the target"),
    "kl": ("#eb6834", r"$\beta_c$, goal KL (Eq. 4)"),
    "boltz": ("#1baf7a", r"$\beta_c$, Boltzmann (Sec. V-B)"),
    "dem": ("#eda100", r"$\beta_d$, DenseTNT plans"),
    "cv": ("#e87ba4", r"$\beta_d$, constant-velocity plan"),
}
TTE_BINS = [0, 0.5, 1, 2, 3, 4, 6]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def auroc_figure(val: Path, out: Path, numbers: dict):
    ev = {k: json.loads((val / p).read_text()) for k, p in
          (("main", "demand/eval.json"), ("cv", "demand_cv/eval.json"), ("boltz", "demand/eval_boltz.json"))}
    rows = [("adv_near1", "near miss\n(1 m)"), ("adv_fairinf", "avoidable\ncollision"), ("adv_cat", "CAT\nattacker"),
            ("adv_maxgap", "timid\n(max gap)"), ("avv", "attacker\nvs victim")]
    pick = {"bs": ("main", "max_bs"), "kl": ("main", "max_bc"), "boltz": ("boltz", "max_bc"),
            "dem": ("main", "max_d"), "cv": ("cv", "max_d")}
    table = {}
    for key, (src, col) in pick.items():
        table[key] = []
        for cond, _ in rows:
            if cond == "avv":
                v = ev[src].get("attacker_vs_victim", {}).get("adv_cat", {}).get(col, [np.nan] * 3)
            else:
                v = ev[src]["sequence"].get(cond, {}).get(col, [np.nan] * 3)
            table[key].append(v)
    numbers["auroc"] = {k: dict(zip([r for r, _ in rows], v)) for k, v in table.items()}
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    n = len(SERIES)
    width = 0.8 / n
    x = np.arange(len(rows))
    for j, (key, (colour, label)) in enumerate(SERIES.items()):
        v = np.array(table[key], dtype=float)
        pos = x - 0.4 + width * (j + 0.5)
        ax.bar(pos, v[:, 0] - 0.5, bottom=0.5, width=width - 0.02, color=colour, label=label, zorder=3)
        ax.errorbar(pos, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor=INK2, lw=0.8,
                    capsize=1.5, zorder=4)
    ax.axhline(0.5, color=INK2, lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([r for _, r in rows], fontsize=8.5, color=INK)
    ax.set_ylim(0.15, 1.0)
    ax.set_ylabel("sequence AUROC vs the logged drive", fontsize=8.5, color=INK)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", frameon=False, labelcolor=INK)
    ax.text(-0.48, 0.505, "chance", fontsize=7, color=INK2, va="bottom", ha="left")
    fig.tight_layout()
    fig.savefig(out / "auroc.png", facecolor=SURFACE)
    plt.close(fig)


def per_window(val: Path, cond: str, n: int, raw: Path, cat: Path) -> pd.DataFrame:
    folder = scene_dir(CONDITIONS[cond].scene_set, val, raw, cat)
    who = CONDITIONS[cond].agent
    rows = {}
    for config, key in (("nms_ooi_court", "kl"), ("nms_ooi_boltz", "boltz")):
        run = val / "runs" / config / cond / "obs"
        for s in range(n):
            p = run / f"{s}.pkl"
            if not p.exists():
                continue
            sc = Scene.load(folder / f"{s}.pkl")
            adv = [i for i in sc.objects_of_interest if i != sc.sdc][0]
            target = sc.track_ids[sc.sdc] if who == "adv" else sc.track_ids[adv]
            with open(p, "rb") as f:
                for o in pickle.load(f):
                    t = o["per_neighbour"].get(target, {})
                    r = rows.setdefault((s, o["step"]), {"scene": s, "step": o["step"]})
                    r[key] = t.get("courtesy") or 0.0
                    if key == "kl":
                        r["bs"] = t.get("safety", 0.0)
    d = pd.DataFrame(rows.values())
    dem = pd.read_csv(val / "demand" / cond / "demand.csv")[["scene", "step", "demand"]]
    dem["demand"] = pd.to_numeric(dem.demand, errors="coerce")
    return d.merge(dem.rename(columns={"demand": "dem"}), on=["scene", "step"], how="left")


def timing_figure(val: Path, out: Path, numbers: dict, n: int, raw: Path, cat: Path):
    ev = event_steps(cat, n)
    frames = {}
    for cond in ("adv_cat", "adv_logged"):
        d = per_window(val, cond, n, raw, cat)
        d["tte"] = (d.scene.map(lambda s: ev[s][0]) - d.step) * 0.1
        d = d[d.tte > 0]
        d["bin"] = pd.cut(d.tte, TTE_BINS, right=True)
        frames[cond] = d
    panels = [("bs", r"$\beta_s$ toward the target (m)"), ("kl", r"$\beta_c$, goal KL (nats)"),
              ("boltz", r"$\beta_c$, Boltzmann (nats)"), ("dem", r"$\beta_d$, braking demand (m/s$^2$)")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    mids = [(a + b) / 2 for a, b in zip(TTE_BINS[:-1], TTE_BINS[1:])]
    numbers["timing"] = {"tte_mid_s": mids}
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        colour = SERIES[key][0]
        for cond, ls, lab in (("adv_cat", "-", "CAT attacker"), ("adv_logged", "--", "same agent, logged")):
            m = frames[cond].groupby("bin", observed=False)[key].mean().to_numpy()
            numbers["timing"][f"{key}_{cond}"] = [None if np.isnan(v) else round(float(v), 4) for v in m]
            ax.plot(mids, m, ls=ls, color=colour if cond == "adv_cat" else INK2, lw=2, marker="o", ms=4, label=lab)
        ax.invert_xaxis()
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("time to CAT's overlap (s)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "timing.png", facecolor=SURFACE)
    plt.close(fig)
    both = pd.concat(frames.values())
    numbers["spearman_with_beta_s"] = {k: round(float(both[["bs", k]].corr(method="spearman").iloc[0, 1]), 3)
                                       for k in ("kl", "boltz", "dem")}
    numbers["spearman_windows"] = int(len(both))


def diversity_figure(val: Path, out: Path, numbers: dict):
    d = pd.read_csv(val / "diversity_sampled_ooi.csv")
    logged = d[d.condition.isin(["adv_logged", "sdc_logged"]) & d.logged_r2.notna()]
    groups = [("human kept speed\n(r2 >= 0.97)", logged[logged.logged_r2 >= 0.97]),
              ("human started slowing\nafter k (r2 < 0.9,\nflat history)", logged[(logged.logged_r2 < 0.9)
                                                                               & (logged.hist_acc > -0.3)]),
              ("human already slowing\n(r2 < 0.9, braking\nin history)", logged[(logged.logged_r2 < 0.9)
                                                                               & (logged.hist_acc <= -0.3)])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=150, gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor(SURFACE)
    style(ax1)
    x = np.arange(len(groups))
    model = [g.mean_r2.mean() for _, g in groups]
    human = [g.logged_r2.mean() for _, g in groups]
    ax1.bar(x - 0.18, human, width=0.34, color=INK2, label="human (logged)", zorder=3)
    ax1.bar(x + 0.18, model, width=0.34, color=SERIES["bs"][0], label="DenseTNT motion set (mean)", zorder=3)
    for i, (h, m) in enumerate(zip(human, model)):
        ax1.text(i - 0.18, h + 0.01, f"{h:.2f}", ha="center", fontsize=7.5, color=INK)
        ax1.text(i + 0.18, m + 0.01, f"{m:.2f}", ha="center", fontsize=7.5, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{name}\nn = {len(g)}" for name, g in groups], fontsize=7.5, color=INK)
    ax1.set_ylim(0.6, 1.15)
    ax1.axhline(1.0, color=INK2, lw=0.8)
    ax1.set_ylabel("2 s progress / constant speed (r2)", fontsize=8, color=INK)
    ax1.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="upper right")
    ax1.set_title("DenseTNT follows braking that is in the history, misses its onset", fontsize=8.5, color=INK)
    style(ax2)
    moving = d[d.condition.isin(["adv_logged", "sdc_logged", "adv_cat"])]
    for cond, colour, lab in (("adv_cat", SERIES["kl"][0], "CAT attacker"),
                              ("adv_logged", INK2, "adversary, logged")):
        v = moving[moving.condition == cond].brake
        ax2.hist(v, bins=np.linspace(0, 1, 21), color=colour, alpha=0.75, label=lab, zorder=3,
                 weights=np.full(len(v), 1 / len(v)))
    ax2.set_xlabel("probability of motion-set members slowing (r2 < 0.75)", fontsize=8, color=INK)
    ax2.set_ylabel("share of windows", fontsize=8, color=INK)
    ax2.legend(fontsize=7.5, frameon=False, labelcolor=INK)
    ax2.set_title("almost no braking alternatives", fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "diversity.png", facecolor=SURFACE)
    plt.close(fig)
    numbers["diversity"] = {name.replace("\n", " "): {"n": int(len(g)), "human_r2": round(float(g.logged_r2.mean()), 3),
                                                      "model_r2": round(float(g.mean_r2.mean()), 3)}
                            for name, g in groups}
    numbers["diversity_brake_mass"] = {c: round(float(moving[moving.condition == c].brake.mean()), 4)
                                       for c in ("adv_cat", "adv_logged", "sdc_logged")}
    numbers["diversity_no_braking_member"] = {c: round(float((~moving[moving.condition == c].any_brake).mean()), 3)
                                              for c in ("adv_cat", "adv_logged", "sdc_logged")}


def reaction_figure(val: Path, out: Path, numbers: dict):
    dt = pd.read_csv(val / "courtesy_probe.csv")
    sm = pd.read_csv(val / "smart_probe.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    numbers["reaction"] = {}
    for df, colour, lab in ((dt, SERIES["kl"][0], "DenseTNT"), (sm, SERIES["bs"][0], "SMART (CAT-K)")):
        g = df[df.src == "CAT"].groupby("lead_s").d_prog.median()
        numbers["reaction"][lab] = {str(k): round(float(v), 3) for k, v in g.items()}
        ax.plot(g.index, g.values, color=colour, lw=2, marker="o", ms=5, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("time before CAT's overlap (s)", fontsize=8, color=INK)
    ax.set_ylabel("ego's predicted progress at 2 s,\nwith minus without the attacker (m)", fontsize=8, color=INK)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK)
    ax.set_title("Does the model predict the ego to hold back?", fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "reaction.png", facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--val-dir", default="logs/responsibility/validity")
    p.add_argument("--raw-scenes", default="raw_scenes_500")
    p.add_argument("--cat-scenes", default="adv_scenes/cat")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out-dir", default="logs/responsibility/validity/report")
    a = p.parse_args(argv)
    val, out = Path(a.val_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {}
    auroc_figure(val, out, numbers)
    timing_figure(val, out, numbers, a.n, Path(a.raw_scenes), Path(a.cat_scenes))
    diversity_figure(val, out, numbers)
    reaction_figure(val, out, numbers)
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
