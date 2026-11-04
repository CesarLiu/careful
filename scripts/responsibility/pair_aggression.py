"""Which of the two objects of interest drove aggressively toward the other?
For scenes where one of them was judged aggressive by hand (the "aggressor";
the other is its "partner"), every measure is taken in both directions, X
toward Y, and a scene counts as told right when the aggressor's value is
the more aggressive one (ties half). No reference population is needed.

Rule-based measures, from the log alone (no motion model), one per kind of
behaviour judged:

  took_row     taking the right of way: the braking (m/s^2) X's entry into
               the conflict zone demanded of Y while Y held the right of way
               (forced_yield.py's "forced"; 0 when X held it or entered last)
  entry_demand the same demand whoever held the right of way (cases without
               a holder: signal unknown, same direction, ...)
  cut_in_hw    cutting in: Y's time headway behind X (s) at and in the 2 s
               after X moved into Y's lane ahead of it; smaller is more
               aggressive (inf: no cut-in)
  follow_hw    following too closely: X's smallest time headway behind Y in
               Y's lane (s, own speed >= 2 m/s); smaller is more aggressive

Counterfactual measures, from responsibility runs of both directions
(--runs-aggressor / --runs-partner, one pair per courtesy estimator): the
largest beta_s toward the other and the largest courtesy toward it over
the scene's windows.

    python -m scripts.responsibility.pair_aggression --labelled adv:logs/responsibility/labelled/scenes_adv \\
        sdc:logs/responsibility/labelled/scenes_sdc --runs boltzmann:logs/responsibility/labelled/runs ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.demand import lane_relation  # noqa: E402
from responsibility.forced_yield import forced_yield  # noqa: E402
from responsibility.results import read_windows  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402

LOWER_IS_AGGRESSIVE = ("cut_in_hw", "follow_hw")


def roles(scene: Scene, role: str):
    """(aggressor, partner) indices for the labelled role."""
    other = [i for i in scene.objects_of_interest if i != scene.sdc][0]
    return (scene.sdc, other) if role == "sdc" else (other, scene.sdc)


def frame(scene: Scene, x: int, y: int, t: int):
    """x's longitudinal offset ahead of y in y's frame (m) and the bumper gap (m)."""
    h = float(scene.heading[y, t])
    d = scene.position[x, t, :2] - scene.position[y, t, :2]
    lon = d[0] * np.cos(h) + d[1] * np.sin(h)
    return lon, abs(lon) - (scene.shape_at(x, t)[0] + scene.shape_at(y, t)[0]) / 2


def lane_measures(scene: Scene, x: int, y: int):
    """(cut_in_hw, follow_hw) of x toward y."""
    steps = [t for t in range(10, scene.n_steps) if scene.valid[x, t] and scene.valid[y, t]]
    rel = {t: lane_relation(scene, x, y, t) for t in steps}
    follow = []
    for t in steps:
        vx = float(np.linalg.norm(scene.velocity[x, t]))
        if rel[t] == "follows" and vx >= 2.0:
            _, gap = frame(scene, x, y, t)
            follow.append(max(gap, 0.0) / vx)
    cut = []
    for t in steps:
        before = [rel.get(s) for s in range(t - 10, t)]
        if rel[t] == "leads" and before and all(r != "leads" for r in before if r is not None) \
                and any(r is None for r in before):
            for s in range(t, min(t + 21, scene.n_steps)):  # the 2 s after the cut-in
                if s in rel and rel[s] == "leads":
                    vy = float(np.linalg.norm(scene.velocity[y, s]))
                    if vy >= 2.0:
                        _, gap = frame(scene, x, y, s)
                        cut.append(max(gap, 0.0) / vy)
    return (min(cut) if cut else np.inf), (min(follow) if follow else np.inf)


def rule_measures(scene: Scene, x: int, y: int) -> dict:
    out = {"row_case": "", "holder": ""}
    try:
        fy = forced_yield(scene, x, y)
        out.update(row_case=fy.case, holder=fy.holder,
                   took_row=float(min(fy.forced, 20.0)) if fy.forced is not None else 0.0,
                   entry_demand=float(min(fy.demand, 20.0)) if fy.demand is not None else 0.0)
    except Exception as e:  # a map the lane matching cannot read
        out.update(row_case=f"error: {type(e).__name__}", took_row=np.nan, entry_demand=np.nan)
    out["cut_in_hw"], out["follow_hw"] = lane_measures(scene, x, y)
    return out


def run_maxima(run_dirs, scene: str, agent_id: str):
    """{"bs": max beta_s, "bc": max courtesy} of agent_id in scene over run_dirs (windows.csv rows)."""
    rows = [r for d in run_dirs for r in _rows(d) if r["scene"] == scene and r["agent_id"] == agent_id]
    if not rows:
        return {"bs": np.nan, "bc": np.nan}
    return {"bs": max(r["safety"] for r in rows), "bc": max(r["courtesy"] for r in rows)}


_CACHE = {}


def _rows(run_dir):
    if run_dir not in _CACHE:
        _CACHE[run_dir] = read_windows(run_dir) if Path(run_dir).exists() else []
    return _CACHE[run_dir]


def told_right(agg, par, lower: bool) -> float:
    """1 when the aggressor's value is the more aggressive, 0 when the partner's, 0.5 for a tie; nan when
    neither direction shows the behaviour (both 0 or both inf)."""
    if np.isnan(agg) or np.isnan(par):
        return np.nan
    if lower:
        if np.isinf(agg) and np.isinf(par):
            return np.nan
        return 1.0 if agg < par else 0.0 if par < agg else 0.5
    if agg == 0 and par == 0:
        return np.nan
    return 1.0 if agg > par else 0.0 if par > agg else 0.5


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labelled", nargs="+", required=True, help="role:folder (role sdc or adv)")
    p.add_argument("--runs", nargs="*", default=[],
                   help="estimator:root, the root holding <role>/ (aggressor) and <role>_partner/ runs")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    rows = []
    for spec in a.labelled:
        role, folder = spec.split(":", 1)
        for path in scene_files(folder):
            sc = Scene.load(path)
            x, y = roles(sc, role)
            row = {"role": role, "scene": path.stem, "aggressor": sc.track_ids[x], "partner": sc.track_ids[y]}
            ra, rp = rule_measures(sc, x, y), rule_measures(sc, y, x)
            row["row_case"], row["holder"] = ra["row_case"], ra["holder"]
            for m in ("took_row", "entry_demand", "cut_in_hw", "follow_hw"):
                row[f"{m}_agg"], row[f"{m}_par"] = ra[m], rp[m]
            for est_spec in a.runs:
                est, root = est_spec.split(":", 1)
                ma = run_maxima([f"{root}/{role}"], path.stem, sc.track_ids[x])
                mp = run_maxima([f"{root}/{role}_partner"], path.stem, sc.track_ids[y])
                row[f"bs_{est}_agg"], row[f"bs_{est}_par"] = ma["bs"], mp["bs"]
                row[f"bc_{est}_agg"], row[f"bc_{est}_par"] = ma["bc"], mp["bc"]
            rows.append(row)
    d = pd.DataFrame(rows)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    d.to_csv(Path(a.out_dir) / "pairs.csv", index=False)
    measures = ["took_row", "entry_demand", "cut_in_hw", "follow_hw"]
    for est_spec in a.runs:
        est = est_spec.split(":", 1)[0]
        measures += [f"bs_{est}", f"bc_{est}"]
    measures = list(dict.fromkeys(measures))
    summary = []
    for m in measures:
        if f"{m}_agg" not in d:
            continue
        lower = m in LOWER_IS_AGGRESSIVE
        d[f"{m}_ok"] = [told_right(r[f"{m}_agg"], r[f"{m}_par"], lower) for _, r in d.iterrows()]
        for role, g in list(d.groupby("role")) + [("all", d)]:
            ok = g[f"{m}_ok"]
            summary.append({"measure": m, "role": role, "scenes": len(g), "shows behaviour": int(ok.notna().sum()),
                            "aggressor more aggressive": round(float(ok.mean()), 2) if ok.notna().any() else np.nan,
                            "over all scenes (none = 0.5)": round(float(ok.fillna(0.5).mean()), 2)})
    s = pd.DataFrame(summary)
    s.to_csv(Path(a.out_dir) / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(s.to_string(index=False))
    print(f"\nwrote {Path(a.out_dir) / 'pairs.csv'}")


if __name__ == "__main__":
    main()
