"""Scores courtesy as demand (compute_demand.py) on the aggression test's
labelled conditions (responsibility/aggression_test.py), side by side with
the KL courtesy and beta_s of the same scenes and with the track-based
forced yield.

Per scene, over the windows before the adversary's first overlap with the
ego (the reference cut at the same step, as evaluate_aggression.py does):

  max beta_d        the largest demand (windows where the pair shares a lane: 0)
  beta_d, not holder   the same, 0 where the agent held the right of way
                    (right_of_way.py over the whole clip)
  forced yield      forced_yield.py on the neighbour's logged track over the
                    whole clip (its demand, inf where it could not stop; not cut)
  max beta_c, max beta_s target   from runs/<--kl-config>/<condition>/obs
                    (KL courtesy and safety toward the other object of
                    interest), same windows

Each condition is scored by AUROC against the same agent's logged drive
(95% interval from 2000 scene bootstraps), and the attacker against its
victim on the same scenes.

    python -m scripts.responsibility.evaluate_demand --cat-scenes /home/Cesar/catspond/adv_scenes/cat
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import CONDITIONS, boot_auroc, event_steps, fmt_ci, scene_dir  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

TTE_BINS = [0, 0.5, 1, 2, 3, 4, 6, 9]
PAIRS = [("adv_cat", "sdc_vs_cat"), ("adv_fairinf", "sdc_vs_fairinf"), ("adv_near1", "sdc_vs_near1")]
COLS = [("max_d", "max beta_d"), ("max_d_row", "beta_d not holder"), ("fy", "forced yield"),
        ("max_bc", "max beta_c (KL)"), ("max_bs", "max beta_s target")]


def load_demand(run: Path) -> pd.DataFrame:
    d = pd.read_csv(run / "demand.csv")
    d["scene"] = d.scene.astype(int)
    d["demand"] = pd.to_numeric(d.demand, errors="coerce")
    d["fy_demand"] = pd.to_numeric(d.fy_demand, errors="coerce")
    return d


def load_kl(run: Path, scenes_dir: Path, agent: str, n: int) -> pd.DataFrame:
    """Per window: KL courtesy and beta_s toward the other object of interest."""
    rows = []
    for s in range(n):
        p = run / "obs" / f"{s}.pkl"
        if not p.exists():
            continue
        sc = Scene.load(scenes_dir / f"{s}.pkl")
        adv = [i for i in sc.objects_of_interest if i != sc.sdc][0]
        target = sc.track_ids[sc.sdc] if agent == "adv" else sc.track_ids[adv]
        with open(p, "rb") as f:
            for o in pickle.load(f):
                t = o["per_neighbour"].get(target, {})
                rows.append({"scene": s, "step": o["step"], "bc": t.get("courtesy") or 0.0,
                             "bs": t.get("safety", 0.0)})
    return pd.DataFrame(rows, columns=["scene", "step", "bc", "bs"])


def seq_scores(d: pd.DataFrame, kl: pd.DataFrame, scenes, cut=None) -> pd.DataFrame:
    """Per scene: the sequence-level scores over the windows before ``cut[scene]``."""
    out = []
    for s in scenes:
        g, k = d[d.scene == s], kl[kl.scene == s]
        if cut is not None and s in cut:
            g, k = g[g.step < cut[s]], k[k.step < cut[s]]
        dem = g.demand.fillna(0.0)
        fy = d[d.scene == s].fy_demand
        out.append({"scene": s, "max_d": dem.max() if len(g) else 0.0,
                    "max_d_row": dem[g.holder != "agent"].max() if (g.holder != "agent").any() else 0.0,
                    "fy": fy.fillna(0.0).max() if len(fy) else 0.0,
                    "measured": g.demand.notna().any(), "unavoidable": (g.p_unavoidable.fillna(0) > 0.5).any(),
                    "max_bc": k.bc.max() if len(k) else np.nan, "max_bs": k.bs.max() if len(k) else np.nan})
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--val-dir", default="logs/responsibility/validity")
    p.add_argument("--demand-dir", default=None, help="default: VAL/demand")
    p.add_argument("--kl-config", default="nms_ooi_court", help="runs/<config> with the KL courtesy")
    p.add_argument("--raw-scenes", default="raw_scenes_500")
    p.add_argument("--cat-scenes", default="adv_scenes/cat")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    val = Path(a.val_dir)
    dem_dir = Path(a.demand_dir) if a.demand_dir else val / "demand"
    rng = np.random.default_rng(0)

    conds = [c for c in CONDITIONS if (dem_dir / c / "demand.csv").exists()]
    folder = {c: scene_dir(CONDITIONS[c].scene_set, val, Path(a.raw_scenes), Path(a.cat_scenes)) for c in conds}
    data = {c: load_demand(dem_dir / c) for c in conds}
    kl = {c: load_kl(val / "runs" / a.kl_config / c, folder[c], CONDITIONS[c].agent, a.n) for c in conds}
    common = sorted(set.intersection(*[set(d.scene) for d in data.values()]))
    print(f"conditions: {conds}\nscenes present in every condition: {len(common)}")
    events = {c: event_steps(folder[c], a.n) for c in conds if CONDITIONS[c].reference is not None}
    cuts = {c: {s: e for s, (e, hit) in ev.items() if hit} for c, ev in events.items()}

    seq, rows, result = {}, [], {"sequence": {}, "attacker_vs_victim": {}}
    for c in conds:
        seq[c] = seq_scores(data[c], kl[c], common, cuts.get(c))
        t = seq[c]
        r = {"condition": c, "label": CONDITIONS[c].label, "measured": round(t.measured.mean(), 2),
             "unavoidable": round(t.unavoidable.mean(), 2), "median max beta_d": round(t.max_d.median(), 2)}
        ref = CONDITIONS[c].reference
        if ref is not None and ref in data:
            tr = seq_scores(data[ref], kl[ref], common, cuts.get(c))
            result["sequence"][c] = {}
            for col, name in COLS:
                v = boot_auroc(t.dropna(subset=[col]), tr.dropna(subset=[col]), col, rng=rng)
                r[name] = fmt_ci(v)
                result["sequence"][c][col] = v
        rows.append(r)
    print("\nSequence level, AUROC vs the same agent's logged drive cut at the same step (95% CI):")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nAttacker vs victim (same scenes, same frames):")
    for att, vic in PAIRS:
        if att in seq and vic in seq:
            v = {col: boot_auroc(seq[att].dropna(subset=[col]), seq[vic].dropna(subset=[col]), col, rng=rng)
                 for col, _ in COLS}
            result["attacker_vs_victim"][att] = v
            print(f"  {att} vs {vic}: " + ", ".join(f"{name} {fmt_ci(v[col])}" for col, name in COLS))

    for c in ("adv_cat", "sdc_vs_cat"):
        if c not in data or CONDITIONS[c].reference not in data:
            continue
        frames = []
        for src in (c, CONDITIONS[c].reference):
            d = data[src].copy()
            d = d[d.scene.isin(common)]
            d["tte"] = (d.scene.map(lambda s: events[c][s][0]) - d.step) * 0.1
            d = d[d.tte > 0]
            d["bin"] = pd.cut(d.tte, TTE_BINS, right=True)
            frames.append(d)
        d, ref = frames
        t = pd.DataFrame({"mean beta_d": d.groupby("bin").demand.mean(),
                          "mean beta_d logged": ref.groupby("bin").demand.mean(),
                          "p(beta_d>0)": d.groupby("bin").demand.apply(lambda x: (x > 0).mean()),
                          "p(beta_d>0) logged": ref.groupby("bin").demand.apply(lambda x: (x > 0).mean()),
                          "lane shared": d.groupby("bin").demand.apply(lambda x: x.isna().mean())})
        print(f"\n{c}: by time to the event (s), windows before it")
        print(t.round(3).to_string())

    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
