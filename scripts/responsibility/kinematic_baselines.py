"""Model-free kinematic baselines for judging a logged driver aggressive:
what the log alone shows, without any motion model, so that a metric's
failure can be told apart from the labels' (if even these do not separate
hand-labelled drivers from logged ones, the counterfactual metrics are not
the only problem).

Per agent and scene, over steps 10 .. 90 where the agent is seen:

  max_decel, max_accel   largest longitudinal deceleration / acceleration
                         (m/s^2, from 0.5 s differences of the speed)
  max_jerk               largest |change of that acceleration| (m/s^3)
  max_lat_acc            largest |speed x yaw rate| (m/s^2)
  max_speed              m/s
  min_gap                closest footprint gap to any other vehicle (m,
                         3-circle covers)
  min_ttc                smallest time to collision with a vehicle ahead in
                         its lane at constant speeds (s; ahead: lateral
                         offset in the agent's frame below half the widths
                         summed, heading within 45 deg)
  min_headway            smallest bumper gap to that vehicle over own speed (s)

Each labelled scene is placed among the reference scenes of the same role
(percentile; for the min_* metrics, small is aggressive) and flagged beyond
that reference's q90 (q10 for min_*). The roles are kept apart because WOMD's
self-driving car is an automated vehicle: its jerk is a fraction of the human
drivers' (reference q90 below). Prints a table and writes OUT/kinematics.csv.

    python -m scripts.responsibility.kinematic_baselines --labelled adv:logs/responsibility/labelled/scenes_adv \\
        sdc:logs/responsibility/labelled/scenes_sdc --reference logs/responsibility/labelled/scenes_ref \\
        --out-dir logs/responsibility/labelled/kinematics
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.interaction import _footprint_gap  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402

DT = 0.1
HIGH = ("max_decel", "max_accel", "max_jerk", "max_lat_acc", "max_speed")
LOW = ("min_gap", "min_ttc", "min_headway")


def pick(scene: Scene, which: str) -> int:
    if which == "sdc":
        return scene.sdc
    return [i for i in scene.objects_of_interest if i != scene.sdc][0]


def kinematics(scene: Scene, a: int) -> dict:
    t = np.arange(10, scene.n_steps)
    t = t[scene.valid[a, t]]
    v = np.linalg.norm(scene.velocity[a, t], axis=-1)
    out = {"max_speed": float(v.max()) if len(v) else np.nan}
    if len(t) > 6:
        acc = (v[5:] - v[:-5]) / 0.5
        out["max_decel"], out["max_accel"] = float(max(0.0, -acc.min())), float(max(0.0, acc.max()))
        out["max_jerk"] = float(np.abs(np.diff(acc)).max() / DT) if len(acc) > 1 else np.nan
        yaw = np.unwrap(scene.heading[a, t])
        rate = np.gradient(yaw, DT)
        out["max_lat_acc"] = float(np.abs(v * rate)[v > 1.0].max()) if (v > 1.0).any() else 0.0
    f = torch.float32
    gaps, ttc, hw = [np.inf], [np.inf], [np.inf]
    la, wa = scene.shape_at(a, int(t[0]))
    for o in range(scene.n_agents):
        if o == a or scene.types[o] != "VEHICLE":
            continue
        tt = t[scene.valid[o, t]]
        if not len(tt):
            continue
        g = _footprint_gap(torch.as_tensor(scene.position[a, tt, :2], dtype=f), torch.as_tensor(scene.heading[a, tt], dtype=f),
                           torch.as_tensor(scene.shape_at(a, int(tt[0]))[None], dtype=f),
                           torch.as_tensor(scene.position[o, tt, :2][None], dtype=f),
                           torch.as_tensor(scene.heading[o, tt][None], dtype=f),
                           torch.as_tensor(scene.shape_at(o, int(tt[0]))[None], dtype=f))[0].numpy()
        gaps.append(float(g.min()))
        lo, wo = scene.shape_at(o, int(tt[0]))
        h = scene.heading[a, tt]
        d = scene.position[o, tt, :2] - scene.position[a, tt, :2]
        lon = d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h)
        lat = -d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h)
        dh = np.abs((scene.heading[o, tt] - h + np.pi) % (2 * np.pi) - np.pi)
        ahead = (lon > 0) & (np.abs(lat) < (wa + wo) / 2) & (dh < np.deg2rad(45))
        if not ahead.any():
            continue
        va = np.linalg.norm(scene.velocity[a, tt], axis=-1)
        vo = (scene.velocity[o, tt, 0] * np.cos(h) + scene.velocity[o, tt, 1] * np.sin(h))
        bumper = lon - (la + lo) / 2
        close = va - vo
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = np.where(ahead & (close > 0.1), np.maximum(bumper, 0) / close, np.inf)
            hd = np.where(ahead & (va > 1.0), np.maximum(bumper, 0) / va, np.inf)
        ttc.append(float(tc.min()))
        hw.append(float(hd.min()))
    out.update(min_gap=min(gaps), min_ttc=min(ttc), min_headway=min(hw))
    return out


def percentile(x, ref, low: bool) -> float:
    ref = np.asarray(ref, float)
    ref = ref[np.isfinite(ref)] if np.isfinite(x) else ref
    if low:
        return 100.0 * (np.mean(ref > x) + 0.5 * np.mean(ref == x))
    return 100.0 * (np.mean(ref < x) + 0.5 * np.mean(ref == x))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labelled", nargs="+", required=True, help="role:folder, role sdc or adv")
    p.add_argument("--reference", required=True, help="folder of reference scenes (both roles measured)")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    ref = []
    for path in scene_files(a.reference):
        sc = Scene.load(path)
        for role in ("sdc", "adv"):
            ref.append({"role": role, "scene": path.stem, **kinematics(sc, pick(sc, role))})
    ref = pd.DataFrame(ref)
    qs = {role: {m: g[m].replace(np.inf, np.nan).quantile(0.1 if m in LOW else 0.9) for m in HIGH + LOW}
          for role, g in ref.groupby("role")}
    rows = []
    for spec in a.labelled:
        role, folder = spec.split(":", 1)
        for path in scene_files(folder):
            sc = Scene.load(path)
            k = kinematics(sc, pick(sc, role))
            q, r = qs[role], ref[ref.role == role]
            row = {"role": role, "scene": path.stem, **{m: round(v, 2) for m, v in k.items()}}
            row["flags"] = ",".join(m for m in HIGH + LOW if (k[m] < q[m] if m in LOW else k[m] > q[m]))
            row["pct"] = round(float(np.median([percentile(k[m], r[m], m in LOW) for m in HIGH + LOW])), 0)
            rows.append(row)
    out = pd.DataFrame(rows)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(a.out_dir) / "kinematics.csv", index=False)
    pd.set_option("display.width", 250)
    for role, q in qs.items():
        print(f"reference thresholds, {role} (q90; q10 for min_*): " + ", ".join(f"{m} {v:.2f}" for m, v in q.items()))
    print(out.to_string(index=False))
    for role, g in out.groupby("role"):
        q = qs[role]
        rates = {m: float(np.mean([m in f.split(",") for f in g["flags"]])) for m in HIGH + LOW}
        r = ref[ref.role == role]
        ref_any = np.mean([any((x[m] < q[m] if m in LOW else x[m] > q[m]) for m in HIGH + LOW) for _, x in r.iterrows()])
        print(f"{role}: {len(g)} scenes; flagged by any metric {np.mean(g['flags'] != ''):.0%} (reference {ref_any:.0%}, "
              f"{len(r)} drives); per metric " + ", ".join(f"{m} {v:.0%}" for m, v in rates.items())
              + f"; median percentile {g['pct'].median():.0f}")


if __name__ == "__main__":
    main()
