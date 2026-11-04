"""How diverse in speed is the queried agent's motion set? Reads the records
of aggression-test runs (compute_responsibility --save-records) and, for
every window where the agent moves (speed >= --min-speed at k), compares
each member of the motion set with what a constant speed would give:

  r2        progress over the next 2 s / (speed at k * 2 s): 1 = constant
            speed, < 1 slowing down, 0 standing
  brake     r2 < 0.75: on average a quarter slower than now over 2 s
            (from 10 m/s, at least ~2.5 m/s^2 for the second half)
  hard      r2 < 0.5
  hist_acc  the agent's speed change over the last 1 s of history (m/s^2);
            a human braking that starts after k ("onset") shows none
  stop      the member's speed drops below 0.5 m/s within its 8 s
  slow goal the goal distribution's mass on goals closer than half the
            constant-speed distance at 8 s (from the record's goals, which
            cover the top 99% of the mass)

Shares are probability-weighted (NMS weights, or uniform for i.i.d.
samples). Per condition it prints the median over windows; for the logged
drives it also asks whether the model covers what the human did: among
windows where the human braked (logged r2 < 0.75), the model's mass on
braking members and the share of windows where no member brakes at all.

    python -m scripts.responsibility.probe_diversity --config sampled_ooi --out logs/responsibility/validity/diversity_sampled.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import CONDITIONS, event_steps, scene_dir  # noqa: E402
from responsibility.records import load_record, scene_of  # noqa: E402

DT = 0.1


def window_rows(record, events=None):
    scene = scene_of(record)
    a = record["agent"]
    rows = []
    for f in record["frames"]:
        k = f["step"]
        v0 = float(f["observation"]["speed"])
        samples = np.asarray(f["samples"], float)
        lp = np.asarray(f["sample_log_prob"], float)
        weighted = record["config"].get("motion_set", "sampled") != "sampled" and f.get("metric_samples", True)
        w = np.exp(lp - lp.max()) if weighted else np.ones(len(samples))
        w = w / w.sum()
        start = scene.position[a, k, :2]
        path = np.concatenate([np.broadcast_to(start, (len(samples), 1, 2)), samples], axis=1)
        step_len = np.linalg.norm(np.diff(path, axis=1), axis=-1)  # [N, 80]
        s2 = step_len[:, :20].sum(1)
        r2 = s2 / max(v0 * 2.0, 1e-6)
        stop = (step_len / DT < 0.5).any(1)
        fut = np.arange(k + 1, min(k + 21, scene.n_steps))
        logged_r2 = np.nan
        if len(fut) == 20 and scene.valid[a, fut].all() and scene.valid[a, k]:
            lp_ = np.concatenate([start[None], scene.position[a, fut, :2]])
            logged_r2 = np.linalg.norm(np.diff(lp_, axis=0), axis=-1).sum() / max(v0 * 2.0, 1e-6)
        g = f["goals"]
        dist = np.linalg.norm(g["points"] - start, axis=1)
        slow_goal = float(g["prob"][dist < 0.5 * v0 * 8.0].sum() / max(g["prob"].sum(), 1e-12))
        v_past = float(np.linalg.norm(scene.velocity[a, k - 10])) if k >= 10 and scene.valid[a, k - 10] else np.nan
        rows.append({"step": k, "speed": v0, "hist_acc": (v0 - v_past) / 1.0, "brake": float(w[r2 < 0.75].sum()), "hard": float(w[r2 < 0.5].sum()),
                     "stop": float(w[stop].sum()), "any_brake": bool((r2 < 0.75).any()),
                     "min_r2": float(r2.min()), "mean_r2": float(np.dot(w, r2)), "logged_r2": logged_r2,
                     "cover": float(np.nanmin(np.abs(r2 - logged_r2))) if np.isfinite(logged_r2) else np.nan,
                     "slow_goal": slow_goal,
                     "tte": None if events is None else (events[0] - k) * DT})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--val-dir", default="logs/responsibility/validity")
    p.add_argument("--config", default="sampled_ooi")
    p.add_argument("--conditions", nargs="+", default=["adv_logged", "sdc_logged", "adv_cat", "sdc_vs_cat"])
    p.add_argument("--raw-scenes", default="raw_scenes_500")
    p.add_argument("--cat-scenes", default="adv_scenes/cat")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--min-speed", type=float, default=3.0, help="m/s at k")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    val = Path(a.val_dir)
    frames = []
    for c in a.conditions:
        run = val / "runs" / a.config / c / "records"
        if not run.exists():
            print(f"{c}: no records in {run}")
            continue
        ref = CONDITIONS[c].reference
        ev = event_steps(scene_dir(CONDITIONS[c].scene_set, val, Path(a.raw_scenes), Path(a.cat_scenes)), a.n) \
            if ref is not None else None
        for s in range(a.n):
            path = run / f"{s}.pkl"
            if not path.exists():
                continue
            rows = window_rows(load_record(path), None if ev is None else ev[s])
            for r in rows:
                r.update(condition=c, scene=s)
            frames.extend(rows)
    d = pd.DataFrame(frames)
    d = d[d.speed >= a.min_speed]
    if a.out:
        d.to_csv(a.out, index=False)
    cols = ["brake", "hard", "stop", "any_brake", "mean_r2", "min_r2", "slow_goal", "cover"]
    print(f"config {a.config}, windows with speed >= {a.min_speed} m/s at k")
    print(d.groupby("condition")[cols].mean().round(3).assign(windows=d.groupby("condition").size()).to_string())
    near = d[d.tte.notna() & (d.tte > 0) & (d.tte <= 3)]
    if len(near):
        print("\nthe 3 s before the event (CAT's first overlap or closest approach):")
        print(near.groupby("condition")[cols].mean().round(3).assign(windows=near.groupby("condition").size()).to_string())
    logged = d[d.condition.isin(["adv_logged", "sdc_logged"]) & d.logged_r2.notna()]
    if len(logged):
        braked = logged[logged.logged_r2 < 0.75]
        kept = logged[logged.logged_r2 >= 0.75]
        print(f"\nlogged drives: the human braked (logged r2 < 0.75) in {len(braked)} of {len(logged)} windows "
              f"({len(braked) / len(logged):.1%})")
        onset = braked[braked.hist_acc > -0.5]  # no deceleration in the last 1 s of history
        ongoing = braked[braked.hist_acc <= -0.5]
        for name, g in (("human braked", braked), ("  already slowing", ongoing), ("  onset after k", onset),
                        ("human kept speed", kept)):
            print(f"  {name:18s} ({len(g):4d}): model mass on braking members {g.brake.mean():.3f} (median {g.brake.median():.3f}), "
                  f"windows with no braking member {(~g.any_brake).mean():.1%}, "
                  f"closest member's r2 off by {g.cover.median():.3f}")


if __name__ == "__main__":
    main()
