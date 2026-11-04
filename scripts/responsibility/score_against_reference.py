"""Scores runs of agents judged aggressive (e.g. by hand) against a logged
reference computed with the same settings: per scene, the largest beta_s
and beta_c over its windows, where each falls among the reference scenes'
largest values (percentile), and whether it exceeds the reference's
q90 (each at 10% false alarms on the reference). Window-level, the share of
the scene's windows above the reference windows' q90 of the positive
values. Prints a table and writes OUT/scores.csv.

beta_c is in the unit of the runs' courtesy estimator (nats for the default
Boltzmann estimate and the goal KL, m/s^2 for braking demand); the runs and
the reference must share it, which config.json is checked for.

    python -m scripts.responsibility.score_against_reference \\
        --runs logs/responsibility/labelled/runs/adv --reference logs/responsibility/labelled/runs/ref_adv \\
        logs/responsibility/labelled/runs/ref_sdc --out-dir logs/responsibility/labelled/scores
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.results import read_config, read_windows  # noqa: E402

COMPARED = ("courtesy_estimator", "motion_set", "n_safety_samples", "metric_horizon", "window_stride", "use_ooi",
            "d_sat", "cvar_alpha")


def table(run_dirs) -> pd.DataFrame:
    frames = []
    for r in run_dirs:
        d = pd.DataFrame(read_windows(r))
        d["run"] = Path(r).name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def percentile(value: float, reference: np.ndarray) -> float:
    """Share of the reference strictly below ``value``, ties counted half (%)."""
    return 100.0 * (np.mean(reference < value) + 0.5 * np.mean(reference == value))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--reference", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    configs = {str(r): (read_config(r) or {}).get("responsibility", {}) for r in a.runs + a.reference}
    first = next(iter(configs.values()))
    for r, c in configs.items():
        diff = [k for k in COMPARED if c.get(k) != first.get(k)]
        if diff:
            raise SystemExit(f"{r} differs from {next(iter(configs))} in {diff}: compare runs made with one setting")
    unit = "m/s^2" if first.get("courtesy_estimator") == "demand" else "nats"
    runs, ref = table(a.runs), table(a.reference)
    per_scene_ref = ref.groupby(["run", "scene"]).agg(safety=("safety", "max"), courtesy=("courtesy", "max"))
    q90 = {k: float(per_scene_ref[k].quantile(0.9)) for k in ("safety", "courtesy")}
    w90 = {k: float(ref[k][ref[k] > 0].quantile(0.9)) for k in ("safety", "courtesy")}
    rows = []
    for (run, scene), g in runs.groupby(["run", "scene"], sort=False):
        i_s, i_c = g.safety.idxmax(), g.courtesy.idxmax()
        row = {"run": run, "scene": scene, "agent_id": g.agent_id.iloc[0], "windows": len(g),
               "max_safety_m": round(g.safety.max(), 3), "at_s": g.loc[i_s, "time"],
               "safety_against": g.loc[i_s, "safety_against"],
               "safety_pct": round(percentile(g.safety.max(), per_scene_ref.safety.to_numpy()), 1),
               "max_courtesy": round(g.courtesy.max(), 3), "at_c": g.loc[i_c, "time"],
               "courtesy_toward": g.loc[i_c, "courtesy_toward"],
               "courtesy_pct": round(percentile(g.courtesy.max(), per_scene_ref.courtesy.to_numpy()), 1),
               "share_safety_hi": round(float((g.safety > w90["safety"]).mean()), 3),
               "share_courtesy_hi": round(float((g.courtesy > w90["courtesy"]).mean()), 3)}
        row["flag_safety"] = row["max_safety_m"] > q90["safety"]
        row["flag_courtesy"] = row["max_courtesy"] > q90["courtesy"]
        row["flag"] = "both" if row["flag_safety"] and row["flag_courtesy"] else \
            "safety" if row["flag_safety"] else "courtesy" if row["flag_courtesy"] else ""
        rows.append(row)
    out = pd.DataFrame(rows)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(a.out_dir) / "scores.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"reference: {len(per_scene_ref)} scenes ({len(ref)} windows) of {', '.join(sorted(ref.run.unique()))}; "
          f"courtesy estimator {first.get('courtesy_estimator', 'kl')} ({unit})")
    print(f"scene q90 (10% of reference scenes above): max beta_s {q90['safety']:.3f} m, "
          f"max beta_c {q90['courtesy']:.3f} {unit}; window q90 of positive values: "
          f"beta_s {w90['safety']:.3f} m, beta_c {w90['courtesy']:.3f} {unit}")
    print(out.drop(columns=["flag_safety", "flag_courtesy"]).to_string(index=False))
    for run, g in out.groupby("run"):
        print(f"{run}: {len(g)} scenes; flagged by max beta_s {g.flag_safety.mean():.0%}, by max beta_c "
              f"{g.flag_courtesy.mean():.0%}, by either {(g.flag != '').mean():.0%} (reference: 10% each, by "
              f"construction); median percentile beta_s {g.safety_pct.median():.0f}, beta_c {g.courtesy_pct.median():.0f}")
    either = (per_scene_ref.safety > q90["safety"]) | (per_scene_ref.courtesy > q90["courtesy"])
    print(f"reference flagged by either: {either.mean():.0%}")


if __name__ == "__main__":
    main()
