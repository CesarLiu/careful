"""Natural references and thresholds for responsibility/aggression_judge.py, from logged driving.

For every scene of a folder it collects what logged drivers do when they owe someone something:

  cut-in    every join between two moving vehicles made by the car that came in ahead
            (forced_merge.cut_in's demand on the car behind)
  tailgate  every following episode between two moving vehicles (forced_merge.tailgate)
  crossing  the pairs whose yielder entered the conflict first (attribution.crossing), with the
            braking it demanded of the holder (forced_yield.forced). By default only the pair the
            scene marks (the self-driving car and the other object of interest, --crossing-pairs
            sdc), since the right of way is slow; --crossing-pairs all takes every pair of moving
            vehicles whose tracks cross.

The threshold of each kind is the --quantile of its reference (0.9: one logged driver in ten asks
more). Logged driving contains some aggressive drivers, so the false-alarm rate on it is an upper
bound. Writes OUT/calibration.json (per kind: quantile, threshold, n, percentiles and the reference
values, which judge_aggression.py uses to place a demand) and OUT/references.csv.

    python -m scripts.responsibility.calibrate_aggression --scenes raw_scenes_500 \\
        --out-dir logs/responsibility/aggression/cat500 --workers 4
"""

import argparse
import csv
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility import attribution as attr  # noqa: E402
from responsibility.aggression_judge import candidate_pairs, same_direction_references  # noqa: E402
from responsibility.forced_yield import forced_yield, track_conflict  # noqa: E402
from responsibility.catk_cache import load as load_scene, scene_files  # noqa: E402
from responsibility.right_of_way import RightOfWayParams  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

BIG = 1e6  # stands for an infinite demand in quantiles


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--quantile", type=float, default=0.9)
    p.add_argument("--crossing-pairs", choices=["sdc", "all"], default="sdc")
    p.add_argument("--n", type=int, default=None, help="Only the first n scenes.")
    p.add_argument("--traffic-side", choices=["right", "left"], default="right",
                   help="left: the mirrored right-of-way table for left-hand traffic (right_of_way.py).")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def crossing_reference(scene: Scene, pairs, params: RightOfWayParams):
    rows = []
    for i, j in pairs:
        if track_conflict(scene, i, j) is None:
            continue
        for x, y in ((i, j), (j, i)):
            cross = attr.crossing(scene, x, y, params)
            if cross is None or cross.actor != "agent":
                continue
            demand = forced_yield(scene, x, y, params=params).forced
            if demand is not None:
                rows.append({"agent": scene.track_ids[x], "other": scene.track_ids[y], "kind": "crossing",
                             "demand": demand, "step": cross.step, "is_sdc": int(x == scene.sdc or y == scene.sdc)})
    return rows


def scan(job):
    path, crossing_pairs, side = job
    params = RightOfWayParams(traffic_side=side)
    try:
        scene = load_scene(path)
        rows = same_direction_references(scene, params)
        if crossing_pairs == "sdc":
            ooi = [i for i in scene.objects_of_interest if i != scene.sdc]
            pairs = [(scene.sdc, ooi[0])] if ooi else []
        else:
            pairs = candidate_pairs(scene)
        rows += crossing_reference(scene, pairs, params)
    except Exception as e:  # one bad scene must not stop a calibration
        print(f"skip {Path(path).name}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return []
    for r in rows:
        r["scene"] = Path(path).stem
    return rows


def summarise(rows, q: float):
    out = {}
    for kind in ("crossing", "cut-in", "tailgate"):
        v = np.array([min(float(r["demand"]), BIG) for r in rows if r["kind"] == kind], dtype=float)
        if not v.size:
            continue
        t = float(np.quantile(v, q, method="higher"))  # an observed value, never between a finite one and inf
        out[kind] = {"quantile": q, "threshold": float("inf") if t >= BIG else t, "n": int(v.size),
                     "n_sdc": int(sum(r["is_sdc"] for r in rows if r["kind"] == kind)),
                     "percentiles": {f"p{p}": round(float(np.quantile(v, p / 100, method="higher")), 3) for p in (10, 25, 50, 75, 90, 95)},
                     "reference": sorted(round(float(x), 4) for x in v)}
    return out


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = scene_files(args.scenes)[: args.n]
    jobs = [(str(f), args.crossing_pairs, args.traffic_side) for f in files]
    results = (map(scan, jobs) if args.workers <= 1
               else multiprocessing.Pool(args.workers).imap_unordered(scan, jobs, chunksize=4))
    rows = [r for rs in results for r in rs]
    with open(out / "references.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "agent", "other", "kind", "demand", "step", "is_sdc"])
        w.writeheader()
        w.writerows(rows)
    cal = {"scenes": str(args.scenes), "n_scenes": len(files), "crossing_pairs": args.crossing_pairs, "traffic_side": args.traffic_side,
           **summarise(rows, args.quantile)}
    (out / "calibration.json").write_text(json.dumps(cal, indent=1))
    for kind in ("crossing", "cut-in", "tailgate"):
        if kind in cal:
            c = cal[kind]
            print(f"{kind}: n {c['n']} (with the SDC {c['n_sdc']}), q{args.quantile:g} = {c['threshold']:.2f} m/s^2, "
                  f"percentiles {c['percentiles']}")
    return cal


if __name__ == "__main__":
    main()
