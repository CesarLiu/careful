"""How the cut-in decision depends on the one free number of its cost: the braking assumed of the car
that moved in ahead (forced_merge.following_demand's front_brake; RightOfWayParams.brake, 4 m/s^2, by
default). A gentler assumption forgives a merging car that is faster than the car behind; RSS's
worst case (brake_max, 8 m/s^2) forgives nothing that is close.

For each assumed braking it recalibrates the threshold on the natural reference (every logged join
made by the car that came in ahead, the --quantile of its demands: calibrate_aggression.py restricted
to cut-ins) and reports, with that threshold, the share flagged among make_cut_in.py's sets (log:
logged lane changes with 1-3 s of headway; advance_h<H>: the car behind advanced to headway H) and,
with --labels, the agreement with the cut-ins of the driving-aggression-judge reviewing agents on the
same scenes' pair. The thresholds are recalibrated, never tuned to the labels or the sets.

    python -m scripts.responsibility.sensitivity_cut_in --scenes raw_scenes_500 \\
        --sets log=$S/log h0.6=$S/advance_h0.6 h0.4=$S/advance_h0.4 h0.25=$S/advance_h0.25 \\
        --labels ../aggression_agent/runs/cat500/labels.csv --out logs/responsibility/aggression/sensitivity_cut_in.md
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
from responsibility.aggression_judge import candidate_pairs  # noqa: E402
from responsibility.forced_merge import cut_in  # noqa: E402
from responsibility.catk_cache import load as load_scene, scene_files  # noqa: E402
from responsibility.scene import sidecar_index  # noqa: E402

BRAKES = (4.0, 6.0, 8.0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--brakes", type=float, nargs="+", default=list(BRAKES))
    p.add_argument("--quantile", type=float, default=0.9)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def worst(scene, merger, rear, brakes, segments):
    """Per assumed braking, the worst demand of merger's own joins in front of rear (None if none)."""
    out = {b: None for b in brakes}
    for k in attr.joins(scene, merger, rear):
        actor, _ = attr.join_actor(scene, merger, rear, k, segments)
        if actor != "a":
            continue
        for b in brakes:
            d = cut_in(scene, merger, rear, k, front_brake=b).demand
            if d is not None and (out[b] is None or d > out[b]):
                out[b] = d
    return out


def reference(job):
    path, brakes = job
    scene = load_scene(path)
    segments = attr.map_segments(scene)
    rows = []
    for i, j in candidate_pairs(scene):
        for m, r in ((i, j), (j, i)):
            w = worst(scene, m, r, brakes, segments)
            if w[brakes[0]] is not None:
                rows.append(w)
    pair = {}
    ooi = [i for i in scene.objects_of_interest if i != scene.sdc]
    if ooi:
        for side, m, r in (("sdc", scene.sdc, ooi[0]), ("ooi", ooi[0], scene.sdc)):
            pair[side] = worst(scene, m, r, brakes, segments)
    return Path(path).stem, rows, pair


def set_demands(job):
    name, path, meta, brakes = job
    scene = load_scene(path)
    w = worst(scene, scene.index(meta["agent"]), scene.index(meta["other"]), brakes, attr.map_segments(scene))
    return name, w


def main(argv=None):
    args = parse_args(argv)
    brakes = tuple(args.brakes)
    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    mapper = pool.imap if pool else map
    refs, pairs = [], {}
    for stem, rows, pair in mapper(reference, [(str(f), brakes) for f in scene_files(args.scenes)]):
        refs += rows
        pairs[stem] = pair
    jobs = []
    for spec in args.sets:
        name, folder = spec.split("=", 1)
        index = json.loads(sidecar_index(folder).read_text())
        jobs += [(name, Path(folder) / f"{k}.pkl", m, brakes) for k, m in index.items() if (Path(folder) / f"{k}.pkl").exists()]
    sets = {}
    for name, w in mapper(set_demands, jobs):
        sets.setdefault(name, []).append(w)
    labels = {}
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            labels = {r["scene"]: r for r in csv.DictReader(f)}
    names = list(dict.fromkeys(s.split("=", 1)[0] for s in args.sets))
    lines = [f"cut-in references: {len(refs)} logged joins made by the car that came in ahead", "",
             "| front braking (m/s^2) | threshold (q%g) | " % args.quantile + " | ".join(f"{n} flagged" for n in names)
             + (" | agents' cut-ins also flagged | flagged, agents not |" if labels else " |"),
             "|---" * (2 + len(names) + (2 if labels else 0)) + "|"]
    for b in brakes:
        v = np.array([min(r[b], 1e6) for r in refs])
        t = float(np.quantile(v, args.quantile, method="higher"))
        cells = []
        for n in names:
            ws = sets.get(n, [])
            flagged = [w[b] is not None and w[b] > t for w in ws]
            cells.append(f"{100 * np.mean(flagged):.0f}% of {len(ws)}" if ws else "–")
        row = f"| {b:g} | {t:.2f} | " + " | ".join(cells)
        if labels:
            hit = miss = extra = 0
            for stem, lab in labels.items():
                for side in ("sdc", "ooi"):
                    w = pairs.get(stem, {}).get(side)
                    if w is None:
                        continue
                    flagged = w[b] is not None and w[b] > t
                    agents = lab.get(side + "_aggressive") == "1" and lab.get(side + "_kind") == "cut_in"
                    hit += agents and flagged
                    miss += agents and not flagged
                    extra += flagged and not agents
            row += f" | {hit} of {hit + miss} | {extra} |"
        else:
            row += " |"
        lines.append(row)
    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
