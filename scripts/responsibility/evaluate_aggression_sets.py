"""Scores responsibility/aggression_judge.py's decision -- the attribution gate and the cost against
its calibrated threshold -- on scene sets whose answer is known by construction.

Each set is a scene folder with a sidecar index naming the judged agent and the other per scene:

  cut-in sets    make_cut_in.py's (an index entry with "join"): log (the source joins, made with at
                 least 1 s of headway) and advance_h<H> (the car behind advanced to headway H)
  crossing sets  make_go_first.py's: log (the holder passed first), delay_pet<P> / retime_pet<P>
                 (the yielder made to go first) and log_yielder_first (the natural reference)

For every scene it reads who caused the conflict (attribution.py: crossing, join or following), the
cost of the duty the agent owed (cut-in: forced_merge.cut_in on the car behind; crossing:
forced_yield.forced, with the right of way of the logged scene a rewrite came from, as
evaluate_forced_yield.py does) and decides: flagged when the attribution names the agent and the
cost is above the threshold. The threshold of each kind is the --quantile of a reference: the named
--reference-set, or else the "reference" of --calibration (calibrate_aggression.py). The summary
gives, per set, how often the agent is named the cause, how often the cost alone is above the
threshold, and how often both are (the decision).

    S=logs/responsibility/aggression/cut_in_scenes
    python -m scripts.responsibility.evaluate_aggression_sets --out-dir logs/responsibility/aggression/eval_cut_in \\
        --calibration logs/responsibility/aggression/cat500/calibration.json \\
        --sets log=$S/log h0.6=$S/advance_h0.6 h0.4=$S/advance_h0.4 h0.25=$S/advance_h0.25
"""

import argparse
import csv
import json
import math
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility import attribution as attr  # noqa: E402
from responsibility.forced_merge import same_direction  # noqa: E402
from responsibility.forced_yield import carry_priority, forced_yield, pair_priority  # noqa: E402
from responsibility.catk_cache import load as load_scene  # noqa: E402
from responsibility.right_of_way import RightOfWayParams  # noqa: E402
from responsibility.scene import sidecar_index  # noqa: E402

FIELDS = ["set", "scene", "kind", "agent", "other", "attribution", "actor", "rule", "demand", "headway", "error"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", required=True, help="name=folder, each with a sidecar index.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--calibration", default=None, help="calibrate_aggression.py's calibration.json.")
    p.add_argument("--reference-set", default=None, help="Take the threshold from this set's costs instead.")
    p.add_argument("--quantile", type=float, default=0.9)
    p.add_argument("--traffic-side", choices=["right", "left"], default="right",
                   help="left: the mirrored right-of-way table for left-hand traffic (right_of_way.py).")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def log_scene(path: Path, meta):
    source = path.parent.parent / "log" / path.name
    return load_scene(source) if "mode" in meta and source.exists() else None


def judge(job):
    name, path, meta, side = job
    p = RightOfWayParams(traffic_side=side)
    kind = "cut-in" if "join" in meta else "crossing"
    row = {**{k: "" for k in FIELDS}, "set": name, "scene": path.stem, "kind": kind}
    try:
        scene = load_scene(path)
        agent, other = scene.index(meta["agent"]), scene.index(meta["other"])
        row.update(agent=meta["agent"], other=meta["other"])
        a = attr.attribute(scene, agent, other, p)
        row.update(attribution=a.kind, actor=a.actor, rule=a.rule)
        if kind == "cut-in":
            m = same_direction(scene, agent, other, p)["cut-in"]
            if m.demand is not None:
                row["demand"] = "inf" if np.isinf(m.demand) else round(float(m.demand), 3)
                row["headway"] = m.evidence.get("headway_at_join", "")
        else:
            cross = attr.crossing(scene, agent, other, p)
            if cross is not None:
                row.update(attribution="crossing", actor=cross.actor, rule=cross.rule)
            log = log_scene(path, meta)
            prio = None if log is None else carry_priority(pair_priority(log, agent, other, params=p), scene, agent, other)
            f = forced_yield(scene, agent, other, params=p, prio=prio).forced
            if f is not None:
                row["demand"] = "inf" if np.isinf(f) else round(float(f), 3)
    except Exception as e:  # one malformed scene must not stop the run
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def value(x):
    return math.inf if x == "inf" else float(x)


def thresholds(rows, args):
    out = {}
    cal = json.loads(Path(args.calibration).read_text()) if args.calibration else {}
    for kind in ("cut-in", "crossing"):
        if args.reference_set:
            ref = [value(r["demand"]) for r in rows if r["set"] == args.reference_set and r["kind"] == kind and r["demand"] != ""]
            if ref:
                out[kind] = (float(np.quantile(ref, args.quantile, method="higher")), f"q{args.quantile:g} of {args.reference_set} (n {len(ref)})")
                continue
        if kind in cal:
            out[kind] = (cal[kind]["threshold"], f"calibration q{cal[kind]['quantile']:g} (n {cal[kind]['n']})")
    return out


def markdown(rows, th) -> str:
    lines = ["| set | kind | pairs | agent named the cause | cost above threshold | both (flagged) | median cost | median headway |",
             "|---|---|---|---|---|---|---|---|"]
    for name in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == name and not r["error"]]
        if not rs:
            continue
        kind = rs[0]["kind"]
        t = th.get(kind, (math.nan, ""))[0]
        named = [r["actor"] == "agent" for r in rs]
        cost = [r["demand"] != "" and value(r["demand"]) > t for r in rs]
        both = [n and c for n, c in zip(named, cost)]
        demands = [value(r["demand"]) for r in rs if r["demand"] != ""]
        heads = [float(r["headway"]) for r in rs if r["headway"] not in ("", None)]

        def pct(xs):
            return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "–"
        lines.append(f"| {name} | {kind} | {len(rs)} | {pct(named)} | {pct(cost)} | {pct(both)} | "
                     f"{np.median(demands):.2f} | " if demands else f"| {name} | {kind} | {len(rs)} | {pct(named)} | {pct(cost)} | {pct(both)} | – | ")
        lines[-1] += f"{np.median(heads):.2f} |" if heads else "– |"
    head = [f"{k} threshold: {v[0]:.2f} m/s^2 ({v[1]})" for k, v in th.items()]
    return "\n".join(head + [""] + lines)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec in args.sets:
        name, folder = spec.split("=", 1)
        index = json.loads(sidecar_index(folder).read_text())
        jobs += [(name, Path(folder) / f"{key}.pkl", meta, args.traffic_side) for key, meta in index.items()
                 if (Path(folder) / f"{key}.pkl").exists()]
    rows = list(map(judge, jobs) if args.workers <= 1
                else multiprocessing.Pool(args.workers).imap(judge, jobs, chunksize=4))
    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    table = markdown(rows, thresholds(rows, args))
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    errors = sum(1 for r in rows if r["error"])
    if errors:
        print(f"{errors} scenes failed; see pairs.csv")
    return rows


if __name__ == "__main__":
    main()
