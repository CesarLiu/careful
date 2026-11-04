"""Judges every scene's pair both ways with responsibility/aggression_judge.py (duty x cost x
attribution against a calibration of calibrate_aggression.py): the self-driving car toward the other
object of interest, and back. One row per scene and direction.

Optional comparisons, each printed as a table and its disagreements:

  --violations SCREEN  the driving-aggression-judge crossing screen's CSV (screen_crossing.py): its
                       right-of-way violations, with whether the other paid for them ("affected").
                       A violation that cost nothing must not be judged aggressive.
  --labels LABELS      the driving-aggression-judge pipeline's labels.csv (the reviewing agents'
                       verdicts). Not ground truth; agreement and the differences are listed.

    python -m scripts.responsibility.judge_aggression --scenes raw_scenes_500 \\
        --calibration logs/responsibility/aggression/cat500/calibration.json \\
        --out-dir logs/responsibility/aggression/judge_cat500 --workers 4 \\
        --violations ../ooi_labels/new_labels/screen_law_us_v4.csv --labels ../aggression_agent/runs/cat500/labels.csv
"""

import argparse
import csv
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responsibility.aggression_judge import KINDS, judge  # noqa: E402
from responsibility.catk_cache import load as load_scene, scene_files  # noqa: E402
from responsibility.right_of_way import RightOfWayParams  # noqa: E402

FIELDS = (["scene", "direction", "agent", "other", "verdict", "kind", "reason"]
          + [f"{k}_{x}" for k in KINDS for x in ("demand", "quantile")]
          + ["crossing_case", "crossing_actor", "cut-in_headway_at_join", "cut-in_speed_difference", "cut-in_join",
             "tailgate_headway", "error"])
LABEL_KIND = {"cut_in": "cut-in", "tailgate": "tailgate", "right_of_way": "crossing"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--violations", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--traffic-side", choices=["right", "left"], default="right",
                   help="left: the mirrored right-of-way table for left-hand traffic (right_of_way.py).")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def one(job):
    path, cal, side = job
    rows = []
    try:
        scene = load_scene(path)
        ooi = [i for i in scene.objects_of_interest if i != scene.sdc]
        if not ooi:
            return rows
        for direction, a, b in (("sdc", scene.sdc, ooi[0]), ("ooi", ooi[0], scene.sdc)):
            row = {k: "" for k in FIELDS}
            row.update(scene=Path(path).stem, direction=direction, agent=scene.track_ids[a], other=scene.track_ids[b])
            row.update({k: v for k, v in judge(scene, a, b, cal, RightOfWayParams(traffic_side=side)).as_row().items() if k in FIELDS})
            rows.append(row)
    except Exception as e:  # one bad scene must not stop a run
        rows.append({**{k: "" for k in FIELDS}, "scene": Path(path).stem, "error": f"{type(e).__name__}: {e}"})
    return rows


def compare_violations(rows, path):
    with open(path, encoding="utf-8") as f:
        screen = {r["scene"]: r for r in csv.DictReader(f)}
    by = {(r["scene"], r["direction"]): r for r in rows}
    out = ["| the other paid for it (screen) | violations | judged aggressive (crossing) | A1 names the violator |",
           "|---|---|---|---|"]
    lines = []
    for affected in ("1", "0", ""):
        n = agg = named = 0
        for scene, s in screen.items():
            for side in ("sdc", "ooi"):
                if s.get(side + "_violation") != "1" or s.get("affected", "") != affected:
                    continue
                r = by.get((scene, side))
                if r is None:
                    continue
                n += 1
                agg += r["verdict"] == "aggressive" and r["kind"] == "crossing"
                named += r["crossing_actor"] == "agent"
                if r["verdict"] == "aggressive" and r["kind"] == "crossing" and affected != "1":
                    lines.append(f"  {scene} {side}: aggressive though the screen saw no cost ({r['reason']})")
        if n:
            out.append(f"| {'yes' if affected == '1' else 'no' if affected == '0' else 'unknown'} | {n} | {agg} | {named} |")
    return out + lines


def compare_labels(rows, path):
    with open(path, encoding="utf-8") as f:
        labels = {r["scene"]: r for r in csv.DictReader(f)}
    by = {(r["scene"], r["direction"]): r for r in rows}
    both = a_only = j_only = neither = 0
    lines = []
    for scene, lab in labels.items():
        for side in ("sdc", "ooi"):
            r = by.get((scene, side))
            if r is None or r["error"]:
                continue
            la = lab.get(side + "_aggressive") == "1"
            ja = r["verdict"] == "aggressive"
            both += la and ja
            a_only += la and not ja
            j_only += ja and not la
            neither += not la and not ja
            if la != ja:
                lines.append(f"  {scene} {side}: agents {LABEL_KIND.get(lab.get(side + '_kind'), lab.get(side + '_kind')) if la else 'not'}"
                             f", judge {r['kind'] + ' (' + r['reason'] + ')' if ja else 'not'}")
    head = [f"agents and judge both aggressive {both}, agents only {a_only}, judge only {j_only}, neither {neither}"]
    return head + lines


def main(argv=None):
    args = parse_args(argv)
    cal = json.loads(Path(args.calibration).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), cal, args.traffic_side) for f in scene_files(args.scenes)[: args.n]]
    rows = [r for rs in (map(one, jobs) if args.workers <= 1
                         else multiprocessing.Pool(args.workers).imap(one, jobs, chunksize=4)) for r in rs]
    with open(out / "judgements.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    report = [f"{len(rows)} judgements -> {out / 'judgements.csv'}",
              "thresholds: " + ", ".join(f"{k} {cal[k]['threshold']:.2f} (q{cal[k]['quantile']:g}, n {cal[k]['n']})"
                                         for k in KINDS if k in cal)]
    for direction in ("sdc", "ooi"):
        rs = [r for r in rows if r["direction"] == direction and not r["error"]]
        count = {}
        for r in rs:
            key = r["verdict"] + (f" ({r['kind']})" if r["verdict"] == "aggressive" else "")
            count[key] = count.get(key, 0) + 1
        report.append(f"{direction}: {dict(sorted(count.items()))}")
    if args.violations:
        report += ["", "crossing screen's violations"] + compare_violations(rows, args.violations)
    if args.labels:
        report += ["", "against the reviewing agents' labels"] + compare_labels(rows, args.labels)
    text = "\n".join(report)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return rows


if __name__ == "__main__":
    main()
