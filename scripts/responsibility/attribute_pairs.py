"""Who caused each scene's conflict (responsibility/attribution.py, the kinematic attribution A1) for
the pair the aggression test judges: the scenario's other object of interest (CAT's adversary in
the adversarial sets) as the agent and the self-driving car as the other. One row per scene.

For E1 of aggression_generalisation_plan.md, run it on every scene set of the aggression test
(aggression_test.SCENE_SETS) into VAL/attribution/<set>.csv; analyze_aggression_offline.py --report
attribution then puts A1 next to the two-sided safety score rel (A2) for attacker against victim.

--check LABELS compares the actor with a labels.csv of the driving-aggression-judge pipeline (the
reviewing agents' verdicts on CAT's 500 scenes; first = the SDC, second = the OOI): for every
cut-in it names, the actor must be the car that cut in, and for every tailgate the follower. The
agents' verdicts are not ground truth either; disagreements are listed for a look.

    python -m scripts.responsibility.attribute_pairs --scenes raw_scenes_500 \\
        --out logs/responsibility/validity/attribution/logged.csv --workers 4
    python -m scripts.responsibility.attribute_pairs --scenes raw_scenes_500 --out /tmp/a1.csv \\
        --check ../aggression_agent/runs/cat500/labels.csv
"""

import argparse
import csv
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responsibility.attribution import attribute  # noqa: E402
from responsibility.catk_cache import load as load_scene, scene_files  # noqa: E402

FIELDS = ["scene", "agent", "other", "kind", "actor", "rule", "step", "ahead", "toward_agent", "toward_other",
          "share_agent", "road_cover", "case", "holder", "first", "error"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=None, help="Only the first n scenes (the aggression test uses 100).")
    p.add_argument("--check", default=None, help="A driving-aggression-judge labels.csv to compare with.")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def one(path):
    row = {k: "" for k in FIELDS}
    row["scene"] = Path(path).stem
    try:
        scene = load_scene(path)
        ooi = [i for i in scene.objects_of_interest if i != scene.sdc]
        if not ooi:
            row["error"] = "no-partner"
            return row
        agent, other = ooi[0], scene.sdc
        row.update(agent=scene.track_ids[agent], other=scene.track_ids[other])
        a = attribute(scene, agent, other)
        row.update({k: v for k, v in a.as_row().items() if k in FIELDS})
    except Exception as e:  # one bad scene must not stop a run
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def check(rows, labels_path):
    """Lines comparing the actor with the agents' cut-in and tailgate verdicts."""
    with open(labels_path, encoding="utf-8") as f:
        labels = {r["scene"]: r for r in csv.DictReader(f)}
    a1 = {r["scene"]: r for r in rows}
    out, agree, total = [], 0, 0
    for scene, lab in labels.items():
        if scene not in a1:
            continue
        r = a1[scene]
        # rows judge (agent = OOI, other = SDC); labels: sdc_* = SDC toward OOI, ooi_* = OOI toward SDC
        for side, who in (("sdc", "other"), ("ooi", "agent")):
            if lab.get(side + "_aggressive") != "1" or lab.get(side + "_kind") not in ("cut_in", "tailgate"):
                continue
            kind = lab[side + "_kind"]
            if kind == "cut_in":
                ok = r["kind"] == "join" and r["actor"] == who and r["ahead"] == who
            else:  # tailgate: the follower, by its own choice (following, or it moved in behind)
                ok = (r["kind"] == "following" and r["actor"] == who) or (r["kind"] == "join" and r["actor"] == who
                                                                           and r["ahead"] != who)
            total += 1
            agree += ok
            if not ok:
                out.append(f"  {scene}: agents say {'SDC' if side == 'sdc' else 'OOI'} {kind}; "
                           f"A1 {r['kind']} actor={r['actor']} ahead={r['ahead']} ({r['rule']})")
    return [f"A1 agrees with the agents' cut-in / tailgate verdicts in {agree} of {total}"] + out


def main(argv=None):
    args = parse_args(argv)
    files = [str(f) for f in scene_files(args.scenes)[: args.n]]
    rows = list(map(one, files) if args.workers <= 1
                else multiprocessing.Pool(args.workers).imap(one, files, chunksize=8))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    kinds = {}
    for r in rows:
        kinds.setdefault(r["kind"] or "error", {}).setdefault(r["actor"] or "-", 0)
        kinds[r["kind"] or "error"][r["actor"] or "-"] += 1
    print(f"{len(rows)} scenes -> {out}")
    for kind, actors in sorted(kinds.items()):
        print(f"  {kind}: {actors}")
    if args.check:
        print("\n".join(check(rows, args.check)))
    return rows


if __name__ == "__main__":
    main()
