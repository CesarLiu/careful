"""Ground truth for cut-ins: logged lane changes made with room to spare, rewritten so that the car
behind is closer when the merging car moves in. The same-direction counterpart of make_go_first.py.

The source is every join in a scene folder (responsibility/attribution.py) in which the car that
came in ahead moved over itself (the actor), the two had travelled the same way for the second
before (a lane change, not a turn into the road, which the right of way judges), it came from the
next lane (within NEXT_LANE sideways), the car behind kept to its lane (moved at most STILL sideways
over the join's window: advancing a car whose track drifts across its lane would change the two
cars' sideways relation, and with it the join) and it left the car behind at least --min-headway s at
the join. For each target headway (--headway) the car behind is advanced along its own logged track: it
is where its log has it SHIFT steps later, so at the join it is closer to the merging car, at the
same speed and on the same path. Past the end of its log it drives on at its last velocity. The
merging car keeps its log, so its own motion -- the lane change itself -- is the logged, lawful one:
only its timing against the car behind makes the merge tight, and a measure of the merging car's
kinematics alone cannot tell the rewrite apart. The shift is the smallest that brings the headway at
the join to the target. A rewrite is dropped when no shift up to --max-shift does, or when the
advanced car's footprint overlaps any other agent's logged one but the merging car's before the end
of the window the cut-in is measured over (forced_merge.JOIN_EFFECT after the join; overlaps its log
also had are not counted). Sources leave at most --max-headway: a longer one needs a shift that runs
the car into the traffic around it. After the window the rewritten scene is not used. Nobody reacts (open loop): the car behind's need to brake is what the measure must read.

Writes OUT/log/<scene>_<merger>_<rear>.pkl (the source scene) and OUT/advance_h<H>/... per target,
and next to each folder <folder>.index.json with the judged agent (the merging car), the other (the
car behind), the join step, the shift and the headway at the join.

    python -m scripts.responsibility.make_cut_in --scenes raw_scenes_500 \\
        --out-dir logs/responsibility/aggression/cut_in_scenes --headway 0.6 0.4 0.25 --workers 4
"""

import argparse
import json
import multiprocessing
import pickle
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility import attribution as attr  # noqa: E402
from responsibility.aggression_judge import candidate_pairs  # noqa: E402
from responsibility.forced_merge import JOIN_EFFECT, following_demand  # noqa: E402
from responsibility.motion_filter import collisions  # noqa: E402
from responsibility.catk_cache import load as load_scene, scene_files  # noqa: E402
from responsibility.scene import Scene, sidecar_index  # noqa: E402

DT = 0.1
FAR = 1e6  # m: where an agent no longer seen is put for the overlap check
LANE_CHANGE = 10  # steps (1 s) before a join the two must travel the same way: a lane change, not a turn in
NEXT_LANE = 6.0  # m: and the merging car must come from within this sideways of the car behind (the next lane)
STILL = 0.3  # m: the car behind may move this much sideways over the join's window (advancing it along a
             # track that drifts across its lane would change the two cars' sideways relation)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--headway", type=float, nargs="+", default=[0.6, 0.4, 0.25], help="s: target headways at the join.")
    p.add_argument("--min-headway", type=float, default=1.0, help="s: sources leave at least this at the join.")
    p.add_argument("--max-headway", type=float, default=3.0,
                   help="s: and at most this (a longer one needs a shift that runs the car into others).")
    p.add_argument("--max-shift", type=int, default=60, help="steps the car behind may be advanced.")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def advance(scene: Scene, r: int, shift: int):
    """r's track ``shift`` steps earlier along its log: (position [T, 2], heading [T], velocity
    [T, 2], valid [T]); past its last logged state at that state's velocity."""
    n = scene.n_steps
    seen = np.flatnonzero(scene.valid[r])
    last = int(seen[-1])
    position, heading = np.zeros((n, 2)), np.zeros(n)
    velocity, valid = np.zeros((n, 2)), np.zeros(n, dtype=bool)
    src = np.arange(n) + shift
    inside = src <= last
    position[inside] = scene.position[r, src[inside], :2]
    heading[inside] = scene.heading[r, src[inside]]
    velocity[inside] = scene.velocity[r, src[inside]]
    valid[inside] = scene.valid[r, src[inside]]
    beyond = np.flatnonzero(~inside)
    v_last = scene.velocity[r, last].astype(float)
    position[beyond] = scene.position[r, last, :2] + (src[beyond] - last)[:, None] * DT * v_last
    heading[beyond], velocity[beyond], valid[beyond] = scene.heading[r, last], v_last, True
    return position, heading, velocity, valid


def headway_at(scene: Scene, rear: int, merger: int, k: int) -> float:
    _, gap, headway = following_demand(scene, rear, merger, np.array([k]))
    return float(headway[0]) if np.isfinite(gap[0]) and gap[0] > 0 else float("nan")


def overlaps(scene: Scene, r: int, m: int, position: np.ndarray, valid: np.ndarray, until: int) -> bool:
    """Whether the advanced car's footprint overlaps any agent but the merging car (which it would
    run into if it did not brake: what the rewrite is for) up to step ``until``."""
    others = [i for i in range(scene.n_agents) if i not in (r, m) and scene.valid[i, 1:until + 1].any()]
    traj = np.where(valid[1:until + 1, None], position[1:until + 1], FAR)
    return bool(collisions(scene, r, 0, traj[None], until, others).any())


def first_ooi(scene: Scene, agent: int) -> Scene:
    return replace(scene, objects_of_interest=[agent] + [i for i in scene.objects_of_interest if i != agent])


def sources(scene: Scene, min_headway: float, max_headway: float = float("inf")):
    """(merger, rear, join step, headway) of the scene's joins made by the car that came in ahead."""
    segments = attr.map_segments(scene)
    out = []
    for i, j in candidate_pairs(scene):
        for m, r in ((i, j), (j, i)):
            for k in attr.joins(scene, m, r):
                before = np.arange(max(0, k - LANE_CHANGE), k + 1)
                if not (scene.valid[m, before] & scene.valid[r, before]).all() or not attr.same_way(scene, m, r, before).all():
                    continue  # a turn into the road, which the right of way judges, not a lane change
                if np.abs(attr.rel(scene, r, m, before)[1]).max() > NEXT_LANE:
                    continue
                actor, ev = attr.join_actor(scene, m, r, k, segments)
                h = headway_at(scene, r, m, k)
                if (actor == "a" and abs(ev.get("toward_other", 0.0)) <= STILL and h == h
                        and min_headway <= h <= max_headway):
                    out.append((m, r, k, h))
    return out


def make(job):
    path, args = job
    scene = load_scene(path)
    written, dropped = [], {}
    for m, r, k, h0 in sources(scene, args.min_headway, args.max_headway):
        sid = f"{Path(path).stem}_{scene.track_ids[m]}_{scene.track_ids[r]}"
        meta = {"agent": scene.track_ids[m], "other": scene.track_ids[r], "join": k, "headway": round(h0, 3),
                "agent_is_sdc": m == scene.sdc, "other_is_sdc": r == scene.sdc}
        written.append(("log", sid, first_ooi(scene, m), meta))
        for target in args.headway:
            name = f"advance_h{target:g}"
            done = False
            for shift in range(1, args.max_shift + 1):
                position, heading, velocity, valid = advance(scene, r, shift)
                if not valid[k]:
                    break
                moved = scene.with_track(r, position, heading.astype(np.float32), velocity.astype(np.float32), valid)
                h = headway_at(moved, r, m, k)
                if h != h:  # the car behind is level with or past the merging car
                    break
                if h <= target:
                    if overlaps(scene, r, m, position, valid, min(k + JOIN_EFFECT, scene.n_steps - 1)):
                        dropped["overlap"] = dropped.get("overlap", 0) + 1
                    else:
                        written.append((name, sid, first_ooi(moved, m), {**meta, "shift": shift, "headway": round(h, 3),
                                                                         "target": target}))
                    done = True
                    break
            if not done:
                dropped.setdefault(name, 0)
                dropped[name] += 1
    return written, dropped


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    files = scene_files(args.scenes)[: args.n]
    jobs = [(str(f), args) for f in files]
    results = (map(make, jobs) if args.workers <= 1
               else multiprocessing.Pool(args.workers).imap_unordered(make, jobs, chunksize=4))
    index, why = {}, {}
    for written, dropped in results:
        for name, sid, scene, meta in written:
            folder = out / name
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / f"{sid}.pkl", "wb") as f:
                pickle.dump(scene.to_description(), f)
            index.setdefault(name, {})[sid] = meta
        for k, v in dropped.items():
            why[k] = why.get(k, 0) + v
    for name, entries in sorted(index.items()):
        sidecar_index(out / name).write_text(json.dumps(entries, indent=1, sort_keys=True))
        print(f"{name}: {len(entries)} scenes", flush=True)
    print(f"dropped: {why}")
    return index, why


if __name__ == "__main__":
    main()
