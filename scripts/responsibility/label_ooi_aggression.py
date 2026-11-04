"""Was either object of interest aggressive toward the other? Evidence,
rule-based candidate labels and pictures for labelling a logged scene set
by eye. Each of the two (the self-driving car and its OOI partner) is
judged toward the other only; below, X is the one judged and Y the other.

Aggressive means X's own action shifted risk or effort onto Y beyond what
the situation required. Holding the right of way does not excuse a
compressed safety margin. The candidate label is aggressive when any of
these holds, each attributable to X's own motion:

  took_row   Y held the right of way (right_of_way.py) and X went
             through the conflict first. Where the two then drive on in
             one lane (Y behind X within 3 s of X's entry: a merge), Y never
             had to stop before the conflict, so it is judged as a cut-in:
             Y's headway below CUT_IN_THW s or its braking above
             CUT_IN_BRAKE m/s^2. Where the paths part (a crossing), X's
             entry demanded more than TOOK_ROW_DEMAND m/s^2 of braking from
             Y (forced_yield.py's a_req; 1.53 is the q90 of logged drivers
             who went first lawfully)
  margin     whoever held the right of way, the margin belongs to whoever
             chose the timing: X went first and its entry demanded more
             than MARGIN_DEMAND m/s^2 (RSS's brake_min) of a Y that was not
             visibly yielding (its speed fell by less than YIELD_DV m/s
             over the YIELD_WINDOW s before, from its fastest to then).
             The PET is the second one's choice and does not count
             against the first; for the second it is not counted either,
             since the zones are drawn wide (a PET of 0.1 s can leave 5 m)
             and creeping in behind a car that is leaving is normal: what
             the second one did shows as close_pass
  cut_in     X came from beside the path Y drives in the clip (lateral
             offset above OTHER_LANE m) onto it (below IN_LANE m) ahead of
             Y within LANE_CHANGE_S s, and left Y a time headway below
             CUT_IN_THW s (the tailgating threshold below) or made it brake
             harder than CUT_IN_BRAKE m/s^2 (from 1 s before to 3 s
             after). A cut-in by a faster X that pulls away still counts:
             a lane change must not endanger the traffic behind (StVO
             7(5) asks that endangering it be ruled out)
  tailgate   X followed Y in its lane (Y on X's own path: lateral offset
             below SAME_LANE m, headings within SAME_HEADING deg) at
             >= TAIL_SPEED m/s with a time
             headway below TAIL_THW s (German practice fines "less than
             half the speedometer", 0.9 s) for TAIL_HOLD s or longer, or
             with a time to collision below TAIL_TTC s. The first SETTLE s
             after Y came in ahead do not count unless X was closing: a
             car that has just been cut off needs time to drop back.
             Following already under way when the clip starts counts
  close_pass a gap between the two boxes below CLOSE_GAP m at a relative
             speed above CLOSE_REL m/s, X moving, caused by X: at a
             crossing (paths at least CROSS_ANGLE deg apart) X went second,
             or went first while Y was already in its zone (X pressed
             past a car encroaching on its path); elsewhere X closed in
             laterally on Y more than Y on X in the second before

Everything else is not aggressive: no interaction, Y acting on X, queues
at walking pace. The rules are a candidate only: lane matching reads
adjacent lanes as one at times and right_of_way.py misreads some maps, so
each figure shows the evidence for checking by eye.

Per scene, OUT gets evidence.csv (one row per scene: the shared measures,
then sdc_* and ooi_*, each toward the other, with the rules they fired),
figures/<scene>.png (map with both tracks, a dot per second; distance to
the conflict point, speeds, gap and headway, and the light or stop sign each
car faced over time) and, with --video,
videos/<scene>.mp4 (real time, 10 Hz). With --labels the pictures show the
final labels and their reasons instead of the candidates.

    python -m scripts.responsibility.label_ooi_aggression --scenes raw_scenes_500 \\
        --out-dir logs/responsibility/ooi_labels --video --workers 8
"""
import argparse
import sys
import textwrap
from multiprocessing import Pool
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from responsibility.forced_yield import forced_yield, pair_priority  # noqa: E402
from responsibility.lanes import lane_graph  # noqa: E402
from responsibility.right_of_way import _lights, _stop_signs, lane_sequence  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402
from scripts.responsibility.visualize_forced_yield import geometry, pet  # noqa: E402
from scripts.responsibility.visualize_responsibility import box, map_segments  # noqa: E402

DT = 0.1  # s per step
FIRST = 10  # first step judged: the 1 s before is history only
SDC_C, OOI_C, GREY = "#1f77b4", "#ff7f0e", "#9a9a9a"
LIGHT_C = {"STOP": "#d62728", "CAUTION": "#ffbf00", "GO": "#2ca02c"}

TOOK_ROW_DEMAND = 1.53  # m/s^2
MARGIN_DEMAND, MOVING, CROSS_ANGLE = 4.0, 2.0, 30.0  # m/s^2, m/s, deg
YIELD_DV, YIELD_WINDOW = 1.0, 3.0  # m/s, s
CUT_IN_THW, CUT_IN_BRAKE = 0.9, 2.0  # s (the tailgating headway), m/s^2
OTHER_LANE, IN_LANE, SAME_LANE, LANE_CHANGE_S, SAME_HEADING = 2.5, 1.0, 1.2, 8.0, 30.0  # m, m, m, s, deg
EXTEND = 50.0  # m: a path is extended this far straight on at both ends
CONFLICT_ANGLE = 20.0  # deg: paths at least this far apart are a crossing or a merge, not a following
TAIL_SPEED, TAIL_THW, TAIL_HOLD, TAIL_TTC, SETTLE = 5.0, 0.9, 1.0, 2.0, 3.0  # m/s, s, s, s, s
CLOSE_GAP, CLOSE_REL = 0.5, 3.0  # m, m/s


def pair(scene: Scene):
    return scene.sdc, [i for i in scene.objects_of_interest if i != scene.sdc][0]


def speed(scene: Scene, x: int) -> np.ndarray:
    return np.linalg.norm(scene.velocity[x], axis=-1)


def decel(scene: Scene, x: int, t0: int, t1: int) -> float:
    """Largest deceleration of x (m/s^2, 0.5 s differences of the speed) over steps t0 .. t1."""
    t = np.arange(max(t0, 0), min(t1, scene.n_steps - 1) + 1)
    t = t[scene.valid[x, t]]
    if len(t) < 6:
        return 0.0
    v = speed(scene, x)[t]
    return float(max(0.0, -((v[5:] - v[:-5]) / 0.5).min()))


def footprint_pet(scene: Scene, a: int, b: int, cell: float = 0.5):
    """Post-encroachment time over the area both boxes swept (SSAM's definition): on a grid of
    ``cell`` m, the smallest time from the first car leaving a cell both occupied to the second
    entering it. Returns (PET (s), "a" or "b" for the one that went first, the step the second
    entered); (nan, "", None) when the swept areas do not overlap."""
    both = np.flatnonzero(scene.valid[a] | scene.valid[b])
    occ = {}
    for x in (a, b):
        cells = {}
        for t in np.flatnonzero(scene.valid[x]):
            c = box(scene.position[x, t, :2], scene.heading[x, t], *scene.shape_at(x, t))
            lo, hi = np.floor(c.min(0) / cell).astype(int), np.ceil(c.max(0) / cell).astype(int)
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
            centres = (np.stack([gx.ravel(), gy.ravel()], -1) + 0.5) * cell
            h = scene.heading[x, t]
            d = centres - scene.position[x, t, :2]
            lon, lat = d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h), -d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h)
            length, width = scene.shape_at(x, t)
            inside = (np.abs(lon) <= length / 2) & (np.abs(lat) <= width / 2)
            for key in map(tuple, np.stack([gx.ravel(), gy.ravel()], -1)[inside]):
                lo_t, hi_t = cells.get(key, (t, t))
                cells[key] = (min(lo_t, t), max(hi_t, t))
        occ[x] = cells
    shared = set(occ[a]) & set(occ[b])
    if not shared or not len(both):
        return np.nan, "", None
    best = (np.inf, "", None)
    for key in shared:
        (a0, a1), (b0, b1) = occ[a][key], occ[b][key]
        if a1 < b0 and (b0 - a1) * DT < best[0]:
            best = ((b0 - a1) * DT, "a", b0)
        elif b1 < a0 and (a0 - b1) * DT < best[0]:
            best = ((a0 - b1) * DT, "b", a0)
        elif not (a1 < b0 or b1 < a0):  # both in the cell at once: they touched
            return 0.0, "a" if a0 <= b0 else "b", max(a0, b0)
    return best


def box_gap(p: np.ndarray, q: np.ndarray) -> float:
    """Distance (m) between two convex polygons [4, 2], 0 when they overlap."""
    for poly in (p, q):  # separating axis test over both boxes' edge normals
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([-edges[:, 1], edges[:, 0]], -1)
        pp, qq = p @ normals.T, q @ normals.T
        if np.any((pp.max(0) < qq.min(0)) | (qq.max(0) < pp.min(0))):
            break
    else:
        return 0.0

    def to_edges(points, poly):
        a, b = poly, np.roll(poly, -1, axis=0)
        ab = b - a
        t = np.clip(((points[:, None] - a[None]) * ab[None]).sum(-1) / (ab ** 2).sum(-1)[None], 0, 1)
        return np.linalg.norm(points[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min()
    return float(min(to_edges(p, q), to_edges(q, p)))


def gaps(scene: Scene, a: int, b: int) -> np.ndarray:
    """Gap (m) between a's and b's boxes at every step, inf where either is unseen."""
    out = np.full(scene.n_steps, np.inf)
    for t in np.flatnonzero(scene.valid[a] & scene.valid[b]):
        boxes = [box(scene.position[x, t, :2], scene.heading[x, t], *scene.shape_at(x, t)) for x in (a, b)]
        out[t] = box_gap(*boxes)
    return out


def path_frame(scene: Scene, ref: int, x: int):
    """x's position along the path ``ref`` drives in the clip (its seen positions, extended EXTEND m
    straight on at both ends) at every step: arc length ahead of ref (m), signed lateral offset (m)
    and heading difference to the path (rad); nan where either is unseen. Measured against the
    path rather than ref's heading, a car in the next lane on a curve stays in the next lane, and
    only x's own moves change its offset."""
    pts = scene.position[ref, scene.valid[ref], :2].astype(float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=-1) > 0.2]) if len(pts) else []
    pts = pts[keep]
    if len(pts) < 2:  # standing: a straight line along its heading
        h = float(scene.heading[ref, np.flatnonzero(scene.valid[ref])[0]])
        pts = pts[:1] + np.array([[0.0, 0.0], [np.cos(h), np.sin(h)]])
    u0 = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
    u1 = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
    pts = np.concatenate([[pts[0] - EXTEND * u0], pts, [pts[-1] + EXTEND * u1]])
    a, b = pts[:-1], pts[1:]
    ab = b - a
    length = np.linalg.norm(ab, axis=-1)
    s0 = np.concatenate([[0.0], np.cumsum(length)])[:-1]

    def project(q):
        """(arc length, signed lateral offset, segment) of the points q [T, 2]."""
        t = np.clip(((q[:, None] - a[None]) * ab[None]).sum(-1) / (length ** 2)[None], 0, 1)
        foot = a[None] + t[..., None] * ab[None]
        dist = np.linalg.norm(q[:, None] - foot, axis=-1)
        k = dist.argmin(1)
        r = np.arange(len(q))
        side = np.sign(ab[k, 0] * (q[:, 1] - foot[r, k, 1]) - ab[k, 1] * (q[:, 0] - foot[r, k, 0]))
        return s0[k] + t[r, k] * length[k], side * dist[r, k], k

    seen = scene.valid[x] & scene.valid[ref]
    s_x, lat, k = project(scene.position[x, :, :2].astype(float))
    s_ref, _, _ = project(scene.position[ref, :, :2].astype(float))
    tangent = np.arctan2(ab[k, 1], ab[k, 0])
    dh = np.abs((scene.heading[x] - tangent + np.pi) % (2 * np.pi) - np.pi)
    return np.where(seen, s_x - s_ref, np.nan), np.where(seen, lat, np.nan), np.where(seen, dh, np.nan)


def following(scene: Scene, x: int, y: int):
    """x behind y in its lane (y on x's own path, lateral offset below SAME_LANE m, headings within
    SAME_HEADING): time headway (s, inf elsewhere or below 1 m/s), time to collision (s, inf unless
    closing) and how long x had been behind y (s; a stretch already under way when both are first
    seen counts as settled) at every step."""
    lon, lat, dh = path_frame(scene, x, y)
    thw, ttc = np.full(scene.n_steps, np.inf), np.full(scene.n_steps, np.inf)
    since = np.zeros(scene.n_steps)
    v_x = speed(scene, x)
    both = np.flatnonzero(scene.valid[x] & scene.valid[y])
    for t in np.flatnonzero((lon > 0) & (np.abs(lat) < SAME_LANE) & (dh < np.deg2rad(SAME_HEADING))):
        if both.size and t == both[0]:
            since[t] = SETTLE
        else:
            since[t] = since[t - 1] + DT if t > 0 and np.isfinite(thw[t - 1]) else 0.0
        h = float(scene.heading[y, t])
        gap = max(lon[t] - (scene.shape_at(x, t)[0] + scene.shape_at(y, t)[0]) / 2, 0.0)
        vy = float(scene.velocity[y, t, 0] * np.cos(h) + scene.velocity[y, t, 1] * np.sin(h))
        vx = float(scene.velocity[x, t, 0] * np.cos(h) + scene.velocity[x, t, 1] * np.sin(h))
        thw[t] = gap / max(v_x[t], 1e-3)
        if vx - vy > 0.1:
            ttc[t] = gap / (vx - vy)
    thw[v_x < 1.0] = np.inf
    return thw, ttc, since


def cut_in(scene: Scene, x: int, y: int):
    """x's worst cut-in ahead of y: x moving, within LANE_CHANGE_S s, from beside the path y drives
    (lateral offset above OTHER_LANE m) onto it (below IN_LANE m) ahead of y. Measured against y's
    path, a lane change by y does not move x's offset. Returns (step x came in, x's lateral move
    (m), y's smallest headway behind x from 1 s before to 3 s after (s), y's hardest braking then
    (m/s^2)); None without one."""
    lon, lat, dh = path_frame(scene, y, x)
    in_lane = (np.abs(lat) < IN_LANE) & (dh < np.deg2rad(SAME_HEADING))
    v_y = speed(scene, y)
    worst = None
    for t in np.flatnonzero(in_lane[1:] & ~in_lane[:-1]) + 1:
        if t < FIRST or not lon[t] > 0:
            continue
        start = max(t - int(LANE_CHANGE_S / DT), 0)
        before = np.flatnonzero(np.abs(lat[start:t]) > OTHER_LANE)
        if not before.size:
            continue
        t0 = start + int(before[-1])
        w = np.arange(max(t - 10, 0), min(t + 31, scene.n_steps))
        w = w[(np.abs(lat[w]) < SAME_LANE) & (lon[w] > 0) & scene.valid[y, w]]
        gap = np.array([max(lon[k] - (scene.shape_at(x, k)[0] + scene.shape_at(y, k)[0]) / 2, 0.0) for k in w])
        hw = float((gap / np.maximum(v_y[w], 1e-3)).min()) if len(w) and v_y[w].max() >= 1.0 else np.inf
        cand = (int(t), float(abs(lat[t0] - lat[t])), hw, decel(scene, y, t - 10, t + 30))
        if worst is None or hw < worst[2]:
            worst = cand
    return worst


def longest(mask: np.ndarray) -> float:
    """Longest run of True (s)."""
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best * DT


def closing(scene: Scene, x: int, y: int, k: int) -> float:
    """How far x moved toward y (m) across y's heading in the second before step k."""
    t0 = max(k - 10, 0)
    if not (scene.valid[x, t0] and scene.valid[y, k]):
        return 0.0
    h = float(scene.heading[y, k])
    normal = np.array([-np.sin(h), np.cos(h)])
    side = np.sign(float((scene.position[x, k, :2] - scene.position[y, k, :2]) @ normal)) or 1.0
    return float(-(scene.position[x, k, :2] - scene.position[x, t0, :2]) @ normal * side)


def reverse(prio):
    """The pair's right of way seen from the other agent (the same holder and conflict point)."""
    if prio is None:
        return None
    conflict = None if prio.conflict is None else replace(prio.conflict, s=prio.conflict.s[::-1])
    approaches = None if prio.approaches is None else prio.approaches[::-1]
    return replace(prio, conflict=conflict, approaches=approaches)


def toward(scene: Scene, x: int, y: int, prio, g: np.ndarray, names):
    """The measures of x toward y (module docstring) and the rules they fire. ``prio``: the pair's
    right of way with x as the agent; ``g``: their footprint gap at every step; ``names``: (x, y)."""
    nx, ny = names
    v_x, v_y = speed(scene, x), speed(scene, y)
    fy, geo = geometry(scene, x, y, prio)
    p = pet(geo, x, y)
    m = {"pet": np.nan if p is None else p,  # s, positive: x went first
         "demand": np.nan if fy.demand is None else min(fy.demand, 20.0),  # m/s^2 x's entry asked of y; 20: y in its zone
         "entry_t": np.nan if fy.entry is None else fy.entry * DT}
    y_at = None
    if geo is not None:
        inside = geo[y]["steps"][geo[y]["s"] >= -geo[y]["h"]]
        y_at = int(inside[0]) if inside.size else None
    m["other_speed_at_its_entry"] = np.nan if y_at is None else float(v_y[y_at])
    x_at = None
    if geo is not None:
        inside = geo[x]["steps"][geo[x]["s"] >= -geo[x]["h"]]
        x_at = int(inside[0]) if inside.size else None
    m["speed_at_own_entry"] = np.nan if x_at is None else float(v_x[x_at])
    e = fy.entry
    w = np.arange(max(e - int(YIELD_WINDOW / DT), 0), e + 1) if e is not None else np.zeros(0, int)
    w = w[scene.valid[y, w]]
    seen = e is not None and scene.valid[y, e]
    m["other_speed_at_entry"] = float(v_y[e]) if seen else np.nan
    m["other_slowed_before_entry"] = float(v_y[w].max() - v_y[e]) if seen and len(w) else np.nan  # m/s

    thw_x, ttc_x, since_x = following(scene, x, y)
    thw_y, _, _ = following(scene, y, x)
    thw_x[:FIRST] = ttc_x[:FIRST] = thw_y[:FIRST] = np.inf
    fast = (v_x >= TAIL_SPEED) & ((since_x >= SETTLE) | np.isfinite(ttc_x))
    m.update(follow_thw=float(np.where(fast, thw_x, np.inf).min()),
             follow_close_s=longest(fast & (thw_x < TAIL_THW)), follow_ttc=float(ttc_x.min()))
    c = cut_in(scene, x, y)
    m.update(cut_in_t=np.nan if c is None else c[0] * DT, cut_in_lat=np.nan if c is None else c[1],
             cut_in_thw=np.nan if c is None else c[2], cut_in_brake=np.nan if c is None else c[3])
    k = int(np.argmin(g))
    rel_v = float(np.linalg.norm(scene.velocity[x, k] - scene.velocity[y, k])) if np.isfinite(g[k]) else np.nan
    event = e if e is not None else c[0] if c is not None else k
    m["event_t"] = event * DT
    m["other_brake_after"] = decel(scene, y, event - 5, event + 30)
    m["max_decel"] = decel(scene, x, FIRST, scene.n_steps - 1)
    m["max_speed"] = float(v_x[scene.valid[x]].max())

    holder = "self" if prio is not None and prio.holder == x else \
        "other" if prio is not None and prio.holder == y else "none"
    angle = np.nan if prio is None or prio.conflict is None else float(np.degrees(prio.conflict.angle))
    fired = []
    if holder == "other" and p is not None and p > 0:
        behind = np.arange(e, min(e + 31, scene.n_steps)) if e is not None else np.zeros(0, int)
        behind = behind[np.isfinite(thw_y[behind])]
        if behind.size:  # a merge: Y followed X in its lane
            hw, brake = float(thw_y[behind].min()), decel(scene, y, e - 5, e + 30)
            if hw < CUT_IN_THW or brake > CUT_IN_BRAKE:
                fired.append(f"took_row (merged {p:.1f} s ahead of {ny}: headway {hw:.2f} s, braking {brake:.1f} m/s²)")
        elif m["demand"] > TOOK_ROW_DEMAND:
            fired.append(f"took_row (a_req {m['demand']:.2f} m/s², PET {p:.1f} s)")
    if MARGIN_DEMAND < m["demand"] < 20.0 and m["other_speed_at_entry"] >= MOVING \
            and m["other_slowed_before_entry"] < YIELD_DV:
        holder_name = {"self": nx, "other": ny}.get(holder, "none")
        fired.append(f"margin (a_req {m['demand']:.1f} m/s² of {ny}, which was not yielding; holder {holder_name})")
    if c is not None and (c[2] < CUT_IN_THW or c[3] > CUT_IN_BRAKE):
        fired.append(f"cut_in at {c[0] * DT:.1f} s ({nx} moved {c[1]:.1f} m onto {ny}'s path; {ny} headway "
                     f"{c[2]:.2f} s, braking {c[3]:.1f} m/s²)")
    if m["follow_close_s"] >= TAIL_HOLD or m["follow_ttc"] < TAIL_TTC:
        fired.append(f"tailgate (headway {m['follow_thw']:.2f} s, {m['follow_close_s']:.1f} s below {TAIL_THW} s; "
                     f"TTC {m['follow_ttc']:.1f} s)")
    if g[k] < CLOSE_GAP and rel_v > CLOSE_REL and v_x[k] > MOVING:
        crossing = p is not None and angle >= CROSS_ANGLE
        lat_x, lat_y = closing(scene, x, y, k), closing(scene, y, x, k)
        pressed = crossing and p > 0 and m["demand"] >= 20.0
        if (crossing and p < 0) or pressed or (not crossing and lat_x > lat_y):
            why = "went second" if crossing and p < 0 else f"pressed past {ny}, already in its zone" if pressed \
                else f"closed in {lat_x:.1f} m laterally, {ny} {lat_y:.1f} m"
            fired.append(f"close_pass (gap {g[k]:.2f} m at {rel_v:.1f} m/s, {nx} at {v_x[k]:.1f} m/s; {why})")
    m["candidate"] = int(bool(fired))
    m["fired"] = "; ".join(fired)
    return m, {"geo": geo, "thw": thw_x, "event": event}


def evidence(scene: Scene):
    """One row of evidence for the pair (shared measures, then sdc_* and ooi_*: each toward the other)
    and what the pictures need."""
    a, b = pair(scene)
    row = {"scenario_id": scene.scenario_id, "sdc_id": scene.track_ids[a], "ooi_id": scene.track_ids[b]}
    prio = pair_priority(scene, a, b)
    row.update(rule="not-seen-together" if prio is None else prio.case,
               conflict="" if prio is None or prio.conflict is None else prio.conflict.kind,
               holder="none" if prio is None or prio.holder is None else "sdc" if prio.holder == a else "ooi",
               angle=np.nan if prio is None or prio.conflict is None else float(np.degrees(prio.conflict.angle)))
    if prio is not None and prio.approaches is not None:
        for name, ap in zip(("sdc", "ooi"), prio.approaches):
            row.update({f"{name}_control": ap.control, f"{name}_turn": ap.turn, f"{name}_changed_lane": ap.changed_lane})
    g = gaps(scene, a, b)
    g[:FIRST] = np.inf
    k = int(np.argmin(g))
    row.update(min_gap=float(g[k]), t_min_gap=k * DT,
               rel_speed_at_min=float(np.linalg.norm(scene.velocity[a, k] - scene.velocity[b, k])))
    fp, first, at = footprint_pet(scene, a, b)
    row.update(footprint_pet=fp, footprint_first={"a": "sdc", "b": "ooi"}.get(first, ""),
               footprint_second_in_t=np.nan if at is None else at * DT)
    m_a, e_a = toward(scene, a, b, prio, g, ("SDC", "OOI"))
    m_b, e_b = toward(scene, b, a, reverse(prio), g, ("OOI", "SDC"))
    row.update({f"sdc_{key}": v for key, v in m_a.items()})
    row.update({f"ooi_{key}": v for key, v in m_b.items()})
    # the map shows the moment of the first rule that fired, else the closest approach
    fired = [e["event"] for m, e in ((m_a, e_a), (m_b, e_b)) if m["candidate"]]
    event = min(fired) if fired else k if np.isfinite(g[k]) else e_a["event"]
    span = None
    if np.isfinite(fp) and (np.isnan(row["angle"]) or row["angle"] >= CONFLICT_ANGLE):
        span = (int(round(at - fp / DT)), int(at))  # the first one leaves the shared area, the second enters
        event = span[0]
    point = e_a["geo"]["point"] if e_a["geo"] is not None else \
        scene.position[a, k, :2] if np.isfinite(g[k]) else scene.position[a, np.flatnonzero(scene.valid[a])[0], :2]
    return row, {"geo": e_a["geo"], "gap": g, "thw_a": e_a["thw"], "thw_b": e_b["thw"], "event": event,
                 "controls": {x: controls(scene, x, point) for x in (a, b)}, "span": span}


# ----------------------------------------------------------------------------- pictures


def draw_map(ax, scene, groups, step, centre, radius, trail_to=None, point=None):
    a, b = pair(scene)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    for (colour, width), lines in groups.items():
        ax.add_collection(LineCollection(lines, colors=colour, linewidths=width, zorder=1))
    for f in scene.map_features.values():
        if str(f.get("type")) == "STOP_SIGN" and "position" in f:
            ax.plot(*np.asarray(f["position"])[:2], marker="8", ms=7, color="#b00000", zorder=2)
    for light in scene.dynamic_map_states.values():
        states = light.get("state", {}).get("object_state", [])
        s = states[step] if step < len(states) and states[step] else "UNKNOWN"
        colour = next((c for k, c in LIGHT_C.items() if s.endswith(k)), "#bbbbbb")
        ax.plot(*np.asarray(light["stop_point"])[:2], marker="s", ms=5, color=colour, mec="black", mew=0.4, zorder=2)
    for i in range(scene.n_agents):
        if i in (a, b) or not scene.valid[i, step]:
            continue
        length, width = scene.shape_at(i, step)
        ax.add_patch(Polygon(box(scene.position[i, step, :2], scene.heading[i, step], length, width),
                             closed=True, facecolor=GREY, edgecolor="none", alpha=0.45, zorder=3))
    if point is not None:
        ax.plot(*np.asarray(point)[:2], marker="*", ms=12, color="crimson", zorder=8)
    for x, colour, name in ((a, SDC_C, "SDC"), (b, OOI_C, "OOI")):
        steps = np.flatnonzero(scene.valid[x])
        if trail_to is not None:
            steps = steps[steps <= trail_to]
        ax.plot(*scene.position[x, steps, :2].T, color=colour, lw=1.2, alpha=0.8, zorder=4)
        for k in steps[steps % 10 == 0]:
            ax.plot(*scene.position[x, k, :2], "o", ms=3, color=colour, zorder=5)
            ax.annotate(f"{k * DT:.0f}", scene.position[x, k, :2], fontsize=6, color=colour,
                        xytext=(2, 2), textcoords="offset points", zorder=5)
        if scene.valid[x, step]:
            length, width = scene.shape_at(x, step)
            ax.add_patch(Polygon(box(scene.position[x, step, :2], scene.heading[x, step], length, width),
                                 closed=True, facecolor=colour, edgecolor="black", lw=0.8, alpha=0.9, zorder=6))
            ax.annotate(name, scene.position[x, step, :2], fontsize=7, xytext=(4, -9), textcoords="offset points",
                        zorder=7)
    ax.set_xticks([])
    ax.set_yticks([])


def view(scene, ext):
    """Centre and radius (m) of a map view showing the pair from 3 s before to 3 s after the event."""
    a, b = pair(scene)
    e = ext["event"]
    w = np.arange(max(e - 30, 0), min(e + 31, scene.n_steps))
    pts = np.concatenate([scene.position[x, w[scene.valid[x, w]], :2] for x in (a, b)])
    now = [scene.position[x, e, :2] for x in (a, b) if scene.valid[x, e]]
    centre = np.mean(now, axis=0) if now else pts.mean(axis=0)
    reach = np.abs(pts - centre).max() + 8.0
    return centre, float(np.clip(reach, 25.0, 60.0))


def controls(scene: Scene, x: int, point) -> dict:
    """The control x met on its way: the signalled or stop-sign lane of its lane sequence nearest to
    ``point`` (P*, or where the pair came closest), else an exit of its last lane that is controlled.
    {"lane", "kind": light / stop / none, "states": the light's state per step (light only),
    "crossed": first step on that lane (its stop line; None: not crossed in the clip)}."""
    graph = lane_graph(scene)
    steps = np.flatnonzero(scene.valid[x])
    out = {"lane": None, "kind": "none", "states": None, "crossed": None}
    if not len(steps) or not graph.points:
        return out
    seq = lane_sequence(graph, scene.position[x, steps, :2].astype(float), scene.heading[x, steps])
    lights = _lights(scene)
    stop_lanes, _ = _stop_signs(scene)
    lit = lambda lane: lane in lights or lane in stop_lanes  # noqa: E731
    candidates = list(dict.fromkeys(lane for lane in seq if lane is not None and lit(lane)))
    last = next((lane for lane in reversed(seq) if lane is not None), None)
    ahead = [e for e in graph.exits.get(last, []) if lit(e)] if last is not None else []
    candidates += [e for e in ahead if e not in candidates]
    if not candidates:
        return out
    near = lambda lane: float(np.linalg.norm(graph.points[lane] - np.asarray(point)[:2], axis=1).min())  # noqa: E731
    lane = min(candidates, key=near)
    on = [int(steps[k]) for k, l in enumerate(seq) if l == lane]
    out.update(lane=lane, crossed=on[0] if on else None)
    if lane in lights:
        out.update(kind="light", states=list(lights[lane]["state"]["object_state"]))
    else:
        out["kind"] = "stop"
    return out


def draw_controls(ax, scene, ext):
    """One bar per car: the light it faced, coloured by state over time (grey: unknown; "A": an arrow),
    or a stop sign; black underneath where the car stood (below 0.3 m/s); a tick where it crossed its
    stop line."""
    a, b = pair(scene)
    for row, (x, name, colour) in enumerate(((a, "SDC", SDC_C), (b, "OOI", OOI_C))):
        c = ext["controls"][x]
        y0 = 1.2 - row * 1.0
        if c["kind"] == "light":
            states = c["states"]
            for k in range(scene.n_steps):
                st = states[k] if k < len(states) and states[k] else "UNKNOWN"
                col = next((v for key, v in LIGHT_C.items() if st.endswith(key)), "#cccccc")
                ax.add_patch(plt.Rectangle((k * DT, y0), DT, 0.6, color=col, lw=0))
            arrow = any(st and "ARROW" in st for st in states)
            label = f"{name}: light{' (arrow)' if arrow else ''}"
        elif c["kind"] == "stop":
            ax.add_patch(plt.Rectangle((0, y0), 9.1, 0.6, color="#e8d0d0", lw=0))
            label = f"{name}: stop sign"
        else:
            label = f"{name}: no control found"
        stood = scene.valid[x] & (speed(scene, x) < 0.3)
        for k in np.flatnonzero(stood):
            ax.add_patch(plt.Rectangle((k * DT, y0 - 0.15), DT, 0.12, color="black", lw=0))
        if c["crossed"] is not None:
            ax.plot([c["crossed"] * DT] * 2, [y0 - 0.05, y0 + 0.65], color=colour, lw=2.5)
        ax.text(0.05, y0 + 0.3, label, fontsize=7, va="center", color="black")
    ax.set_ylim(-0.1, 2.0)
    ax.set_yticks([])
    ax.set_ylabel("control")


def conflict_point(ext):
    return None if ext["geo"] is None else ext["geo"]["point"]


def draw_series(axes, scene, ext, cursor=None):
    a, b = pair(scene)
    t = np.arange(scene.n_steps) * DT
    ax = axes[0]
    geo = ext["geo"]
    if geo is not None:
        for x, colour, name in ((a, SDC_C, "SDC"), (b, OOI_C, "OOI")):
            ax.plot(geo[x]["steps"] * DT, geo[x]["s"], color=colour, lw=1.5, label=name)
            if np.isfinite(geo[x]["h"]):
                ax.axhspan(-geo[x]["h"], geo[x]["h"], color=colour, alpha=0.08)
        ax.axhline(0, color="crimson", lw=0.8)
        ax.set_ylim(-60, 40)
        ax.set_ylabel("past conflict point (m)")
    else:
        ax.text(0.5, 0.5, "tracks do not cross", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax2 = axes[1]
    for x, colour, name in ((a, SDC_C, "SDC"), (b, OOI_C, "OOI")):
        v = np.where(scene.valid[x], speed(scene, x), np.nan)
        ax2.plot(t, v, color=colour, lw=1.5, label=f"{name} speed")
    ax2.set_ylabel("speed (m/s)")
    ax3 = axes[2]
    g = np.where(np.isfinite(ext["gap"]), ext["gap"], np.nan)
    ax3.plot(t, g, color="black", lw=1.5, label="gap SDC–OOI (m)")
    ax3.set_ylim(0, max(10.0, min(40.0, np.nanmax(g) if np.isfinite(g).any() else 10.0)))
    ax3.set_ylabel("gap (m)")
    ax4 = ax3.twinx()
    for thw, colour, name in ((ext["thw_a"], SDC_C, "SDC behind OOI"), (ext["thw_b"], OOI_C, "OOI behind SDC")):
        h = np.where(np.isfinite(thw), thw, np.nan)
        if np.isfinite(h).any():
            ax4.plot(t, h, color=colour, lw=1.0, ls="--", label=f"headway, {name}")
    ax4.axhline(TAIL_THW, color="grey", lw=0.6, ls=":")
    ax4.set_ylim(0, 4)
    ax4.set_ylabel("time headway (s)")
    for axis in axes[:3]:
        if ext.get("span"):  # the PET: from the first one leaving the shared area to the second entering it
            axis.axvspan(ext["span"][0] * DT, ext["span"][1] * DT, color="#d62728", alpha=0.12, lw=0)
        axis.set_xlim(0, 9.0)
        axis.grid(alpha=0.3)
        axis.axvline(ext["event"] * DT, color="black", lw=0.8, ls=":")
        if cursor is not None:
            axis.axvline(cursor * DT, color="grey", lw=1.0)
        axis.legend(fontsize=6.5, loc="upper left")
    if np.isfinite(ext["thw_a"]).any() or np.isfinite(ext["thw_b"]).any():
        ax4.legend(fontsize=6.5, loc="upper right")
    draw_controls(axes[3], scene, ext)
    axes[3].set_xlim(0, 9.1)
    axes[3].axvline(ext["event"] * DT, color="black", lw=0.8, ls=":")
    if cursor is not None:
        axes[3].axvline(cursor * DT, color="grey", lw=1.0)
    axes[3].set_xlabel("time (s)")


def caption(sid, row) -> str:
    ctrl = "" if pd.isna(row.get("sdc_control", np.nan)) else \
        f" · SDC {row['sdc_control']} {row['sdc_turn']}, OOI {row['ooi_control']} {row['ooi_turn']}"
    pet_s = "–" if np.isnan(row["footprint_pet"]) else \
        f"{row['footprint_pet']:.1f} s, {row['footprint_first'].upper()} first"
    lines = [f"scene {sid} · rule {row['rule']} ({row['conflict'] or 'no crossing'}), holder {row['holder']}{ctrl} · "
             f"PET {pet_s} (swept boxes) · a_req SDC→OOI {row['sdc_demand']:.2f}, OOI→SDC {row['ooi_demand']:.2f} m/s²"]
    for who in ("sdc", "ooi"):
        if row.get("hide_candidates"):
            continue
        if f"{who}_aggressive" in row:  # the final label, when given
            verdict = "AGGRESSIVE" if row[f"{who}_aggressive"] else "not aggressive"
            lines.append(f"{who.upper()} label: {verdict}: {textwrap.shorten(str(row[f'{who}_reason']), 190)}")
        else:
            lines.append(f"{who.upper()} candidate: " + ("AGGRESSIVE: " + row[f"{who}_fired"]
                                                         if row[f"{who}_candidate"] else "not aggressive"))
    return "\n".join(lines)


def figure(scene, sid, row, ext, out: Path):
    fig = plt.figure(figsize=(15, 8.4))
    grid = fig.add_gridspec(4, 2, width_ratios=[1.15, 1], height_ratios=[1, 1, 1, 0.6])
    ax_map = fig.add_subplot(grid[:, 0])
    centre, radius = view(scene, ext)
    draw_map(ax_map, scene, map_segments(scene), ext["event"], centre, radius, point=conflict_point(ext))
    ax_map.set_title(f"footprints at the event, t = {ext['event'] * DT:.1f} s (red band: the PET) "
                     "(dots: every 1 s; star: first crossing of the tracks)", fontsize=9)
    draw_series([fig.add_subplot(grid[i, 1]) for i in range(4)], scene, ext)
    fig.suptitle(caption(sid, row), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def video(scene, sid, row, ext, out: Path, fps: float = 10.0):
    import cv2

    groups = map_segments(scene)
    pair_ = pair(scene)
    centre = view(scene, ext)[0]
    writer = None
    for step in range(scene.n_steps):
        # the view follows the midpoint of the pair (the one still seen when the other is not)
        now = [scene.position[x, step, :2] for x in pair_ if scene.valid[x, step]]
        centre = np.mean(now, axis=0) if now else centre
        gap = np.linalg.norm(now[0] - now[1]) if len(now) == 2 else 0.0
        radius = float(np.clip(gap / 2 + 20.0, 30.0, 60.0))
        fig = plt.figure(figsize=(14, 7.6))
        grid = fig.add_gridspec(4, 2, width_ratios=[1.15, 1], height_ratios=[1, 1, 1, 0.6])
        ax_map = fig.add_subplot(grid[:, 0])
        draw_map(ax_map, scene, groups, step, centre, radius, trail_to=step, point=conflict_point(ext))
        ax_map.set_title(f"t = {step * DT:.1f} s", fontsize=10)
        draw_series([fig.add_subplot(grid[i, 1]) for i in range(4)], scene, ext, cursor=step)
        fig.suptitle(caption(sid, row), fontsize=8.5)
        fig.tight_layout()
        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        if writer is None:
            h, w = im.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    writer.release()


def one(job):
    path, out, make_video, label, hide = job
    scene = Scene.load(path)
    row, ext = evidence(scene)
    row = {"scene": path.stem, **row}
    shown = {**row, **(label or {}), "hide_candidates": hide and not label}
    figure(scene, path.stem, shown, ext, out / "figures" / f"{path.stem}.png")
    if make_video:
        video(scene, path.stem, shown, ext, out / "videos" / f"{path.stem}.mp4")
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenes", required=True, help="folder of scenes whose SDC is an object of interest")
    p.add_argument("--scene", nargs="+", default=None, help="only these scene names")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--video", action="store_true", help="also write videos/<scene>.mp4")
    p.add_argument("--labels", default=None,
                   help="labels CSV (scene, sdc_aggressive, sdc_reason, ooi_aggressive, ooi_reason): the pictures "
                        "show these instead of the candidates")
    p.add_argument("--hide-candidates", action="store_true",
                   help="leave the rule candidates out of the captions (for labelling without anchoring on them)")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    for sub in ("figures", "videos") if a.video else ("figures",):
        (out / sub).mkdir(parents=True, exist_ok=True)
    files = [f for f in scene_files(a.scenes) if a.scene is None or f.stem in a.scene]
    labels = {}
    if a.labels:
        for r in pd.read_csv(a.labels, dtype={"scene": str}).to_dict("records"):
            labels[r["scene"]] = {k: r[k] for k in ("sdc_aggressive", "sdc_reason", "ooi_aggressive", "ooi_reason")}
    jobs = [(f, out, a.video, labels.get(f.stem), a.hide_candidates) for f in files]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            rows = pool.map(one, jobs, chunksize=2)
    else:
        rows = [one(j) for j in jobs]
    d = pd.DataFrame(rows)
    d = d.iloc[np.argsort([int(s) if s.isdigit() else 0 for s in d.scene])]
    d.to_csv(out / "evidence.csv", index=False)
    for who in ("sdc", "ooi"):
        fired = d[f"{who}_fired"].fillna("")
        print(f"{who}: {len(d)} scenes, candidate aggressive {int(d[f'{who}_candidate'].sum())} ("
              + ", ".join(f"{r} {int(fired.str.contains(r).sum())}" for r in
                          ("took_row", "margin", "cut_in", "tailgate", "close_pass")) + ")")


if __name__ == "__main__":
    main()
