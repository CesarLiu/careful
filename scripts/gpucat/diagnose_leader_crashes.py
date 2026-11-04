"""Why do the trained GPUDrive policies run into the vehicle ahead? (the
"leader" collisions of gpucat.env.PARTNER_CLASSES, half of all collisions
on the test scenes, gpucat/PLAN.md, "反应式背景交通").

Re-runs evaluate.py's episodes (same order, deterministic actions, the log
replay unless --background reactive), recording the SDC and every agent at
every step. Each collision's outcome and partner must equal evaluate.py's
CSV (checked when --check-against is given).

For every collision with a leader, at the step of contact k:
  advance       how many steps earlier than its log the SDC reached the same
                arc length along its route (k_log - k; > 0 means faster
                than the log); empty when the log never got that far
  v_sdc, v_log  the SDC's speed (m/s) and its logged speed at the same arc length
  v_leader      the leader's speed (m/s)
and HORIZON s (20 steps) before contact, in the SDC's frame at that time:
  visible       the leader within GPUDrive's 50 m observation radius and valid
  lat           the leader's lateral offset (m); |lat| > LANE (2.5 m) means it
                was not yet ahead in the SDC's lane: a cut-in
  gap, closing  bumper gap (m, along the SDC's heading) and closing speed (m/s)
  need          the constant deceleration (m/s^2) that would have stopped the
                closing within the gap: closing^2 / (2 gap); <= 6 m/s^2 (the
                SDC's own limit) means a braking SDC avoids it
  brake         the SDC's hardest commanded acceleration (m/s^2) over those
                HORIZON steps, and the steps it commanded < -1 m/s^2

Classes: stopped (leader below 0.5 m/s at contact), cut_in (|lat| > LANE or
not visible HORIZON steps before), following (otherwise: a leader in the lane,
in sight, the SDC closing on it).

Also, per test, over the episodes that end in a collision and over the
others: the SDC's distance along its route over the episode against the
log's over the same steps (median ratio; pace_vs_log.py's measure), the
share of steps it drives more than 2 m/s faster than the log at the same arc
length, and the share of steps it commands braking harder than -1 m/s^2
(and harder than -3 m/s^2).

Writes OUT/<run>.csv (one row per leader collision) and OUT/summary.json.

    PYTHONPATH=~/gpudrive/build:~/gpudrive:. LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.diagnose_leader_crashes \\
        --runs replay_nav_s0 cat_nav_s0 fair_nav_s0 fair_valid_nav_s0 --worlds 25 \\
        --out logs/gpucat/leader_crashes --check-against logs/gpucat/eval_bg/replay
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from gpucat.env import CRASHED, EPISODE_STEPS, PARTNER_CLASSES, CatEnv, EnvSettings  # noqa: E402
from gpucat.geometry import route_coords  # noqa: E402
from gpucat.ppo import ActorCritic  # noqa: E402
from gpucat.reactive import POS, TRAJ_LEN, VALID, VEL  # noqa: E402
from scripts.gpucat.evaluate import OUTCOMES, Policy  # noqa: E402
from scripts.gpucat.train import TEST  # noqa: E402

HORIZON = 20  # steps (2 s) before contact
LANE = 2.5  # m
OBS_RADIUS = 50.0  # m, GPUDrive's EnvConfig.obs_radius
DT = 0.1
LEADER = PARTNER_CLASSES.index("leader")


def run_episode(env, policy, adversarial, rule, opts):
    """train.rollout with a record of every step: the SDC's position [W, S, 2]
    (raw), heading [W, S], commanded acceleration [W, S] (m/s^2), and every
    agent's position [W, S, 64, 2] (raw), heading, speed and validity, S
    steps (index k-1 after k steps)."""
    W, s = env.W, env.s
    obs = env.begin_episode(adversarial, rule, opts.tau, opts.rho, evaluation=True)
    rec = {k: [] for k in ("pos", "yaw", "speed", "valid", "accel")}
    for _ in range(EPISODE_STEPS):
        with torch.no_grad():
            a, _, _ = policy.net.act(obs, True)
        acc = s.accel_low + 0.5 * (a[:, 0].clamp(-1, 1) + 1.0) * (s.accel_high - s.accel_low)
        obs, _, _, _ = env.step(a)
        g = env.sim.absolute_self_observation_tensor().to_torch()
        r = (env.t - 1).clamp(min=0)
        rows = env.traj[..., :VALID + TRAJ_LEN]
        vel = rows[..., VEL:VEL + 2 * TRAJ_LEN].unflatten(-1, (TRAJ_LEN, 2))
        vel = vel.gather(2, r[:, None, None, None].expand(-1, 64, 1, 2)).squeeze(2)
        valid = rows[..., VALID:VALID + TRAJ_LEN].gather(2, r[:, None, None].expand(-1, 64, 1)).squeeze(2) > 0.5
        rec["pos"].append(g[..., :2] + env.mean[:, None])
        rec["yaw"].append(g[..., 7].clone())
        rec["speed"].append(vel.norm(dim=-1))
        rec["valid"].append(valid)
        rec["accel"].append(acc)
        if bool(env.done.all()):
            break
    out = env.end_episode(True)
    rec = {k: torch.stack(v, 1) for k, v in rec.items()}
    # the SDC's speed from its positions (its own row is not its motion)
    p0 = rec["pos"][:, :, 0]
    prev = torch.cat([p0[:, :1], p0[:, :-1]], 1)
    rec["speed"][:, :, 0] = (p0 - prev).norm(dim=-1) / DT
    rec["valid"][:, :, 0] = True
    return out, rec


def logged_arc(env):
    """[W, T] arc length of the logged SDC along its route at each row (nan
    where invalid)."""
    pos = env.orig_rows[:, 0, POS:POS + 2 * TRAJ_LEN].unflatten(-1, (TRAJ_LEN, 2)) + env.mean[:, None]
    valid = env.orig_rows[:, 0, VALID:VALID + TRAJ_LEN] > 0.5
    s = torch.stack([route_coords(pos[:, t], env.route, env.route_n)[1] for t in range(TRAJ_LEN)], 1)
    return torch.where(valid, s, torch.full_like(s, float("nan")))


def frame(p, h, q):
    """q's (lon, lat) in the frame at p with heading h."""
    dx, dy = float(q[0] - p[0]), float(q[1] - p[1])
    return dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h)


def analyse(env, i, out, rec, s_log, s_sdc):
    """The leader collision of world i (a dict), or None."""
    if int(out["outcome"][i]) != CRASHED or int(out["partner_class"][i]) != LEADER:
        return None
    j = int(out["partner"][i])
    steps = int((~torch.isnan(s_sdc[i])).sum())
    k = steps  # contact after k steps: record index k-1
    c = k - 1
    sl = s_log[i]
    reach = torch.nonzero(sl >= s_sdc[i, c]).flatten()
    k_log = int(reach[0]) if len(reach) else None
    row = {"steps": k, "partner": j}
    row["advance"] = "" if k_log is None else k_log - k
    row["v_sdc"] = round(float(rec["speed"][i, c, 0]), 2)
    if k_log is not None and k_log >= 1:
        row["v_log"] = round(float(sl[k_log] - sl[k_log - 1]) / DT, 2)
    else:
        row["v_log"] = ""
    row["v_leader"] = round(float(rec["speed"][i, c, j]), 2)
    b = max(c - HORIZON, 0)
    p, h = rec["pos"][i, b, 0], float(rec["yaw"][i, b, 0])
    q = rec["pos"][i, b, j]
    lon, lat = frame(p, h, q)
    vis = bool(rec["valid"][i, b, j]) and math.hypot(lon, lat) <= OBS_RADIUS
    gap = lon - 0.5 * float(env.sizes[i, 0, 0] + env.sizes[i, j, 0])
    hj = float(rec["yaw"][i, b, j])
    closing = float(rec["speed"][i, b, 0]) - float(rec["speed"][i, b, j]) * math.cos(hj - h)
    row.update(visible=int(vis), lat=round(lat, 2), gap=round(gap, 2), closing=round(closing, 2))
    row["need"] = round(closing ** 2 / (2 * gap), 2) if gap > 0 and closing > 0 else ("0" if closing <= 0 else "inf")
    acc = rec["accel"][i, b:c + 1]
    row["brake"] = round(float(acc.min()), 2)
    row["brake_steps"] = int((acc < -1.0).sum())
    if row["v_leader"] < 0.5:
        row["class"] = "stopped"
    elif abs(lat) > LANE or not vis:
        row["class"] = "cut_in"
    else:
        row["class"] = "following"
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--root", default="logs/gpucat/runs")
    p.add_argument("--adversaries", nargs="*", default=["cat", "fair", "fair_valid"])
    p.add_argument("--scenes", default="logs/gpucat/scenes")
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--valid", default="logs/gpucat/valid_samples.npz")
    p.add_argument("--background", choices=["replay", "reactive"], default="replay")
    p.add_argument("--worlds", type=int, default=25)
    p.add_argument("--device", default="cuda", help="cpu runs GPUDrive's CPU backend (no GPU memory)")
    p.add_argument("--out", default="logs/gpucat/leader_crashes")
    p.add_argument("--check-against", default=None, help="evaluate.py's output directory for the same runs")
    args = p.parse_args(argv)
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = {r: json.loads((REPO / args.root / r / "config.json").read_text()) for r in args.runs}
    W = min(args.worlds, len(TEST))
    env = CatEnv(EnvSettings(worlds=W, navigation=True, background=args.background), REPO / args.scenes,
                 REPO / args.static, REPO / args.bank, device=args.device, valid_path=REPO / args.valid)
    policies = {}
    for run in args.runs:
        ck = torch.load(REPO / args.root / run / "model.pt", map_location=env.device)
        net = ActorCritic(ck["obs_dim"]).to(env.device)
        net.load_state_dict(ck["net"])
        policies[run] = Policy(net)
    tests = [("none", False, "cat")] + [(rule, True, rule) for rule in args.adversaries]
    rows = {r: [] for r in args.runs}
    outcomes = {r: {} for r in args.runs}
    pace = {r: {t[0]: {f"{m}_{k}": [] for m in ("ratio", "fast", "brake1", "brake3") for k in ("crash", "other")}
                for t in tests} for r in args.runs}
    for b0 in range(0, len(TEST), W):
        batch = TEST[b0:b0 + W]
        env.load((batch * W)[:W])
        s_log = logged_arc(env)
        for run in args.runs:
            a = configs[run]["args"]
            opts = argparse.Namespace(tau=a["tau"], rho=a["rho"])
            for name, adversarial, rule in tests:
                out, rec = run_episode(env, policies[run], torch.full((W,), adversarial, device=env.device), rule, opts)
                S = rec["pos"].shape[1]
                s_sdc = torch.stack([route_coords(rec["pos"][:, k, 0], env.route, env.route_n)[1] for k in range(S)], 1)
                steps = torch.arange(1, S + 1, device=env.device)
                alive = steps[None] <= env.t[:, None]
                s_sdc = torch.where(alive, s_sdc, torch.full_like(s_sdc, float("nan")))
                for i, stem in enumerate(batch):
                    outcomes[run][(stem, name)] = (OUTCOMES[int(out["outcome"][i])], int(out["partner"][i]))
                    n = int(env.t[i])
                    sl = s_log[i, :n + 1]
                    ok = torch.nonzero(~torch.isnan(sl)).flatten()
                    # logged speed at the SDC's arc length: interpolate the log's speed profile
                    v_log = torch.diff(s_log[i], prepend=s_log[i, :1]) / DT
                    v_log = torch.nan_to_num(v_log, nan=0.0)
                    idx = torch.searchsorted(torch.nan_to_num(s_log[i], nan=-1.0).cummax(0).values.contiguous(),
                                             s_sdc[i, :n].contiguous()).clamp(max=TRAJ_LEN - 1)
                    fast = (rec["speed"][i, :n, 0] > v_log[idx] + 2.0).float().mean()
                    key = "crash" if int(out["outcome"][i]) == CRASHED else "other"
                    pc = pace[run][name]
                    if len(ok) and n >= 10:
                        last = int(ok[-1])  # the log's distance over the same steps (to its end if shorter)
                        logged = float(sl[last] - s_log[i, 0])
                        if logged > 5.0:
                            pc["ratio_" + key].append(float(s_sdc[i, min(n, last) - 1] - s_log[i, 0]) / logged)
                    pc["fast_" + key].append(float(fast))
                    acc = rec["accel"][i, :n]
                    pc["brake1_" + key].append(float((acc < -1.0).float().mean()))
                    pc["brake3_" + key].append(float((acc < -3.0).float().mean()))
                    r = analyse(env, i, out, rec, s_log, s_sdc)
                    if r is not None:
                        rows[run].append({"scene": stem, "test_adversary": name, **r})
            print(f"{run}: scenes {batch[0]}-{batch[-1]} done", flush=True)
    summary = {}
    for run in args.runs:
        if args.check_against:
            ref = {(x["scene"], x["test_adversary"]): (x["outcome"], int(x["partner"]))
                   for x in csv.DictReader(open(REPO / args.check_against / f"{run}.csv"))
                   if x["test_adversary"] in {t[0] for t in tests}}
            same = sum(outcomes[run][k] == v for k, v in ref.items())
            summary.setdefault("check", {})[run] = f"{same}/{len(ref)} episodes match evaluate.py"
        fields = ["scene", "test_adversary", "class", "steps", "partner", "advance", "v_sdc", "v_log", "v_leader",
                  "visible", "lat", "gap", "closing", "need", "brake", "brake_steps"]
        with open(out_dir / f"{run}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows[run])
        summary[run] = {}
        for name, *_ in tests:
            rr = [x for x in rows[run] if x["test_adversary"] == name]
            pc = pace[run][name]
            mean = lambda v: round(sum(v) / len(v), 3) if v else None  # noqa: E731
            median = lambda v: round(sorted(v)[len(v) // 2], 3) if v else None  # noqa: E731
            summary[run][name] = {
                "leader_crashes": len(rr), "classes": dict(Counter(x["class"] for x in rr)),
                **{f"{m}_{k}": (median if m == "ratio" else mean)(pc[f"{m}_{k}"])
                   for m in ("ratio", "fast", "brake1", "brake3") for k in ("crash", "other")}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
