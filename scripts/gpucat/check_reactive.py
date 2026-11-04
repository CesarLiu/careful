"""Checks the reactive background traffic (gpucat/reactive.py) in GPUDrive
on CAT's test scenes (400-499), with the SDC driven by the simulator itself
(switched to an expert, as check_env.py does) along rewritten logs:

  logged   the SDC's own log: the reactive traffic must be the log replay
           exactly (every agent's position at every step, 0 m apart, and no
           agent off its log)
  slow     the SDC along its logged path at half its logged pace (its log
           at time t/2)
  stopped  the SDC stopping dead at step 20 and staying there

For each, under the log replay and under the reactive traffic: the SDC's
collisions and their partners' classes (gpucat.env.PARTNER_CLASSES), the
agents that left their logs, and, over the steps the episode runs, the
pairs of valid background vehicles whose full-size boxes overlap (the log
itself has some) and those of them involving an agent off its log (the
overlaps the reactive traffic makes). Every reactive collision with a
follower is listed: whether the follower had left its log, and its logged
speed and bumper gap to the SDC when the SDC stopped. Also the time a step
takes.

Run with GPUDrive's environment, from this repository's root:
    PYTHONPATH=~/gpudrive/build:~/gpudrive:. LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.check_reactive --worlds 25 --out logs/gpucat/check_reactive.json
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from gpucat.env import CRASHED, EPISODE_STEPS, PARTNER_CLASSES, CatEnv, EnvSettings  # noqa: E402
from gpucat.geometry import boxes_overlap  # noqa: E402
from gpucat.reactive import POS, TRAJ_LEN, VALID, VEL, YAW  # noqa: E402
from scripts.gpucat.train import TEST  # noqa: E402

T = TRAJ_LEN


def sdc_rows(row: torch.Tensor, mode: str) -> torch.Tensor:
    """The SDC's trajectory rows [W, 1456] rewritten for the scenario."""
    if mode == "logged":
        return row
    new = row.clone()
    pos = row[:, POS:POS + 2 * T].view(-1, T, 2)
    vel = row[:, VEL:VEL + 2 * T].view(-1, T, 2)
    yaw, valid = row[:, YAW:YAW + T], row[:, VALID:VALID + T]
    if mode == "slow":
        tt = torch.arange(T, device=row.device) / 2.0
        lo, hi = tt.floor().long(), tt.ceil().long()
        u = (tt - lo)[None, :, None]
        new_pos = pos[:, lo] + u * (pos[:, hi] - pos[:, lo])
        d = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
        dd = d[:, lo] + u * (d[:, hi] - d[:, lo])
        new_yaw = torch.atan2(dd[..., 1], dd[..., 0])
        new_vel = 0.5 * (vel[:, lo] + u * (vel[:, hi] - vel[:, lo]))
        new_valid = valid[:, lo] * valid[:, hi]
    else:  # stopped from step 20 on
        k = torch.arange(T, device=row.device).clamp(max=20)
        new_pos, new_yaw, new_valid = pos[:, k], yaw[:, k], valid[:, k]
        new_vel = torch.where((torch.arange(T, device=row.device) < 20)[None, :, None], vel[:, k], torch.zeros_like(vel))
    new[:, POS:POS + 2 * T] = new_pos.flatten(1)
    new[:, VEL:VEL + 2 * T] = new_vel.flatten(1)
    new[:, YAW:YAW + T] = new_yaw
    new[:, VALID:VALID + T] = new_valid
    return new


def background_overlaps(env, g, involve: torch.Tensor) -> torch.Tensor:
    """[2, W]: pairs of valid background vehicles (not the SDC) whose boxes
    overlap now, and those of them with a vehicle in involve [W, 64]."""
    pos, yaw = g[..., :2], g[..., 7]
    r = (env.t - 1).clamp(min=0)
    valid = env.traj[:, :, VALID:VALID + T].gather(2, r[:, None, None].expand(-1, 64, 1)).squeeze(2) > 0.5
    slot = torch.arange(64, device=env.device)[None]
    veh = valid & (env.types == 1) & (slot >= 1) & (slot < env.n_agents[:, None])
    count = torch.zeros(2, env.W, dtype=torch.long, device=env.device)
    for i in range(1, 64):
        if not bool(veh[:, i].any()):
            continue
        ov = boxes_overlap(pos[:, i], yaw[:, i], env.sizes[:, i], pos, yaw, env.sizes) & veh & (slot > i)
        ov &= veh[:, i:i + 1]
        count[0] += ov.sum(-1)
        count[1] += (ov & (involve | involve[:, i:i + 1])).sum(-1)
    return count


def follower_crashes(env, out, batch, mode):
    """The reactive episode's collisions with a follower: the follower's slot,
    whether it had left its log, and (stopped) its logged speed (m/s) and
    bumper gap to the SDC (m, along the SDC's heading) at step 20."""
    rows = []
    for i, stem in enumerate(batch):
        if int(out["outcome"][i]) != CRASHED or PARTNER_CLASSES[int(out["partner_class"][i])] != "follower":
            continue
        j = int(out["partner"][i])
        row = {"mode": mode, "scene": stem, "partner": j, "left_its_log": bool(env.reactive.ever[i, j]),
               "reactive_agent": bool(env.reactive.reactive[i, j])}
        if mode == "stopped":
            sdc = env.orig_rows[i, 0, POS + 40:POS + 42]
            h = float(env.orig_rows[i, 0, YAW + 20])
            pj = env.orig_rows[i, j, POS + 40:POS + 42]
            lon = float((sdc - pj) @ torch.tensor([math.cos(h), math.sin(h)], device=env.device))
            row["log_speed"] = round(float(env.reactive.v_at[i, j, 20]), 2)
            row["bumper_gap"] = round(lon - 0.5 * float(env.sizes[i, 0, 0] + env.sizes[i, j, 0]), 2)
        rows.append(row)
    return rows


def episode(env, mode, background):
    env.s.background = background
    control = env.sim.controlled_state_tensor().to_torch()
    control[:, 0] = 0  # the SDC replayed by the simulator from its (rewritten) rows
    orig = env.traj[:, 0].clone()
    env.traj[:, 0] = sdc_rows(orig, mode)
    env.begin_episode(torch.zeros(env.W, dtype=torch.bool, device=env.device))
    positions, overlaps, react_s = [], torch.zeros(2, env.W, dtype=torch.long, device=env.device), 0.0
    for _ in range(EPISODE_STEPS):
        torch.cuda.synchronize()
        t0 = time.time()
        _, _, _, alive = env.step(torch.zeros(env.W, 2, device=env.device))
        torch.cuda.synchronize()
        react_s += time.time() - t0
        g = env.sim.absolute_self_observation_tensor().to_torch()
        positions.append(g[..., :2].clone())
        off = env.reactive.off if background == "reactive" else torch.zeros_like(env.reactive.off)
        overlaps += background_overlaps(env, g, off) * alive
    out = env.end_episode()
    control[:, 0] = 1
    env.traj[:, 0] = orig
    return out, torch.stack(positions, 1), overlaps, react_s / EPISODE_STEPS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worlds", type=int, default=25)
    p.add_argument("--scenes", default="logs/gpucat/scenes")
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--out", default="logs/gpucat/check_reactive.json")
    args = p.parse_args(argv)
    W = args.worlds
    env = CatEnv(EnvSettings(worlds=W), REPO / args.scenes, REPO / args.static, REPO / args.bank)
    res = {m: {b: {"crash": 0, "partners": Counter(), "agents_off": 0, "max_lag": 0.0, "bg_overlaps": 0,
                   "bg_overlaps_off_log": 0, "step_ms": []} for b in ("replay", "reactive")}
           for m in ("logged", "slow", "stopped")}
    followers = []
    max_diff = 0.0
    for b in range(0, len(TEST), W):
        batch = TEST[b:b + W]
        n = len(batch)
        env.load((batch * W)[:W])
        for mode in res:
            pos = {}
            for bg in ("replay", "reactive"):
                out, pos[bg], ov, ms = episode(env, mode, bg)
                r = res[mode][bg]
                crash = out["outcome"][:n] == CRASHED
                r["crash"] += int(crash.sum())
                r["partners"].update(PARTNER_CLASSES[int(c)] for c in out["partner_class"][:n][crash])
                r["agents_off"] += int(out["agents_off"][:n].sum())
                r["max_lag"] = max(r["max_lag"], float(out["max_lag"][:n].max()))
                r["bg_overlaps"] += int(ov[0, :n].sum())
                r["bg_overlaps_off_log"] += int(ov[1, :n].sum())
                if bg == "reactive":
                    followers += follower_crashes(env, out, batch, mode)
                r["step_ms"].append(1000 * ms)
            if mode == "logged":
                max_diff = max(max_diff, float((pos["replay"][:n] - pos["reactive"][:n]).abs().max()))
        print(f"scenes {batch[0]}-{batch[-1]} done", flush=True)
    for mode in res:
        for bg, r in res[mode].items():
            r["partners"] = dict(r["partners"])
            r["step_ms"] = round(sum(r["step_ms"]) / len(r["step_ms"]), 2)
            r["max_lag"] = round(r["max_lag"], 2)
    summary = {"scenes": len(TEST), "worlds": W, "logged_max_position_difference_m": max_diff, **res,
               "reactive_follower_crashes": followers}
    print(json.dumps(summary, indent=1))
    (REPO / args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
