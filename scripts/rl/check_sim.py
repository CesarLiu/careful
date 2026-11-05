"""Checks the log-replay simulator (rl/sim.py) on real scenes and measures
its speed. Prints PASS/FAIL per check and ends with ALL CHECKS PASSED.

  1. replaying the SDC's log with the bicycle's inverse (expert_action,
     unclipped) keeps it on its logged positions: median over scenes of the
     largest error below --tol m (steps where the log needs a negative speed
     cannot be followed and are counted, not failed)
  2. the logged drive is a good episode: with the expert actions clipped to
     [-1, 1], most scenes end arrived or at the time limit, not out of road
     (CAT's scenes are logged human driving: out of road there means the
     route, edges or yellow lines are built wrongly)
  3. without an adversary, nothing collides with the replayed SDC more than
     rarely (the log is collision-free up to annotation noise)
  4. with CAT's adversary (needs --candidates), its plan is followed: slot 1
     sits on the chosen candidate a step late
  5. throughput: world steps per second with --worlds worlds

    python -m scripts.rl.check_sim --bank logs/rl/scenes.npz --candidates logs/rl/candidates.npz --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from rl.adversary import CandidateBank  # noqa: E402
from rl.scenes import SceneBank  # noqa: E402
from rl.sim import ARRIVED, OUT_OF_ROAD, OUTCOMES, TIMEOUT, LogReplaySim, SimSettings, obs_dim  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def replay(sim, rows, clip: bool, adversarial: bool = False):
    sim.begin(rows, torch.full((sim.W,), adversarial, device=sim.device))
    err = torch.zeros(sim.W, device=sim.device)
    outside = torch.zeros(sim.W, device=sim.device)
    log = sim.bank.position[rows, 0]
    while not bool(sim.done.all()):
        a = sim.expert_action()
        outside += ((a.abs() > 1.0).any(-1) & ~sim.done).float()
        sim.step(a, clip=clip)
        live = sim.t < log.shape[1]
        e = (sim.p - log[torch.arange(sim.W), sim.t.clamp(max=log.shape[1] - 1)]).norm(dim=-1)
        err = torch.maximum(err, torch.where(live, e, torch.zeros_like(e)))
    return sim.end_episode(), err, outside


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bank", required=True)
    p.add_argument("--candidates", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--worlds", type=int, default=100)
    p.add_argument("--tol", type=float, default=0.05)
    args = p.parse_args(argv)

    bank = SceneBank(args.bank, args.device)
    cand = CandidateBank(args.candidates, args.device) if args.candidates else None
    W = min(args.worlds, len(bank))
    rows = torch.arange(W, device=args.device)
    sim = LogReplaySim(SimSettings(worlds=W), bank, cand, args.device)
    print(f"{len(bank)} scenes, {W} worlds, observation {obs_dim(sim.s)} values, up to {sim.A} agents")

    _, err, outside = replay(sim, rows, clip=False)
    med = float(err.median())
    check("1 the inverse replays the log", med < args.tol,
          f"max error per scene: median {med:.4f} m, worst {float(err.max()):.3f} m; "
          f"{int((outside > 0).sum())} scenes need actions outside [-1, 1] somewhere")

    rep, _, _ = replay(sim, rows, clip=True)
    counts = {OUTCOMES[k]: int((rep["outcome"] == k).sum()) for k in range(1, len(OUTCOMES))}
    good = int(((rep["outcome"] == ARRIVED) | (rep["outcome"] == TIMEOUT)).sum())
    check("2 the logged drive is a good episode", good >= 0.9 * W, f"{counts}")
    bad = torch.nonzero(rep["outcome"] == OUT_OF_ROAD).flatten().tolist()
    if bad:
        print(f"    out of road: scenes {[bank.stems[int(rows[i])] for i in bad][:10]}")
    check("3 no collisions in the replayed log", float(rep["crashed"].float().mean()) <= 0.1,
          f"{int(rep['crashed'].sum())} of {W} scenes touch a vehicle")

    if cand is not None:
        sim.rule = "cat"
        has = sim.has_adversary(rows)
        rep, _, _ = replay(sim, rows, clip=True, adversarial=True)
        chosen = rep["chosen"]
        n = int(has.sum())
        check("4 CAT's adversary is chosen and injected", bool((chosen[has] >= 0).all()) and n > 0,
              f"{n} worlds with an adversary; collisions {int(rep['crashed'][has].sum())}")
        sim.begin(rows, has)
        for _ in range(30):
            sim.step(sim.expert_action())
        pos, _, _, _ = sim.agents()
        c = sim.crow_of[rows]
        want = cand.cand[c, sim.chosen.clamp(min=0), 30 - sim.s.adversary_lag - 11]
        gap = (pos[:, 1] - want).norm(dim=-1)[has]
        check("4 the adversary follows its plan a step late", float(gap.max()) < 1e-3, f"max {float(gap.max()):.2e} m")

    sim2 = LogReplaySim(SimSettings(worlds=args.worlds), bank, cand, args.device)
    sim2.begin(torch.randint(0, len(bank), (args.worlds,), device=args.device),
               torch.zeros(args.worlds, dtype=torch.bool, device=args.device), auto_reset=True)
    a = torch.zeros(args.worlds, 2, device=args.device)
    for _ in range(5):
        sim2.step(a)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0, n = time.time(), 200
    for _ in range(n):
        sim2.step(a)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    rate = n * args.worlds / (time.time() - t0)
    check("5 throughput", rate > 0, f"{rate:.0f} world steps/s with {args.worlds} worlds on {args.device}")
    print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
