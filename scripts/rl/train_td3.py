"""Trains the SDC policy with TD3 in CAT's setting on the log-replay
simulator (rl/sim.py), without MetaDrive: cat_RLtrain.py's loop with the
environment stepping --worlds worlds at once.

  mode replay   no adversary
  mode cat      CAT's adversary (--adv-rule cat) or the fair one (--adv-rule
                fair, --resp-threshold tau, --resp-avoid rho) in a share of
                the episodes that grows as in cat_RLtrain.py: an episode is
                adversarial with probability
                1 - max(1 - 2 t / T (1 - min_prob), min_prob)

As in cat_RLtrain.py: scenes 0..n_train-1 train (one drawn at random per
episode), the rest test; --start-timesteps random actions first, then the
policy with Gaussian noise (std expl_noise); one TD3 update per transition
(--utd); a transition is done when its episode ended, at the time limit too
(cat_RLtrain stores done = float(done)); collisions do not end a training
episode (crash_done off) and do end an evaluation one. Every --eval-freq
transitions, the policy (deterministic) drives every test scene twice: a
normal episode, whose trajectory becomes the scene's evaluation history,
then one against the adversary chosen against it (CAT's eval_policy).

Writes OUT/progress.csv (per evaluation: route completion and crash rate,
normal and adversarial, as CAT's logger, plus arrival and out-of-road
rates), OUT/train.csv (the training episodes between evaluations),
OUT/config.json, with --save-model OUT/model.pt, and with
--blame-weighting OUT/blame.csv (every attributed collision; rl/blame.py:
share weights the penalty by the counterfactual blame, with any predictor
of responsibility/predictors, rss by RSS).

    python -m scripts.rl.train_td3 --bank logs/rl/scenes.npz --candidates logs/rl/candidates.npz \\
        --mode cat --out logs/rl/runs/cat_s0 --seed 0 --device cuda --save-model
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.adversary import CandidateBank  # noqa: E402
from rl.scenes import SceneBank  # noqa: E402
from rl.sim import ARRIVED, OUT_OF_ROAD, LogReplaySim, SimSettings, obs_dim, run_lockstep  # noqa: E402
from rl.td3 import TD3, ReplayBuffer, TD3Settings  # noqa: E402

EVAL_FIELDS = ["timesteps", "route_completion_normal", "crash_rate_normal", "arrive_normal", "out_of_road_normal",
               "route_completion_adv", "crash_rate_adv", "arrive_adv", "out_of_road_adv", "minutes"]
TRAIN_FIELDS = ["timesteps", "episodes", "adversarial", "route_completion", "crash_rate", "arrive", "out_of_road",
                "p_adv"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", required=True, help="scripts/rl/build_scenes.py's .npz")
    p.add_argument("--candidates", default=None, help="scripts/rl/precompute_candidates.py's .npz (mode cat)")
    p.add_argument("--mode", choices=["replay", "cat"], default="replay")
    p.add_argument("--adv-rule", choices=["cat", "fair"], default="cat")
    p.add_argument("--resp-threshold", type=float, default=2.0, help="fair: tau, m")
    p.add_argument("--resp-avoid", type=float, default=0.1, help="fair: rho")
    p.add_argument("--min-prob", type=float, default=0.1)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--max-timesteps", type=float, default=1e6)
    p.add_argument("--start-timesteps", type=int, default=10000)
    p.add_argument("--eval-freq", type=int, default=25000)
    p.add_argument("--expl-noise", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--utd", type=float, default=1.0, help="TD3 updates per transition (cat_RLtrain: 1).")
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--buffer", type=int, default=int(1e6))
    g = p.add_argument_group("blame-weighted collision penalty (cat_RLtrain.py --blame_weighting)")
    g.add_argument("--blame-weighting", choices=["none", "share", "rss"], default="none")
    g.add_argument("--scenes", default="raw_scenes_500", help="the scene files the bank was built from (attribution)")
    g.add_argument("--blame-model", default="densetnt", help="share: the predictor (responsibility/predictors).")
    g.add_argument("--blame-checkpoint", default=None)
    g.add_argument("--blame-unitraj-method", default=None)
    g.add_argument("--blame-catk-root", default=None)
    g.add_argument("--blame-margin", type=float, default=None, help="m; default responsibility.blame.MARGIN")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--save-model", action="store_true")
    return p.parse_args(argv)


def p_adversarial(t: int, args) -> float:
    if args.mode != "cat":
        return 0.0
    return 1.0 - max(1.0 - (2.0 * t / args.max_timesteps) * (1.0 - args.min_prob), args.min_prob)


def summary(rep, prefix: str = "") -> dict:
    def mean(x):
        return float(x.float().mean()) if len(x) else float("nan")
    return {f"route_completion{prefix}": mean(rep["completion"]), f"crash_rate{prefix}": mean(rep["crashed"]),
            f"arrive{prefix}": mean(rep["outcome"] == ARRIVED), f"out_of_road{prefix}": mean(rep["outcome"] == OUT_OF_ROAD)}


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device

    bank = SceneBank(args.bank, dev)
    cand = CandidateBank(args.candidates, dev) if args.candidates else None
    if args.mode == "cat" and cand is None:
        raise SystemExit("--mode cat needs --candidates (scripts/rl/precompute_candidates.py)")
    S = len(bank)
    n_train = min(args.n_train, S)
    train_rows = torch.arange(n_train, device=dev)
    test_rows = torch.arange(n_train, S, device=dev)
    rule = args.adv_rule

    train_set = SimSettings(worlds=args.worlds, crash_done=False)
    sim = LogReplaySim(train_set, bank, cand, dev)
    eval_sim = LogReplaySim(SimSettings(worlds=max(1, len(test_rows)), crash_done=True), bank, cand, dev)
    weighting = None
    if args.blame_weighting != "none":
        from types import SimpleNamespace

        from responsibility.blame import MARGIN
        from rl.blame import SimBlameWeighting

        model = None
        if args.blame_weighting == "share":
            from responsibility.predictors import load_predictor

            model = load_predictor(args.blame_model, SimpleNamespace(
                model=args.blame_model, checkpoint=args.blame_checkpoint, unitraj_method=args.blame_unitraj_method,
                catk_root=args.blame_catk_root), dev)
        weighting = SimBlameWeighting(model, args.scenes, train_set.crash_penalty, args.blame_weighting,
                                      margin=MARGIN if args.blame_margin is None else args.blame_margin,
                                      log_path=out / "blame.csv")
        sim.before_reset = weighting
    state_dim, action_dim = obs_dim(train_set), 2
    policy = TD3(state_dim, action_dim, TD3Settings(), dev)
    buffer = ReplayBuffer(state_dim, action_dim, args.buffer, dev)
    (out / "config.json").write_text(json.dumps({"args": vars(args), "sim": asdict(train_set),
                                                 "td3": asdict(policy.s), "obs_dim": state_dim,
                                                 "n_train": n_train, "n_test": len(test_rows)}, indent=1))

    def pick(n):
        return train_rows[torch.randint(0, n_train, (n,), device=dev)]

    def evaluate(t):
        act = lambda o: policy.select_action(o)  # noqa: E731
        row = {"timesteps": t}
        if len(test_rows):
            normal = run_lockstep(eval_sim, act, test_rows, False, rule=rule, tau=args.resp_threshold,
                                  rho=args.resp_avoid, evaluation=True)
            row.update(summary(normal, "_normal"))
            if cand is not None:
                adv = run_lockstep(eval_sim, act, test_rows, True, rule=rule, tau=args.resp_threshold,
                                   rho=args.resp_avoid, evaluation=True)
                row.update(summary(adv, "_adv"))
        row["minutes"] = round((time.time() - t0) / 60, 2)
        with open(out / "progress.csv", "a", newline="") as h:
            w = csv.DictWriter(h, fieldnames=EVAL_FIELDS, extrasaction="ignore")
            if h.tell() == 0:
                w.writeheader()
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        print(f"[eval {t}] " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if isinstance(v, float)), flush=True)
        if args.save_model:
            policy.save(out / "model.pt")

    t0 = time.time()
    obs = sim.begin(pick(args.worlds), torch.zeros(args.worlds, dtype=torch.bool, device=dev), rule=rule,
                    tau=args.resp_threshold, rho=args.resp_avoid, auto_reset=True, p_adv=p_adversarial(0, args),
                    pick=pick)
    T, t, credit, next_eval = int(args.max_timesteps), 0, 0.0, args.eval_freq
    episodes = []
    while t < T:
        sim.p_adv = p_adversarial(t, args)
        if t < args.start_timesteps:
            action = torch.rand(args.worlds, action_dim, device=dev) * 2.0 - 1.0
        else:
            action = (policy.select_action(obs) + torch.randn(args.worlds, action_dim, device=dev) * args.expl_noise
                      ).clamp(-1.0, 1.0)
        if weighting is not None:
            weighting.total_steps = t
        next_obs, reward, ended, _ = sim.step(action)
        buffer.add(obs, action, next_obs, reward, ended)
        obs = next_obs
        t += args.worlds
        if t >= args.start_timesteps:
            credit += args.worlds * args.utd
            while credit >= 1.0:
                policy.train(buffer, args.batch_size)
                credit -= 1.0
        fin = sim.take_finished()
        if fin:
            episodes.append(fin)
        if t >= next_eval or t >= T:
            if episodes:
                rep = {k: torch.cat([e[k] for e in episodes]) for k in episodes[0]}
                row = {"timesteps": t, "episodes": len(rep["outcome"]),
                       "adversarial": float(rep["adversarial"].float().mean()), "p_adv": round(sim.p_adv, 4),
                       **summary(rep)}
                with open(out / "train.csv", "a", newline="") as h:
                    w = csv.DictWriter(h, fieldnames=TRAIN_FIELDS, extrasaction="ignore")
                    if h.tell() == 0:
                        w.writeheader()
                    w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
                print(f"[train {t}] {row['episodes']} episodes, adversarial {row['adversarial']:.2f}, completion "
                      f"{row['route_completion']:.3f}, crash {row['crash_rate']:.3f}, arrive {row['arrive']:.3f}, "
                      f"out of road {row['out_of_road']:.3f}", flush=True)
                episodes = []
            evaluate(t)
            if weighting is not None:
                print(weighting.summary(), flush=True)
            next_eval += args.eval_freq
    print(f"done: {t} transitions in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
