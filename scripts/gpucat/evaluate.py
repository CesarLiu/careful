"""Evaluates trained GPUDrive policies scene by scene (gpucat/PLAN.md, M3):
each run's model.pt on the 100 test scenes (400-499), deterministic actions,
as train.py's evaluation does -- a normal episode first (which becomes each
scene's evaluation history, as CAT's eval_policy), then one against each
test adversary chosen against it. Unlike train.py's eval.csv, it keeps every
scene's outcome, so runs can be compared scene by scene (paired).

--background reactive evaluates against the reactive background traffic
(gpucat/reactive.py) instead of the log replay; --worlds evaluates the test
scenes in batches of that many worlds (less GPU memory), with the same
results as one batch.

Writes OUT/<run>.csv (scene, test_adversary, outcome, completion, adversary
present, chosen candidate, the first collision partner's slot and class
(gpucat.env.PARTNER_CLASSES), and how many background agents left their
logs and the largest lag behind a log, m) and prints the rates.

    PYTHONPATH=~/gpudrive/build:~/gpudrive:. LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.evaluate --runs replay_nav_s0 cat_nav_s0 --out logs/gpucat/eval
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from gpucat.env import PARTNER_CLASSES, CatEnv, EnvSettings  # noqa: E402
from gpucat.ppo import ActorCritic  # noqa: E402
from scripts.gpucat.train import TEST, rollout  # noqa: E402

OUTCOMES = {0: "running", 1: "arrive", 2: "out_of_road", 3: "crash", 4: "timeout"}


class Policy:
    """train.rollout's view of a PPO object: .net."""

    def __init__(self, net):
        self.net = net


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="Run directories under --root.")
    p.add_argument("--root", default="logs/gpucat/runs")
    p.add_argument("--adversaries", nargs="+", default=["cat", "fair", "fair_valid"])
    p.add_argument("--scenes", default="logs/gpucat/scenes")
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--valid", default="logs/gpucat/valid_samples.npz")
    p.add_argument("--out", default="logs/gpucat/eval")
    p.add_argument("--background", choices=["replay", "reactive"], default="replay")
    p.add_argument("--worlds", type=int, default=len(TEST), help="Test scenes evaluated at once.")
    args = p.parse_args(argv)
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    configs = {r: json.loads((REPO / args.root / r / "config.json").read_text()) for r in args.runs}
    navigation = {c["env"].get("navigation", False) for c in configs.values()}
    if len(navigation) != 1:
        raise ValueError("evaluate runs with and without navigation observations in separate calls")
    W = min(args.worlds, len(TEST))
    env = CatEnv(EnvSettings(worlds=W, navigation=navigation.pop(), background=args.background), REPO / args.scenes,
                 REPO / args.static, REPO / args.bank, valid_path=REPO / args.valid)
    policies, steps, rows = {}, {}, {r: [] for r in args.runs}
    for run in args.runs:
        ck = torch.load(REPO / args.root / run / "model.pt", map_location=env.device)
        net = ActorCritic(ck["obs_dim"]).to(env.device)
        net.load_state_dict(ck["net"])
        policies[run], steps[run] = Policy(net), ck["steps"]
    tests = [("none", False, "cat")] + [(rule, True, rule) for rule in args.adversaries]
    for b in range(0, len(TEST), W):
        batch = TEST[b:b + W]
        env.load((batch * W)[:W])  # the last batch padded with repeats, which are dropped
        for run in args.runs:
            a = configs[run]["args"]
            opts = argparse.Namespace(tau=a["tau"], rho=a["rho"])
            for name, adversarial, rule in tests:
                _, _, res = rollout(env, policies[run], torch.full((W,), adversarial, device=env.device), rule, opts,
                                    evaluation=True, deterministic=True)
                for i, stem in enumerate(batch):
                    rows[run].append({"scene": stem, "test_adversary": name,
                                      "outcome": OUTCOMES[int(res["outcome"][i])],
                                      "completion": round(float(res["completion"][i]), 4),
                                      "adversarial": int(res["adversarial"][i]), "chosen": int(res["chosen"][i]),
                                      "partner": int(res["partner"][i]),
                                      "partner_class": PARTNER_CLASSES[int(res["partner_class"][i])],
                                      "agents_off": int(res["agents_off"][i]),
                                      "max_lag": round(float(res["max_lag"][i]), 2)})
    for run in args.runs:
        with open(out / f"{run}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[run][0]))
            w.writeheader()
            w.writerows(rows[run])
        print(f"{run} ({steps[run]} steps, {args.background} background)")
        for name, *_ in tests:
            r = [x for x in rows[run] if x["test_adversary"] == name]
            rate = lambda o: sum(x["outcome"] == o for x in r) / len(r)  # noqa: E731
            crashes = Counter(x["partner_class"] for x in r if x["outcome"] == "crash")
            print(f"  {name:10s} arrive {rate('arrive'):.2f} crash {rate('crash'):.2f} out_of_road "
                  f"{rate('out_of_road'):.2f} completion {sum(x['completion'] for x in r) / len(r):.3f}  "
                  f"crash partners {dict(crashes)}  agents off their logs {sum(x['agents_off'] for x in r)}")


if __name__ == "__main__":
    main()
