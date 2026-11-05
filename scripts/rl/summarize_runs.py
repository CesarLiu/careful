"""Summarises train_td3.py runs: per setting (the run directory's name
without its _s<seed> suffix, as compare_policies.py groups CAT's models),
the mean and standard deviation over seeds of the last --last evaluations'
mean (route completion, crash rate, arrival, out of road; normal and
adversarial test episodes).

    python -m scripts.rl.summarize_runs logs/rl/runs/* --out logs/rl/summary.md
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["route_completion_normal", "crash_rate_normal", "arrive_normal", "out_of_road_normal",
           "route_completion_adv", "crash_rate_adv", "arrive_adv", "out_of_road_adv"]


def setting_of(name: str) -> str:
    return re.sub(r"_s\d+$", "", name)


def run_values(run: Path, last: int):
    rows = list(csv.DictReader(open(run / "progress.csv")))
    if not rows:
        return None, 0
    tail = rows[-last:]
    out = {}
    for m in METRICS:
        v = [float(r[m]) for r in tail if r.get(m) not in (None, "", "nan")]
        out[m] = float(np.mean(v)) if v else float("nan")
    return out, int(rows[-1]["timesteps"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--last", type=int, default=4, help="evaluations averaged at the end of each run")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    groups = defaultdict(list)
    for r in map(Path, args.runs):
        if (r / "progress.csv").exists():
            values, steps = run_values(r, args.last)
            if values is not None:
                groups[setting_of(r.name)].append((r.name, values, steps))
    lines = [f"mean +- std over seeds of each run's last {args.last} evaluations", "",
             "| setting | seeds | steps | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 3) + "|"]
    for setting, runs in sorted(groups.items()):
        cells = []
        for m in METRICS:
            v = np.array([x[1][m] for x in runs])
            v = v[np.isfinite(v)]
            cells.append("–" if len(v) == 0 else (f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"))
        lines.append(f"| {setting} | {len(runs)} | {min(x[2] for x in runs)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    sys.exit(main())
