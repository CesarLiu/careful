"""Checks that this code reproduces an existing run window for window: the
settings in config.json are accepted, and the recomputed beta_s / beta_c of
every common (scene, agent, step) equal the stored ones.

It copies OLD's config.json into a fresh directory NEW, so a run there with
the same flags must be accepted by compute_responsibility (it refuses other
settings), then compares the windows of the two runs. Run the
compute_responsibility command in between, e.g. for a DenseTNT run made by
run_h200.sh:

    python -m scripts.responsibility.regress_run prepare --old ~/catspond/logs/responsibility/sdc \\
        --new logs/regress/sdc
    python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 --n 5 --agent sdc \\
        --out-dir logs/regress/sdc --stride 5 --n-samples 40 --device cuda      # the old run's flags
    python -m scripts.responsibility.regress_run compare --old ~/catspond/logs/responsibility/sdc \\
        --new logs/regress/sdc

``compare`` prints the number of common windows and the largest absolute
differences (m, nats), and ends with IDENTICAL (all within --atol, default
1e-4: windows.csv rounds beta_s to 1e-5 m and beta_c to 1e-6 nats) or
DIFFERENT. Windows only in OLD are expected (NEW holds a subset of scenes);
windows only in NEW mean the window steps changed and are reported.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responsibility.results import read_windows  # noqa: E402

KEY = ("scene", "agent_id", "step")


def prepare(old: Path, new: Path) -> None:
    if (new / "config.json").exists():
        raise SystemExit(f"{new} already holds a run")
    new.mkdir(parents=True, exist_ok=True)
    shutil.copy(old / "config.json", new / "config.json")
    settings = json.loads((old / "config.json").read_text())
    print(f"copied {old / 'config.json'} to {new}; the old run's settings:")
    print(json.dumps({k: settings[k] for k in settings if k != "responsibility"}, indent=1))
    print(json.dumps(settings.get("responsibility", {}), indent=1))


def compare(old: Path, new: Path, atol: float) -> bool:
    a = {tuple(str(r[k]) for k in KEY): r for r in read_windows(old)}
    b = {tuple(str(r[k]) for k in KEY): r for r in read_windows(new)}
    common = sorted(set(a) & set(b))
    only_new = sorted(set(b) - set(a))
    scenes = sorted({k[0] for k in b})
    missing = [k for k in a if k[0] in scenes and k not in b]
    print(f"{len(b)} windows in NEW ({len(scenes)} scenes), {len(common)} in common with OLD")
    worst = {}
    for metric in ("safety", "courtesy"):
        diffs = [(abs(float(a[k][metric]) - float(b[k][metric])), k) for k in common]
        worst[metric] = max(diffs, default=(0.0, None))
        print(f"  {metric}: max |new - old| = {worst[metric][0]:.6g} at {worst[metric][1]}")
    if only_new:
        print(f"  {len(only_new)} windows only in NEW (window steps changed?), e.g. {only_new[:3]}")
    if missing:
        print(f"  {len(missing)} windows of these scenes only in OLD, e.g. {missing[:3]}")
    same = bool(common) and not only_new and not missing and all(w[0] <= atol for w in worst.values())
    print("IDENTICAL" if same else "DIFFERENT")
    return same


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["prepare", "compare"])
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args(argv)
    old, new = Path(args.old).expanduser(), Path(args.new).expanduser()
    if args.action == "prepare":
        prepare(old, new)
    else:
        sys.exit(0 if compare(old, new, args.atol) else 1)


if __name__ == "__main__":
    main()
