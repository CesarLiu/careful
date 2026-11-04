"""Mirrors scene files left to right (y -> -y), turning right-hand traffic into left-hand traffic:
E4's test that the aggression judge carries to another traffic code by changing only the duty table
(aggression_generalisation_plan.md).

Every position, velocity and map point has its y negated and every heading its sign; a lane's left
and right neighbours and boundaries swap. The drivers, the gaps and the speeds stay the same, so a
mirrored scene judged with RightOfWayParams(traffic_side="left") must get the holders, attributions
and demands of the original under the right-hand table. The difference measures the code's
asymmetries, not how well the table fits a real left-hand country.

A folder's sidecar index (<folder>.index.json, e.g. make_go_first.py's or make_cut_in.py's) is
copied as it is: track ids do not change.

    python -m scripts.responsibility.mirror_scenes --scenes raw_scenes_500 \\
        --out-dir logs/responsibility/aggression/mirror/raw_scenes_500
"""

import argparse
import copy
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.scene import sidecar_index  # noqa: E402

SWAP = (("left_neighbor", "right_neighbor"), ("left_boundaries", "right_boundaries"))


def _flip(points):
    a = np.array(points, copy=True)
    if a.ndim >= 1 and a.shape[-1] >= 2:
        a[..., 1] = -a[..., 1]
    return a


def mirror(sd: dict) -> dict:
    """A ScenarioNet scene description mirrored across the x axis."""
    out = copy.deepcopy(sd)
    for track in out["tracks"].values():
        state = track["state"]
        state["position"] = _flip(state["position"])
        state["velocity"] = _flip(state["velocity"])
        state["heading"] = -np.asarray(state["heading"])
    for feature in out["map_features"].values():
        for key in ("polyline", "polygon", "position"):
            if key in feature:
                feature[key] = _flip(feature[key])
        for left, right in SWAP:
            if left in feature or right in feature:
                feature[left], feature[right] = feature.get(right, []), feature.get(left, [])
    for light in out["dynamic_map_states"].values():
        if "stop_point" in light:
            light["stop_point"] = _flip(light["stop_point"])
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True, help="Folder of ScenarioNet scene .pkl files.")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    src, out = Path(args.scenes), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.pkl"))
    for f in files:
        with open(f, "rb") as fh:
            sd = pickle.load(fh)
        with open(out / f.name, "wb") as fh:
            pickle.dump(mirror(sd), fh)
    if sidecar_index(src).exists():
        shutil.copy(sidecar_index(src), sidecar_index(out))
    print(f"{len(files)} scenes mirrored -> {out}")


if __name__ == "__main__":
    main()
