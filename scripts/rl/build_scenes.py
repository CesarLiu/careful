"""Writes rl/scenes.py's per-scene tensors for every scene of a folder to one
.npz (logged tracks in slot order, routes, road-edge and yellow-line
segments), which rl.scenes.SceneBank loads.

    python -m scripts.rl.build_scenes --scenes raw_scenes_500 --out logs/rl/scenes.npz
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from rl.scenes import build_bank  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--max-agents", type=int, default=128)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    t0 = time.time()
    bank = build_bank(args.scenes, args.first, args.n, args.max_agents)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **bank)
    print(f"{len(bank['stems'])} scenes -> {args.out} in {time.time() - t0:.0f} s: agents <= {bank['position'].shape[1]} "
          f"(median {int(np.median(bank['n_agents']))}), road-edge segments median {int(np.median(bank['edges_n']))}, "
          f"yellow segments median {int(np.median(bank['yellow_n']))}, scenes without road edges "
          f"{int((bank['edges_n'] == 0).sum())}, without an adversary {int((~bank['has_adversary']).sum())}; "
          f"route length median {np.median(bank['route_len']):.0f} m")


if __name__ == "__main__":
    main()
