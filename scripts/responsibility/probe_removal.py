"""Does removing an agent change DenseTNT's prediction of a neighbour b,
and more so the closer it is? Three probes on the logged scenes, b = the
self-driving car at context steps where it moves (>= --min-speed):

  order     b's goal distribution with the other object of interest a in
            the partner slot (second in the agent list, as CAT builds it)
            against a in the rest of the list (as with_and_without does):
            DenseTNT's goal scoring reads the target's hidden state and
            attends over every element, with no slot features, so the KL
            should be ~0
  removal   for a and for up to --n-others other vehicles c within 60 m:
            KL(b with c || b without c), and the change in b's expected
            8 s goal progress along its heading (m; < 0: c holds b back),
            by c's distance to b and whether c leads b in its lane
  obstacle  a stationary vehicle placed on b's own logged path, --ahead
            seconds of b's travel ahead (another far vehicle re-placed, so
            the agent count stays): b's expected progress and the mass of
            its goals short of the obstacle, with it and without it

    python -m scripts.responsibility.probe_removal --n 100 --out logs/responsibility/validity/removal_probe.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.demand import lane_relation  # noqa: E402
from responsibility.metrics import goal_kl  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402


def progress(dist, scene: Scene, b: int, step: int):
    """(expected goal progress along b's heading at step (m), goals' progress [G], probabilities [G])."""
    h = float(scene.heading[b, step])
    d = dist.goals_global - scene.position[b, step, :2]
    along = d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h)
    p = dist.log_prob.exp().double().cpu().numpy()
    return float(np.dot(p, along)), along, p


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--steps", type=int, nargs="+", default=[20, 30, 40, 50])
    p.add_argument("--min-speed", type=float, default=4.0, help="m/s of b at k")
    p.add_argument("--n-others", type=int, default=6)
    p.add_argument("--ahead", type=float, default=1.5, help="s of b's logged travel to the obstacle")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    from responsibility.predictors.densetnt import DenseTNT

    model = DenseTNT(device=a.device)
    rows = []
    for path in scene_files(a.scenes)[: a.n]:
        sc = Scene.load(path)
        b = sc.sdc
        others_ooi = [i for i in sc.objects_of_interest if i != b]
        for k in a.steps:
            if k + int(a.ahead * 10) >= sc.n_steps or not sc.valid[b, k]:
                continue
            speed = float(np.linalg.norm(sc.velocity[b, k]))
            if speed < a.min_speed:
                continue
            base = {"scene": path.stem, "step": k, "speed": speed}
            # order: the other object of interest in the partner slot, or in the rest of the list
            if others_ooi and sc.valid[others_ooi[0], k]:
                ooi = others_ooi[0]
                rest = [i for i in range(sc.n_agents) if i not in (b, ooi)]
                anchor = sc.position[b, k, :2]
                rest.sort(key=lambda i: np.linalg.norm(sc.position[i, k, :2] - anchor) if sc.valid[i, :k + 1].any()
                          else np.inf)
                m1 = model.instance(sc, k, b, order=[b, ooi] + rest)
                default = model.distribution(sc, k, b, avoid_partner=(ooi,))
                if m1 is not None and default is not None:
                    d1 = model.goal_distributions([m1])[0]
                    rows.append({**base, "probe": "order", "kl": goal_kl(d1.log_prob, default.log_prob)})
            # removal of a and of other nearby vehicles
            cands = [i for i in range(sc.n_agents) if i != b and sc.valid[i, k] and sc.types[i] == "VEHICLE"
                     and np.linalg.norm(sc.position[i, k, :2] - sc.position[b, k, :2]) <= 60.0]
            cands.sort(key=lambda i: np.linalg.norm(sc.position[i, k, :2] - sc.position[b, k, :2]))
            picked = cands[: a.n_others] + [i for i in others_ooi if i in cands[a.n_others:]]
            for c in picked:
                pair = model.with_and_without(sc, k, b, c)
                if pair is None:
                    continue
                pw, _, _ = progress(pair[0], sc, b, k)
                po, _, _ = progress(pair[1], sc, b, k)
                rows.append({**base, "probe": "removal", "other": sc.track_ids[c], "is_ooi": c in others_ooi,
                             "dist": float(np.linalg.norm(sc.position[c, k, :2] - sc.position[b, k, :2])),
                             "relation": lane_relation(sc, c, b, k) or "", "kl": goal_kl(pair[0].log_prob,
                                                                                      pair[1].log_prob),
                             "d_prog": pw - po})
            # obstacle on b's own path, re-placing the farthest vehicle (not an object of interest)
            spare = [i for i in range(sc.n_agents) if i != b and i not in sc.objects_of_interest
                     and sc.types[i] == "VEHICLE"]
            ahead = k + int(round(a.ahead * 10))
            if not spare or not sc.valid[b, ahead]:
                continue
            c = max(spare, key=lambda i: np.linalg.norm(sc.position[i, k, :2] - sc.position[b, k, :2])
                    if sc.valid[i, k] else 1e9)
            spot = sc.position[b, ahead].copy()
            T = sc.n_steps
            with_obs = sc.with_track(c, np.tile(spot, (T, 1)), np.full(T, sc.heading[b, ahead]), np.zeros((T, 2)),
                                     valid=np.ones(T, bool))
            without = sc.with_track(c, sc.position[c], sc.heading[c], sc.velocity[c], valid=np.zeros(T, bool))
            dw = model.distribution(with_obs, k, b, avoid_partner=(c,))
            do = model.distribution(without, k, b, avoid_partner=(c,))
            if dw is None or do is None:
                continue
            gap = float(np.dot(spot[:2] - sc.position[b, k, :2],
                               [np.cos(sc.heading[b, k]), np.sin(sc.heading[b, k])]))
            pw, aw, ww = progress(dw, sc, b, k)
            po, ao, wo = progress(do, sc, b, k)
            short = gap - 3.0  # goals ending before the obstacle's rear
            rows.append({**base, "probe": "obstacle", "dist": gap, "kl": goal_kl(dw.log_prob, do.log_prob),
                         "d_prog": pw - po, "prog_without": po, "prog_with": pw,
                         "short_without": float(wo[ao < short].sum()), "short_with": float(ww[aw < short].sum())})
        print(f"{path.name}: {len(rows)} rows", flush=True)
    d = pd.DataFrame(rows)
    if a.out:
        d.to_csv(a.out, index=False)
    o = d[d.probe == "order"]
    print(f"\norder: KL between a in the partner slot and a in the rest, {len(o)} windows: "
          f"median {o.kl.median():.2e}, max {o.kl.max():.2e} nats")
    r = d[d.probe == "removal"].copy()
    r["bin"] = pd.cut(r.dist, [0, 10, 20, 40, 60])
    r["who"] = np.where(r.relation == "leads", "leads b in its lane", np.where(r.is_ooi, "other object of interest",
                                                                             "other vehicle"))
    print("\nremoval: KL (nats) and change in b's expected 8 s progress (m) by the removed vehicle's distance")
    print(r.groupby(["who", "bin"], observed=True).agg(n=("kl", "size"), kl_median=("kl", "median"),
                                                       kl_p90=("kl", lambda x: x.quantile(0.9)),
                                                       d_prog_median=("d_prog", "median")).round(3).to_string())
    print(f"Spearman(distance, KL) = {r[['dist', 'kl']].corr(method='spearman').iloc[0, 1]:.3f}")
    ob = d[d.probe == "obstacle"]
    if len(ob):
        print(f"\nobstacle {a.ahead:.1f} s ahead on b's path ({len(ob)} windows, median {ob.dist.median():.1f} m ahead):"
              f"\n  b's expected 8 s progress without {ob.prog_without.median():.1f} m, with {ob.prog_with.median():.1f} m"
              f" (median change {ob.d_prog.median():+.2f} m)"
              f"\n  goal mass short of the obstacle: without {ob.short_without.median():.3f}, "
              f"with {ob.short_with.median():.3f} (mean {ob.short_without.mean():.3f} -> {ob.short_with.mean():.3f})"
              f"\n  KL median {ob.kl.median():.3f} nats")


if __name__ == "__main__":
    main()
