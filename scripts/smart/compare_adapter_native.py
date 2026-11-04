"""Compares SMART's two paths to responsibility on the same scenes, windows
and neighbours: CAT-K's native batched pipeline
(responsibility/predictors/smart/native, as catk4test ran it) and the
predictor-agnostic metrics over SmartPredictor (--model smart). It answers
two questions:

1. Did the merge change SMART's numbers? Both paths draw from the same
   model with the same sampling scheme, so they can only agree in
   distribution. The yardstick is Monte-Carlo noise: each path is run
   --repeats times with different seeds, and the native-vs-adapter
   differences are compared with the seed-to-seed differences of each path.
2. Where does the SDC's high SMART courtesy come from
   (responsibility/SMART_PLAN.md: median beta_c 0.54 nats for the SDC,
   0.06 for the adversary)? The per-role table reports both paths' medians,
   and the native beta_c both as computed (a max over every neighbour,
   pedestrians and cyclists included) and over vehicle neighbours only,
   which is what the predictor-agnostic metrics measure.

What is held equal. Neighbours: for every window, the native pipeline's
own selection (CAT-K's interaction.py) is handed to responsibility_at, so
the two differ only in how each (agent, neighbour) value is computed.
Settings: the native ResponsibilityConfig's (40 safety and 20 courtesy
samples, CVaR alpha 0.1, D_g saturated at 20 m, 2 s horizon, Boltzmann
courtesy with no skip distance) on both sides; rollouts of 2 s (the native
rollout_horizon); --neighbor-future on both. Windows: the native window
steps (--stride).

What remains different, by design (and listed in the summary):
  - context: the native cache keeps the agents observed at 1.0 s and the
    lights of 1.0 s for every window; the adapter rebuilds the batch at
    each step k (agents observed at k, lights of k). --adapter-context
    native makes the adapter use the 1.0 s batch too, isolating the rest.
  - Boltzmann scoring: native scores b's plans against every agent and
    SMART's own vehicle-lane points, with SMART's rollout headings; the
    adapter against agents within 50 m of b and the scene's lane graph,
    headings from the plan's steps.
  - courtesy targets: native measures toward every neighbour, the adapter
    toward vehicles only (pairs with a non-vehicle neighbour are compared
    on safety only).

Writes OUT/pairs.csv (one row per window and neighbour), OUT/windows.csv
(one row per window, beta = max over the same neighbours), OUT/summary.md
and OUT/config.json; OUT/scenes/<scene>.pkl makes a run resumable.

Runs in CAT-K's environment, from this repository's root:

    CATK_ROOT=~/catk python -m scripts.smart.compare_adapter_native --ckpt ~/catk/ckpts/clsft_E9.ckpt \\
        --scenes raw_scenes_500 --n 20 --agent sdc adv --out-dir logs/smart/compare_adapter_native
    CATK_ROOT=~/catk python -m scripts.smart.compare_adapter_native --ckpt random --scenes <dir> --n 2 \\
        --model-config smart_nano_1M --out-dir /tmp/compare   # plumbing only

The summary ends with CONSISTENT or DIFFERENT per metric: CONSISTENT when
the median absolute native-vs-adapter difference is at most --tol times the
larger of the two paths' seed-to-seed ones, and the native-vs-adapter
Spearman correlation is at most 0.1 below the smaller seed-to-seed one
(needs --repeats >= 2). Scene folders whose maps SMART's preprocessing
drops (polylines spaced over 3 m) are warned about; their courtesy is not
comparable.
"""

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

ROLE_HELP = "sdc, adv (the other object of interest) or a track id"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True, help="CAT scene files (<n>.pkl).")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--agent", nargs="+", default=["sdc", "adv"], help=ROLE_HELP)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt", required=True, help="SMART checkpoint, or 'random' (plumbing check only).")
    p.add_argument("--model-config", default="smart")
    p.add_argument("--catk-root", default=os.environ.get("CATK_ROOT"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=32, help="Scene copies per model call.")
    p.add_argument("--stride", type=int, default=5, help="Context steps between windows (0.1 s each).")
    p.add_argument("--neighbor-future", default="hidden", choices=["hidden", "logged", "simulated"])
    p.add_argument("--adapter-context", default="step", choices=["step", "native"],
                   help="step: the adapter as deployed (batch rebuilt at each k); native: the 1.0 s batch.")
    p.add_argument("--repeats", type=int, default=2, help="Seeds per path (>= 2 for the noise yardstick).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1.5)
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# statistics (no model needed; tested in tests/smart/test_compare_adapter_native.py)
# ----------------------------------------------------------------------

def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):  # average ranks of ties
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return math.nan
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def pair_stats(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """n, Spearman, median |y - x| (robust to the heavy tails courtesy has),
    mean |y - x| and median (y - x), over the rows where both are finite."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return {"n": 0, "spearman": math.nan, "medad": math.nan, "mad": math.nan, "bias": math.nan}
    d = y - x
    return {"n": int(len(x)), "spearman": spearman(x, y), "medad": float(np.median(np.abs(d))),
            "mad": float(np.abs(d).mean()), "bias": float(np.median(d))}


def comparisons(rows: List[Dict], metric: str, repeats: int) -> Dict[str, Dict[str, float]]:
    """native vs native and adapter vs adapter (seed noise; pairs of repeats)
    and native vs adapter (same repeat index, pooled over repeats)."""
    def col(path, r):
        return [_float(row.get(f"{path}_{metric}_{r}")) for row in rows]

    out = {}
    if repeats >= 2:
        for path in ("native", "adapter"):
            xs, ys = [], []
            for r in range(repeats - 1):
                xs += col(path, r)
                ys += col(path, r + 1)
            out[f"{path} vs {path} (seed noise)"] = pair_stats(xs, ys)
    xs, ys = [], []
    for r in range(repeats):
        xs += col("native", r)
        ys += col("adapter", r)
    out["native vs adapter"] = pair_stats(xs, ys)
    return out


RANK_SLACK = 0.1  # how much lower than the paths' own seed-to-seed Spearman the cross one may be


def verdict(stats: Dict[str, Dict[str, float]], tol: float) -> Optional[bool]:
    """True when native and adapter differ no more than each path differs
    from itself under another seed: the median |difference| at most ``tol``
    times the larger seed-noise one, and the Spearman correlation at most
    RANK_SLACK below the smaller seed-noise one. None without a yardstick."""
    noise = [v for k, v in stats.items() if "noise" in k and v["n"] > 0]
    cross = stats["native vs adapter"]
    if not noise or cross["n"] == 0:
        return None
    close = cross["medad"] <= tol * max(max(v["medad"] for v in noise), 1e-9)
    ranks = [v["spearman"] for v in noise if np.isfinite(v["spearman"])]
    ordered = (not ranks or not np.isfinite(cross["spearman"])
               or cross["spearman"] >= min(ranks) - RANK_SLACK)
    return bool(close and ordered)


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _median(values) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    return float(np.median(v)) if len(v) else math.nan


def summarize(pairs: List[Dict], windows: List[Dict], repeats: int, tol: float, settings: Dict) -> str:
    lines = ["# SMART: native pipeline vs SmartPredictor", "",
             f"{len({r['scene'] for r in windows})} scenes, {len(windows)} windows, {len(pairs)} (window, neighbour) "
             f"pairs, {repeats} seed(s) per path, neighbour future `{settings['neighbor_future']}`, adapter context "
             f"`{settings['adapter_context']}`.", ""]
    verdicts = {}
    for level, rows in (("pair", pairs), ("window", windows)):
        for metric, unit in (("safety", "m"), ("courtesy", "nats")):
            use = rows
            if level == "pair" and metric == "courtesy":  # the adapter measures courtesy toward vehicles only
                use = [r for r in rows if r["neighbour_type"] == "VEHICLE"]
            stats = comparisons(use, metric, repeats)
            lines += [f"## {metric} ({unit}), per {level}", "", "| comparison | n | Spearman | median abs diff | "
                      "mean abs diff | median diff (2nd - 1st) |", "|---|---|---|---|---|---|"]
            for name, s in stats.items():
                lines.append(f"| {name} | {s['n']} | {s['spearman']:.3f} | {s['medad']:.4f} | {s['mad']:.4f} | "
                             f"{s['bias']:+.4f} |")
            if level == "pair":
                v = verdict(stats, tol)
                verdicts[metric] = v
                lines += ["", "verdict: " + ("no seed-noise yardstick (--repeats 1)" if v is None else
                                             ("CONSISTENT" if v else "DIFFERENT")
                                             + f" (median abs diff within {tol} x seed noise, Spearman within "
                                               f"{RANK_SLACK} of it)")]
            lines.append("")
    lines += ["## distributions (per pair, seed 0)", "", "| metric | path | median | q90 | max |", "|---|---|---|---|---|"]
    for metric in ("safety", "courtesy"):
        for path in ("native", "adapter"):
            v = np.asarray([_float(r.get(f"{path}_{metric}_0")) for r in pairs], float)
            v = v[np.isfinite(v)]
            if len(v):
                lines.append(f"| {metric} | {path} | {np.median(v):.4f} | {np.quantile(v, 0.9):.4f} | {v.max():.4f} |")
    lines += ["", "Courtesy of a pooled Boltzmann pair is at most log(2n) = 3.69 nats when P2 is nearly uniform; "
              "values far above that mean the road-deviation or collision cost separates b's plans by tens of "
              "nats (e.g. a map that SMART's preprocessing dropped).", ""]
    lines += ["## per role, window medians (seed 0)", "",
              "| role | windows | native beta_s | adapter beta_s | native beta_c (all neighbours) | "
              "native beta_c (vehicles) | adapter beta_c |", "|---|---|---|---|---|---|---|"]
    for role in sorted({r["role"] for r in windows}):
        w = [r for r in windows if r["role"] == role]
        med = {k: _median(_float(r[k]) for r in w) for k in
               ("native_safety_0", "adapter_safety_0", "native_courtesy_0", "native_courtesy_vehicles_0",
                "adapter_courtesy_0")}
        lines.append(f"| {role} | {len(w)} | {med['native_safety_0']:+.3f} | {med['adapter_safety_0']:+.3f} | "
                     f"{med['native_courtesy_0']:.4f} | {med['native_courtesy_vehicles_0']:.4f} | "
                     f"{med['adapter_courtesy_0']:.4f} |")
    non_vehicle = [r for r in pairs if r["neighbour_type"] != "VEHICLE"]
    lines += ["", f"{len(non_vehicle)} of {len(pairs)} pairs have a non-vehicle neighbour: native measures courtesy "
              "toward them, the adapter does not. If the native beta_c (all neighbours) is much higher than its "
              "vehicles-only value for one role, those pairs explain that role's high courtesy.", "",
              "Differences by design: the context (" + settings["adapter_context"] + "), the agents and map "
              "points in the Boltzmann score, the plan headings (module docstring).", "",
              "RESULT: " + ", ".join(f"{m} {'n/a' if v is None else ('CONSISTENT' if v else 'DIFFERENT')}"
                                     for m, v in verdicts.items())]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# the two paths
# ----------------------------------------------------------------------

def pick_agent(scene, which: str) -> int:
    if which == "sdc":
        return scene.sdc
    if which == "adv":
        others = [i for i in scene.objects_of_interest if i != scene.sdc]
        if not others:
            raise ValueError("no second object of interest")
        return others[0]
    return scene.index(which)


PAIR_FIELDS = ["scene", "role", "agent_id", "k", "neighbour_id", "neighbour_type"]
WINDOW_FIELDS = ["scene", "role", "agent_id", "k", "n_neighbours"]


def value_fields(repeats: int, window: bool) -> List[str]:
    out = []
    for r in range(repeats):
        out += [f"native_safety_{r}", f"native_courtesy_{r}", f"adapter_safety_{r}", f"adapter_courtesy_{r}"]
        if window:
            out.append(f"native_courtesy_vehicles_{r}")
    return out


def scene_rows(stem: str, scene, data, roles, args, native, adapter_predictor, adapter_cfgs, catk) -> Dict:
    """pairs and windows of one scene, both paths, every repeat."""
    import torch

    from responsibility.catk_export import numeric_id
    from responsibility.metrics import responsibility_at

    builder, shifted_context, ncfg, synthesizer = catk["builder"], catk["shifted_context"], catk["cfg"], native
    ids = [int(i) for i in data["agent"]["id"].tolist()]
    index_of = {numeric_id(t): i for i, t in enumerate(scene.track_ids)}
    steps = builder.window_steps(data, ncfg, synthesizer, args.stride)
    scene_number = int(stem) if stem.isdigit() else 0
    pairs, windows = {}, {}
    for role_idx, role in enumerate(roles):
        try:
            a = pick_agent(scene, role)
        except ValueError as e:
            print(f"  {stem}/{role}: skipped ({e})", flush=True)
            continue
        a_id = numeric_id(scene.track_ids[a])
        if a_id not in ids:
            print(f"  {stem}/{role}: not in the native batch (not observed at 1.0 s)", flush=True)
            continue
        a_row = ids.index(a_id)
        for r in range(args.repeats):
            seed = (args.seed + 1000 * r) * 100003 + scene_number * 131 + role_idx
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            cfg = adapter_cfgs[r]
            for k in steps:
                with shifted_context(data, k, ncfg.num_historical_steps):
                    found = builder._responsibility_at_current_step(synthesizer, data, ncfg, stem, k, None, None,
                                                                   agents=[a_row])
                if not found or not found[0].per_neighbour:
                    continue
                obs_n = found[0]
                neighbours = {index_of[nid]: (obs_n.interaction or {}).get(nid, {})
                              for nid in obs_n.per_neighbour if nid in index_of}
                obs_a = responsibility_at(adapter_predictor, scene, a, k, cfg, generator, neighbours=neighbours)
                wkey = (role, k)
                w = windows.setdefault(wkey, {"scene": stem, "role": role, "agent_id": scene.track_ids[a], "k": k,
                                              "n_neighbours": len(neighbours)})
                veh = [c for nid, (_, c) in obs_n.per_neighbour.items()
                       if nid in index_of and scene.types[index_of[nid]] == "VEHICLE"]
                w[f"native_safety_{r}"] = float(obs_n.beta[0])
                w[f"native_courtesy_{r}"] = float(obs_n.beta[1])
                w[f"native_courtesy_vehicles_{r}"] = max(veh, default=0.0)
                w[f"adapter_safety_{r}"] = math.nan if obs_a is None else obs_a.safety
                w[f"adapter_courtesy_{r}"] = math.nan if obs_a is None else obs_a.courtesy
                for nid, (s, c) in obs_n.per_neighbour.items():
                    if nid not in index_of:
                        continue
                    b = index_of[nid]
                    tid = scene.track_ids[b]
                    p = pairs.setdefault((role, k, tid), {"scene": stem, "role": role,
                                                          "agent_id": scene.track_ids[a], "k": k,
                                                          "neighbour_id": tid, "neighbour_type": scene.types[b]})
                    entry = None if obs_a is None else obs_a.per_neighbour.get(tid)
                    p[f"native_safety_{r}"] = float(s)
                    p[f"native_courtesy_{r}"] = float(c)
                    p[f"adapter_safety_{r}"] = math.nan if entry is None else entry["safety"]
                    p[f"adapter_courtesy_{r}"] = (math.nan if entry is None or entry.get("courtesy") is None
                                                  else entry["courtesy"])
    return {"pairs": list(pairs.values()), "windows": list(windows.values())}


def main(argv=None):
    args = parse_args(argv)
    from responsibility.predictors.smart.catk import require_catk
    from responsibility.scene import Scene, scene_files

    root = require_catk(args.catk_root)
    scenes_dir = Path(args.scenes).resolve()
    out = Path(args.out_dir).resolve()
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    files = scene_files(scenes_dir)[args.first:args.first + args.n]
    todo = [f for f in files if not (out / "scenes" / f"{f.stem}.pkl").exists()]
    print(f"{len(files)} scenes, {len(files) - len(todo)} done, {len(todo)} to go", flush=True)

    settings = {"scenes": str(scenes_dir), "agent": args.agent, "ckpt": args.ckpt, "model_config": args.model_config,
                "stride": args.stride, "neighbor_future": args.neighbor_future,
                "adapter_context": args.adapter_context, "repeats": args.repeats, "seed": args.seed}
    config_path = out / "config.json"
    if todo:
        if config_path.exists() and json.loads(config_path.read_text())["settings"] != settings:
            raise SystemExit(f"{out} holds a comparison with other settings; use a new --out-dir")

        import responsibility.predictors.smart.native.dataset_builder as builder
        import responsibility.predictors.smart.native.metrics as nm
        from responsibility.metrics import ResponsibilityConfig
        from responsibility.predictors.smart import load_smart_predictor
        from responsibility.predictors.smart.native.interaction import InteractionConfig
        from responsibility.predictors.smart.native.motion_set import MotionSetSynthesizer, shifted_context
        from responsibility.predictors.smart.native.pipeline import (default_sampling_scheme, iter_scenarios,
                                                                       resolve_device)
        from responsibility.predictors.smart.native.results_io import scenario_files
        from scripts.responsibility import export_catk

        device = resolve_device(args.device)
        export_catk.main(["--catk-root", str(root), "--scenes", str(scenes_dir), "--out-dir", str(out),
                          "--first", str(args.first), "--n", str(args.n)])
        predictor = load_smart_predictor(args.ckpt, root, args.model_config, device,
                                         neighbor_future=args.neighbor_future, batch_size=args.batch)
        model = predictor.synthesizer.model
        ncfg = nm.ResponsibilityConfig(num_historical_steps=predictor.history, window_stride=args.stride,
                                       neighbor_future=args.neighbor_future, courtesy_estimator="boltzmann",
                                       interaction=InteractionConfig())
        predictor.horizon = ncfg.metric_horizon  # the native rollout_horizon: 2 s
        native = MotionSetSynthesizer(model, default_sampling_scheme(), rollout_batch_size=args.batch)
        adapter_cfgs = [ResponsibilityConfig(
            n_safety_samples=ncfg.n_safety_samples, cvar_alpha=ncfg.cvar_alpha, d_sat=ncfg.d_sat,
            metric_horizon=ncfg.metric_horizon, window_stride=args.stride, courtesy=True,
            courtesy_estimator="boltzmann", n_courtesy_samples=ncfg.n_courtesy_samples,
            courtesy_skip_distance=ncfg.courtesy_skip_distance, seed=args.seed + 1000 * r)
            for r in range(args.repeats)]
        config_path.write_text(json.dumps({"settings": settings, "native": asdict(ncfg),
                                           "adapter": asdict(adapter_cfgs[0])}, indent=2, default=str))
        catk = {"builder": builder, "shifted_context": shifted_context, "cfg": ncfg}

        cache = [f.stem for f in scenario_files(str(out / "cache"))]
        wanted = {f.stem: f for f in todo}
        indices = [i for i, stem in enumerate(cache) if stem in wanted]
        for n, (index, _, data) in enumerate(iter_scenarios(str(out / "cache"), device, indices=indices), 1):
            stem = cache[index]
            scene = Scene.load(wanted[stem])
            if args.adapter_context == "native":
                predictor.build_batch = lambda s, current, _data=data: _data  # the 1.0 s batch for every k
                predictor._cache.clear()
            rows = scene_rows(stem, scene, data, args.agent, args, native, predictor, adapter_cfgs, catk)
            tmp = out / "scenes" / f"{stem}.pkl.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump(rows, f)
            os.replace(tmp, out / "scenes" / f"{stem}.pkl")
            print(f"[{n}/{len(indices)}] {stem}: {len(rows['windows'])} windows, {len(rows['pairs'])} pairs",
                  flush=True)

    pairs, windows = [], []
    for f in files:
        path = out / "scenes" / f"{f.stem}.pkl"
        if path.exists():
            with open(path, "rb") as h:
                rows = pickle.load(h)
            pairs += rows["pairs"]
            windows += rows["windows"]
    for name, rows, fields in (("pairs.csv", pairs, PAIR_FIELDS + value_fields(args.repeats, False)),
                               ("windows.csv", windows, WINDOW_FIELDS + value_fields(args.repeats, True))):
        with open(out / name, "w", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    text = summarize(pairs, windows, args.repeats, args.tol, settings)
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
