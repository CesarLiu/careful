"""CAT's scenes as the tensors the log-replay simulator (rl/sim.py) steps:
the logged tracks of every agent, in a fixed slot order, and the static
data the episode checks need. Built from the scene files directly
(``responsibility.scene.Scene``), with no simulator in between.

  slots     the SDC first, CAT's adversary (the other object of interest)
            second, every other object by its distance to the SDC at step 10,
            at most ``max_agents``. Vehicles MetaDrive deems static (valid
            positions spread by at most 3 m) are left out, as CAT runs
            MetaDrive with no_static_vehicles (never the SDC or the adversary)
  tracks    position [A, 91, 2], heading [A, 91], velocity [A, 91, 2],
            validity [A, 91], footprint [A, 2] (length, width at the step
            nearest to 10), type (1 vehicle, 2 cyclist, 3 pedestrian, 0 other
            or empty)
  route     the SDC's logged path extended EXTENSION m along its last
            heading: MetaDrive's navigation reference, against which
            progress (the driving reward), route completion and the 10 m
            out-of-route test are measured; route_len is the logged part's
            length (completion's denominator)
  edges     road-edge segments (boundary and median) within NEAR_ROUTE m of
            the route: MetaDrive turns every road edge into a sidewalk, and
            touching one is out of road
  yellow    segments (also within NEAR_ROUTE m) of the yellow lines a
            vehicle must not cross (solid single, solid double, passing
            double): MetaDrive builds them as continuous yellow lines

The slot order, static-vehicle test and route/edge/yellow construction are
those of the GPUDrive port (gpucat/scenes.py and gpucat/static.py on main),
without GPUDrive's cap of 64 agents.

``build_bank`` turns a folder of scenes into one dict of padded arrays
(saved by scripts/rl/build_scenes.py as an .npz); ``SceneBank`` holds it on
a device, rows by scene stem.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from responsibility.scene import Scene, scene_files

STEPS = 91  # 9.1 s at 10 Hz
EXTENSION = 100.0  # m past the log's last point
NEAR_ROUTE = 20.0  # m: segments farther from the route are never reached (out of route beyond 10 m)
STATIC_THRESHOLD = 3.0  # m: MetaDrive's ScenarioTrafficManager.STATIC_THRESHOLD
EDGE_TYPES = ("ROAD_EDGE_BOUNDARY", "ROAD_EDGE_MEDIAN")
YELLOW_TYPES = ("ROAD_LINE_SOLID_SINGLE_YELLOW", "ROAD_LINE_SOLID_DOUBLE_YELLOW", "ROAD_LINE_PASSING_DOUBLE_YELLOW")
TYPE_CODES = {"VEHICLE": 1, "CYCLIST": 2, "PEDESTRIAN": 3}
VEHICLE = 1


def cat_adversary(scene: Scene) -> Optional[int]:
    others = [i for i in scene.objects_of_interest if i != scene.sdc]
    return others[0] if others else None


def is_static_vehicle(scene: Scene, i: int) -> bool:
    pts = scene.position[i, scene.valid[i], :2]
    return scene.types[i] == "VEHICLE" and len(pts) > 0 and float(np.std(pts, axis=0).max()) <= STATIC_THRESHOLD


def agent_order(scene: Scene, max_agents: Optional[int] = None) -> List[int]:
    """Scene agent indices in slot order (module docstring)."""
    adv = cat_adversary(scene)
    t0 = 10
    anchor = scene.position[scene.sdc, t0, :2]

    def dist(i):
        valid = np.flatnonzero(scene.valid[i])
        if len(valid) == 0:
            return np.inf
        t = valid[np.argmin(np.abs(valid - t0))]
        return float(np.linalg.norm(scene.position[i, t, :2] - anchor))

    rest = sorted((i for i in range(scene.n_agents)
                   if i not in (scene.sdc, adv) and not is_static_vehicle(scene, i)), key=dist)
    order = [scene.sdc] + ([adv] if adv is not None else []) + rest
    return order[:max_agents] if max_agents else order


def segments(map_features: Dict, types) -> np.ndarray:
    """[N, 4] (x0, y0, x1, y1) of every polyline segment of the given types."""
    out = []
    for f in map_features.values():
        poly = np.asarray(f.get("polyline", []), dtype=float)
        if f.get("type") in types and poly.ndim == 2 and len(poly) >= 2:
            out.append(np.concatenate([poly[:-1, :2], poly[1:, :2]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 4))


def near(segs: np.ndarray, route: np.ndarray, radius: float) -> np.ndarray:
    """The segments [N, 4] with an end or midpoint within radius of the route
    (densified to 1 m)."""
    if len(segs) == 0:
        return segs
    pts = np.concatenate([segs[:, :2], segs[:, 2:], 0.5 * (segs[:, :2] + segs[:, 2:])])  # [3N, 2]
    dense = np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b - a))) + 1))
                            for a, b in zip(route[:-1], route[1:])]) if len(route) > 1 else route
    d = np.full(len(pts), np.inf)
    for k in range(0, len(dense), 512):
        d = np.minimum(d, np.linalg.norm(pts[:, None] - dense[None, k:k + 512], axis=-1).min(1))
    return segs[(d.reshape(3, -1) <= radius).any(0)]


def scene_arrays(scene: Scene, max_agents: Optional[int] = None) -> Dict[str, np.ndarray]:
    """One scene's tracks and static data (module docstring)."""
    if scene.n_steps != STEPS:
        raise ValueError(f"{scene.scenario_id}: {scene.n_steps} steps, the simulator needs {STEPS}")
    order = agent_order(scene, max_agents)
    sdc = scene.sdc
    valid = np.flatnonzero(scene.valid[sdc])
    path = scene.position[sdc, valid, :2].astype(float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-3])  # drop repeats
    path = path[keep]
    route_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
    h = float(scene.heading[sdc, valid[-1]])
    route = np.concatenate([path, path[-1:] + EXTENSION * np.array([[np.cos(h), np.sin(h)]])])
    size = np.stack([scene.shape_at(i, 10) for i in order]).astype(float)
    return {
        "position": scene.position[order, :, :2].astype(np.float32),
        "heading": scene.heading[order].astype(np.float32),
        "velocity": scene.velocity[order, :, :2].astype(np.float32),
        "valid": scene.valid[order].astype(bool),
        "size": size.astype(np.float32),
        "type": np.array([TYPE_CODES.get(scene.types[i], 0) for i in order], dtype=np.int64),
        "track_id": np.array([scene.track_ids[i] for i in order]),
        "route": route.astype(np.float32), "route_len": np.float32(route_len),
        "edges": near(segments(scene.map_features, EDGE_TYPES), route, NEAR_ROUTE).astype(np.float32),
        "yellow": near(segments(scene.map_features, YELLOW_TYPES), route, NEAR_ROUTE).astype(np.float32),
        "has_adversary": cat_adversary(scene) is not None,
    }


def _pad(arrays: Sequence[np.ndarray], fill=0) -> np.ndarray:
    n = max(1, max(len(a) for a in arrays))
    out = np.full((len(arrays), n) + arrays[0].shape[1:], fill, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


def build_bank(scenes_dir, first: int = 0, n: Optional[int] = None, max_agents: Optional[int] = 128) -> Dict:
    files = scene_files(scenes_dir)[first:]
    files = files[:n] if n is not None else files
    rows = [scene_arrays(Scene.load(f), max_agents) for f in files]
    out = {"stems": np.array([f.stem for f in files])}
    for key in ("position", "heading", "velocity", "valid", "size", "type", "route", "edges", "yellow"):
        out[key] = _pad([r[key] for r in rows], fill=False if key == "valid" else 0)
    out["track_id"] = _pad([r["track_id"].astype("U32") for r in rows], fill="")
    out["n_agents"] = np.array([len(r["type"]) for r in rows])
    out["route_n"] = np.array([len(r["route"]) for r in rows])
    out["edges_n"] = np.array([len(r["edges"]) for r in rows])
    out["yellow_n"] = np.array([len(r["yellow"]) for r in rows])
    out["route_len"] = np.array([r["route_len"] for r in rows], dtype=np.float32)
    out["has_adversary"] = np.array([r["has_adversary"] for r in rows])
    return out


class SceneBank:
    """build_bank's arrays (or its .npz) on a device; ``index`` maps stem -> row."""

    def __init__(self, source, device: str = "cpu"):
        import torch

        z = dict(np.load(source, allow_pickle=False)) if isinstance(source, (str, Path)) else source
        self.stems = [str(s) for s in z["stems"]]
        self.index = {s: i for i, s in enumerate(self.stems)}
        self.track_id = z["track_id"]
        f = torch.float32
        t = lambda k, dtype=f: torch.as_tensor(np.asarray(z[k]), dtype=dtype, device=device)  # noqa: E731
        self.position, self.heading, self.velocity = t("position"), t("heading"), t("velocity")
        self.valid = t("valid", torch.bool)
        self.size, self.type, self.n_agents = t("size"), t("type", torch.long), t("n_agents", torch.long)
        self.route, self.route_n, self.route_len = t("route"), t("route_n", torch.long), t("route_len")
        self.edges, self.edges_n = t("edges"), t("edges_n", torch.long)
        self.yellow, self.yellow_n = t("yellow"), t("yellow_n", torch.long)
        self.has_adversary = t("has_adversary", torch.bool)

    def __len__(self) -> int:
        return len(self.stems)

    def rows(self, stems: Sequence[str]):
        import torch

        return torch.tensor([self.index[str(s)] for s in stems], device=self.position.device)
