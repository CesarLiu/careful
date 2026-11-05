"""A ``responsibility.scene.Scene`` as the PyG batch CAT-K's SMART reads,
in memory, anchored at any context step.

CAT-K caches a WOMD scenario in two stages (its catk/data_preprocess.py):
protobuf decoders, then ``get_agent_features``, ``get_map_features`` /
``process_dynamic_map`` and ``preprocess_map``. ``responsibility.catk_export``
rebuilds the decoders' output from a Scene; the second stage is CAT-K's own
code, run unchanged here (``catk_stage_two``; also used by
scripts/responsibility/export_catk.py to write cache files).

The second stage keeps only agents observed at the "current" step and takes
the traffic lights of that step, so the batch is built with ``current`` =
the context step k: every agent observed at k is in it, and the lights are
k's. ``SmartPredictor`` then re-anchors it with
``native.motion_set.shifted_context`` so that SMART's 11 history steps end
at k. Positions stay in the scene frame, as in CAT-K's cache.

CAT-K's map preprocessing assumes WOMD's 0.5 m point spacing: it cuts a
polyline wherever two points are more than 3 m apart and drops pieces of
fewer than two points. A map whose polylines are sparser (a synthetic map,
another dataset's coarse lanes) therefore loses those polylines, and with
none left SMART gets a one-token placeholder at (6e4, 6e4) m. ``build`` warns
when a scene that has lanes comes out with that placeholder.
"""

import warnings
from typing import Callable, Iterable

import numpy as np

from responsibility.catk_export import dynamic_map_infos, map_infos, track_infos
from responsibility.scene import Scene

PLACEHOLDER_MAP = 6e4  # CAT-K's "dummy empty map" position (catk/smart/utils/preprocess.py)


def check_map(scene: Scene, data: dict) -> bool:
    """False (with a warning) if CAT-K's preprocessing left ``scene``'s
    lanes out entirely, i.e. SMART would roll out on an empty map."""
    has_lanes = any(str(f.get("type", "")).startswith("LANE") for f in scene.map_features.values())
    pos = data["map_save"]["traj_pos"]
    pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
    empty = pos.shape[0] == 1 and bool(np.all(pos == PLACEHOLDER_MAP))
    if has_lanes and empty:
        warnings.warn(f"{scene.scenario_id}: SMART sees no map -- CAT-K's preprocessing dropped every polyline "
                      "(it cuts polylines at gaps over 3 m and expects WOMD's 0.5 m point spacing)")
        return False
    return True


def catk_stage_two() -> Callable:
    """``build(scene, predict=(), stem=None, current=10) -> dict``: CAT-K's
    cached-scenario dict of ``scene`` (what its pickles hold)."""
    from catk.data_preprocess import get_agent_features, get_map_features, process_dynamic_map
    from catk.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map

    def build(scene: Scene, predict: Iterable[str] = (), stem=None, current: int = 10) -> dict:
        lights = process_dynamic_map(dynamic_map_infos(scene.dynamic_map_states, scene.n_steps))
        maps = map_infos(scene.map_features, get_polylines_from_polygon)
        data = preprocess_map(get_map_features(maps, lights.loc[lights["time_step"] == current]))
        data["agent"] = get_agent_features(track_infos(scene, predict), split="validation",
                                           num_historical_steps=current + 1, num_steps=scene.n_steps)
        data["scenario_id"] = stem if stem is not None else scene.scenario_id
        check_map(scene, data)
        return data

    return build


def catk_batch_builder(device: str = "cpu") -> Callable:
    """``build(scene, current) -> torch_geometric Batch`` of one scenario, as
    CAT-K's batch_size=1 DataLoader yields it (WaymoTargetBuilderVal is
    ``HeteroData(data)``), on ``device``."""
    stage_two = catk_stage_two()
    from torch_geometric.data import Batch, HeteroData

    def build(scene: Scene, current: int):
        data = stage_two(scene, scene.track_ids, current=current)
        return Batch.from_data_list([HeteroData(data)]).to(device)

    return build
