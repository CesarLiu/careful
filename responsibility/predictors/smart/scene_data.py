"""A ``responsibility.scene.Scene`` as the PyG batch CAT-K's SMART reads,
in memory, anchored at any context step.

CAT-K caches a WOMD scenario in two stages (its src/data_preprocess.py):
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
"""

from typing import Callable, Iterable

from responsibility.catk_export import dynamic_map_infos, map_infos, track_infos
from responsibility.predictors.smart.catk import require_catk
from responsibility.scene import Scene


def catk_stage_two(catk_root=None) -> Callable:
    """``build(scene, predict=(), stem=None, current=10) -> dict``: CAT-K's
    cached-scenario dict of ``scene`` (what its pickles hold)."""
    require_catk(catk_root)
    from src.data_preprocess import get_agent_features, get_map_features, process_dynamic_map
    from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map

    def build(scene: Scene, predict: Iterable[str] = (), stem=None, current: int = 10) -> dict:
        lights = process_dynamic_map(dynamic_map_infos(scene.dynamic_map_states, scene.n_steps))
        maps = map_infos(scene.map_features, get_polylines_from_polygon)
        data = preprocess_map(get_map_features(maps, lights.loc[lights["time_step"] == current]))
        data["agent"] = get_agent_features(track_infos(scene, predict), split="validation",
                                           num_historical_steps=current + 1, num_steps=scene.n_steps)
        data["scenario_id"] = stem if stem is not None else scene.scenario_id
        return data

    return build


def catk_batch_builder(catk_root=None, device: str = "cpu") -> Callable:
    """``build(scene, current) -> torch_geometric Batch`` of one scenario, as
    CAT-K's batch_size=1 DataLoader yields it (WaymoTargetBuilderVal is
    ``HeteroData(data)``), on ``device``."""
    stage_two = catk_stage_two(catk_root)
    from torch_geometric.data import Batch, HeteroData

    def build(scene: Scene, current: int):
        data = stage_two(scene, scene.track_ids, current=current)
        return Batch.from_data_list([HeteroData(data)]).to(device)

    return build
