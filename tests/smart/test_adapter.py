"""SmartPredictor: SMART behind the predictor interface, on CAT-K's stub
SMART (conftest.FakeSMART) and a stand-in for the Scene -> PyG conversion,
so neither CAT-K nor PyG is needed. What is checked is ours: track ids map
to the batch's rows, removed agents leave the scene, the context is cached
per (scene, step), sampling is reproducible from the generator, and the
predictor-agnostic metrics run on it."""

import numpy as np
import pytest
import torch

from responsibility.catk_export import numeric_id
from responsibility.metrics import ResponsibilityConfig, responsibility_at
from responsibility.predictors.base import GOAL_KL, SAMPLE, capabilities, require
from responsibility.predictors.smart import SmartPredictor
from tests.responsibility.conftest import make_scene, track
from tests.smart.conftest import FakeSMART


def scene_of_three():
    return make_scene({"0": track((0.0, 0.0), (10.0, 0.0)), "1": track((15.0, 0.0), (8.0, 0.0)),
                       "2": track((40.0, 3.5), (9.0, 0.0))})


class Batches:
    """Builds a minimal CAT-K batch: the scene's agents observed at ``current``,
    in reversed order (so rows differ from scene indices), agent 2 left out."""

    def __init__(self):
        self.calls = []

    def __call__(self, scene, current):
        self.calls.append(current)
        keep = [i for i in reversed(range(scene.n_agents)) if scene.valid[i, current] and i != 2]
        return {"agent": {
            "id": torch.tensor([numeric_id(scene.track_ids[i]) for i in keep]),
            "position": torch.as_tensor(scene.position[keep]),
            "heading": torch.as_tensor(scene.heading[keep]),
            "velocity": torch.as_tensor(scene.velocity[keep]),
            "valid_mask": torch.as_tensor(scene.valid[keep]),
        }}


@pytest.fixture
def predictor(sampling_scheme):
    return SmartPredictor(FakeSMART(), sampling_scheme, build_batch=Batches(), neighbor_future="hidden",
                          batch_size=4)


def test_rows_removal_and_none(predictor):
    scene = scene_of_three()
    dist = predictor.distribution(scene, 20, 0, excluded=(1, 2))
    assert dist.row == 1  # reversed order of [0, 1]
    assert dist.removed_rows == (0,)  # agent 1; agent 2 is not in the batch
    assert predictor.distribution(scene, 20, 2) is None  # not in the batch
    assert predictor.distribution(scene, 5, 0) is None  # less than SMART's 1.0 s of history
    late = make_scene({"0": track((0, 0), (1, 0), valid=np.arange(91) > 30), "1": track((5, 0), (1, 0))})
    assert predictor.distribution(late, 20, 0) is None


def test_context_is_built_once_per_scene_and_step(predictor):
    scene = scene_of_three()
    predictor.distribution(scene, 20, 0)
    predictor.distribution(scene, 20, 1, excluded=(0,))
    predictor.distribution(scene, 25, 0)
    assert predictor.build_batch.calls == [20, 25]
    predictor.distribution(scene_of_three(), 20, 0)  # another scene object: rebuilt
    assert predictor.build_batch.calls == [20, 25, 20]


def test_samples_and_their_log_probability(predictor):
    scene = scene_of_three()
    dist = predictor.distribution(scene, 20, 0)
    idx, log_prob, trajs = predictor.sample(dist, 6, torch.Generator().manual_seed(1))
    assert idx is None
    assert trajs.shape == (6, 70, 2) and trajs.dtype == np.float64  # from step 20 the clip holds 7 s more
    assert log_prob.shape == (6,) and torch.isfinite(log_prob).all()
    # the stub's token-0 logit counts the agents in the scene: removing one changes the free agent's odds
    without = predictor.sample(predictor.distribution(scene, 20, 0, excluded=(1,)), 6)[1]
    assert not torch.allclose(log_prob, without)


def test_horizon_is_rounded_to_whole_tokens(sampling_scheme):
    p = SmartPredictor(FakeSMART(), sampling_scheme, build_batch=Batches(), horizon=12)
    trajs = p.sample(p.distribution(scene_of_three(), 20, 0), 3)[2]
    assert trajs.shape[1] == 15  # 12 steps -> three 0.5 s tokens


def test_rollouts_stop_where_the_log_does(predictor):
    # SMART reads the logged token of every step it rolls out: 8 s from 1.0 s, a token less per 0.5 s later
    assert [predictor.max_horizon(91, k) for k in (10, 11, 15, 16, 20, 80, 85)] == [80, 75, 75, 70, 70, 10, 5]
    assert predictor.rollout_horizon(91, 20) == 70
    scene = scene_of_three()
    assert predictor.sample(predictor.distribution(scene, 80, 0), 2)[2].shape[1] == 10
    assert predictor.distribution(scene, 86, 0) is None


def test_capabilities_and_metrics(predictor):
    assert capabilities(predictor) == {SAMPLE}
    assert GOAL_KL not in capabilities(predictor)
    with pytest.raises(ValueError):
        require(predictor, "sampled", "kl")
    require(predictor, "sampled", "boltzmann")
    scene = scene_of_three()
    cfg = ResponsibilityConfig(n_safety_samples=8, n_courtesy_samples=4)
    obs = responsibility_at(predictor, scene, 0, 20, cfg, torch.Generator().manual_seed(0))
    assert obs is not None and "1" in obs.per_neighbour
    assert np.isfinite(obs.safety) and np.isfinite(obs.courtesy)


def test_bad_neighbor_future(sampling_scheme):
    with pytest.raises(ValueError, match="neighbor_future"):
        SmartPredictor(FakeSMART(), sampling_scheme, build_batch=Batches(), neighbor_future="nope")
