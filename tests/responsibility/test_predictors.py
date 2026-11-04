"""The predictor registry, the capability checks, the constant-velocity
reference predictor, and a whole compute_responsibility run with it (no
TensorFlow, UniTraj, CAT-K or checkpoint needed)."""

import csv
import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from responsibility.metrics import ResponsibilityConfig, responsibility_at
from responsibility.models import load_model, model_settings
from responsibility.predictors import (capabilities, load_predictor, predictor_names, predictor_settings,
                                       register_predictor, require)
from responsibility.predictors.base import GOAL_KL, NMS_SET, SAMPLE, WEIGHTED_SET, MotionPredictor
from responsibility.predictors.constant_velocity import ConstantVelocity
from tests.responsibility.conftest import make_scene, track


def two_cars():
    # a follows b in the same lane, 12 m behind, closing at 3 m/s
    return make_scene({"0": track((0.0, 0.0), (11.0, 0.0)), "1": track((12.0, 0.0), (8.0, 0.0))})


def test_builtin_predictors_are_registered():
    assert {"densetnt", "mtr", "unitraj", "smart", "constant_velocity"} <= set(predictor_names())


def test_settings_of_older_runs_are_unchanged():
    base = SimpleNamespace(model="densetnt", checkpoint=None, unitraj_root=None, unitraj_method=None,
                           temperature=None)
    assert model_settings(base) == {"name": "densetnt"}
    mtr = SimpleNamespace(**{**vars(base), "model": "mtr", "temperature": 2.0})
    assert model_settings(mtr) == {"name": "mtr", "checkpoint": None, "method": "MTR_womd", "temperature": 2.0}
    way = SimpleNamespace(**{**vars(base), "model": "unitraj", "unitraj_method": "autobot", "temperature": 1.0})
    assert model_settings(way)["method"] == "autobot"
    smart = SimpleNamespace(model="smart", checkpoint=None)
    assert model_settings(smart) == {"name": "smart", "checkpoint": None, "config": "smart",
                                     "neighbor_future": "hidden"}


def test_models_that_need_a_checkpoint_say_so():
    for name in ("mtr", "unitraj", "smart"):
        with pytest.raises(SystemExit, match="checkpoint"):
            load_model(SimpleNamespace(model=name, checkpoint=None), "cpu")


def test_a_predictor_of_ones_own_by_name_and_by_registration():
    register_predictor("cv_test", lambda args, device: ConstantVelocity(horizon=30))
    model = load_predictor("cv_test", SimpleNamespace(), "cpu")
    assert model.horizon == 30
    assert predictor_settings("cv_test", SimpleNamespace()) == {"name": "cv_test"}
    by_path = load_predictor("tests.responsibility.test_predictors:factory", SimpleNamespace(), "cpu")
    assert isinstance(by_path, ConstantVelocity)
    with pytest.raises(ValueError, match="unknown predictor"):
        load_predictor("nope", SimpleNamespace(), "cpu")


def factory(args, device):
    return ConstantVelocity()


def test_capabilities_and_requirements():
    cv = ConstantVelocity()
    assert isinstance(cv, MotionPredictor)
    assert capabilities(cv) == {SAMPLE, WEIGHTED_SET, "topk", NMS_SET, GOAL_KL}

    class SamplesOnly:
        name = "samples_only"

        def distribution(self, scene, step, agent, excluded=(), avoid_partner=()):
            return None

        def sample(self, dist, n, generator=None):
            return None

    assert capabilities(SamplesOnly()) == {SAMPLE}
    require(SamplesOnly(), "sampled", "boltzmann")
    require(SamplesOnly(), "sampled", "demand")
    with pytest.raises(ValueError, match="kl"):
        require(SamplesOnly(), "sampled", "kl")
    with pytest.raises(ValueError, match="nms"):
        require(SamplesOnly(), "nms", "boltzmann")
    with pytest.raises(TypeError):
        capabilities(object())


def test_constant_velocity_distribution():
    scene = two_cars()
    cv = ConstantVelocity()
    dist = cv.distribution(scene, 20, 0)
    assert dist.trajectories.shape == (25, 80, 2)
    assert float(dist.log_prob.exp().sum()) == pytest.approx(1.0, abs=1e-5)
    # the most probable mode keeps speed and heading: 11 m/s for 8 s from x(step 20) = 11 m
    best = dist.trajectories[int(dist.log_prob.argmax())]
    assert best[-1] == pytest.approx([11.0 + 88.0, 0.0], abs=1e-6)
    idx, lp, trajs = cv.sample(dist, 7, torch.Generator().manual_seed(0))
    assert trajs.shape == (7, 80, 2) and lp.shape == (7,)
    trajs, probs = cv.motion_set(dist, top_k=5)
    assert len(trajs) == 5 and probs.sum() < 1
    trajs, w = cv.nms_motion_set(dist, 4)
    assert len(trajs) == len(w) <= 4
    with_a, without_a = cv.with_and_without(scene, 20, 1, 0)
    assert torch.equal(with_a.log_prob, without_a.log_prob)
    invalid = make_scene({"0": track((0, 0), (1, 0), valid=np.arange(91) > 50), "1": track((5, 0), (1, 0))})
    assert cv.distribution(invalid, 20, 0) is None


@pytest.mark.parametrize("estimator,motion_set", [("boltzmann", "sampled"), ("kl", "weighted"), ("demand", "nms")])
def test_metrics_run_on_the_constant_velocity_predictor(estimator, motion_set):
    scene = two_cars()
    cfg = ResponsibilityConfig(courtesy_estimator=estimator, motion_set=motion_set)
    obs = responsibility_at(ConstantVelocity(), scene, 0, 20, cfg, torch.Generator().manual_seed(0))
    assert obs is not None and "1" in obs.per_neighbour
    assert np.isfinite(obs.safety)
    # closing on the leader: braking alternatives keep more distance than the log
    assert obs.safety > 0
    if estimator == "kl":
        assert obs.courtesy == 0.0  # a predictor that ignores everyone is courtesy-blind


def test_compute_responsibility_end_to_end(tmp_path, monkeypatch):
    from scripts.responsibility import compute_responsibility

    scenes = tmp_path / "scenes"
    scenes.mkdir()
    for i in range(2):
        with open(scenes / f"{i}.pkl", "wb") as f:
            pickle.dump(two_cars().to_description(), f)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["compute_responsibility", "--scenes", str(scenes), "--out-dir", str(out),
                                      "--model", "constant_velocity", "--device", "cpu", "--save-records"])
    compute_responsibility.main()
    rows = list(csv.DictReader(open(out / "windows.csv")))
    assert len(rows) > 0 and {r["scene"] for r in rows} == {"0", "1"}
    assert json.loads((out / "config.json").read_text())["model"] == {"name": "constant_velocity"}
    assert (out / "records" / "0.pkl").exists()
    # a predictor without a finite motion set is refused up front
    monkeypatch.setattr(sys, "argv", ["compute_responsibility", "--scenes", str(scenes), "--out-dir",
                                      str(tmp_path / "run2"), "--model", "tests.responsibility.test_predictors:"
                                      "samples_only", "--motion-set", "nms", "--device", "cpu"])
    with pytest.raises(ValueError, match="finite motion set"):
        compute_responsibility.main()


class _SamplesOnly(ConstantVelocity):
    motion_set = None
    nms_motion_set = None
    with_and_without = None


def samples_only(args, device):
    return _SamplesOnly()


def test_a_run_regresses_against_itself(tmp_path, monkeypatch, capsys):
    from scripts.responsibility import compute_responsibility, regress_run

    scenes = tmp_path / "scenes"
    scenes.mkdir()
    for i in range(3):
        with open(scenes / f"{i}.pkl", "wb") as f:
            pickle.dump(two_cars().to_description(), f)
    flags = ["--scenes", str(scenes), "--model", "constant_velocity", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", ["c", *flags, "--out-dir", str(tmp_path / "old")])
    compute_responsibility.main()
    regress_run.prepare(tmp_path / "old", tmp_path / "new")
    monkeypatch.setattr(sys, "argv", ["c", *flags, "--n", "2", "--out-dir", str(tmp_path / "new")])
    compute_responsibility.main()
    assert regress_run.compare(tmp_path / "old", tmp_path / "new", 1e-4)
    assert capsys.readouterr().out.strip().endswith("IDENTICAL")
    # other settings are refused by compute_responsibility itself
    monkeypatch.setattr(sys, "argv", ["c", *flags, "--n", "2", "--stride", "10", "--out-dir", str(tmp_path / "new")])
    with pytest.raises(SystemExit, match="other settings"):
        compute_responsibility.main()
