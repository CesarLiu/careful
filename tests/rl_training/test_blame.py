"""Blame-weighted crash penalties in the log-replay simulator (rl/blame.py)."""

import csv

import numpy as np
import pytest
import torch

from responsibility.predictors.constant_velocity import ConstantVelocity
from rl.blame import SimBlameWeighting
from rl.sim import LogReplaySim, SimSettings
from tests.rl_training.test_sim import bank_of, scene
from tests.responsibility.conftest import track


def run(scn, action, weighting, steps=60):
    sim = LogReplaySim(SimSettings(worlds=1), bank_of(scn))
    sim.before_reset = weighting
    sim.begin(torch.tensor([0]), torch.tensor([False]))
    rewards, flags = [], []
    for _ in range(steps):
        _, r, *_ = sim.step(torch.tensor([action]))
        rewards.append(float(r))
        flags.append((bool(sim.flags["crash"]), float(sim.step_reward)))
    return sim, rewards, flags


def scenes_dir(tmp_path, scn):
    import pickle

    d = tmp_path / "scenes"
    d.mkdir(exist_ok=True)
    with open(d / "0.pkl", "wb") as f:
        pickle.dump(scn.to_description(), f)
    return d


def test_rss_keeps_the_penalty_when_the_sdc_rear_ends(tmp_path):
    scn = scene({"0": track((0, 0), (10, 0)), "1": track((15, 0), (2, 0))})
    weighting = SimBlameWeighting(None, scenes_dir(tmp_path, scn), 1.0, mode="rss", log_path=tmp_path / "b.csv")
    _, rewards, flags = run(scn, [1.0, 0.0], weighting)
    crash = [i for i, (c, _) in enumerate(flags) if c]
    assert crash and all(rewards[i] == -1.0 for i in crash)
    rows = list(csv.DictReader(open(tmp_path / "b.csv")))
    assert len(rows) == 1 and rows[0]["rss"] == "ego" and float(rows[0]["weight"]) == 1.0


def test_rss_lifts_the_penalty_when_a_follower_runs_into_the_sdc(tmp_path):
    # the SDC (10 m/s) brakes hard; a follower 50 m behind, at a safe RSS distance, keeps 12 m/s (its log)
    # and does not respond. (10 m behind it would be unsafe from the first step: RSS gives no verdict then)
    scn = scene({"0": track((0, 0), (10, 0)), "1": track((-50, 0), (12, 0))})
    weighting = SimBlameWeighting(None, scenes_dir(tmp_path, scn), 1.0, mode="rss", log_path=tmp_path / "b.csv")
    _, rewards, flags = run(scn, [-1.0, 0.0], weighting, steps=90)
    crash = [i for i, (c, _) in enumerate(flags) if c]
    assert crash
    rows = list(csv.DictReader(open(tmp_path / "b.csv")))
    assert rows[0]["rss"] == "other" and float(rows[0]["weight"]) == 0.0
    for i in crash:  # the step keeps the progress it made instead of -1
        assert rewards[i] == pytest.approx(flags[i][1])


def test_share_attributes_with_any_predictor_and_rebuilds_the_episode(tmp_path):
    scn = scene({"0": track((0, 0), (10, 0)), "1": track((15, 0), (2, 0))})
    weighting = SimBlameWeighting(ConstantVelocity(), scenes_dir(tmp_path, scn), 1.0, mode="share",
                                  log_path=tmp_path / "b.csv")
    sim, rewards, flags = run(scn, [1.0, 0.0], weighting, steps=20)
    rows = list(csv.DictReader(open(tmp_path / "b.csv")))
    assert len(rows) == 1 and rows[0]["verdict"] != "error"
    w = float(rows[0]["weight"])
    assert 0.0 <= w <= 1.0
    first = next(i for i, (c, _) in enumerate(flags) if c)
    assert rewards[first] == pytest.approx(w * -1.0 + (1 - w) * flags[first][1])
    # the rollout behind the attribution is the simulated episode
    rebuilt, rollout = weighting.rollout(sim, 0) if bool(sim.contact[0] >= 0) else (None, None)
    if rebuilt is not None:
        n = int(sim.t[0]) + 1
        assert np.allclose(rebuilt.position[rebuilt.sdc, :n, :2], sim.ego_buf[0, :n].numpy(), atol=1e-5)
        assert rollout["end"]["crash_with"] == "1"
