"""The log-replay simulator (rl/sim.py), its scene tensors (rl/scenes.py)
and TD3 (rl/td3.py) on synthetic scenes with known answers."""

import pickle
import sys

import numpy as np
import pytest
import torch

from rl.adversary import CandidateBank
from rl.scenes import SceneBank, agent_order, build_bank, scene_arrays
from rl.sim import (ARRIVED, CRASHED, EGO_DIM, NAV_DIM, OUT_OF_ROAD, TIMEOUT, LogReplaySim, SimSettings, obs_dim,
                    run_lockstep)
from rl.td3 import TD3, ReplayBuffer
from tests.responsibility.conftest import make_scene, track


def lane(y=0.0, x0=-50.0, x1=250.0, kind="LANE_SURFACE_STREET"):
    return {"type": kind, "polyline": np.stack([np.linspace(x0, x1, 601), np.full(601, y), np.zeros(601)], 1)}


def road(extra=None):
    m = {"100": lane(0.0), "101": lane(3.5), "102": lane(-2.0, kind="ROAD_EDGE_BOUNDARY"),
         "103": lane(5.5, kind="ROAD_LINE_SOLID_SINGLE_YELLOW")}
    m.update(extra or {})
    return m


def scene(tracks, extra_map=None, ooi=("0", "1")):
    return make_scene(tracks, ooi=ooi, map_features=road(extra_map), lights={})


def bank_of(*scenes):
    """build_bank's arrays, from Scene objects instead of files."""
    import rl.scenes as rs

    rows = [scene_arrays(s) for s in scenes]
    out = {"stems": np.array([str(i) for i in range(len(rows))])}
    for key in ("position", "heading", "velocity", "valid", "size", "type", "route", "edges", "yellow"):
        out[key] = rs._pad([r[key] for r in rows], fill=False if key == "valid" else 0)
    out["track_id"] = rs._pad([r["track_id"].astype("U32") for r in rows], fill="")
    for k, src in (("n_agents", "type"), ("route_n", "route"), ("edges_n", "edges"), ("yellow_n", "yellow")):
        out[k] = np.array([len(r[src]) for r in rows])
    out["route_len"] = np.array([r["route_len"] for r in rows], dtype=np.float32)
    out["has_adversary"] = np.array([r["has_adversary"] for r in rows])
    return SceneBank(out)


def test_slot_order_drops_static_vehicles_and_puts_the_adversary_second():
    s = scene({"0": track((0, 0), (10, 0)), "1": track((60, 3.5), (8, 0)), "2": track((20, 3.5), (8, 0)),
               "3": track((5, 3.5), (0.1, 0))})  # 3: parked (spread < 3 m)
    assert agent_order(s) == [0, 1, 2]
    a = scene_arrays(s)
    assert a["position"].shape == (3, 91, 2) and a["has_adversary"]
    assert len(a["edges"]) > 0 and len(a["yellow"]) > 0  # within 20 m of the route


def test_the_inverse_replays_the_log_and_the_drive_arrives():
    s = scene({"0": track((0, 0), (10, 0)), "1": track((40, 3.5), (9, 0))})
    sim = LogReplaySim(SimSettings(worlds=1), bank_of(s))
    sim.begin(torch.tensor([0]), torch.tensor([False]))
    err, total = 0.0, 0.0
    while not bool(sim.done.all()):
        obs, r, *_ = sim.step(sim.expert_action(), clip=False)
        total += float(r)
        err = max(err, float((sim.p - sim.bank.position[0, 0, sim.t.clamp(max=90)]).norm()))
    rep = sim.end_episode()
    assert err < 1e-3
    assert int(rep["outcome"]) == ARRIVED and not bool(rep["crashed"])
    # 1 m of progress a step for 85 steps; the 86th passes 95% of the logged 90 m and pays +10 instead
    assert total == pytest.approx(85.0 + 10.0, abs=1e-3)
    assert obs.shape == (1, obs_dim(sim.s))


def test_a_curve_is_replayed_too():
    t = (np.arange(91) - 10) * 0.1
    R, w = 30.0, 0.3  # 9 m/s on a 30 m circle
    pos = np.stack([R * np.sin(w * (t + 1)), R - R * np.cos(w * (t + 1)), np.zeros(91)], 1).astype(np.float32)
    tr = track((0, 0), (9, 0))
    tr["state"]["position"] = pos
    tr["state"]["heading"] = (w * (t + 1)).astype(np.float32)
    tr["state"]["velocity"] = np.stack([9 * np.cos(w * (t + 1)), 9 * np.sin(w * (t + 1))], 1).astype(np.float32)
    s = make_scene({"0": tr, "1": track((200, 200), (5, 0))}, map_features={"100": lane(0.0)}, lights={})
    sim = LogReplaySim(SimSettings(worlds=1), bank_of(s))
    sim.begin(torch.tensor([0]), torch.tensor([False]))
    err = 0.0
    for _ in range(80):
        sim.step(sim.expert_action(), clip=False)
        err = max(err, float((sim.p - sim.bank.position[0, 0, sim.t]).norm()))
    assert err < 1e-3
    assert float(sim.expert_action()[0, 1]) == pytest.approx((1 / R) / 0.3, rel=0.05)  # curvature 1/30 of 0.3 1/m


def test_collisions_cost_and_end_only_with_crash_done():
    # a slow leader 15 m ahead in the SDC's lane; full throttle runs into it
    s = scene({"0": track((0, 0), (10, 0)), "1": track((15, 0), (2, 0))})
    for crash_done, want in ((False, None), (True, CRASHED)):
        sim = LogReplaySim(SimSettings(worlds=1, crash_done=crash_done), bank_of(s))
        sim.begin(torch.tensor([0]), torch.tensor([False]))
        rewards = []
        while not bool(sim.done.all()):
            _, r, *_ = sim.step(torch.tensor([[1.0, 0.0]]))
            rewards.append(float(r))
            if bool(sim.flags["crash"]):
                assert rewards[-1] == -1.0
        rep = sim.end_episode()
        assert bool(rep["crashed"]) and int(rep["partner"]) == 1
        if want is not None:
            assert int(rep["outcome"]) == want and int(rep["steps"]) == int(rep["first_crash"])
        else:
            assert int(rep["outcome"]) != CRASHED and int(rep["steps"]) > int(rep["first_crash"])


def test_out_of_road_at_an_edge_and_a_yellow_line():
    s = scene({"0": track((0, 0), (10, 0)), "1": track((200, 200), (5, 0))})
    for steer in (-1.0, 1.0):  # right: the road edge at y = -2; left: the yellow line at y = 5.5
        sim = LogReplaySim(SimSettings(worlds=1), bank_of(s))
        sim.begin(torch.tensor([0]), torch.tensor([False]))
        while not bool(sim.done.all()):
            sim.step(torch.tensor([[0.0, steer]]))
            flags = dict(sim.flags)
        rep = sim.end_episode()
        assert int(rep["outcome"]) == OUT_OF_ROAD
        assert bool(flags["edge"] if steer < 0 else flags["yellow"])


def test_observations_put_the_nearest_partner_first_in_the_sdc_frame():
    s = scene({"0": track((0, 0), (10, 0)), "1": track((30, 3.5), (8, 0)), "2": track((-12, 0), (10, 0))})
    sim = LogReplaySim(SimSettings(worlds=1, partners=4), bank_of(s))
    obs = sim.begin(torch.tensor([0]), torch.tensor([False]))
    assert obs.shape == (1, EGO_DIM + NAV_DIM + 9 * 4)
    parts = obs[0, EGO_DIM + NAV_DIM:].reshape(4, 9)
    # at step 0 (t = -1 s): the SDC at x = -10, agent 2 at x = -22, agent 1 at x = 22
    assert parts[0, :2].tolist() == pytest.approx([-12 / 50, 0.0], abs=1e-5)
    assert parts[1, :2].tolist() == pytest.approx([32 / 50, 3.5 / 50], abs=1e-5)
    assert parts[0, 8] == 1.0 and parts[2:, 8].sum() == 0.0


def write_candidates(tmp_path, bank, plans=((3.5, 8.0), (0.0, 2.0))):
    """A candidate bank whose adversary (slot 1, 30 m ahead at step 10) can
    follow one of two futures (lane y, speed): on in its own lane, or cut
    into the SDC's lane and crawl, which the logged SDC (10 m/s) runs into."""
    S = len(bank.stems)
    K = len(plans)
    t = np.arange(11, 91) * 0.1
    cand = np.stack([np.stack([30 + v * (t - 1), np.full(80, y)], -1) for y, v in plans])
    z = {"stems": np.array(bank.stems), "adv_id": np.array(["1"] * S),
         "candidates": np.repeat(cand[None], S, 0).astype(np.float32),
         "log_scores": np.repeat(np.log(np.full((1, K), 1.0 / K)), S, 0),
         "adv_past": np.repeat(bank.position[:, 1, :11].cpu().numpy()[:1], S, 0).astype(np.float64),
         "ov_size": np.repeat([[4.8, 2.0]], S, 0), "av_size": np.repeat([[4.8, 2.0]], S, 0),
         "ego_route": bank.position[:, 0, 11:].cpu().numpy().astype(np.float64),
         "samples": np.repeat(cand[None, :1].repeat(4, 1), S, 0).astype(np.float32),
         "avoid": np.ones((S, K))}
    np.savez(tmp_path / "cand.npz", **z)
    return CandidateBank(str(tmp_path / "cand.npz"))


def test_cat_adversary_is_chosen_against_the_history_and_followed_a_step_late(tmp_path):
    s = scene({"0": track((0, 0), (10, 0)), "1": track((30, 3.5), (8, 0))})
    bank = bank_of(s)
    cand = write_candidates(tmp_path, bank)
    sim = LogReplaySim(SimSettings(worlds=1), bank, cand)
    sim.begin(torch.tensor([0]), torch.tensor([True]), rule="cat")
    assert int(sim.chosen) == 1  # the cut-in hits the logged SDC; its own lane does not
    for _ in range(30):
        sim.step(sim.expert_action())
    pos, _, _, present = sim.agents()
    assert bool(present[0, 1])
    assert pos[0, 1].tolist() == pytest.approx(cand.cand[0, 1, 30 - 1 - 11].tolist(), abs=1e-5)
    rep = run_lockstep(sim, lambda o: sim.expert_action(), torch.tensor([0]), True, rule="cat")
    assert bool(rep["crashed"]) and int(rep["partner"]) == 1


def test_finished_episodes_restart_and_update_the_history(tmp_path):
    s = scene({"0": track((0, 0), (10, 0)), "1": track((30, 3.5), (8, 0))})
    bank = bank_of(s, s)
    cand = write_candidates(tmp_path, bank)
    sim = LogReplaySim(SimSettings(worlds=2), bank, cand)
    picks = []

    def pick(n):
        picks.append(n)
        return torch.ones(n, dtype=torch.long)

    sim.begin(torch.tensor([0, 1]), torch.tensor([False, False]), auto_reset=True, pick=pick)
    before = sim.hist[0].clone()
    for _ in range(95):
        sim.step(torch.tensor([[0.2, 0.0], [0.2, 0.0]]))
    fin = sim.take_finished()
    assert len(fin["outcome"]) == 2 and set(fin["outcome"].tolist()) <= {ARRIVED, TIMEOUT}
    assert picks == [2] and sim.rows.tolist() == [1, 1]
    assert not torch.equal(sim.hist[0, -1], before[-1])  # the newest history slot is the driven trajectory
    assert torch.equal(sim.hist[0, 0], before[1])  # the deque moved on by one


def test_td3_buffer_and_update():
    buf = ReplayBuffer(5, 2, max_size=8)
    idx = buf.add(torch.ones(6, 5), torch.zeros(6, 2), torch.ones(6, 5), torch.ones(6), torch.zeros(6, dtype=torch.bool))
    assert idx.tolist() == list(range(6))
    idx = buf.add(torch.ones(4, 5), torch.zeros(4, 2), torch.ones(4, 5), torch.ones(4), torch.ones(4, dtype=torch.bool))
    assert idx.tolist() == [6, 7, 0, 1] and buf.size == 8
    policy = TD3(5, 2)
    before = [p.clone() for p in policy.critic.parameters()]
    for _ in range(4):
        policy.train(buf, 4)
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.critic.parameters()))
    assert policy.select_action(torch.zeros(3, 5)).shape == (3, 2)


def test_training_runs_end_to_end(tmp_path, monkeypatch):
    from scripts.rl import train_td3

    scenes = tmp_path / "scenes"
    scenes.mkdir()
    for i in range(3):
        d = scene({"0": track((0, 0), (10, 0)), "1": track((30, 3.5), (8, 0))}).to_description()
        with open(scenes / f"{i}.pkl", "wb") as f:
            pickle.dump(d, f)
    bank = build_bank(scenes)
    np.savez(tmp_path / "bank.npz", **bank)
    cand = write_candidates(tmp_path, SceneBank(str(tmp_path / "bank.npz")))
    assert cand is not None
    out = tmp_path / "run"
    train_td3.main(["--bank", str(tmp_path / "bank.npz"), "--candidates", str(tmp_path / "cand.npz"),
                    "--mode", "cat", "--n-train", "2", "--max-timesteps", "800", "--start-timesteps", "200",
                    "--eval-freq", "400", "--worlds", "8", "--utd", "0.25", "--batch-size", "32",
                    "--device", "cpu", "--out", str(out), "--save-model"])
    rows = (out / "progress.csv").read_text().splitlines()
    assert rows[0].startswith("timesteps,route_completion_normal") and len(rows) == 3
    assert (out / "model.pt").exists() and (out / "train.csv").exists()
