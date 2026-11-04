from types import SimpleNamespace

import numpy as np
import torch

from responsibility.demand import DemandConfig
from responsibility.metrics import ResponsibilityConfig, responsibility_at, scene_responsibility
from responsibility.records import load_record, run_scene, save_record, scene_of, sparse_goals
from scripts.responsibility import visualize_responsibility as vis
from tests.responsibility.conftest import make_scene, track


class GoalModel:
    """Stand-in model with goal distributions: 50 goals around each agent,
    samples are straight lines to sampled goals; courtesy for vehicle 1."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def _dist(self, scene, step, agent, shift=0.0):
        origin = scene.position[agent, step, :2]
        goals = np.stack([np.linspace(5, 40, 50), np.linspace(-5, 5, 50) + shift], -1)
        logits = torch.linspace(-1, 1, 50) * (1 + shift)
        return SimpleNamespace(goals=goals, log_prob=torch.log_softmax(logits, -1),
                               to_global=lambda pts, o=origin: np.asarray(pts) + o)

    def distribution(self, scene, step, agent, excluded=(), avoid_partner=()):
        return self._dist(scene, step, agent)

    def sample(self, dist, n, generator=None):
        idx = torch.multinomial(dist.log_prob.exp(), n, replacement=True, generator=generator)
        ends = dist.to_global(dist.goals[idx.numpy()])
        start = dist.to_global(np.zeros((1, 2)))[0]
        frac = np.linspace(1 / 80, 1, 80)[None, :, None]
        return idx, dist.log_prob[idx], start + (ends[:, None] - start) * frac

    def with_and_without(self, scene, step, b, a):
        if scene.types[b] != "VEHICLE":
            return None
        return self._dist(scene, step, b), self._dist(scene, step, b, shift=0.5)


def _scene():
    return make_scene({
        "0": track((0, 0), (10, 0)),
        "1": track((5, 3.5), (10, 0)),
        "2": track((-30, 40), (0, 0)),  # parked far away: no neighbour in some windows
    })


def test_record_captures_what_the_values_were_computed_from():
    record = {}
    obs = responsibility_at(GoalModel(), _scene(), 0, 10, ResponsibilityConfig(n_safety_samples=3), record=record)
    assert record["samples"].shape == (3, 80, 2) and record["sample_log_prob"].shape == (3,)
    assert set(record["neighbours"]) == {1} and set(record["courtesy"]) == {1}
    assert obs.courtesy > 0


def test_records_keep_the_run_values_and_survive_a_round_trip(tmp_path):
    scene = _scene()
    cfg = ResponsibilityConfig(n_safety_samples=6, window_stride=20, courtesy_estimator="kl")
    plain = scene_responsibility(GoalModel(), scene, 0, cfg)
    observations, record = run_scene(GoalModel(), scene, 0, cfg)
    assert [(o.step, o.safety, o.courtesy) for o in observations] == [(o.step, o.safety, o.courtesy) for o in plain]
    save_record(record, tmp_path / "r.pkl")
    loaded = load_record(tmp_path / "r.pkl")
    assert scene_of(loaded).track_ids == scene.track_ids and loaded["agent_id"] == "0"
    frame = loaded["frames"][0]
    assert frame["samples"].dtype == np.float32 and frame["samples"].shape == (6, 80, 2)
    assert set(frame["courtesy"]) == {"1"} and frame["courtesy"]["1"]["kl"] == frame["observation"]["courtesy"]
    assert abs(frame["goals"]["prob"].sum() - frame["goals"]["mass"]) < 1e-5


def test_sparse_goals_keep_the_requested_mass():
    dist = GoalModel()._dist(_scene(), 10, 0)
    kept = sparse_goals(dist, top_mass=0.5)
    assert kept["mass"] >= 0.5 and len(kept["prob"]) < 50
    assert np.all(np.diff(kept["prob"]) <= 0)


def test_windows_without_neighbours_get_display_samples_off_the_metric_stream():
    scene = make_scene({"0": track((0, 0), (10, 0)), "1": track((-30, 40), (0, 0))})
    cfg = ResponsibilityConfig(n_safety_samples=4, window_stride=30)
    observations, record = run_scene(GoalModel(), scene, 0, cfg)
    frame = record["frames"][0]
    assert not frame["metric_samples"] and frame["samples"].shape[0] > 0
    assert observations[0].per_neighbour == {}


def test_offline_rendering_from_a_record(tmp_path):
    cfg = ResponsibilityConfig(n_safety_samples=6, window_stride=30)
    _, record = run_scene(GoalModel(), _scene(), 0, cfg)
    record["scene_file"] = "7"
    save_record(record, tmp_path / "7.pkl")
    levels = tmp_path / "levels.csv"
    levels.write_text("run,scene,agent_id,step,level,aggressive\n" + "".join(
        f"sdc,7,0,{f['step']},{i % 2},{i % 2}\n" for i, f in enumerate(record["frames"])))
    (gif, mp4), radius = vis.render_record(load_record(tmp_path / "7.pkl"), tmp_path / "video", levels,
                                           ego_heatmap=True)
    assert gif.exists() and radius >= 30.0
    assert len(list((tmp_path / "video" / "frames").glob("t_*.png"))) == len(record["frames"])


def test_live_settings_take_use_ooi_and_a_runs_filters(tmp_path):
    import json
    from dataclasses import asdict

    import pytest

    from responsibility.motion_filter import MotionFilterConfig

    args = SimpleNamespace(run=None, n_samples=40, horizon=20, stride=5, seed=0, motion_set="sampled", use_ooi=True,
                           courtesy_same_mode=None, courtesy_estimator="kl", demand_horizon=50, demand_rho=1.0)
    assert vis.config_from(args).use_ooi
    cfg = ResponsibilityConfig(filter=MotionFilterConfig(drivable_edges=True))
    (tmp_path / "config.json").write_text(json.dumps({"responsibility": asdict(cfg)}))
    run = SimpleNamespace(**{**vars(args), "run": str(tmp_path)})
    with pytest.raises(SystemExit, match="without --use-ooi"):
        vis.config_from(run)
    loaded = vis.config_from(SimpleNamespace(**{**vars(run), "use_ooi": False}))
    assert isinstance(loaded.filter, MotionFilterConfig) and loaded.filter.drivable_edges


def test_live_mode_loads_the_model_once_and_swaps_the_objects_of_interest(tmp_path, monkeypatch):
    import pickle

    import responsibility.models

    scene = make_scene({"0": track((0.0, 0.0), (10.0, 0.0)), "1": track((30.0, 4.0), (10.0, 0.0))})
    for stem in ("3", "5"):
        with open(tmp_path / f"{stem}.pkl", "wb") as f:
            pickle.dump(scene.to_description(), f)
    loads = []

    def load_model(args, device):
        loads.append(device)
        return GoalModel()

    monkeypatch.setattr(responsibility.models, "load_model", load_model)
    args = SimpleNamespace(scene=["3", "5"], scenes=str(tmp_path), agent=["sdc", "adv"], use_ooi=True, run=None,
                           device="cpu", n_samples=8, horizon=20, stride=20, seed=0, motion_set="sampled",
                           courtesy_same_mode=None, courtesy_estimator="kl", demand_horizon=50, demand_rho=1.0)
    records = list(vis.live_records(args))
    assert [name for name, _ in records] == ["3_sdc", "3_adv", "5_sdc", "5_adv"]
    assert loads == ["cpu"]  # once for all four
    assert all(len(r) == 1 for _, r in records)  # one estimator: one record each
    sdc, adv = records[0][1][0], records[1][1][0]
    assert sdc["agent_id"] == "0" and adv["agent_id"] == "1"
    neighbours = [set(f["observation"]["per_neighbour"]) for f in adv["frames"]]
    assert neighbours and all(n == {"0"} for n in neighbours)  # the roles swapped


def test_records_mark_the_goals_a_restricted_courtesy_kl_was_taken_over(tmp_path):
    cfg = ResponsibilityConfig(window_stride=20, courtesy_same_mode="path", courtesy_path_lateral=2.0,
                               courtesy_estimator="kl")
    _, record = run_scene(GoalModel(), _scene(), 0, cfg)
    pairs = [pair for f in record["frames"] for pair in f["courtesy"].values()]
    assert pairs
    for pair in pairs:
        for side in ("with", "without"):
            inside = pair[side]["in_support"]
            assert inside.dtype == bool and len(inside) == len(pair[side]["prob"])
        assert 0.0 < pair["with"]["support_mass"] < 1.0  # a 2 m band leaves part of the goals out
    plain = run_scene(GoalModel(), _scene(), 0, ResponsibilityConfig(window_stride=20, courtesy_estimator="kl"))[1]
    assert all("in_support" not in p["with"] for f in plain["frames"] for p in f["courtesy"].values())
    save_record(record, tmp_path / "r.pkl")
    (gif, _), _ = vis.render_record(load_record(tmp_path / "r.pkl"), tmp_path / "video")
    assert gif.exists()


def test_demand_courtesy_is_recorded_rendered_and_resumes_its_settings(tmp_path):
    import json
    from dataclasses import asdict

    from responsibility.demand import DemandConfig

    scene = make_scene({"0": track((0.0, 0.0), (10.0, 0.0), heading=0.0),
                        "1": track((30.0, -40.0), (0.0, 10.0), heading=np.pi / 2)})  # b crosses a's path
    cfg = ResponsibilityConfig(window_stride=20, motion_set="sampled", n_safety_samples=6,
                               courtesy_estimator="demand", use_ooi=True)
    observations, record = run_scene(GoalModel(), scene, 0, cfg)
    assert observations and record["frames"]
    entries = [e for f in record["frames"] for e in f["courtesy"].values()]
    assert entries and all(e["kind"] == "demand" and e["plans"].shape[0] == 6 for e in entries)
    for f in record["frames"]:
        for tid, e in f["courtesy"].items():
            assert f["observation"]["per_neighbour"][tid]["courtesy"] == e["value"] >= 0.0
            assert e["delta"].shape == e["weights"].shape == (6,) and abs(e["weights"].sum() - 1) < 1e-5
    save_record(record, tmp_path / "r.pkl")
    (gif, _), _ = vis.render_record(load_record(tmp_path / "r.pkl"), tmp_path / "video")
    assert gif.exists() and vis.unit_of(record["config"]) == "m/s^2"
    # a run's saved settings rebuild the demand configuration
    (tmp_path / "config.json").write_text(json.dumps({"responsibility": asdict(cfg)}))
    args = SimpleNamespace(run=str(tmp_path), use_ooi=True)
    loaded = vis.config_from(args)
    assert loaded.courtesy_estimator == "demand" and isinstance(loaded.demand, DemandConfig)


def test_demand_courtesy_skips_a_neighbour_in_the_same_lane_and_the_metric_stream_is_unchanged():
    scene = make_scene({"0": track((0.0, 0.0), (10.0, 0.0)), "1": track((20.0, 0.2), (5.0, 0.0))})
    kl = ResponsibilityConfig(n_safety_samples=6, use_ooi=True)
    demand = ResponsibilityConfig(n_safety_samples=6, use_ooi=True, courtesy_estimator="demand")
    a, b = (responsibility_at(GoalModel(), scene, 0, 10, c, torch.Generator().manual_seed(0)) for c in (kl, demand))
    assert b.per_neighbour["1"]["courtesy"] is None and b.per_neighbour["1"]["courtesy_relation"] == "follows"
    assert b.courtesy == 0.0 and a.safety == b.safety  # beta_s draws the same samples either way


def test_boltzmann_courtesy_rises_when_the_agent_crosses_the_neighbours_samples(tmp_path):
    # the agent (0) drives north through (15, 0) after 1.5 s, where the neighbour (1) samples (slow: 0.5-4 m in 2 s) end
    crossing = make_scene({"0": track((15.0, -15.0), (0.0, 10.0), heading=np.pi / 2),
                           "1": track((11.0, 0.0), (10.0, 0.0), heading=0.0)})
    apart = make_scene({"0": track((15.0, -80.0), (0.0, 10.0), heading=np.pi / 2),
                        "1": track((11.0, 0.0), (10.0, 0.0), heading=0.0)})
    cfg = ResponsibilityConfig(window_stride=20, n_safety_samples=6, n_courtesy_samples=5,
                               courtesy_estimator="boltzmann", use_ooi=True)
    record = {}
    near = responsibility_at(GoalModel(), crossing, 0, 10, cfg, torch.Generator().manual_seed(0), record=record)
    far = responsibility_at(GoalModel(), apart, 0, 10, cfg, torch.Generator().manual_seed(0))
    assert near.per_neighbour["1"]["courtesy"] > 0.0 == far.per_neighbour["1"]["courtesy"]  # skipped: > 20 m apart
    kl, pooled, log_p1, log_p2 = record["courtesy"][1]
    assert pooled.shape == (10, 20, 2) and abs(np.exp(log_p1).sum() - 1) < 1e-5
    _, rec = run_scene(GoalModel(), crossing, 0, cfg)
    save_record(rec, tmp_path / "r.pkl")
    (gif, _), _ = vis.render_record(load_record(tmp_path / "r.pkl"), tmp_path / "video")
    assert gif.exists() and all(e["kind"] == "boltzmann" for f in rec["frames"] for e in f["courtesy"].values())


def test_comparison_renders_one_row_per_estimator_with_the_same_beta_s(tmp_path, monkeypatch):
    import pickle

    import responsibility.models

    scene = make_scene({"0": track((15.0, -15.0), (0.0, 10.0), heading=np.pi / 2),
                        "1": track((11.0, 0.0), (10.0, 0.0), heading=0.0)})
    with open(tmp_path / "3.pkl", "wb") as f:
        pickle.dump(scene.to_description(), f)
    monkeypatch.setattr(responsibility.models, "load_model", lambda args, device: GoalModel())
    args = SimpleNamespace(scene=["3"], scenes=str(tmp_path), agent=["sdc"], use_ooi=True, run=None,
                           device="cpu", n_samples=6, horizon=20, stride=20, seed=0, motion_set="sampled",
                           courtesy_same_mode=None, courtesy_estimator=["kl", "boltzmann", "demand"],
                           demand_horizon=50, demand_rho=1.0)
    [(name, records)] = list(vis.live_records(args))
    assert [vis.estimator_of(r) for r in records] == ["kl", "boltzmann", "demand"]
    safety = [[f["observation"]["safety"] for f in r["frames"]] for r in records]
    assert safety[0] == safety[1] == safety[2]  # courtesy draws from its own generator
    (gif, _), _ = vis.render_comparison_records(records, tmp_path / "compare")
    assert gif.exists() and len(list((tmp_path / "compare" / "frames").glob("t_*.png"))) == len(records[0]["frames"])


def test_courtesy_panels_draw_the_neighbour_where_it_is():
    """The demand and Boltzmann panels frame the plans and the agent's future, so the view centre is not
    the neighbour's position; its box must still be drawn at that position."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    scene = make_scene({"0": track((15.0, -15.0), (0.0, 10.0), heading=np.pi / 2),
                        "1": track((11.0, 0.0), (10.0, 0.0), heading=0.0)})
    for est in ("demand", "boltzmann"):
        cfg = ResponsibilityConfig(window_stride=20, n_safety_samples=6, n_courtesy_samples=5,
                                   courtesy_estimator=est, use_ooi=True)
        _, record = run_scene(GoalModel(), scene, 0, cfg)
        frame = next(f for f in record["frames"] if "1" in f["courtesy"])
        fig, axes = plt.subplots(1, 2)
        vis.draw_courtesy(axes, scene, 0, frame, {}, 50.0, 20, DemandConfig() if est == "demand" else None)
        boxes = [p.get_xy()[:4].mean(0) for p in axes[0].patches if isinstance(p, Polygon)]
        plt.close(fig)
        assert any(np.allclose(c, scene.position[1, frame["step"], :2], atol=1e-3) for c in boxes), est


def test_courtesy_panel_says_why_courtesy_is_missing():
    import matplotlib.pyplot as plt

    scene = make_scene({"0": track((0.0, 0.0), (10.0, 0.0)), "1": track((5.0, 3.5), (10.0, 0.0))})
    cases = {"no prediction": {"1": {"type": "VEHICLE", "courtesy": None, "safety": 0.0}},
             "too far apart": {"1": {"type": "VEHICLE", "courtesy": 0.0, "safety": 0.0}},
             "vehicles only": {"1": {"type": "PEDESTRIAN", "courtesy": None, "safety": 0.0}},
             "no neighbour": {}}
    for expected, pn in cases.items():
        fig, axes = plt.subplots(1, 2)
        frame = {"step": 10, "observation": {"per_neighbour": pn}, "courtesy": {}}
        vis.draw_courtesy(axes, scene, 0, frame, {}, 50.0, 20)
        text = " ".join(t.get_text() for t in axes[0].texts).replace("\n", " ")
        plt.close(fig)
        assert expected in text, (expected, text)


def test_boltzmann_courtesy_samples_far_pairs_without_the_skip_distance():
    apart = make_scene({"0": track((15.0, -80.0), (0.0, 10.0), heading=np.pi / 2),
                        "1": track((11.0, 0.0), (10.0, 0.0), heading=0.0)})
    record = {}
    cfg = ResponsibilityConfig(n_safety_samples=6, n_courtesy_samples=5, courtesy_estimator="boltzmann",
                               use_ooi=True, courtesy_skip_distance=None)
    obs = responsibility_at(GoalModel(), apart, 0, 10, cfg, torch.Generator().manual_seed(0), record=record)
    assert 1 in record["courtesy"] and record["courtesy"][1][1] is not None  # sampled, not skipped
    assert obs.per_neighbour["1"]["courtesy"] == 0.0  # 80 m apart: nothing to block
