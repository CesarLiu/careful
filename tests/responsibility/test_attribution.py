import numpy as np
import pytest

from responsibility import attribution as attr
from responsibility.forced_merge import cut_in, following_demand, same_direction, tailgate
from responsibility.forced_yield import pair_priority
from responsibility.scene import Scene
from tests.responsibility.conftest import N_STEPS, make_scene, track

DT = 0.1
SPEED = 15.0


def road(*lanes_y, branch=None):
    """Straight lanes along x at the given y, road edges outside them, and an optional branch lane
    [(x0, y0), (x1, y1)]."""
    feats = {}
    for n, y in enumerate(lanes_y):
        feats[f"l{n}"] = {"type": "LANE_SURFACE_STREET",
                          "polyline": np.array([[x, y, 0.0] for x in np.arange(-50.0, 260.0, 10.0)])}
    lo, hi = min(lanes_y) - 1.8, max(lanes_y) + 1.8
    for name, y in (("e0", lo), ("e1", hi)):
        feats[name] = {"type": "ROAD_EDGE_BOUNDARY", "polyline": np.array([[x, y, 0.0] for x in np.arange(-50.0, 260.0, 10.0)])}
    if branch is not None:
        (x0, y0), (x1, y1) = branch
        xs = np.linspace(x0, x1, 12)
        feats["b"] = {"type": "LANE_SURFACE_STREET", "polyline": np.stack([xs, np.linspace(y0, y1, 12), 0 * xs], 1)}
    return feats


def lane_change(x0, y0, y1, t_start, t_end, speed=SPEED):
    """A track along x at constant speed that moves from y0 to y1 between t_start and t_end (s)."""
    t = (np.arange(N_STEPS) - 10) * DT
    x = x0 + speed * t
    w = np.clip((t - t_start) / (t_end - t_start), 0, 1)
    y = y0 + (y1 - y0) * (3 * w ** 2 - 2 * w ** 3)
    vy = np.gradient(y, DT)
    tr = track((x0, y0), (speed, 0.0))
    tr["state"]["position"][:, 0], tr["state"]["position"][:, 1] = x, y
    tr["state"]["velocity"][:, 1] = vy
    tr["state"]["heading"][:] = np.arctan2(vy, speed)
    return tr


def scene_of(tracks, feats):
    return make_scene(tracks, sdc="0", ooi=("0", "1"), map_features=feats, lights={})


def test_cut_in_by_the_agent_is_attributed_to_it():
    s = scene_of({"0": lane_change(15.0, 3.5, 0.0, 2.0, 4.0), "1": track((0.0, 0.0), (SPEED, 0.0))}, road(0.0, 3.5))
    a = attr.attribute(s, 0, 1)
    assert a.kind == "join" and a.actor == "agent" and a.evidence["ahead"] == "agent"
    assert a.evidence["toward_agent"] > 3.0 and abs(a.evidence["toward_other"]) < 0.5


def test_moving_in_behind_is_attributed_to_the_car_behind():
    s = scene_of({"0": track((15.0, 0.0), (SPEED, 0.0)), "1": lane_change(0.0, 3.5, 0.0, 2.0, 4.0)}, road(0.0, 3.5))
    a = attr.attribute(s, 0, 1)
    assert a.kind == "join" and a.actor == "other" and a.evidence["ahead"] == "agent"


def test_a_lane_branching_across_the_road_does_not_hide_the_move():
    # the agent follows a lane that runs diagonally from the left lane into the right one
    s = scene_of({"0": lane_change(15.0, 3.5, 0.0, 2.0, 4.0), "1": track((0.0, 0.0), (SPEED, 0.0))},
                 road(0.0, 3.5, branch=((40.0, 3.5), (90.0, 0.0))))
    assert attr.attribute(s, 0, 1).actor == "agent"


def test_following_is_the_followers_doing():
    s = scene_of({"0": track((0.0, 0.0), (SPEED, 0.0)), "1": track((20.0, 0.0), (SPEED, 0.0))}, road(0.0, 3.5))
    a = attr.attribute(s, 0, 1)
    assert a.kind == "following" and a.actor == "agent"


def test_crossing_without_a_holder_is_unknown():
    s = scene_of({"0": track((-20.0, 0.0), (10.0, 0.0)), "1": track((0.0, -20.0), (0.0, 10.0), heading=np.pi / 2)},
                 road(0.0))
    a = attr.attribute(s, 0, 1)
    assert a.kind == "crossing" and a.actor == "unknown"


def test_oncoming_cars_passing_a_lane_apart_are_no_crossing():
    # their tracks never cross, so the conflict is their closest approach (WOMD scenes 115beb8f, 83d69d7f)
    s = scene_of({"0": track((-20.0, 0.0), (3.0, 0.0)), "1": track((20.0, 3.5), (-3.0, 0.0), heading=np.pi)},
                 road(0.0, 3.5))
    assert pair_priority(s, 0, 1).conflict.kind == "closest"
    assert attr.crossing(s, 0, 1) is None
    assert attr.attribute(s, 0, 1).kind != "crossing"


def test_following_demand_equals_the_braking_at_one_response_time():
    # equal speeds and a headway of rho (1 s): the rear car must brake exactly as hard as the front car
    gap = SPEED * 1.0
    length = 4.8
    s = scene_of({"0": track((0.0, 0.0), (SPEED, 0.0)), "1": track((gap + length, 0.0), (SPEED, 0.0))}, road(0.0))
    demand, g, headway = following_demand(s, 0, 1, np.array([20]))
    assert g[0] == pytest.approx(gap, abs=1e-3)
    assert headway[0] == pytest.approx(1.0, abs=1e-3)
    assert demand[0] == pytest.approx(4.0, rel=1e-3)


def test_a_tighter_cut_in_demands_more():
    def demand(x_rear):
        s = scene_of({"0": lane_change(15.0, 3.5, 0.0, 2.0, 4.0), "1": track((x_rear, 0.0), (SPEED, 0.0))}, road(0.0, 3.5))
        return same_direction(s, 0, 1)["cut-in"].demand
    assert demand(0.0) > demand(-10.0) > 0


def test_tailgating_right_after_being_cut_in_on_is_excused():
    # the other cuts in 6 m ahead of the agent at about 3 s: the short gap after it is not the agent's doing
    s = scene_of({"0": track((0.0, 0.0), (SPEED, 0.0)), "1": lane_change(10.8, 3.5, 0.0, 2.0, 3.5)}, road(0.0, 3.5))
    k = attr.joins(s, 1, 0)[0]
    excused = tailgate(s, 0, 1, excused_from=[k])
    blamed = tailgate(s, 0, 1)
    assert blamed.demand is not None and blamed.demand > 4.0
    assert excused.demand is None or excused.step >= k + 30


def test_advance_moves_the_car_along_its_own_track():
    from scripts.responsibility.make_cut_in import advance
    s = scene_of({"0": track((0.0, 0.0), (SPEED, 0.0)), "1": track((30.0, 0.0), (SPEED, 0.0))}, road(0.0))
    position, heading, velocity, valid = advance(s, 0, 10)
    assert np.allclose(position[20], s.position[0, 30, :2])
    assert np.allclose(position[-1, 0], s.position[0, -1, 0] + SPEED * 1.0, atol=1e-3)  # driven on past the log
    assert valid.all()


def test_the_judge_flags_a_tight_cut_in_and_not_a_lawful_follower():
    from responsibility.aggression_judge import judge
    cal = {"cut-in": {"threshold": 5.0, "quantile": 0.9, "reference": [1.0, 2.0, 3.0, 5.0]},
           "tailgate": {"threshold": 4.5, "quantile": 0.9, "reference": [1.0, 2.0, 4.5]},
           "crossing": {"threshold": 2.5, "quantile": 0.9, "reference": [0.5, 2.5]}}
    tight = scene_of({"0": lane_change(8.0, 3.5, 0.0, 2.0, 4.0, speed=SPEED - 1.0), "1": track((0.0, 0.0), (SPEED, 0.0))},
                     road(0.0, 3.5))
    j = judge(tight, 0, 1, cal)
    assert j.verdict == "aggressive" and j.kind == "cut-in"
    assert judge(tight, 1, 0, cal).verdict == "not aggressive"  # the car cut in on is not the aggressor
    calm = scene_of({"0": track((0.0, 0.0), (SPEED, 0.0)), "1": track((40.0, 0.0), (SPEED, 0.0))}, road(0.0, 3.5))
    assert judge(calm, 0, 1, cal).verdict == "not aggressive"
