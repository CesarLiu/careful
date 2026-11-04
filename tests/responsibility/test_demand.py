"""Courtesy as demand (responsibility/demand.py) at a synthetic crossing: b
drives east along y = 0 at 10 m/s and reaches x = 0 at step 50; a drives
north along x = 0 at 10 m/s and reaches y = 0 at step ``a_at``. b's plan is
its own constant-velocity continuation, as if a were not there."""

import numpy as np
import pytest

from responsibility.demand import DT, DemandConfig, braking_demand, demand_at, lane_relation, retime
from tests.responsibility.conftest import make_scene, track

B_AT = 50


def _scene(a_at: int, a_track=None):
    b = track((-10.0 * (B_AT - 10) * DT, 0.0), (10.0, 0.0), heading=0.0)
    a = a_track or track((0.0, -10.0 * (a_at - 10) * DT), (0.0, 10.0), heading=np.pi / 2)
    return make_scene({"0": b, "1": a}, sdc="0", ooi=("0", "1"))


def _plan(scene, b: int, step: int, n: int = 80):
    """b's constant-velocity continuation from ``step`` [1, n, 2]."""
    t = np.arange(1, n + 1)[:, None] * DT
    return (scene.position[b, step, :2] + t * scene.velocity[b, step])[None]


def test_retime_brakes_along_the_plan():
    path = np.stack([np.arange(81) * 1.0, np.zeros(81)], axis=1)  # 10 m/s east
    pos, head = retime(path, 0.0, np.array([0.0, 5.0]), react=10)
    assert np.allclose(pos[0], path[1:])  # no braking: the plan itself
    # 5 m/s^2 from step 10 (x = 10 m) at 10 m/s stops 10 m further on, after 2 s
    assert pos[1, 9, 0] == pytest.approx(10.0) and pos[1, -1, 0] == pytest.approx(20.0)
    assert np.all(np.diff(pos[1, :, 0]) >= -1e-9) and np.allclose(head, 0.0)


def test_demand_grows_as_a_crosses_closer_ahead_of_b():
    cfg = DemandConfig()

    def demand(a_at, k):
        sc = _scene(a_at)
        return demand_at(_plan(sc, 0, k), None, sc, 1, 0, k, cfg)
    # judged 3 s before b would cross (k = 20): a long gone, then a crossing 0.6 s, 0.2 s
    # before and 0.2 s after b would need gentle, growing braking
    assert demand(10, 20).demand == 0.0 and demand(10, 20).p_conflict == 0.0
    assert 0.0 < demand(44, 20).demand < demand(48, 20).demand < demand(52, 20).demand < 4.0
    # the same crossings judged 2 s before: past RSS's 4 m/s^2; 1 s before: unavoidable
    assert demand(46, 30).demand > 4.0
    late = demand(46, 40)
    assert late.p_unavoidable == 1.0 and late.demand == cfg.unavoidable
    # a crossing 0.8 s after b would: b has passed, no demand
    assert demand(58, 20).demand == 0.0


def test_demand_is_directional():
    """b yielded in the log (braked to a stop before the crossing at 4.5 m/s^2 from step 20);
    its plan without a did not. a's motion demanded that braking; b's demanded nothing of a."""
    sc = _scene(48)
    t = np.clip(np.arange(91) - 20, 0, None) * DT
    t_stop = 10.0 / 4.5
    tb = np.minimum(t, t_stop)
    x = np.where(np.arange(91) < 20, sc.position[0, :, 0], -30.0 + 10.0 * tb - 4.5 * tb ** 2 / 2)
    sc.position[0, :, 0] = x
    sc.velocity[0, :, 0] = np.where(np.arange(91) < 20, 10.0, np.maximum(10.0 - 4.5 * t, 0.0))
    k = 20
    plan_b = (sc.position[0, k, :2] + np.arange(1, 81)[:, None] * DT * [10.0, 0.0])[None]
    on_b = demand_at(plan_b, None, sc, 1, 0, k, DemandConfig())
    on_a = demand_at(_plan(sc, 1, k), None, sc, 0, 1, k, DemandConfig())
    assert on_b.demand > 0.0 and on_a.demand == 0.0


def test_same_lane_is_not_taken():
    b = track((0.0, 0.0), (10.0, 0.0))
    ahead = track((20.0, 0.3), (5.0, 0.0))  # slower leader in b's lane
    beside = track((20.0, 3.5), (5.0, 0.0))  # the next lane
    sc = make_scene({"0": b, "1": ahead, "2": beside}, ooi=("0", "1"))
    assert lane_relation(sc, 1, 0, 10) == "leads" and lane_relation(sc, 0, 1, 10) == "follows"
    assert lane_relation(sc, 2, 0, 10) is None
    assert demand_at(_plan(sc, 0, 10), None, sc, 1, 0, 10, DemandConfig()) is None


def test_weights_and_invalid_future():
    sc = _scene(45)
    plans = np.concatenate([_plan(sc, 0, 10), _plan(sc, 0, 10) * 0.0 + sc.position[0, 10, :2]])  # go / stand
    d = demand_at(plans, np.array([0.25, 0.75]), sc, 1, 0, 10, DemandConfig())
    assert d.per_member[1] == 0.0 and d.p_conflict == pytest.approx(0.25)
    assert d.mean == pytest.approx(0.25 * d.per_member[0])
    gone = braking_demand(plans[:, :50], sc.position[0, 10, :2], 0.0, sc.shape_at(0, 10),
                          sc.position[1, 11:61, :2], sc.heading[1, 11:61], np.zeros(50, bool),
                          sc.shape_at(1, 10), DemandConfig())
    assert np.all(gone == 0.0)
