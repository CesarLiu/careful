"""gpucat/reactive.py on stand-in worlds (no GPUDrive): straight logged
drives along x, and an SDC placed by the test. The stand-in steps as
GPUDrive does: after t steps the experts sit at row t-1 of the trajectory
rows."""

import math

import torch

from gpucat.geometry import boxes_overlap
from gpucat.reactive import POS, ROW_COLUMNS, TRAJ_LEN, VALID, VEL, YAW, ReactiveSettings, ReactiveTraffic, along, \
    logged_paths

T = TRAJ_LEN
SIZE = (4.5, 2.0)


def rows_of(tracks):
    """Trajectory rows [1, A, ROW_COLUMNS] of tracks: (x [T], y [T], valid [T]) per slot, heading +x."""
    A = len(tracks)
    rows = torch.zeros(1, A, ROW_COLUMNS)
    for a, (x, y, v) in enumerate(tracks):
        p = torch.stack([torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)], -1)
        p = torch.where(torch.as_tensor(v)[:, None], p, torch.full_like(p, -1e4))
        rows[0, a, POS:POS + 2 * T] = p.flatten()
        speed = torch.cat([torch.diff(p[:, 0]), torch.diff(p[:, 0])[-1:]]) / 0.1
        rows[0, a, VEL:VEL + 2 * T:2] = speed
        rows[0, a, YAW:YAW + T] = 0.0
        rows[0, a, VALID:VALID + T] = torch.as_tensor(v, dtype=torch.float32)
    return rows


def straight(x0, speed, y=0.0, valid=None):
    x = x0 + speed * 0.1 * torch.arange(T, dtype=torch.float32)
    return x, torch.full((T,), y), torch.ones(T, dtype=torch.bool) if valid is None else valid


def run(tracks, ego_at, settings=ReactiveSettings(), reactive=True):
    """Steps a world with these logged tracks (slot 0 the SDC's log) and the
    SDC at ego_at(t) [2] after t steps. Returns the traffic, the rows and,
    per step, whether the SDC's box overlaps each agent's, and the agents'
    positions [T-1, A, 2]."""
    rows = rows_of(tracks)
    traj = rows.clone()
    A = len(tracks)
    tr = ReactiveTraffic(settings, "cpu")
    tr.load(rows, torch.tensor(SIZE).expand(1, A, 2), torch.tensor([[1] * A]), torch.tensor([A]))
    hits, places = [], []
    for t in range(T - 1):
        e = torch.as_tensor(ego_at(t), dtype=torch.float32)[None]
        if reactive:
            tr.step(traj, torch.tensor([t]), torch.tensor([True]), e, torch.zeros(1), torch.tensor([False]))
        p = traj[0, :, POS + 2 * t:POS + 2 * t + 2]  # the experts after step t+1
        e1 = torch.as_tensor(ego_at(t + 1), dtype=torch.float32)
        ov = boxes_overlap(e1[None], torch.zeros(1), torch.tensor([SIZE]), p[None], traj[0, :, YAW + t][None],
                           torch.tensor(SIZE).expand(1, A, 2))[0]
        ov &= traj[0, :, VALID + t] > 0.5
        ov[0] = False
        hits.append(ov)
        places.append(p.clone())
    return tr, rows, traj, torch.stack(hits), torch.stack(places)


def logged(rows, a):
    return rows[0, a, POS:POS + 2 * T].view(T, 2)


def test_along_interpolates_holds_and_extrapolates():
    S = torch.tensor([[0.0, 1.0, 1.0, 3.0]])
    P = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 2.0]]])
    Y = torch.tensor([[0.0, 0.0, math.pi / 2, math.pi / 2]])
    p, y = along(S, P, Y, torch.tensor([3]), torch.tensor([[0.5, 1.0, 2.0, 4.0]]))
    assert torch.allclose(p[0], torch.tensor([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 3.0]]), atol=1e-5)
    assert math.isclose(float(y[0, 2]), math.pi / 2, abs_tol=1e-5) and math.isclose(float(y[0, 3]), math.pi / 2, abs_tol=1e-5)


def test_logged_paths_fill_gaps_and_hold_the_last_speed():
    valid = torch.ones(T, dtype=torch.bool)
    valid[:5] = False
    valid[20:23] = False
    valid[60:] = False
    rows = rows_of([straight(0.0, 10.0, valid=valid)])
    P, Y, S, V, first, last, v_at = logged_paths(rows)
    assert int(first) == 5 and int(last) == 59
    assert torch.allclose(P[0, 0, 21], torch.tensor([21.0, 0.0]), atol=1e-4)  # interpolated over the gap
    assert torch.allclose(P[0, 0, 0], P[0, 0, 5]) and torch.allclose(P[0, 0, 80], P[0, 0, 59])
    assert math.isclose(float(S[0, 0, 59] - S[0, 0, 5]), 54.0, abs_tol=1e-3)
    assert math.isclose(float(v_at[0, 0, 80]), 10.0, abs_tol=1e-3)  # after the log ends: its last speed


def test_sdc_on_its_log_leaves_the_replay_untouched():
    """A follower 1 s behind the SDC (closer than the rule's gap at 10 m/s):
    with the SDC on its log, nothing is off its log and every row is the log."""
    tracks = [straight(30.0, 10.0), straight(15.5, 10.0)]
    sdc = logged(rows_of(tracks), 0)
    tr, rows, traj, hits, _ = run(tracks, lambda t: sdc[max(t - 1, 0)])
    assert torch.equal(traj, rows)
    assert not bool(tr.ever.any())


def test_follower_stops_behind_a_stopped_sdc():
    tracks = [straight(50.0, 10.0), straight(20.0, 10.0)]
    tr, rows, traj, hits, places = run(tracks, lambda t: (50.0, 0.0))
    _, _, _, replay_hits, _ = run(tracks, lambda t: (50.0, 0.0), reactive=False)
    assert bool(replay_hits[:, 1].any())  # the log replay runs into the SDC
    assert not bool(hits[:, 1].any())
    bumper_gap = 50.0 - float(places[-1, 1, 0]) - SIZE[0]
    assert 1.0 <= bumper_gap <= 6.0
    assert bool(tr.ever[0, 1]) and bool((traj[0, 1, VALID:VALID + T] > 0.5).all())


def test_cascade_leader_and_follower_both_stop():
    tracks = [straight(80.0, 10.0), straight(40.0, 10.0), straight(25.0, 10.0)]
    tr, rows, traj, hits, places = run(tracks, lambda t: (70.0, 0.0))
    assert not bool(hits.any())
    gap = float(places[-1, 1, 0] - places[-1, 2, 0]) - SIZE[0]
    assert gap >= 1.0  # the follower stops behind the leader that stopped for the SDC
    assert bool(tr.ever[0, 1]) and bool(tr.ever[0, 2])


def test_cut_in_closer_than_the_stopping_distance_is_still_hit():
    """At 15 m/s the agent needs 18.75 m to stop at 6 m/s^2; the SDC appears
    standing 5 m ahead of its front bumper."""
    tracks = [straight(500.0, 0.0), straight(0.0, 15.0)]
    rows = rows_of(tracks)
    agent = logged(rows, 1)

    def ego(t):
        if t < 20:
            return (500.0, 50.0)  # far away
        return (float(agent[19, 0]) + SIZE[0] + 5.0, 0.0)

    tr, _, _, hits, _ = run(tracks, ego)
    assert bool(hits[:, 1].any())
    assert bool(tr.ever[0, 1])  # it braked, but could not stop in time


def test_sdc_behind_the_agent_is_ignored():
    tracks = [straight(0.0, 10.0), straight(20.0, 10.0)]
    sdc = logged(rows_of(tracks), 0)
    tr, rows, traj, _, _ = run(tracks, lambda t: (float(sdc[max(t - 1, 0), 0]) + 9.0, 0.0))  # tailgating, off its log
    assert torch.equal(traj, rows)
    assert not bool(tr.ever.any())


def test_agent_catches_up_and_rejoins_its_log():
    tracks = [straight(500.0, 0.0), straight(0.0, 8.0)]
    rows = rows_of(tracks)

    def ego(t):
        return (35.0, 0.0) if 10 <= t < 30 else (500.0, 80.0)  # blocks the lane for 2 s, then leaves

    tr, rows, traj, hits, places = run(tracks, ego)
    assert not bool(hits[:, 1].any())
    assert bool(tr.ever[0, 1]) and not bool(tr.off[0, 1])
    assert torch.allclose(places[-1, 1], logged(rows, 1)[T - 2])  # back on its log by the end


def test_partner_classes_in_the_sdc_frame():
    from gpucat.env import PARTNER_CLASSES, partner_class

    pos = torch.tensor([[[0.0, 0.0], [-5.0, 0.0], [5.0, 0.0], [0.5, 2.0], [5.0, 0.5], [3.0, 3.0]]])
    yaw = torch.tensor([[0.0, 0.1, -0.1, 0.0, math.pi, math.pi / 2]])
    names = [PARTNER_CLASSES[int(partner_class(pos[:, 0], yaw[:, 0], torch.tensor([SIZE]), pos, yaw, torch.tensor([j])))]
             for j in range(1, 6)]
    assert names == ["follower", "leader", "side", "opposite", "crossing"]
