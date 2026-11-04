"""Courtesy as demand: how hard agent a's actual motion would have made a
neighbour b brake, had b driven what it intended without a.

The KL courtesy of metrics.py (Hsu et al., Eq. 4) compares b's predicted
goals with and without a in the scene, so it can only be as large as the
motion model's reaction to a. CAT's DenseTNT hardly reacts: 0.5 s before
CAT's attacker overlaps the ego, the ego's predicted progress changes by a
median +0.07 m and the KL is 0.061 against 0.042 nats in the log
(scripts/responsibility/probe_courtesy.py). And the goal KL is blind to
timing: a left turn across b's path leaves b's goals on the same lane, only
later.

This measure asks the model only for what it does predict well, b's intent,
and lets a's actual future (logged, CAT's, or a policy's) do the rest. At
context step k:

    M_b      b's motion set predicted without a (trajectories tau_i with
             probabilities w_i; b's history up to k, a left out)
    xi_a     a's actual future over the next H steps
    delta_i  the smallest constant deceleration (m/s^2) which, applied from
             k + rho on along tau_i's own path, keeps b's footprint clear of
             a's along xi_a for H steps; 0 when tau_i is clear as it is
    beta_d(a, b) = CVaR_alpha[ delta_i ],  tau_i ~ M_b             [m/s^2]

Re-timed, b keeps tau_i's path and goes no faster than tau_i: its arc
length is min(s_i(t), s_i(r) + v_r (t - r) dt - delta ((t - r) dt)^2 / 2),
held once it stops, where r = rho / dt and v_r is tau_i's speed at r. The
footprints are 3-circle covers (interaction._footprint_gap). delta runs
over a grid up to ``brake_max``; a member no braking up to it clears (a
too close, or coming from behind, where braking does not help) counts as
``unavoidable``.

The reaction time rho (RSS's 1 s, as in right_of_way.py) keeps a from being
charged with braking b could only have done before a's motion showed, and
lets an entry close ahead of b register as unavoidable. Where a and b
share a lane at k (headings within 45 degrees, lateral offset in b's frame
below half their widths summed) the measure is not taken: a leader slowing
down demands braking of its follower, but keeping the distance is the
follower's duty (CVC 21703; the rear-end rule of blame.py), and a follower
closing in is beta_s's case. A cut-in, from the next lane at k, is taken.

beta_d is directional, unlike beta_s's shared distance: it charges a for b's
adaptation, never b for a's. Who held the right of way is not used here;
compute_demand.py records right_of_way.py's holder alongside, so that a
demand on a b that owed a the right of way can be discounted.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from responsibility.interaction import _footprint_gap
from responsibility.scene import Scene

DT = 0.1  # s per step
SAME_LANE_HEADING = np.deg2rad(45.0)  # rad: headings closer than this share a direction of travel


@dataclass
class DemandConfig:
    horizon: int = 50  # 10 Hz steps of a's future judged (5 s)
    rho: float = 1.0  # s: b follows its plan unchanged this long (RSS's response time)
    decel_step: float = 0.25  # m/s^2: grid of the decelerations tried
    brake_max: float = 9.0  # m/s^2: the hardest braking tried (motion_filter.py's max_decel)
    unavoidable: float = 10.0  # m/s^2: the value of a member no braking up to brake_max clears
    margin: float = 0.0  # m: footprint gap (circle covers) below which b and a conflict
    cvar_alpha: float = 0.1  # upper-tail CVaR level, as beta_s's

    @property
    def grid(self) -> np.ndarray:
        return np.arange(0.0, self.brake_max + 1e-9, self.decel_step)


@dataclass
class Demand:
    demand: float  # beta_d: CVaR_alpha of the members' deceleration (m/s^2)
    mean: float  # their probability-weighted mean (m/s^2)
    p_conflict: float  # probability of the members that conflict with a unchanged
    p_unavoidable: float  # probability of the members no braking up to brake_max clears
    horizon: int  # steps judged
    per_member: np.ndarray  # [N] each member's deceleration (unavoidable ones: cfg.unavoidable)


def lane_relation(scene: Scene, a: int, b: int, step: int) -> Optional[str]:
    """"leads" / "follows" when a is ahead of / behind b in b's lane at
    ``step`` (headings within SAME_LANE_HEADING, lateral offset in b's
    frame below half their widths summed), else None."""
    dh = (scene.heading[a, step] - scene.heading[b, step] + np.pi) % (2 * np.pi) - np.pi
    if abs(dh) >= SAME_LANE_HEADING:
        return None
    h = float(scene.heading[b, step])
    d = scene.position[a, step, :2] - scene.position[b, step, :2]
    lon, lat = d[0] * np.cos(h) + d[1] * np.sin(h), -d[0] * np.sin(h) + d[1] * np.cos(h)
    half = (scene.shape_at(a, step)[1] + scene.shape_at(b, step)[1]) / 2
    if abs(lat) >= half:
        return None
    return "leads" if lon > 0 else "follows"


def retime(path: np.ndarray, start_heading: float, decels: np.ndarray, react: int):
    """The plan ``path`` [T + 1, 2] (b's position at k, then its planned
    positions at k + 1 .. k + T) re-timed along itself for each deceleration
    of ``decels`` [D] applied from step ``react`` on: positions [D, T, 2]
    and headings [D, T]. decels[d] == 0 leaves the plan unchanged."""
    seg = np.diff(path, axis=0)  # [T, 2]
    length = np.linalg.norm(seg, axis=-1)
    s = np.concatenate([[0.0], np.cumsum(length)])  # [T + 1] planned arc length at k .. k + T
    T = len(path) - 1
    t = np.arange(1, T + 1)
    r = min(react, T)
    v_r = (s[max(r, 1)] - s[max(r, 1) - 1]) / DT  # the plan's speed when braking begins
    tau = np.clip(t - r, 0, None) * DT  # [T] s since braking began
    with np.errstate(divide="ignore", invalid="ignore"):
        stop_time = np.where(decels > 0, v_r / decels, np.inf)  # [D]
    tb = np.minimum(tau[None, :], stop_time[:, None])  # [D, T]
    brake = s[r] + v_r * tb - decels[:, None] * tb ** 2 / 2
    progress = np.where(decels[:, None] > 0, np.minimum(s[1:][None, :], brake), s[1:][None, :])  # [D, T]
    # heading of the segment each point lies on; a standing plan keeps the last heading it had
    seg_heading = np.arctan2(seg[:, 1], seg[:, 0])
    moving = length > 1e-3
    idx = np.where(moving, np.arange(T), -1)
    idx = np.maximum.accumulate(idx)
    seg_heading = np.where(idx >= 0, seg_heading[np.maximum(idx, 0)], start_heading)
    pos = np.stack([np.interp(progress, s, path[:, 0]), np.interp(progress, s, path[:, 1])], axis=-1)
    k = np.clip(np.searchsorted(s, progress, side="right") - 1, 0, T - 1)  # segment index of each point
    return pos, seg_heading[k]


def braking_demand(plans: np.ndarray, start: np.ndarray, start_heading: float, shape_b: np.ndarray,
                   a_pos: np.ndarray, a_heading: np.ndarray, a_valid: np.ndarray, shape_a: np.ndarray,
                   cfg: DemandConfig) -> np.ndarray:
    """Each plan's deceleration delta_i (m/s^2; cfg.unavoidable where no
    braking up to cfg.brake_max clears a): ``plans`` [N, H, 2] b's planned
    positions at k + 1 .. k + H, ``start`` [2] its position at k; a's actual
    positions [H, 2], headings [H] and validity [H] over the same steps;
    shapes (length, width)."""
    n, horizon = plans.shape[:2]
    grid = cfg.grid
    react = int(round(cfg.rho / DT))
    f = torch.float32
    out = np.full(n, cfg.unavoidable)
    if not np.any(a_valid):
        return np.zeros(n)
    pos, head = zip(*(retime(np.concatenate([start[None], p]), start_heading, grid, react) for p in plans))
    pos = torch.as_tensor(np.concatenate(pos), dtype=f)  # [N * D, H, 2]
    head = torch.as_tensor(np.concatenate(head), dtype=f)
    gap = _footprint_gap(torch.as_tensor(a_pos, dtype=f), torch.as_tensor(a_heading, dtype=f),
                         torch.as_tensor(np.asarray(shape_a)[None], dtype=f), pos, head,
                         torch.as_tensor(np.tile(shape_b, (len(pos), 1)), dtype=f))  # [N * D, H]
    gap = gap.masked_fill(~torch.as_tensor(a_valid)[None], float("inf"))
    clear = (gap.amin(-1) > cfg.margin).numpy().reshape(n, len(grid))
    first = np.argmax(clear, axis=1)
    ok = clear.any(axis=1)
    out[ok] = grid[first[ok]]
    return out


def demand_at(plans: np.ndarray, weights: Optional[np.ndarray], scene: Scene, a: int, b: int, step: int,
              cfg: DemandConfig) -> Optional[Demand]:
    """beta_d(a, b) at context step ``step`` from b's motion set without a
    (``plans`` [N, >= H, 2] in the scene frame, positions at step + 1 ...;
    ``weights`` [N] or None for i.i.d. samples), against a's future in
    ``scene``. None where a and b share a lane at ``step`` (lane_relation)."""
    if lane_relation(scene, a, b, step) is not None:
        return None
    horizon = min(cfg.horizon, plans.shape[1], scene.n_steps - 1 - step)
    if horizon <= 0:
        return None
    fut = slice(step + 1, step + 1 + horizon)
    delta = braking_demand(np.asarray(plans)[:, :horizon, :2], scene.position[b, step, :2].astype(float),
                           float(scene.heading[b, step]), scene.shape_at(b, step),
                           scene.position[a, fut, :2].astype(float), scene.heading[a, fut].astype(float),
                           scene.valid[a, fut], scene.shape_at(a, step), cfg)
    from responsibility.metrics import weighted_cvar  # metrics imports this module's DemandConfig

    w = np.full(len(delta), 1.0 / len(delta)) if weights is None else np.asarray(weights, float) / np.sum(weights)
    cvar = float(weighted_cvar(torch.as_tensor(delta, dtype=torch.float64), torch.as_tensor(w), cfg.cvar_alpha))
    return Demand(demand=cvar, mean=float(np.dot(w, delta)), p_conflict=float(w[delta > 0].sum()),
                  p_unavoidable=float(w[delta > cfg.brake_max].sum()), horizon=horizon, per_member=delta)
