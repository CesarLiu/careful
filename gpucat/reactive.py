"""Reactive background traffic, for evaluation only (gpucat/PLAN.md,
"反应式背景交通").

Every vehicle and cyclist other than the SDC replays its log exactly, until
following the log would take it too close to an object that has left its
own log. CAT's adversary in an adversarial episode is not reactive either.
The objects that count as having left their logs are the SDC, a planned
adversary, and another agent that has already reacted. When that happens,
the agent brakes along its logged path.

Waymax's IDMRoutePolicy also keeps each agent on its own logged path and
changes only its speed. But it drives every agent by IDM towards one desired
speed (30 m/s), so even undisturbed agents leave their logs. Here the
free-road behaviour is the log itself. The interaction is a safe-speed limit
(Gipps's, as in Krauss's car-following model) instead of IDM's interaction
term:

  v_safe = -b tau + sqrt(b^2 tau^2 + 2 b (gap - min_gap) + v_lead^2)

This is the highest speed from which the agent can still stop min_gap short
of the point where a leader braking at the same rate would stop, reacting
after tau and braking at b.
  gap     how far the agent's box can move along its path before it overlaps
          the obstacle's current box (boxes at lookahead points, as in Waymax)
  v_lead  the obstacle's speed along the path at that point, at least 0

An agent keeps its log while the log's next step is within v_safe.
Otherwise it leaves the log, and from then on, at each step,

  v' = clip(min(v_log + clip(k (s_log - s), 0, dv_max), v_safe), v - a_brake dt, v + a_max dt),  v' >= 0
  s' = s + (v + v') dt / 2

It catches up with its log (k, dv_max) and takes the logged row again once
it reaches it. It cannot brake harder than a_brake, so an object that cuts in
closer than its stopping distance is still hit.

An agent on its log treats as obstacles only the objects that have left
theirs. The logged agents are consistent with one another, and a logged gap
shorter than the rule allows (a queue at a light) must not move an agent off
its log on its own. An agent off its log reacts to every valid object ahead
of it. So when the SDC follows its log, the traffic is exactly the log
replay.

Pedestrians are replayed. The rows are those of GPUDrive's expert
trajectory tensor [W, 64, 1456] (gpucat/env.py). GPUDrive places an expert
at row t-1 after t steps, so before step t+1 the row t is written from the
state at row t-1.
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from gpucat.geometry import boxes_overlap

TRAJ_LEN = 91
DT = 0.1  # s
POS, VEL, YAW, VALID = 0, 182, 364, 455  # first columns of the trajectory rows (gpucat/env.py)
ROW_COLUMNS = VALID + TRAJ_LEN


@dataclass
class ReactiveSettings:
    decel: float = 4.0  # m/s^2, b: the braking the safe speed assumes, of the agent and the leader (Waymax's max_decel)
    max_brake: float = 6.0  # m/s^2, a_brake: the hardest braking (the SDC's own, -EnvSettings.accel_low)
    max_accel: float = 2.0  # m/s^2, a_max (Waymax's max_accel)
    reaction: float = 0.5  # s, tau
    min_gap: float = 2.0  # m between the boxes when stopped (Waymax's min_spacing)
    catch_up: float = 0.5  # 1/s, k
    max_catch_up: float = 3.0  # m/s above the log's speed, dv_max
    lookahead_margin: float = 10.0  # m beyond the stopping distance
    max_lookahead: float = 100.0  # m
    points: int = 40  # lookahead boxes, from the current position to the lookahead
    off_log: float = 0.5  # m: an SDC or adversary farther than this from its logged position has left its log
    chunk: int = 1 << 19  # lookahead boxes tested at once


def _unit(yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)


def along(S: torch.Tensor, P: torch.Tensor, Y: torch.Tensor, last: torch.Tensor,
          q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Points [N, K, 2] and headings [N, K] at arc lengths q [N, K] along N
    logged paths: S [N, T] arc length at each row, P [N, T, 2] and Y [N, T]
    the (gap-filled) positions and headings, last [N] the last valid row.
    Past the last valid row, the path goes straight on along its last
    heading."""
    T = S.shape[1]
    end = S.gather(1, last[:, None])  # [N, 1]
    k = torch.searchsorted(S.contiguous(), q.clamp(max=end).contiguous(), right=True) - 1
    k = torch.minimum(k.clamp(min=0), (last[:, None] - 1).clamp(min=0)).clamp(max=T - 2)
    a, b = S.gather(1, k), S.gather(1, k + 1)
    u = ((q - a) / (b - a).clamp(min=1e-6)).clamp(0.0, 1.0)
    pa = P.gather(1, k[..., None].expand(-1, -1, 2))
    pb = P.gather(1, (k + 1)[..., None].expand(-1, -1, 2))
    da, db = _unit(Y.gather(1, k)), _unit(Y.gather(1, k + 1))
    p = pa + u[..., None] * (pb - pa)
    d = da + u[..., None] * (db - da)
    yaw = torch.atan2(d[..., 1], d[..., 0])
    beyond = q > end
    p_end = P.gather(1, last[:, None, None].expand(-1, 1, 2))  # [N, 1, 2]
    y_end = Y.gather(1, last[:, None])  # [N, 1]
    p = torch.where(beyond[..., None], p_end + (q - end)[..., None] * _unit(y_end), p)
    yaw = torch.where(beyond, y_end.expand_as(yaw), yaw)
    return p, yaw


def logged_paths(rows: torch.Tensor):
    """The logged paths of the agents in trajectory rows [W, A, >= ROW_COLUMNS]:
    positions P and headings Y [W, A, T] with the invalid rows filled
    (interpolated in time between two valid rows, held before the first and
    after the last), the arc length S at each row, the valid mask V, the
    first and last valid rows, and v_at [W, A, T]: the logged speed over the
    step that ends at each row (the first step's at row 0; the last step's
    after the last valid row)."""
    T = TRAJ_LEN
    P = rows[..., POS:POS + 2 * T].unflatten(-1, (T, 2)).float()
    Y = rows[..., YAW:YAW + T].float()
    V = rows[..., VALID:VALID + T] > 0.5
    idx = torch.arange(T, device=rows.device)
    prev = torch.where(V, idx, torch.full_like(idx, -1)).cummax(-1).values
    nxt = torch.where(V, idx, torch.full_like(idx, T)).flip(-1).cummin(-1).values.flip(-1)
    lo = torch.where(prev >= 0, prev, nxt).clamp(0, T - 1)
    hi = torch.where(nxt < T, nxt, prev).clamp(0, T - 1)
    w = ((idx - lo).float() / (hi - lo).clamp(min=1).float()).clamp(0.0, 1.0)
    g2 = lambda x, i: x.gather(-2, i[..., None].expand(*i.shape, 2))  # noqa: E731
    P = g2(P, lo) + w[..., None] * (g2(P, hi) - g2(P, lo))
    ua, ub = _unit(Y.gather(-1, lo)), _unit(Y.gather(-1, hi))
    d = ua + w[..., None] * (ub - ua)
    Y = torch.atan2(d[..., 1], d[..., 0])
    step = (P[..., 1:, :] - P[..., :-1, :]).norm(dim=-1)
    S = torch.cat([torch.zeros_like(step[..., :1]), torch.cumsum(step, -1)], -1)
    first, last = nxt[..., 0], prev[..., -1]
    v = torch.cat([step[..., :1], step], -1) / DT
    v_end = v.gather(-1, last.clamp(min=0)[..., None])
    v_at = torch.where(idx > last[..., None], v_end, v)
    return P, Y, S, V, first, last, v_at


class ReactiveTraffic:
    """The reactive background of W worlds (module docstring). load() takes
    the logged rows and the agents of a scene batch, reset() starts episodes
    in some worlds, step() writes the next rows before GPUDrive's step."""

    def __init__(self, settings: ReactiveSettings, device: str = "cuda"):
        self.r, self.device = settings, device

    def load(self, rows: torch.Tensor, sizes: torch.Tensor, types: torch.Tensor, n_agents: torch.Tensor) -> None:
        """rows [W, A, >= ROW_COLUMNS]: the logged trajectory rows, sim
        coordinates; sizes [W, A, 2] length and width; types [W, A] (1
        vehicle, 2 cyclist, 3 pedestrian); n_agents [W]."""
        self.W, self.A = rows.shape[:2]
        self.P, self.Y, self.S, self.V, self.first, self.last, self.v_at = logged_paths(rows)
        self.P_log = rows[..., POS:POS + 2 * TRAJ_LEN].unflatten(-1, (TRAJ_LEN, 2)).float()
        self.sizes = sizes.float()
        slot = torch.arange(self.A, device=rows.device)[None]
        self.reactive = ((types == 1) | (types == 2)) & (slot >= 1) & (slot < n_agents[:, None]) & (self.last >= 0)
        self.reset(torch.arange(self.W, device=rows.device))

    def reset(self, worlds: torch.Tensor) -> None:
        if not hasattr(self, "off"):
            W, A, dev = self.W, self.A, self.S.device
            self.off = torch.zeros(W, A, dtype=torch.bool, device=dev)
            self.s, self.v = torch.zeros(W, A, device=dev), torch.zeros(W, A, device=dev)
            self.ever = torch.zeros(W, A, dtype=torch.bool, device=dev)
            self.max_lag = torch.zeros(W, A, device=dev)
            self.ego_prev = torch.full((W, 2), float("nan"), device=dev)
        self.off[worlds] = False
        self.s[worlds], self.v[worlds] = 0.0, 0.0
        self.ever[worlds] = False
        self.max_lag[worlds] = 0.0
        self.ego_prev[worlds] = float("nan")

    def _row(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """x [W, A, T, ...] at the row r [W] of each world: [W, A, ...]."""
        idx = r[:, None, None].expand(-1, self.A, 1)
        if x.dim() == 4:
            return x.gather(2, idx[..., None].expand(-1, -1, -1, x.shape[-1])).squeeze(2)
        return x.gather(2, idx).squeeze(2)

    def step(self, traj: torch.Tensor, t: torch.Tensor, alive: torch.Tensor, ego_pos: torch.Tensor,
             ego_yaw: torch.Tensor, adversarial: torch.Tensor) -> None:
        """Writes the row t [W] (the steps each world has taken) of the agents
        off their logs into traj [W, A, >= ROW_COLUMNS] (sim coordinates),
        from the state at row t-1 and the SDC at ego_pos [W, 2] / ego_yaw [W]
        (sim coordinates); worlds that are not alive or have taken no step
        yet are left alone. adversarial [W]: slot 1 follows a plan."""
        r_, W, A, dev = self.r, self.W, self.A, traj.device
        ego_vel = torch.nan_to_num((ego_pos - self.ego_prev) / DT, nan=0.0)
        self.ego_prev = ego_pos.clone()
        act = alive & (t >= 1) & (t < TRAJ_LEN)
        if not bool(act.any()):
            return
        r = (t - 1).clamp(0, TRAJ_LEN - 1)
        n = t.clamp(1, TRAJ_LEN - 1)
        T = TRAJ_LEN
        rows_pos = traj[..., POS:POS + 2 * T].unflatten(-1, (T, 2))
        pos = self._row(rows_pos, r).float()  # [W, A, 2] where each object is now
        yaw = self._row(traj[..., YAW:YAW + T], r).float()
        vel = self._row(traj[..., VEL:VEL + 2 * T].unflatten(-1, (T, 2)), r).float()
        valid = self._row(traj[..., VALID:VALID + T], r) > 0.5
        pos[:, 0], yaw[:, 0], vel[:, 0], valid[:, 0] = ego_pos, ego_yaw, ego_vel, True
        # objects off their logs: the SDC, a planned adversary, reactive agents that left
        log_pos, log_valid = self._row(self.P_log, r), self._row(self.V, r)
        dev_log = (pos - log_pos).norm(dim=-1)
        moved = (dev_log > r_.off_log) | ~log_valid
        obst_off = self.off.clone()
        obst_off[:, 0] = moved[:, 0]
        obst_off[:, 1] |= adversarial & valid[:, 1] & moved[:, 1]
        reactive = self.reactive & act[:, None]
        reactive[:, 1] &= ~adversarial
        on_now = valid & (self.off | log_valid)
        s_cur = torch.where(self.off, self.s, self._row(self.S, r))
        v_cur = torch.where(self.off, self.v, self._row(self.v_at, r))
        b, tau = r_.decel, r_.reaction
        L = (v_cur * tau + v_cur ** 2 / (2 * b) + r_.lookahead_margin).clamp(max=r_.max_lookahead)
        # which obstacle can each reactive agent meet ahead along its path
        rel = pos[:, None, :, :] - pos[:, :, None, :]  # [W, i, j, 2]: p_j - p_i
        ahead = (rel * _unit(yaw)[:, :, None]).sum(-1) > 0
        reach = L[:, :, None] + 0.5 * (self.sizes[:, :, None, 0] + self.sizes[:, None, :, 0]) + 1.0
        cand = (reactive & on_now)[:, :, None] & valid[:, None, :] & ahead & (rel.norm(dim=-1) <= reach)
        cand &= self.off[:, :, None] | obst_off[:, None, :]
        cand &= ~torch.eye(A, dtype=torch.bool, device=dev)[None]
        v_safe = torch.full((W * A,), float("inf"), device=dev)
        pw, pi, pj = cand.nonzero(as_tuple=True)
        if len(pw):
            f = pw * A + pi
            uf, inv = torch.unique(f, return_inverse=True)
            K = r_.points
            d = L.view(-1)[uf][:, None] * torch.linspace(0.0, 1.0, K, device=dev)[None]  # [U, K]
            q = s_cur.view(-1)[uf][:, None] + d
            lp, ly = along(self.S.view(W * A, T)[uf], self.P.view(W * A, T, 2)[uf], self.Y.view(W * A, T)[uf],
                           self.last.view(-1)[uf].clamp(min=0), q)
            size_i = self.sizes[pw, pi]
            hit = torch.zeros(len(pw), K, dtype=torch.bool, device=dev)
            step_pairs = max(1, r_.chunk // K)
            for c in range(0, len(pw), step_pairs):
                sl = slice(c, c + step_pairs)
                m = len(pw[sl])
                ov = boxes_overlap(lp[inv[sl]].reshape(-1, 2), ly[inv[sl]].reshape(-1),
                                   size_i[sl, None].expand(m, K, 2).reshape(-1, 2),
                                   pos[pw[sl], pj[sl]][:, None].expand(m, K, 2).reshape(-1, 1, 2),
                                   yaw[pw[sl], pj[sl]][:, None].expand(m, K).reshape(-1, 1),
                                   self.sizes[pw[sl], pj[sl]][:, None].expand(m, K, 2).reshape(-1, 1, 2))
                hit[sl] = ov.view(m, K)
            any_hit = hit.any(1)
            k = hit.float().argmax(1)
            gap = torch.where(k > 0, d[inv, (k - 1).clamp(min=0)], torch.zeros_like(k, dtype=torch.float32))
            v_lead = (vel[pw, pj] * _unit(ly[inv, k])).sum(-1).clamp(min=0.0)
            slack = (gap - r_.min_gap).clamp(min=0.0)
            vs = -b * tau + torch.sqrt((b * tau) ** 2 + 2 * b * slack + v_lead ** 2)
            vs = torch.where(any_hit, vs, torch.full_like(vs, float("inf")))
            v_safe = v_safe.scatter_reduce(0, f, vs, reduce="amin")
        v_safe = v_safe.view(W, A)
        # agents on their logs leave them when the log's next step is too fast
        S_n, v_n = self._row(self.S, n), self._row(self.v_at, n)
        valid_n = self._row(self.V, n)
        leave = reactive & ~self.off & log_valid & valid_n & valid & (v_n > v_safe + 1e-3)
        self.s = torch.where(leave, s_cur, self.s)
        self.v = torch.where(leave, v_cur, self.v)
        off = self.off | leave
        # agents off their logs: brake, or catch up with the log
        go = off & reactive
        v_des = v_n + (r_.catch_up * (S_n - self.s)).clamp(0.0, r_.max_catch_up)
        v_new = torch.minimum(v_des, v_safe)
        v_new = torch.maximum(torch.minimum(v_new, self.v + r_.max_accel * DT), self.v - r_.max_brake * DT).clamp(min=0.0)
        s_new = self.s + 0.5 * (self.v + v_new) * DT
        last = self.last
        S_end = self.S.gather(2, last.clamp(min=0)[..., None]).squeeze(-1)
        rejoin = go & (n[:, None] <= last) & valid_n & (s_new >= S_n)
        finished = go & (n[:, None] > last) & (s_new >= S_end)
        still = go & ~rejoin & ~finished
        self.s = torch.where(go, s_new, self.s)
        self.v = torch.where(go, v_new, self.v)
        self.off = torch.where(go, still, off)
        self.ever |= still
        self.max_lag = torch.where(still, torch.maximum(self.max_lag, S_n - s_new), self.max_lag)
        ww, aa = still.nonzero(as_tuple=True)
        if len(ww):
            p, y = along(self.S[ww, aa], self.P[ww, aa], self.Y[ww, aa], last[ww, aa].clamp(min=0), s_new[ww, aa][:, None])
            p, y = p[:, 0], y[:, 0]
            col, sp = n[ww], v_new[ww, aa]
            traj[ww, aa, POS + 2 * col] = p[:, 0].to(traj.dtype)
            traj[ww, aa, POS + 2 * col + 1] = p[:, 1].to(traj.dtype)
            traj[ww, aa, VEL + 2 * col] = (sp * torch.cos(y)).to(traj.dtype)
            traj[ww, aa, VEL + 2 * col + 1] = (sp * torch.sin(y)).to(traj.dtype)
            traj[ww, aa, YAW + col] = y.to(traj.dtype)
            traj[ww, aa, VALID + col] = 1.0
