"""A batched log-replay driving simulator in plain torch, in place of
MetaDrive (CAT's simulator, which does not run on Python 3.11) and of
GPUDrive (the earlier port, gpucat/ on main): W worlds stepped together,
each the SDC under policy control in one of CAT's scenes, everyone else
replayed from the log, and -- in an adversarial episode -- CAT's adversary
driving the plan CAT (or the fair rule) chooses against the SDC's recent
trajectories in that scene (rl/adversary.py).

What it reproduces of MetaDrive's ScenarioEnv as CAT's cat_RLtrain.py
configures it (reactive_traffic False, no_static_vehicles, no_light), and
what it replaces:

  dynamics    a kinematic bicycle at 10 Hz: acceleration in
              [accel_low, accel_high] m/s^2 and curvature within +-curvature
              1/m from the two actions in [-1, 1]; speed in [0, max_speed]
              (replaces MetaDrive's physics vehicle)
  traffic     every other agent at its logged state at step t; the adversary
              at its plan's row t - adversary_lag (CAT's plan is followed a
              step late in MetaDrive), the steps before it appears in the log
              left as logged (absent)
  reward      driving_reward x the progress along the SDC's route (arc
              length, rl.geometry.route_coords), replaced by +success_reward
              on arrival, -out_of_road_penalty out of road, -crash_penalty on
              a step in contact with a vehicle (that precedence)
  arrival     route completion > arrive_completion
  out of road farther than out_of_route m from the route, or the footprint
              touching a road-edge (MetaDrive's sidewalk) or a solid yellow
              line segment
  collision   the SDC's footprint overlapping a vehicle's (full-size boxes,
              separating axes); it ends the episode only with crash_done
              (False in CAT's training, True in its evaluation)
  observation [ego 5 | navigation 60 | partners 9 x K] (``obs_dim``), all in
              the SDC's frame (replaces MetaDrive's lidar, side and lane-line
              detectors):
                ego         speed / 30, length / 10, width / 5, last action (2)
                navigation  the route 5, 10, 20, 30, 50 m ahead (/50), the
                            distance from the route (/out_of_route), route
                            completion, the distances along 24 rays (every
                            15 degrees, /50) to road edges and, separately,
                            to yellow lines (gpucat's navigation observation)
                partners    the K nearest agents within partner_radius m:
                            position (/50), heading (cos, sin), velocity
                            (/30), length / 10, width / 5, a presence flag;
                            zeros for empty slots

CAT's adversary bookkeeping is AdvGenerator's: per scene, the SDC's last
``history`` trajectories (steps 11 on; episodes shorter than 10 steps after
step 11 are ignored) start as the logged route and are what the candidates
are scored against; evaluation keeps one of its own (AV_trajs_eval).

The results are not comparable with MetaDrive's: dynamics, observations and
collision geometry differ. What is kept is CAT's setting (scenes, log replay,
the adversary and its schedule, the reward and the termination rules).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch

from rl import adversary as ga
from rl.geometry import boxes_overlap, points_along, ray_distances, route_coords, segments_touch_box
from rl.scenes import STEPS, VEHICLE, SceneBank

DT = 0.1
FIRST_STORED = 11  # AdvGenerator stores ego_traj[11:91]
MIN_STORED = 10  # and ignores shorter ones
RUNNING, ARRIVED, OUT_OF_ROAD, CRASHED, TIMEOUT = 0, 1, 2, 3, 4
OUTCOMES = ("running", "arrived", "out_of_road", "crashed", "timeout")
ROUTE_AHEAD = (5.0, 10.0, 20.0, 30.0, 50.0)  # m along the route
RAYS = 24
RAY_RANGE = 50.0  # m
EGO_DIM, NAV_DIM, PARTNER_DIM = 5, 2 * len(ROUTE_AHEAD) + 2 + 2 * RAYS, 9


@dataclass
class SimSettings:
    worlds: int = 64
    accel_low: float = -6.0  # m/s^2
    accel_high: float = 4.0
    curvature: float = 0.3  # 1/m
    max_speed: float = 40.0  # m/s
    driving_reward: float = 1.0
    success_reward: float = 10.0
    out_of_road_penalty: float = 10.0
    crash_penalty: float = 1.0  # MetaDrive's crash_vehicle_penalty
    crash_done: bool = False  # CAT: False in training, True in evaluation
    out_of_route: float = 10.0  # m
    arrive_completion: float = 0.95
    episode_steps: int = STEPS - 1  # 9 s
    history: int = 5  # CAT's AV_traj_num
    collision_radius: float = 30.0  # m: vehicles farther from the SDC are not checked
    partners: int = 16
    partner_radius: float = 50.0  # m
    adversary_lag: int = 1  # steps: the adversary is at plan row t - lag after t steps


def obs_dim(settings: SimSettings) -> int:
    return EGO_DIM + NAV_DIM + PARTNER_DIM * settings.partners


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi


class LogReplaySim:
    """W worlds over the scenes of ``bank`` (rl.scenes.SceneBank). Each
    world plays one scene row at a time: ``scene_rows`` [W] at begin(); with
    auto_reset, a world whose episode ends draws its next scene with
    ``pick(n) -> rows [n]`` (default: the same scene). ``candidates``
    (rl.adversary.CandidateBank, its rows matched by stem) is needed for
    adversarial episodes."""

    def __init__(self, settings: SimSettings, bank: SceneBank, candidates: Optional[ga.CandidateBank] = None,
                 device: str = "cpu"):
        self.s, self.bank, self.cand, self.device, self.W = settings, bank, candidates, device, settings.worlds
        S = len(bank)
        self.crow_of = torch.full((S,), -1, dtype=torch.long, device=device)  # scene row -> candidate row
        if candidates is not None:
            for stem, r in bank.index.items():
                if stem in candidates.index:
                    self.crow_of[r] = candidates.index[stem]
            M = settings.history
            route = candidates.ego_route
            self.hist = route[:, None].repeat(1, M, 1, 1).clone()
            self.hist_len = torch.full((len(candidates.stems), M), route.shape[1], device=device, dtype=torch.long)
            self.hist_prob = torch.ones(len(candidates.stems), M, device=device)
            self.hist_yaw = ga.subsampled_yaw(self.hist.double(), self.hist_len).float()
            self.eval_hist = route[:, None].clone()
            self.eval_len = torch.full((len(candidates.stems), 1), route.shape[1], device=device, dtype=torch.long)
            self.eval_yaw = ga.subsampled_yaw(self.eval_hist.double(), self.eval_len).float()
        self.A = bank.position.shape[1]
        self.pick: Optional[Callable[[int], torch.Tensor]] = None
        # called after every step's checks and before any world restarts, with
        # (sim, ended, reward); may rewrite reward in place (rl/blame.py)
        self.before_reset: Optional[Callable] = None

    # ------------------------------------------------------------------ episodes

    def begin(self, scene_rows: torch.Tensor, adversarial: torch.Tensor, rule: str = "cat", tau: float = 2.0,
              rho: float = 0.1, evaluation: bool = False, auto_reset: bool = False, p_adv: float = 0.0,
              pick: Optional[Callable[[int], torch.Tensor]] = None) -> torch.Tensor:
        """Starts an episode in every world; returns the SDC's observations [W, D].
        Lockstep (evaluation): a world whose episode ends stays done until the
        next begin and end_episode reports every world. auto_reset (training):
        an ended episode is reported in ``finished``, its SDC trajectory stored,
        and the world restarts at once (a scene from ``pick``, adversarial with
        probability ``p_adv``)."""
        dev, W = self.device, self.W
        self.rule, self.tau, self.rho = rule, tau, rho
        self.evaluation, self.auto_reset, self.p_adv, self.pick = evaluation, auto_reset, p_adv, pick
        self.rows = torch.as_tensor(scene_rows, device=dev).long().clone()
        self.done = torch.zeros(W, dtype=torch.bool, device=dev)
        self.t = torch.zeros(W, dtype=torch.long, device=dev)
        self.p = torch.zeros(W, 2, device=dev)
        self.h = torch.zeros(W, device=dev)
        self.v = torch.zeros(W, device=dev)
        self.last_action = torch.zeros(W, 2, device=dev)
        self.outcome = torch.zeros(W, dtype=torch.long, device=dev)
        self.completion = torch.zeros(W, device=dev)
        self.crashed = torch.zeros(W, dtype=torch.bool, device=dev)
        self.first_crash = torch.full((W,), -1, dtype=torch.long, device=dev)
        self.partner = torch.full((W,), -1, dtype=torch.long, device=dev)
        self.adversarial = torch.zeros(W, dtype=torch.bool, device=dev)
        self.chosen = torch.full((W,), -1, dtype=torch.long, device=dev)
        self.plan = torch.zeros(W, STEPS, 5, device=dev)
        self.prev_s = torch.zeros(W, device=dev)
        self.ego_buf = torch.full((W, STEPS, 2), float("nan"), device=dev)
        self.head_buf = torch.full((W, STEPS), float("nan"), device=dev)
        self.vel_buf = torch.full((W, STEPS, 2), float("nan"), device=dev)
        self.contact_buf = torch.full((W, STEPS), -1, dtype=torch.long, device=dev)  # slot in contact, -1 none
        self.finished: List[Dict[str, torch.Tensor]] = []
        self._restart(torch.arange(W, device=dev), torch.as_tensor(adversarial, device=dev).bool())
        return self.obs()

    def has_adversary(self, rows: torch.Tensor) -> torch.Tensor:
        return self.bank.has_adversary[rows] & (self.crow_of[rows] >= 0)

    def _restart(self, worlds: torch.Tensor, adversarial: torch.Tensor) -> None:
        b = self.bank
        rows = self.rows[worlds]
        adversarial = adversarial & self.has_adversary(rows)
        self.adversarial[worlds] = adversarial
        self.chosen[worlds] = -1
        adv = worlds[adversarial]
        if len(adv):
            plans, chosen = self.choose_plans(adv)
            self.plan[adv] = plans
            self.chosen[adv] = chosen
        self.t[worlds] = 0
        self.p[worlds] = b.position[rows, 0, 0]
        self.h[worlds] = b.heading[rows, 0, 0]
        self.v[worlds] = b.velocity[rows, 0, 0].norm(dim=-1)
        self.last_action[worlds] = 0.0
        _, s = route_coords(self.p[worlds], b.route[rows], b.route_n[rows])
        self.prev_s[worlds] = s
        self.outcome[worlds] = RUNNING
        self.completion[worlds] = 0.0
        self.crashed[worlds] = False
        self.first_crash[worlds] = -1
        self.partner[worlds] = -1
        self.ego_buf[worlds] = float("nan")
        self.ego_buf[worlds, 0] = self.p[worlds]
        self.head_buf[worlds] = float("nan")
        self.head_buf[worlds, 0] = self.h[worlds]
        self.vel_buf[worlds] = float("nan")
        self.vel_buf[worlds, 0] = self.v[worlds, None] * torch.stack([torch.cos(self.h[worlds]),
                                                                       torch.sin(self.h[worlds])], -1)
        self.contact_buf[worlds] = -1

    def choose_plans(self, worlds: torch.Tensor):
        """CAT's (or the fair rule's) plan [n, 91, 5] for these worlds against
        their scenes' stored SDC trajectories, and the chosen candidates [n]."""
        c, cb = self.crow_of[self.rows[worlds]], self.cand
        if self.evaluation:
            hist, hlen, hyaw = self.eval_hist[c], self.eval_len[c], self.eval_yaw[c]
            hprob = torch.ones(len(c), 1, device=self.device)
        else:
            hist, hlen, hyaw, hprob = self.hist[c], self.hist_len[c], self.hist_yaw[c], self.hist_prob[c]
        score, md = ga.cat_scores(cb.cand[c], cb.probs_ov[c], hist, hlen, hprob, cb.ov_size[c], cb.av_size[c],
                                  yaw_ov=cb.yaw_ov[c], yaw_av=hyaw)
        if self.rule == "cat":
            chosen = ga.select("cat", score, md)
        elif self.rule == "fair":
            if cb.samples is None or cb.avoid is None:
                raise ValueError("the fair rule needs the adversary's samples and the ego avoidability in the bank")
            beta = ga.adversary_beta(cb.samples[c], cb.cand[c], hist, hlen, hprob)
            chosen = ga.select("fair", score, md, beta, cb.avoid[c], self.tau, self.rho)
        else:
            raise ValueError(f"unknown rule {self.rule!r}: cat or fair")
        return ga.plan(cb.adv_past[c], cb.cand[c, chosen]), chosen

    # ------------------------------------------------------------------ state

    def agents(self):
        """Every slot's state at each world's step t: position [W, A, 2],
        heading [W, A], velocity [W, A, 2], present [W, A] (slot 0, the SDC,
        is its logged state and not present: the policy drives it)."""
        b, rows, t = self.bank, self.rows, self.t
        a = torch.arange(self.A, device=self.device)
        r, tt = rows[:, None], t[:, None]
        pos, yaw = b.position[r, a[None], tt], b.heading[r, a[None], tt]
        vel, present = b.velocity[r, a[None], tt], b.valid[r, a[None], tt]
        if bool(self.adversarial.any()):  # slot 1 follows its plan where it is in the scene
            w = torch.nonzero(self.adversarial).flatten()
            k = (t[w] - self.s.adversary_lag).clamp(min=0)
            row = self.plan[w, k]
            # absent where its log is, up to step 10 (it has not appeared yet); on its plan everywhere else
            on = b.valid[rows[w], 1, t[w]] | (t[w] > 10)
            pos[w, 1] = torch.where(on[:, None], row[:, :2], pos[w, 1])
            vel[w, 1] = torch.where(on[:, None], row[:, 2:4], vel[w, 1])
            yaw[w, 1] = torch.where(on, row[:, 4], yaw[w, 1])
            present[w, 1] = on
        present = present & (a[None] < b.n_agents[rows, None])
        present[:, 0] = False
        return pos, yaw, vel, present

    def obs(self) -> torch.Tensor:
        s, b, rows = self.s, self.bank, self.rows
        p, h = self.p, self.h
        cos, sin = torch.cos(h)[:, None], torch.sin(h)[:, None]

        def local(d):  # [W, K, 2] scene -> SDC frame
            return torch.stack([d[..., 0] * cos + d[..., 1] * sin, -d[..., 0] * sin + d[..., 1] * cos], -1)

        size = b.size[rows, 0]
        ego = torch.stack([self.v / 30.0, size[:, 0] / 10.0, size[:, 1] / 5.0], -1)
        ego = torch.cat([ego, self.last_action], -1)
        route, route_n = b.route[rows], b.route_n[rows]
        lateral, sarc = route_coords(p, route, route_n)
        ahead = local(points_along(route, route_n, sarc, torch.tensor(ROUTE_AHEAD, device=self.device)) - p[:, None])
        angles = torch.arange(RAYS, device=self.device) * (2.0 * torch.pi / RAYS)
        edge = ray_distances(p, h, b.edges[rows], b.edges_n[rows], angles, RAY_RANGE)
        yellow = ray_distances(p, h, b.yellow[rows], b.yellow_n[rows], angles, RAY_RANGE)
        completion = (sarc / b.route_len[rows].clamp(min=1.0)).clamp(0.0, 1.0)
        nav = torch.cat([ahead.flatten(1) / RAY_RANGE, (lateral / s.out_of_route)[:, None], completion[:, None],
                         edge / RAY_RANGE, yellow / RAY_RANGE], -1)
        pos, yaw, vel, present = self.agents()
        dist = (pos - p[:, None]).norm(dim=-1)
        dist = torch.where(present & (dist < s.partner_radius), dist, torch.full_like(dist, float("inf")))
        k = min(s.partners, self.A)
        d, j = torch.topk(dist, k, dim=-1, largest=False)
        w = torch.arange(self.W, device=self.device)[:, None]
        on = torch.isfinite(d)
        rel = local(pos[w, j] - p[:, None]) / RAY_RANGE
        dh = yaw[w, j] - h[:, None]
        rv = local(vel[w, j]) / 30.0
        osize = b.size[rows[:, None], j]
        partners = torch.cat([rel, torch.cos(dh)[..., None], torch.sin(dh)[..., None], rv,
                              (osize[..., 0] / 10.0)[..., None], (osize[..., 1] / 5.0)[..., None],
                              on[..., None].float()], -1) * on[..., None]
        if k < s.partners:
            partners = torch.cat([partners, partners.new_zeros(self.W, s.partners - k, PARTNER_DIM)], 1)
        return torch.cat([ego, nav, partners.flatten(1)], -1).float()

    # ------------------------------------------------------------------ dynamics

    def controls(self, action: torch.Tensor):
        s = self.s
        accel = s.accel_low + 0.5 * (action[:, 0] + 1.0) * (s.accel_high - s.accel_low)
        return accel, action[:, 1] * s.curvature

    def expert_action(self) -> torch.Tensor:
        """The action [W, 2] that takes each SDC from its current state to its
        logged position at t + 1 (the bicycle's exact inverse), in the
        policy's scale -- outside [-1, 1] where the log needs more."""
        s = self.s
        q = self.bank.position[self.rows, 0, (self.t + 1).clamp(max=STEPS - 1)]
        d = q - self.p
        ds = d.norm(dim=-1)
        v1 = 2.0 * ds / DT - self.v
        accel = (v1 - self.v) / DT
        dh = 2.0 * wrap(torch.atan2(d[:, 1], d[:, 0]) - self.h)
        kappa = torch.where(ds > 1e-4, dh / ds.clamp(min=1e-4), torch.zeros_like(ds))
        return torch.stack([2.0 * (accel - s.accel_low) / (s.accel_high - s.accel_low) - 1.0, kappa / s.curvature], -1)

    def step(self, action: torch.Tensor, clip: bool = True):
        """action [W, 2] in [-1, 1] (not clipped with clip=False: replaying the
        log). Returns (obs [W, D] -- with auto_reset, the next episode's first
        for worlds that just ended --, reward [W], ended [W], alive [W]);
        ``flags`` holds this step's tests and ``terminal`` [W] whether the step
        ended the episode for a reason other than the time limit."""
        s, b = self.s, self.bank
        alive = ~self.done
        a = action.clamp(-1.0, 1.0) if clip else action
        accel, kappa = self.controls(a)
        v1 = (self.v + accel * DT).clamp(0.0, s.max_speed)
        ds = 0.5 * (self.v + v1) * DT
        hm = self.h + 0.5 * kappa * ds
        step = torch.stack([torch.cos(hm), torch.sin(hm)], -1) * ds[:, None]
        keep = alive[:, None]
        self.p = torch.where(keep, self.p + step, self.p)
        self.h = torch.where(alive, wrap(self.h + kappa * ds), self.h)
        self.v = torch.where(alive, v1, self.v)
        self.last_action = torch.where(keep, a.clamp(-1.0, 1.0), self.last_action)
        self.t = self.t + alive.long()
        rows, p, h = self.rows, self.p, self.h
        size = b.size[rows, 0]
        pos, yaw, _, present = self.agents()
        others = present & (b.type[rows] == VEHICLE) & ((pos - p[:, None]).norm(dim=-1) < s.collision_radius)
        touching = boxes_overlap(p, h, size, pos, yaw, b.size[rows]) & others
        crash = touching.any(-1) & alive
        first = crash & (self.first_crash < 0)
        self.first_crash = torch.where(first, self.t, self.first_crash)
        self.partner = torch.where(first, touching.float().argmax(-1), self.partner)
        self.crashed |= crash
        route, route_n = b.route[rows], b.route_n[rows]
        lateral, sarc = route_coords(p, route, route_n)
        edge = segments_touch_box(p, h, size[:, 0], size[:, 1], b.edges[rows], b.edges_n[rows])
        yellow = segments_touch_box(p, h, size[:, 0], size[:, 1], b.yellow[rows], b.yellow_n[rows])
        out_of_road = (lateral > s.out_of_route) | edge | yellow
        completion = sarc / b.route_len[rows].clamp(min=1.0)
        arrive = (completion > s.arrive_completion) | (b.route_len[rows] < 1.0)
        self.flags = {"crash": crash, "edge": edge, "yellow": yellow, "off_route": lateral > s.out_of_route,
                      "arrive": arrive, "lateral": lateral}
        reward = s.driving_reward * (sarc - self.prev_s)
        self.step_reward = reward.clone()  # what a crash penalty replaces (rl/blame.py)
        self.penalised = crash & ~out_of_road & ~arrive  # the step's reward is the crash penalty
        reward = torch.where(crash, torch.full_like(reward, -s.crash_penalty), reward)
        reward = torch.where(out_of_road, torch.full_like(reward, -s.out_of_road_penalty), reward)
        reward = torch.where(arrive, torch.full_like(reward, s.success_reward), reward)
        reward = torch.where(alive, reward, torch.zeros_like(reward))
        terminal = alive & (arrive | out_of_road | (crash & s.crash_done))
        ended = terminal | (alive & (self.t >= s.episode_steps))
        reason = torch.where(arrive, ARRIVED, torch.where(out_of_road, OUT_OF_ROAD,
                             torch.where(crash & s.crash_done, CRASHED, TIMEOUT)))
        self.outcome = torch.where(ended, reason, self.outcome)
        self.completion = torch.where(alive, completion.clamp(0.0, 1.0), self.completion)
        self.prev_s = torch.where(alive, sarc, self.prev_s)
        self.terminal = terminal
        w = torch.nonzero(alive).flatten()
        tw = self.t[w].clamp(max=STEPS - 1)
        self.ego_buf[w, tw] = p[w]
        self.head_buf[w, tw] = h[w]
        self.vel_buf[w, tw] = self.v[w, None] * torch.stack([torch.cos(h[w]), torch.sin(h[w])], -1)
        self.contact = torch.where(crash, touching.float().argmax(-1), torch.full_like(self.partner, -1))
        self.contact_buf[w, tw] = self.contact[w]
        if self.before_reset is not None:
            self.before_reset(self, ended, reward)
        if self.auto_reset:
            idx = torch.nonzero(ended).flatten()
            if len(idx):
                self.finished.append(self._report(idx))
                self._store(idx)
                if self.pick is not None:
                    self.rows[idx] = self.pick(len(idx))
                self._restart(idx, torch.rand(len(idx), device=self.device) < self.p_adv)
        else:
            self.done |= ended
        return self.obs(), reward, ended, alive

    # ------------------------------------------------------------------ bookkeeping

    def _report(self, worlds: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"row": self.rows[worlds].clone(), "outcome": self.outcome[worlds].clone(),
                "completion": self.completion[worlds].clone(), "crashed": self.crashed[worlds].clone(),
                "adversarial": self.adversarial[worlds].clone(), "chosen": self.chosen[worlds].clone(),
                "first_crash": self.first_crash[worlds].clone(), "partner": self.partner[worlds].clone(),
                "ego": self.ego_buf[worlds].clone(), "steps": self.t[worlds].clone()}

    def _store(self, worlds: torch.Tensor) -> None:
        """AdvGenerator.after_episode: the SDC's positions from step 11 on (at
        most 80; episodes with fewer than 10 are ignored) become the newest of
        its scene's history (evaluation: its one evaluation trajectory)."""
        if self.cand is None:
            return
        c = self.crow_of[self.rows[worlds]]
        stored = self.ego_buf[worlds, FIRST_STORED:FIRST_STORED + 80]
        length = (~torch.isnan(stored[..., 0])).sum(1)
        ok = (length >= MIN_STORED) & (c >= 0)
        if not bool(ok.any()):
            return
        c, new, n = c[ok], torch.nan_to_num(stored[ok]), length[ok]
        yaw = ga.subsampled_yaw(new.double(), n).float()
        if self.evaluation:
            self.eval_hist[c, 0], self.eval_len[c, 0], self.eval_yaw[c, 0] = new, n, yaw
        else:  # a deque: drop the oldest
            self.hist[c] = torch.cat([self.hist[c, 1:], new[:, None]], 1)
            self.hist_len[c] = torch.cat([self.hist_len[c, 1:], n[:, None]], 1)
            self.hist_yaw[c] = torch.cat([self.hist_yaw[c, 1:], yaw[:, None]], 1)

    def end_episode(self) -> Dict[str, torch.Tensor]:
        """Lockstep: stores every world's SDC trajectory and reports the episode."""
        worlds = torch.arange(self.W, device=self.device)
        self._store(worlds)
        return self._report(worlds)

    def take_finished(self) -> Dict[str, torch.Tensor]:
        """With auto_reset: the episodes finished since the last call, concatenated."""
        done, self.finished = self.finished, []
        if not done:
            return {}
        return {k: torch.cat([d[k] for d in done]) for k in done[0]}


def run_lockstep(sim: LogReplaySim, policy: Callable[[torch.Tensor], torch.Tensor], scene_rows: torch.Tensor,
                 adversarial: bool, **begin) -> Dict[str, torch.Tensor]:
    """One episode in every world with ``policy(obs) -> action``; the report."""
    obs = sim.begin(scene_rows, torch.full((sim.W,), adversarial, device=sim.device), **begin)
    while not bool(sim.done.all()):
        obs, *_ = sim.step(policy(obs))
    return sim.end_episode()
