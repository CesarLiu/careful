"""Collision penalties weighted by responsibility in the log-replay
simulator (rl/sim.py): responsibility/blame_reward.py's rule (cat_RLtrain.py
--blame_weighting share|rss) for training episodes that run many worlds at
once, without MetaDrive.

  event    a run of consecutive steps in contact with the same vehicle,
           attributed once, at its first step c, in the scene rebuilt from
           the episode up to c (responsibility.rollouts.scene_from_rollout):
           the SDC and, in an adversarial episode, the adversary as
           simulated; the other slots as replayed; objects the simulator
           leaves out (static vehicles, slots beyond max_agents) absent
  weight   share: the SDC's share beta_ego+ / (beta_ego+ + beta_other+) when
           the counterfactual attribution (responsibility/blame.py) tells
           the two sides apart by more than ``margin`` m, else 1;
           rss: 0 when RSS puts the collision on the other alone, else 1
           (responsibility.blame_reward.penalty_weight / rss.rss_weight)
  reward   at every penalised step of the event
               r' = w * (-penalty) + (1 - w) * step_reward
           step_reward being the progress reward the penalty replaced

The attribution needs only the episode up to the first contact, so the
weight is known at that step and every transition reaches the replay buffer
already weighted (blame_reward.py holds a MetaDrive episode back until it
ends instead). Every event is logged to ``log_path`` with
blame_reward.LOG_FIELDS, RSS and right-of-way verdicts included in both modes.
"""

import csv
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from responsibility.blame import MARGIN, rollout_blame
from responsibility.blame_reward import LOG_FIELDS, MODES, penalty_weight
from responsibility.metrics import ResponsibilityConfig
from responsibility.right_of_way import rollout_right_of_way
from responsibility.rollouts import make_rollout, scene_from_rollout
from responsibility.rss import rollout_rss, rss_weight
from responsibility.scene import Scene

_np = lambda x: x.detach().cpu().numpy()  # noqa: E731


class SimBlameWeighting:
    """``sim.before_reset = SimBlameWeighting(...)`` weights the crash
    penalties of every world's steps in place (module docstring)."""

    def __init__(self, model, scenes_dir, penalty: float, mode: str = "share", cfg: Optional[ResponsibilityConfig] = None,
                 margin: float = MARGIN, log_path=None, cache: int = 32, verbose: bool = False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if model is None and mode == "share":
            raise ValueError("the share weighting needs a motion model")
        self.model, self.mode, self.penalty = model, mode, float(penalty)
        self.scenes = Path(scenes_dir)
        self.cfg = cfg or ResponsibilityConfig()
        self.margin = margin
        self.log_path = None if log_path is None else Path(log_path)
        self.cache: "OrderedDict[str, Scene]" = OrderedDict()
        self.cache_size = cache
        self.verbose = verbose
        self.weight: Optional[torch.Tensor] = None
        self.weights = []
        self.total_steps = 0

    # ------------------------------------------------------------------ the hook

    def __call__(self, sim, ended: torch.Tensor, reward: torch.Tensor) -> None:
        if self.weight is None or len(self.weight) != sim.W:
            self.weight = torch.ones(sim.W, device=sim.device)
        contact = sim.contact
        prev_t = (sim.t - 1).clamp(min=0)
        prev = sim.contact_buf[torch.arange(sim.W, device=sim.device), prev_t]
        prev = torch.where(sim.t > 1, prev, torch.full_like(prev, -1))
        new = (contact >= 0) & (contact != prev)
        self.weight = torch.where(contact >= 0, self.weight, torch.ones_like(self.weight))
        for w in torch.nonzero(new).flatten().tolist():
            self.weight[w] = self.attribute(sim, w)
        pen = sim.penalised
        if bool(pen.any()):
            weighted = self.weight * -self.penalty + (1.0 - self.weight) * sim.step_reward
            reward.copy_(torch.where(pen, weighted, reward))

    # ------------------------------------------------------------------ attribution

    def _scene(self, stem: str) -> Scene:
        if stem in self.cache:
            self.cache.move_to_end(stem)
            return self.cache[stem]
        scene = Scene.load(self.scenes / f"{stem}.pkl")
        self.cache[stem] = scene
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return scene

    def rollout(self, sim, w: int) -> Tuple[Scene, Dict]:
        """The scene and the rollout record of world w's episode up to now,
        ended in a collision with the slot it is in contact with."""
        b = sim.bank
        row, t = int(sim.rows[w]), int(sim.t[w])
        stem = b.stems[row]
        scene = self._scene(stem)
        tids = [str(x) for x in b.track_id[row]]
        n_agents = int(b.n_agents[row])
        n = t + 1
        ego = {"position": _np(sim.ego_buf[w, :n]), "heading": _np(sim.head_buf[w, :n]),
               "velocity": _np(sim.vel_buf[w, :n]), "size": tuple(_np(b.size[row, 0]).tolist())}
        adversary, first_logged = None, 1
        if bool(sim.adversarial[w]):
            k = torch.arange(n, device=sim.device)
            on = b.valid[row, 1, :n] | (k > 10)
            plan = sim.plan[w, (k - sim.s.adversary_lag).clamp(min=0)]
            nan = torch.full_like(plan, float("nan"))
            plan = torch.where(on[:, None], plan, nan)
            adversary = {"position": _np(plan[:, :2]), "heading": _np(plan[:, 4]), "velocity": _np(plan[:, 2:4]),
                         "size": tuple(_np(b.size[row, 1]).tolist()), "track_id": tids[1], "planned": _np(sim.plan[w])}
            first_logged = 2
        present = {"track_ids": tids[first_logged:n_agents], "mask": _np(b.valid[row, first_logged:n_agents, :n])}
        partner = int(sim.contact[w])
        end = {"step": t, "reason": "crash_vehicle", "crash_vehicle": True,
               "crash_with": tids[partner] if 0 < partner < n_agents else None,
               "route_completion": float(sim.completion[w])}
        rollout = make_rollout(stem, scene.scenario_id, "td3", "cat" if adversary is not None else "none", ego, end,
                               adversary, present, no_static_vehicles=True)
        return scene_from_rollout(scene, rollout), rollout

    def attribute(self, sim, w: int) -> float:
        t0 = time.time()
        try:
            scene, rollout = self.rollout(sim, w)
            blame = None if self.model is None else rollout_blame(self.model, scene, rollout, self.cfg,
                                                                   margin=self.margin)
            rss, row_of_way = rollout_rss(scene, rollout), rollout_right_of_way(scene, rollout)
            error = None
        except Exception as e:  # a long training run must not stop on one attribution
            blame = rss = row_of_way = None
            error = e
        weight = rss_weight(rss) if self.mode == "rss" else penalty_weight(blame)
        self.weights.append(weight)
        self._log(sim, w, blame, rss, row_of_way, weight, time.time() - t0, error)
        return weight

    def _log(self, sim, w, blame, rss, row_of_way, weight, seconds, error) -> None:
        b = sim.bank
        row = {"episode": "", "total_steps": self.total_steps, "scene": b.stems[int(sim.rows[w])],
               "scenario_index": int(sim.rows[w]), "adversary": "1" if bool(sim.adversarial[w]) else "",
               "crash_step": int(sim.t[w]), "penalised_steps": "", "other_id": "", "other_type": "", "window": "",
               "beta_ego": "", "beta_other": "", "share": "", "verdict": "error" if error is not None else "unknown",
               "rule": "n/a", "rss": "n/a", "rss_case": "", "right_of_way": "n/a", "right_of_way_case": "",
               "priority": "none", "weight": round(float(weight), 4), "seconds": round(seconds, 2)}
        if blame is not None:
            r = blame.as_row()
            row.update({k: r[k] for k in ("other_id", "other_type", "window", "verdict", "rule")})
            row.update({k: "" if r[k] is None else round(r[k], 4) for k in ("beta_ego", "beta_other", "share")})
        for baseline in (rss, row_of_way):
            if baseline is not None:
                row.update(baseline.as_row())
        if error is not None:
            print(f"warning: collision in scene {row['scene']} at step {row['crash_step']} not attributed "
                  f"({type(error).__name__}: {error}); full penalty kept", flush=True)
        elif self.verbose:
            print(f"collision in scene {row['scene']} at step {row['crash_step']}: {row['verdict']}, RSS {row['rss']}, "
                  f"right of way {row['right_of_way']}, weight {weight:.2f}", flush=True)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.log_path.exists()
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)

    def summary(self) -> str:
        if not self.weights:
            return "no collisions attributed yet"
        w = np.array(self.weights)
        return (f"{len(w)} collisions attributed: mean penalty weight {w.mean():.2f}, "
                f"{100 * np.mean(w < 0.5):.0f}% mostly the other's fault, {100 * np.mean(w == 1.0):.0f}% full penalty")
