"""A dependency-free reference predictor: constant velocity with Gaussian
noise on speed and yaw rate.

It is not a traffic model -- it ignores the map and the other agents, so its
counterfactuals do not react to anyone and courtesy measured with it is 0 by
construction (KL of identical distributions). It exists for three reasons:

- a smoke test of the whole pipeline (compute_responsibility, records,
  visualisation, summaries) on a machine without TensorFlow, UniTraj, CAT-K
  or a checkpoint;
- a kinematic baseline for safety responsibility: how much of beta_s a
  learned predictor adds over "keep doing roughly what you are doing";
- the template for a new adapter (responsibility/predictors/base.py).

A distribution is K = n_speed x n_yaw modes on a fixed grid of speed
factors and yaw rates, with Gaussian weights (``speed_sigma`` relative,
``yaw_rate_sigma`` rad/s). Because the grid is the same with and without any
agent, ``with_and_without`` gives two identical distributions and
``--courtesy-estimator kl`` is supported (and is 0).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from responsibility.modes import NMS_THRESHOLD, nms_select, speed_scale_factor
from responsibility.scene import Scene

DT = 0.1


@dataclass
class ModeDistribution:
    agent: int
    step: int
    excluded: Tuple[int, ...]
    log_prob: torch.Tensor  # [K]
    trajectories: np.ndarray  # [K, T, 2], scene frame
    extras: Dict = field(default_factory=dict)

    @property
    def goals_global(self) -> np.ndarray:
        return self.trajectories[:, -1, :]


class ConstantVelocity:
    name = "constant_velocity"

    def __init__(self, horizon: int = 80, speed_factors: Sequence[float] = (0.0, 0.5, 0.8, 1.0, 1.2),
                 yaw_rates: Sequence[float] = (-0.3, -0.15, 0.0, 0.15, 0.3), speed_sigma: float = 0.3,
                 yaw_rate_sigma: float = 0.15, types: Sequence[str] = ("VEHICLE", "CYCLIST", "PEDESTRIAN")):
        self.horizon = horizon
        self.speed_factors = np.asarray(speed_factors, dtype=np.float64)
        self.yaw_rates = np.asarray(yaw_rates, dtype=np.float64)
        self.speed_sigma = speed_sigma
        self.yaw_rate_sigma = yaw_rate_sigma
        self.types = set(types)

    def _log_weights(self) -> torch.Tensor:
        s = -0.5 * ((self.speed_factors - 1.0) / self.speed_sigma) ** 2
        w = -0.5 * (self.yaw_rates / self.yaw_rate_sigma) ** 2
        logits = torch.as_tensor((s[:, None] + w[None, :]).reshape(-1), dtype=torch.float32)
        return torch.log_softmax(logits, -1)

    def _rollout(self, xy: np.ndarray, heading: float, speed: float) -> np.ndarray:
        t = np.arange(1, self.horizon + 1) * DT
        out = []
        for f in self.speed_factors:
            for w in self.yaw_rates:
                yaw = heading + w * t
                v = f * speed
                if abs(w) < 1e-9:
                    dx, dy = v * t * np.cos(heading), v * t * np.sin(heading)
                else:
                    dx = v / w * (np.sin(yaw) - np.sin(heading))
                    dy = -v / w * (np.cos(yaw) - np.cos(heading))
                out.append(np.stack([xy[0] + dx, xy[1] + dy], -1))
        return np.stack(out)

    def distribution(self, scene: Scene, step: int, agent: int, excluded: Sequence[int] = (),
                     avoid_partner: Sequence[int] = ()) -> Optional[ModeDistribution]:
        if not scene.valid[agent, step] or scene.types[agent] not in self.types:
            return None
        speed = float(np.linalg.norm(scene.velocity[agent, step]))
        trajs = self._rollout(scene.position[agent, step, :2].astype(np.float64), float(scene.heading[agent, step]),
                              speed)
        return ModeDistribution(agent, step, tuple(excluded), self._log_weights(), trajs, {"speed": speed})

    def with_and_without(self, scene: Scene, step: int, target: int, removed: int):
        with_a = self.distribution(scene, step, target)
        if with_a is None:
            return None
        return with_a, self.distribution(scene, step, target, excluded=(removed,))

    def sample(self, dist: ModeDistribution, n: int, generator: Optional[torch.Generator] = None):
        idx = torch.multinomial(dist.log_prob.double().exp(), n, replacement=True, generator=generator)
        return idx, dist.log_prob[idx], dist.trajectories[idx.numpy()]

    def motion_set(self, dist: ModeDistribution, top_k: Optional[int] = None):
        probs = dist.log_prob.double().exp().numpy()
        if top_k is None or top_k >= len(probs):
            return dist.trajectories, probs
        keep = np.sort(np.argsort(-probs, kind="stable")[:top_k])
        return dist.trajectories[keep], probs[keep]

    def nms_motion_set(self, dist: ModeDistribution, n: int):
        threshold = NMS_THRESHOLD * speed_scale_factor(dist.extras["speed"])
        idx, weights = nms_select(dist.goals_global, dist.log_prob.double().exp().numpy(), n, threshold)
        return dist.trajectories[idx], weights
