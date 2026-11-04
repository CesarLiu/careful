"""The interface every motion predictor implements for the responsibility
metrics, and the capabilities the metrics look for.

The metrics (responsibility.metrics, .blame, .forced_yield, .demand) never
touch a model's tensors. They ask four questions of it, all on a
``responsibility.scene.Scene`` (one 10 Hz clip, scene frame, metres):

  distribution(scene, step, agent, excluded=(), avoid_partner=())
      What could ``agent`` do from context step ``step``, given the scene's
      history up to ``step`` (never anyone's future), with the agents in
      ``excluded`` left out of the scene? Returns an opaque distribution
      object, or None where the model does not predict the agent (not
      observed at ``step``, a type the model was not trained on, too little
      history). ``avoid_partner`` names agents the model should not pick as
      an auxiliary "partner" input (DenseTNT's second agent of interest);
      models without such an input ignore it.

  sample(dist, n, generator=None) -> (index [n], log_prob [n], trajectories [n, T, 2])
      n i.i.d. members of the motion set (Eq. 1a), in the scene frame, the
      first row being the step after ``step``. ``index`` identifies the mode
      or goal a sample came from (None if the model has no discrete
      support); ``log_prob`` is the model's log-probability of each sample
      (per mode, per goal, or the mean token log-probability for SMART).

Optional capabilities (checked with ``hasattr``; the metrics refuse an
option the model does not support, with a message naming it):

  motion_set(dist, top_k=None) -> (trajectories [K, T, 2], probabilities [K])
      the whole finite, probability-weighted motion set (``--motion-set
      weighted``) or its ``top_k`` most probable members (``topk``).
  nms_motion_set(dist, n) -> (trajectories [n, T, 2], weights [n])
      n members spread by CAT's goal NMS (``--motion-set nms``, modes.py).
  with_and_without(scene, step, b, a) -> (dist, dist) | None
      b's distributions with and without a over one shared discrete support
      (``dist.log_prob`` [K] on both), so that courtesy can be the exact KL
      of Eq. 4 (``--courtesy-estimator kl``).

A goal-based distribution may also expose ``goals_global`` [G, 2] (or
``goals`` with ``to_global``) and ``log_prob`` [G]; the lane-route filter
and the courtesy goal restrictions then act on its goals before sampling.
Distributions without goals pass through those filters unchanged and are
filtered sample by sample instead (motion_filter.MotionFilter).

``MotionPredictor`` below is the structural type of the required part;
adapters need not inherit from anything. ``capabilities`` reports what a
predictor offers, so scripts can fail early and say why.
"""

from typing import Any, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

import numpy as np
import torch

from responsibility.scene import Scene

SAMPLE = "sample"
WEIGHTED_SET = "weighted"
TOPK_SET = "topk"
NMS_SET = "nms"
GOAL_KL = "goal_kl"


@runtime_checkable
class MotionPredictor(Protocol):
    def distribution(self, scene: Scene, step: int, agent: int, excluded: Sequence[int] = (),
                     avoid_partner: Sequence[int] = ()) -> Optional[Any]:
        ...

    def sample(self, dist: Any, n: int, generator: Optional[torch.Generator] = None
               ) -> Tuple[Optional[torch.Tensor], torch.Tensor, np.ndarray]:
        ...


def capabilities(model) -> Set[str]:
    """What ``model`` supports: SAMPLE always (it is required), WEIGHTED_SET
    and TOPK_SET with ``motion_set``, NMS_SET with ``nms_motion_set``,
    GOAL_KL with ``with_and_without``."""
    if not isinstance(model, MotionPredictor):
        raise TypeError(f"{type(model).__name__} is not a motion predictor: it needs distribution() and sample()")
    caps = {SAMPLE}
    if callable(getattr(model, "motion_set", None)):
        caps |= {WEIGHTED_SET, TOPK_SET}
    if callable(getattr(model, "nms_motion_set", None)):
        caps.add(NMS_SET)
    if callable(getattr(model, "with_and_without", None)):
        caps.add(GOAL_KL)
    return caps


def require(model, motion_set: str = "sampled", courtesy_estimator: str = "boltzmann") -> None:
    """Raises ValueError if ``model`` cannot compute the metrics with this
    motion set and courtesy estimator (ResponsibilityConfig's names)."""
    caps = capabilities(model)
    name = getattr(model, "name", type(model).__name__)
    if motion_set in (WEIGHTED_SET, TOPK_SET, NMS_SET) and motion_set not in caps:
        raise ValueError(f"--motion-set {motion_set} needs a predictor with a finite motion set; {name} has "
                         f"{sorted(caps)}")
    if courtesy_estimator == "kl" and GOAL_KL not in caps:
        raise ValueError(f"--courtesy-estimator kl needs distributions over a shared discrete support "
                         f"(with_and_without); {name} has none: use boltzmann or demand")
