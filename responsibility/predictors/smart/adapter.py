"""CAT-K's SMART behind the predictor interface (predictors/base.py), so
that every predictor-agnostic tool -- responsibility.metrics (safety, the
Boltzmann and demand courtesy), blame, forced yields, the policy rollouts
-- runs on SMART exactly as on DenseTNT or UniTraj.

SMART is a jointly autoregressive token model: there is no closed-form
distribution over an agent's futures, only rollouts. So here a
"distribution" is a recipe -- the scene anchored at step k, the free agent,
the agents removed -- and ``sample`` draws n rollouts of it with
``native.motion_set.MotionSetSynthesizer``, the same machinery as CAT-K's
own pipeline:

  free agent   sampled from the policy (sampling scheme ``topk_prob`` over
               the whole 2048-token vocabulary at temperature 1, i.e.
               genuine sampling from pi_theta)
  the others   teacher-forced onto their logged tokens, and with
               ``neighbor_future="hidden"`` (default) dropped from the free
               agent's interaction graph after step k: the open-loop reading
               DenseTNT and UniTraj have by construction. "logged" lets the
               free agent see their logged futures; "simulated" samples them.
  excluded     validity zeroed, so they are in no attention edge (x^{-a})

``sample`` returns (None, mean token log-probability [n], trajectories
[n, T, 2]) in the scene frame, T = ``horizon`` 10 Hz steps after k
(rounded up to SMART's 0.5 s tokens; default 8 s), cut to what the rest of
the clip supports: SMART reads the logged token of every step it rolls out,
so from step k of a 91-step clip T is at most 80 - 5 ceil((k - 10) / 5). There is no shared discrete support,
so ``with_and_without`` and ``motion_set`` are not offered: courtesy uses
the Boltzmann estimate (the paper's Sec. V-B, the default), and CAT-K's
exact per-token KL stays in the batched native pipeline
(scripts/smart/compute_responsibility.py --courtesy-estimator exact).

Each (scene, step) is converted and tokenized once and cached
(``cache_size`` most recent), since the metrics ask for the same context
for the agent and for each neighbour.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from responsibility.catk_export import numeric_id
from responsibility.scene import Scene

DEFAULT_HISTORY = 11  # SMART's num_historical_steps: 1.0 s of history plus the current step


@dataclass
class RolloutDistribution:
    """What ``sample`` rolls out: the context (scene, step), the free
    agent's row in the batch and the rows removed."""

    scene: Scene
    step: int
    agent: int
    excluded: Tuple[int, ...]
    row: int
    removed_rows: Tuple[int, ...]
    extras: Dict = field(default_factory=dict)


@dataclass
class _Context:
    scene: Scene
    data: object
    tokenized_map: Dict
    tokenized_agent: Dict
    rows: Dict[int, int]  # numeric track id -> row


class SmartPredictor:
    name = "smart"

    def __init__(self, model, sampling_scheme=None, build_batch: Optional[Callable] = None,
                 neighbor_future: str = "hidden", batch_size: int = 32, horizon: Optional[int] = None,
                 num_historical_steps: int = DEFAULT_HISTORY, cache_size: int = 4):
        from responsibility.predictors.smart.native.motion_set import (NEIGHBOR_FUTURE_MODES,
                                                                        MotionSetSynthesizer)

        if neighbor_future not in NEIGHBOR_FUTURE_MODES:
            raise ValueError(f"neighbor_future must be one of {NEIGHBOR_FUTURE_MODES}, not {neighbor_future!r}")
        if sampling_scheme is None:
            from omegaconf import OmegaConf

            sampling_scheme = OmegaConf.create({"criterium": "topk_prob", "num_k": 2048, "temp": 1.0})
        self.synthesizer = MotionSetSynthesizer(model, sampling_scheme, rollout_batch_size=batch_size)
        if build_batch is None:
            from responsibility.predictors.smart.scene_data import catk_batch_builder

            device = next(iter(model.parameters())).device if hasattr(model, "parameters") else "cpu"
            build_batch = catk_batch_builder(device=str(device))
        self.build_batch = build_batch
        self.neighbor_future = neighbor_future
        self.horizon = horizon
        self.history = num_historical_steps
        self.cache_size = cache_size
        self._cache: "OrderedDict[Tuple[int, int], _Context]" = OrderedDict()

    # ------------------------------------------------------------------
    def _context(self, scene: Scene, step: int) -> _Context:
        from responsibility.predictors.smart.native.motion_set import shifted_context

        key = (id(scene), step)
        ctx = self._cache.get(key)
        if ctx is not None and ctx.scene is scene:
            self._cache.move_to_end(key)
            return ctx
        data = self.build_batch(scene, step)
        with shifted_context(data, step, self.history):
            tokenized_map, tokenized_agent = self.synthesizer.tokenize(data)
        ids = [int(i) for i in data["agent"]["id"].tolist()]
        ctx = _Context(scene, data, tokenized_map, tokenized_agent, {tid: r for r, tid in enumerate(ids)})
        self._cache[key] = ctx
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return ctx

    def max_horizon(self, n_steps: int, step: int) -> int:
        """The longest rollout (10 Hz steps, a whole number of tokens) the
        clip's remaining log supports from ``step``: SMART's decoder reads
        the logged token of every step it rolls out (for teacher forcing),
        so it cannot run past the end of the clip
        (native.dataset_builder.last_usable_window_step, solved for the
        horizon)."""
        shift = self.synthesizer.token_shift
        available = n_steps + self.history - 1 - step  # steps of the re-anchored clip
        return max(0, ((available - 1) // shift - 2) * shift)

    def rollout_horizon(self, n_steps: int, step: int) -> int:
        full = self.max_horizon(n_steps, self.history - 1)
        wanted = self.synthesizer.round_horizon(self.horizon) if self.horizon else full
        return min(wanted, self.max_horizon(n_steps, step))

    def distribution(self, scene: Scene, step: int, agent: int, excluded: Sequence[int] = (),
                     avoid_partner: Sequence[int] = ()) -> Optional[RolloutDistribution]:
        if step < self.history - 1 or not scene.valid[agent, step]:
            return None
        if self.rollout_horizon(scene.n_steps, step) <= 0:  # the clip ends within one token
            return None
        ctx = self._context(scene, step)
        row = ctx.rows.get(numeric_id(scene.track_ids[agent]))
        if row is None:
            return None
        removed = tuple(r for r in (ctx.rows.get(numeric_id(scene.track_ids[e])) for e in excluded)
                        if r is not None and r != row)
        speed = float(np.linalg.norm(scene.velocity[agent, step]))
        return RolloutDistribution(scene, step, agent, tuple(excluded), row, removed, {"speed": speed})

    @torch.no_grad()
    def rollouts(self, dist: RolloutDistribution, n: int, generator: Optional[torch.Generator] = None) -> Dict:
        """CAT-K's raw rollout result for ``dist`` (pred_traj_10hz [n, 1, T, 2],
        pred_head_10hz, next_token_idx, next_token_valid, token_log_prob)."""
        from responsibility.predictors.smart.native.motion_set import RolloutJob, shifted_context

        ctx = self._context(dist.scene, dist.step)
        job = RolloutJob(free=(dist.row,), n_samples=n, removed=dist.removed_rows)
        horizon = self.rollout_horizon(dist.scene.n_steps, dist.step)
        seed = None if generator is None else int(torch.randint(0, 2 ** 62, (1,), generator=generator))
        devices = [ctx.tokenized_agent["valid_mask"].device] if ctx.tokenized_agent["valid_mask"].is_cuda else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
            with shifted_context(ctx.data, dist.step, self.history):
                return self.synthesizer.run_jobs(ctx.tokenized_map, ctx.tokenized_agent, [job],
                                                 rollout_horizon=horizon, neighbor_future=self.neighbor_future)[0]

    def sample(self, dist: RolloutDistribution, n: int, generator: Optional[torch.Generator] = None):
        from responsibility.predictors.smart.native.reward_model import mean_token_log_prob

        result = self.rollouts(dist, n, generator)
        trajs = result["pred_traj_10hz"][:, 0].float().cpu().numpy().astype(np.float64)
        log_prob = mean_token_log_prob(result["token_log_prob"][:, 0], result["next_token_valid"][:, 0])
        return None, log_prob.float().cpu(), trajs


def load_smart_predictor(checkpoint: Optional[str], catk_root=None, model_config: str = "smart",
                         device: str = "cpu", neighbor_future: str = "hidden", batch_size: int = 32,
                         horizon: Optional[int] = None) -> SmartPredictor:
    """A trained SMART checkpoint (CAT-K's Lightning .ckpt, with the
    configs/model/<model_config>.yaml it was trained with) as a predictor."""
    import os

    from responsibility.predictors.smart.catk import require_catk

    if not checkpoint:
        raise SystemExit("--model smart needs --checkpoint (a CAT-K SMART .ckpt)")
    root = require_catk(catk_root)
    from responsibility.predictors.smart.native.pipeline import (compose_responsibility_config,
                                                                   default_sampling_scheme, load_smart_model,
                                                                   resolve_device)
    from responsibility.predictors.smart.scene_data import catk_batch_builder

    device = resolve_device(device)
    checkpoint = os.path.abspath(checkpoint)
    cwd = os.getcwd()
    os.chdir(root)  # SMART reads its token files relative to the CAT-K root
    try:
        cfg = compose_responsibility_config(overrides=[f"model={model_config}"])
        model = load_smart_model(cfg.model.model_config, checkpoint, device=device)
    finally:
        os.chdir(cwd)
    history = int(cfg.model.model_config.decoder.num_historical_steps)
    return SmartPredictor(model, default_sampling_scheme(), catk_batch_builder(root, device),
                          neighbor_future=neighbor_future, batch_size=batch_size, horizon=horizon,
                          num_historical_steps=history)
