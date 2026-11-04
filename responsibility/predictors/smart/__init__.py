"""CAT-K's SMART as a motion predictor.

  adapter.SmartPredictor   the predictor interface (base.py) over SMART
                           rollouts, for every predictor-agnostic tool
  scene_data               Scene -> CAT-K's PyG batch, in memory
  native/                  CAT-K's own batched pipeline (exact courtesy,
                           reward networks, reward fine-tuning)
  catk                     finding the CAT-K checkout ($CATK_ROOT)
  configs/                 the Hydra config composed over CAT-K's configs

Importing this package needs neither CAT-K nor PyG; building a predictor does.
"""

from responsibility.predictors.smart.adapter import RolloutDistribution, SmartPredictor, load_smart_predictor  # noqa: F401
