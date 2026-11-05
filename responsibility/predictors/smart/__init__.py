"""CAT-K's SMART as a motion predictor.

  adapter.SmartPredictor   the predictor interface (base.py) over SMART
                           rollouts, for every predictor-agnostic tool
  scene_data               Scene -> CAT-K's PyG batch, in memory
  native/                  CAT-K's own batched pipeline (exact courtesy,
                           reward networks, reward fine-tuning)

SMART's own code is the repository's ``catk`` package. Importing this package
needs no PyG; building a predictor does.
"""

from responsibility.predictors.smart.adapter import RolloutDistribution, SmartPredictor, load_smart_predictor  # noqa: F401
