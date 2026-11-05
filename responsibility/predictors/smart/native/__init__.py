"""CAT-K's SMART responsibility pipeline, batched and token-exact
(moved here from the catk4test repository's src/responsibility).

Implements Hsu, Leung, Chen, Fisac & Pavone, "Interpretable Trajectory
Prediction for Autonomous Vehicles via Counterfactual Responsibility"
(IROS 2023) directly on SMART's tokenized scenes (PyG ``HeteroData`` from
CAT-K's WOMD cache). Next to the predictor-agnostic metrics in
``responsibility.metrics`` (which reach SMART through
``responsibility.predictors.smart.SmartPredictor``), this package keeps what
only a token model can do: many counterfactual rollout jobs packed into one
model call, the exact per-token courtesy KL (``--courtesy-estimator exact``),
the per-level reward networks, responsibility-aware candidate selection and
the reward fine-tuning of SMART (finetune.py, the RL part on SMART).

The SMART model code itself is CAT-K's, in this repository's ``catk``
package (``catk.smart``, ``catk.data_preprocess``). ``geometry``, ``risk`` and
``hmm`` are the shared ones in ``responsibility/`` (they were identical
copies).

Submodules
----------
motion_set      Counterfactual motion-set synthesis via SMART rollouts.
metrics         Safety (Eq. 3) and courtesy (Eq. 4/7) responsibility.
interaction     Interacting-neighbour selection on HeteroData.
reward_model    Per-level IRL-style reward networks (Sec IV-B).
dataset_builder Responsibility-augmented dataset construction.
selection       Online responsibility-aware trajectory selection.
finetune        Reward fine-tuning of SMART toward a responsibility level.
scenario_tags   Interaction / scene type tags of agent pairs.
pipeline        End-to-end orchestration used by scripts/smart/*.
"""
