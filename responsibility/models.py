"""The command-line options that choose the motion predictor
(responsibility/predictors), shared by every script that loads one:

  --model densetnt           CAT's pretrained DenseTNT (default)
  --model mtr                UniTraj's MTR, --checkpoint
  --model unitraj            any UniTraj method, --checkpoint --unitraj-method <wayformer|autobot|...>
  --model smart              CAT-K's SMART (the catk package), --checkpoint
  --model constant_velocity  kinematic reference (no model, no dependencies)
  --model pkg.module:factory a predictor of your own

``model_settings`` is what identifies the predictor's results in a run's
config.json; for densetnt and mtr it is unchanged from before the predictors
were split out, so older runs resume.
"""

from typing import Dict

from responsibility.predictors import (load_predictor, predictor_names, predictor_settings,  # noqa: F401
                                       resolved_temperature)

MODELS = predictor_names()


def add_model_arguments(parser) -> None:
    g = parser.add_argument_group("motion predictor")
    g.add_argument("--model", default="densetnt",
                   help=f"One of {', '.join(predictor_names())}, or package.module:factory.")
    g.add_argument("--checkpoint", default=None, help="mtr / unitraj / smart: the trained model.")
    g.add_argument("--unitraj-root", default=None, help="mtr / unitraj: the UniTraj checkout (default $UNITRAJ_ROOT).")
    g.add_argument("--unitraj-method", default=None,
                   help="mtr / unitraj: the method config it was trained with (MTR_womd, wayformer, autobot, ...).")
    g.add_argument("--temperature", type=float, default=None,
                   help="mtr / unitraj: softmax temperature of the mode scores (default: the checkpoint's "
                        "calibration file, else 1).")
    g.add_argument("--smart-config", default="smart", help="smart: the configs/model/<name>.yaml of the checkpoint.")
    g.add_argument("--smart-neighbor-future", default="hidden", choices=["hidden", "logged", "simulated"],
                   help="smart: what the sampled agent sees of the others' future (hidden = open loop).")
    g.add_argument("--smart-batch", type=int, default=32, help="smart: scene copies per model call.")


def model_settings(args) -> Dict:
    return predictor_settings(args.model, args)


def load_model(args, device: str):
    return load_predictor(args.model, args, device)
