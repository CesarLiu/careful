"""Motion predictors for the responsibility metrics, behind one interface
(base.py) and one registry.

  constant_velocity  dependency-free kinematic reference (constant_velocity.py)
  densetnt           CAT's pretrained DenseTNT (densetnt.py; TensorFlow + advgen/)
  mtr                UniTraj's MTR with all 64 intention queries (unitraj.py)
  unitraj            any other UniTraj method -- Wayformer, AutoBot, EMP, ... --
                     through its own forward pass (unitraj.py, --unitraj-method)
  smart              CAT-K's SMART, counterfactual rollouts with the other agents
                     teacher-forced or hidden (smart/ and the catk package)

Each is imported only when chosen, so none brings in another's dependencies.
A predictor of your own is registered with ``register_predictor`` or named
on the command line as ``--model package.module:factory``, where
``factory(args, device)`` returns an object with ``distribution`` and
``sample`` (base.MotionPredictor).
"""

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from responsibility.predictors.base import MotionPredictor, capabilities, require  # noqa: F401


@dataclass
class PredictorEntry:
    load: Callable  # (args, device) -> predictor
    settings: Callable  # args -> dict stored in a run's config.json
    help: str = ""


_REGISTRY: Dict[str, PredictorEntry] = {}


def register_predictor(name: str, load: Callable, settings: Optional[Callable] = None, help: str = "") -> None:
    _REGISTRY[name] = PredictorEntry(load, settings or (lambda args: {"name": name}), help)


def predictor_names():
    return tuple(_REGISTRY)


def entry(name: str) -> PredictorEntry:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if ":" in name:  # package.module:factory
        module, attr = name.split(":", 1)
        factory = getattr(importlib.import_module(module), attr)
        return PredictorEntry(factory, lambda args: {"name": name})
    raise ValueError(f"unknown predictor {name!r}: one of {predictor_names()} or package.module:factory")


def load_predictor(name: str, args, device: str):
    return entry(name).load(args, device)


def predictor_settings(name: str, args) -> Dict:
    return entry(name).settings(args)


# ----------------------------------------------------------------------
# the built-in predictors
# ----------------------------------------------------------------------

def _arg(args, name, default=None):
    return getattr(args, name, default) if getattr(args, name, None) is not None else default


def _load_cv(args, device):
    from responsibility.predictors.constant_velocity import ConstantVelocity

    return ConstantVelocity()


def _load_densetnt(args, device):
    from responsibility.predictors.densetnt import DenseTNT

    return DenseTNT(device=device)


def _unitraj_method(args, default):
    return _arg(args, "unitraj_method", default)


def resolved_temperature(args) -> Optional[float]:
    if getattr(args, "model", None) not in ("mtr", "unitraj"):
        return None
    if _arg(args, "temperature") is not None:
        return float(args.temperature)
    from responsibility.predictors.unitraj import load_calibration

    return load_calibration(args.checkpoint) if _arg(args, "checkpoint") else 1.0


def _load_unitraj(args, device, default_method):
    if not _arg(args, "checkpoint"):
        raise SystemExit(f"--model {args.model} needs --checkpoint")
    from responsibility.predictors.unitraj import (Inputs, UniTrajModel, load_config, mtr_predictor,
                                                   unitraj_predictor)

    root = _arg(args, "unitraj_root")
    config = load_config(method=_unitraj_method(args, default_method), root=root)
    if config["model_name"] == "MTR":
        predictor = mtr_predictor(args.checkpoint, config, device=device, root=root)
    else:
        predictor = unitraj_predictor(args.checkpoint, config, device=device, root=root)
    return UniTrajModel(predictor, Inputs(config, root=root), temperature=resolved_temperature(args))


def _unitraj_settings(args, default_method):
    from pathlib import Path

    ckpt = _arg(args, "checkpoint")
    return {"name": args.model, "checkpoint": str(Path(ckpt).resolve()) if ckpt else None,
            "method": _unitraj_method(args, default_method), "temperature": resolved_temperature(args)}


def _load_smart(args, device):
    from responsibility.predictors.smart import load_smart_predictor

    return load_smart_predictor(checkpoint=_arg(args, "checkpoint"), model_config=_arg(args, "smart_config", "smart"), device=device,
                                neighbor_future=_arg(args, "smart_neighbor_future", "hidden"),
                                batch_size=_arg(args, "smart_batch", 32))


def _smart_settings(args):
    from pathlib import Path

    ckpt = _arg(args, "checkpoint")
    return {"name": "smart", "checkpoint": str(Path(ckpt).resolve()) if ckpt else None,
            "config": _arg(args, "smart_config", "smart"),
            "neighbor_future": _arg(args, "smart_neighbor_future", "hidden")}


register_predictor("densetnt", _load_densetnt, help="CAT's pretrained DenseTNT")
register_predictor("mtr", lambda a, d: _load_unitraj(a, d, "MTR_womd"),
                   lambda a: _unitraj_settings(a, "MTR_womd"), help="UniTraj MTR, all 64 intentions")
register_predictor("unitraj", lambda a, d: _load_unitraj(a, d, "wayformer"),
                   lambda a: _unitraj_settings(a, "wayformer"), help="any UniTraj method (--unitraj-method)")
register_predictor("smart", _load_smart, _smart_settings, help="CAT-K SMART (the catk package)")
register_predictor("constant_velocity", _load_cv, help="kinematic reference, no dependencies")
