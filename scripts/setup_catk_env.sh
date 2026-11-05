#!/usr/bin/env bash
# Prepares CAT-K's environment (Python 3.11; CAT-K's install/setup_server.sh)
# to run all of this repository: the predictor-agnostic metrics, DenseTNT,
# SMART, UniTraj's methods and the MetaDrive-free TD3 training (rl/).
#
#   source ~/venvs/catk/bin/activate && bash scripts/setup_catk_env.sh
#
# What it adds to CAT-K's packages (torch 2.4.1, PyG 2.6.1, torch_cluster,
# TensorFlow 2.12, torchvision, scipy, matplotlib, hydra):
#   - cython < 3, and with it advgen/utils_cython for this Python (DenseTNT's
#     input parsing; the repository ships the Python 3.9 build only)
#   - h5py, einops, pytorch_lightning: UniTraj's dataset and model code
#     (WITH_UNITRAJ=0 skips them)
# and then runs the tests. MetaDrive is not needed and not installed: it does
# not support Python 3.11, and training runs on rl/sim.py instead.
#
#   PIP      the installer (default: uv pip if uv is on PATH, else python -m pip)
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'EOF'
import sys
if sys.version_info[:2] != (3, 11):
    sys.exit(f"expected CAT-K's Python 3.11 environment, got {sys.version.split()[0]}")
EOF
if [ -z "${PIP:-}" ]; then
  if command -v uv >/dev/null 2>&1; then PIP="uv pip"; else PIP="python -m pip"; fi
fi
$PIP install "cython>=0.29.34,<3"
if [ "${WITH_UNITRAJ:-1}" = 1 ]; then
  $PIP install h5py einops "pytorch_lightning==2.4.0"
fi
python scripts/responsibility/build_cython.py
python - <<'EOF'
import importlib
for m in ("torch", "torch_geometric", "torch_cluster", "tensorflow", "torchvision", "scipy", "omegaconf"):
    mod = importlib.import_module(m)
    print(f"{m:16s} {getattr(mod, '__version__', '')}")
import pickle, sys
sys.modules.setdefault("pickle5", pickle)
import advgen.utils_cython  # noqa: F401
print("advgen.utils_cython OK")
EOF
python -m pytest tests -q
