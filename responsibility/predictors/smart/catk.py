"""Locating a CAT-K checkout, whose ``src.smart`` (the SMART model, its
tokenizer and data pipeline) and ``configs/`` the SMART predictor uses.

CAT-K is not vendored: its code is under NVIDIA's Source Code License (and
SMART's Apache 2.0) and its environment (Python 3.11, PyG 2.6.1,
torch_cluster) differs from DenseTNT's (Python 3.9, TensorFlow 2.12). The
checkout is found at, in order: the ``root`` argument, ``$CATK_ROOT``,
``~/catk``. Any CAT-K fork works, e.g. CesarLiu/catk4test.
"""

import os
import sys
from pathlib import Path
from typing import Optional

SMART_CONFIGS = Path(__file__).resolve().parent / "configs"


def catk_root(root=None) -> Optional[Path]:
    for candidate in (root, os.environ.get("CATK_ROOT"), Path.home() / "catk"):
        if candidate and (Path(candidate) / "src" / "smart").is_dir():
            return Path(candidate).expanduser().resolve()
    return None


def ensure_catk_on_path(root=None) -> Optional[Path]:
    """Appends the CAT-K checkout to sys.path (once) and returns it, or None
    if there is none -- importing ``src.smart`` then fails, unless a test has
    registered stand-ins."""
    found = catk_root(root)
    if found is not None and str(found) not in sys.path:
        sys.path.append(str(found))
    return found


def require_catk(root=None) -> Path:
    found = ensure_catk_on_path(root)
    if found is None:
        raise ImportError("CAT-K not found: set CATK_ROOT (or --catk-root) to a CAT-K checkout "
                          "(the directory holding src/smart and configs/)")
    return found
