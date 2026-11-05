"""CAT-K (Zhang et al., "Closed-Loop Supervised Fine-Tuning of Tokenized
Traffic Models", CVPR 2025) as forked in CesarLiu/catk4test @ 1bc7e22: the
SMART tokenized traffic model (smart/), WOMD preprocessing
(data_preprocess.py), training and evaluation (run.py with configs/), and
utilities (utils/). Copied into this repository unchanged except for the
package name (``src.`` -> ``catk.``), since CAT-K is no longer updated.
Under the NVIDIA Source Code License for CAT-K (LICENSE.CATK at the
repository root) and SMART's Apache 2.0 (smart/LICENSE.SMART).

    python -m catk.data_preprocess --split validation --input_dir <womd> --output_dir data/womd_cache
    python -m catk.run experiment=pre_bc            # training (scripts/catk/train.sh)
"""

from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / "configs"
