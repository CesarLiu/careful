"""tests/rl_training (not tests/rl: pytest would put tests/ first on sys.path and shadow the rl package): the repository root on sys.path, and the pickle5 alias CAT's
advgen expects (advgen.structs imports pickle5; plain pickle does the same
on Python >= 3.8), as responsibility/predictors/densetnt.py sets it."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.modules.setdefault("pickle5", pickle)
