#!/usr/bin/env bash
# E1 of aggression_generalisation_plan.md: the kinematic attribution A1 (responsibility/attribution.py)
# of every scene set of the aggression test, next to the two-sided safety score rel (A2) of its
# offline tables. Needs the scene sets and offline tables of run_aggression_test.sh (the server);
# no model, CPU only, a few minutes.
#
# usage (from the repository root):
#   bash scripts/responsibility/run_attribution_e1.sh
#   CONFIG=nms_ooi_boltz bash scripts/responsibility/run_attribution_e1.sh
#
# variables (defaults): VAL (logs/responsibility/validity), SCENES (raw_scenes_500),
#   CAT_SCENES (adv_scenes/cat), N (100), CONFIG (nms_ooi), WORKERS (8)
set -euo pipefail
VAL=${VAL:-logs/responsibility/validity}
SCENES=${SCENES:-raw_scenes_500}
CAT_SCENES=${CAT_SCENES:-adv_scenes/cat}
N=${N:-100}
CONFIG=${CONFIG:-nms_ooi}
WORKERS=${WORKERS:-8}
PY=${PYTHON:-python}
mkdir -p "$VAL/attribution"
for set in $($PY -c "from responsibility.aggression_test import SCENE_SETS; print(' '.join(SCENE_SETS))"); do
  folder=$($PY -c "from responsibility.aggression_test import scene_dir; print(scene_dir('$set', '$VAL', '$SCENES', '$CAT_SCENES'))")
  if [ ! -d "$folder" ]; then echo "skip $set: no $folder"; continue; fi
  [ -f "$VAL/attribution/$set.csv" ] || \
    $PY -m scripts.responsibility.attribute_pairs --scenes "$folder" --n "$N" --out "$VAL/attribution/$set.csv" --workers "$WORKERS"
done
$PY -m scripts.responsibility.analyze_aggression_offline --val-dir "$VAL" --config "$CONFIG" --report attribution
