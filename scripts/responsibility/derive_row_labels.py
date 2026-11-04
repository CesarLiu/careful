"""Aggressive or not, from recorded facts about right-of-way conflicts.

An annotator looks at a conflict, two cars sweeping the same patch of road
one after the other (label_ooi_aggression.py's conflicts), and records
facts, not a verdict: who held the right of way and why, whether the car
that went first broke its own control, how the second car's speed changed
and why. This script turns the facts and the measured numbers into one
verdict per car, so that the threshold sits in one place and every
conflict is judged by it.

Let F be the car that went first, S the second. Thresholds after the
literature the research pass collected (responsibility/labels/README.md):
PET below 1.5 s is a serious conflict (Hyden 1977; SSAM's default), a
braking of 3.4 m/s^2 is what drivers find comfortable at most (AASHTO) and
what DRAC takes as a conflict (3.35 m/s^2, Archer 2005), and a headway
below 0.9 s is the tailgating limit used for the lane rules.

F is aggressive when the reaction was forced on S by F's going first, S
being the one that held the right of way, or nobody did:
  clear       S held the right of way (F took it), or none did, and the
              conflict was close: PET below PET_SERIOUS s at a crossing
              (HEADWAY s at a merge, where the paths run on together; not
              at two cars passing in opposite directions) or
              S's attributable reaction was `hard` (a peak braking of at
              least HARD m/s^2 over more than one frame) or `stopped`
  middle      PET up to PET_MIDDLE s, or an attributable `slowed`, at a
              crossing: only when F also broke its own control (a red, a
              stop sign it did not stop at, going out of turn at an
              all-way stop). At a stop sign the violation needs
              MIN_APPROACH s of record before the line and a car that
              slowed on the way
  with right  F held the right of way and still forced S to brake `hard` to
              a near stop (below NEAR_STOP m/s) or to stop, at PET below
              PET_MIDDLE s: holding it does not excuse a margin that cost
              the other that much
Otherwise not aggressive: a gap taken with S slowing only a little, or S
slowing for its own reasons, is assertive driving at most. S is never
aggressive in a conflict it entered second.

The size of the reaction is measured, not read off a figure: in the window
from WINDOW s before F entered the shared area to S entering it, `slowed` is
a fall of at least SLOWED m/s from S's running peak, `hard` a peak braking
(0.5 s differences, so one-frame speed spikes drop out) of at least HARD
m/s^2, `stopped` a stop from at least 5 m/s (conflicts_v2.csv). The
annotator gives only the cause. A reaction counts when its cause is
`first_car`; any other cause (its own turn, its own red or stop sign, a lead
vehicle) is the second car's own business. Where the numbers show a
reaction but the annotator recorded none, the cause is `unreviewed` until an
override settles it (--overrides: scene, reaction_cause). `holder` unclear is
judged only by a hard reaction, and flagged. The same holds where the right
of way was read from a stop sign that F passed at NO_STOP_SPEED m/s without
ever slowing (the sign was matched to the wrong approach), or from the rule
that a left turn yields to oncoming traffic while F's own light is not in
the map (a protected left turn cannot be ruled out).

    python -m scripts.responsibility.derive_row_labels --conflicts conflicts_v2.csv --facts "review/main_*.csv" \\
        --duplicates "review/dup_*.csv" --out-dir OUT
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PET_SERIOUS = 1.5  # s
PET_MIDDLE = 2.0  # s
HEADWAY = 0.9  # s: the PET below which a merge (paths within MERGE_ANGLE) is too close
MERGE_ANGLE = 45.0  # deg: paths closer than this run on together (a merge)
OPPOSITE_ANGLE = 165.0  # deg: paths this far apart are two cars passing each other, not a crossing
MIN_APPROACH = 3.0  # s of record before a stop line needed to see whether the car stopped
NO_STOP_SPEED = 5.0  # m/s: a car that never went below this before the line was not stopping for a sign
NEAR_STOP = 2.0  # m/s: a reaction that ends below this has all but stopped the second car
HARD = 3.4  # m/s^2
SLOWED = 1.0  # m/s
WINDOW = 1.0  # s: how long before F entered the shared area the reaction window opens


def reaction_of(c: dict) -> str:
    """none / slowed / hard / stopped from the measured numbers of conflicts_v2.csv."""
    if bool(c.get("second_stopped_from_5", False)):
        return "stopped"
    if float(c.get("second_peak_decel_w", 0.0)) >= HARD:
        return "hard"
    return "slowed" if float(c.get("second_drop", 0.0)) >= SLOWED else "none"


def sign_violation_visible(c: dict) -> bool:
    """Whether a `first_violated_control` of yes can be believed. At a stop sign it needs MIN_APPROACH s of
    record before the line (a record that starts at the line cannot show a stop that came earlier) and a car
    that slowed on the way: one that held NO_STOP_SPEED m/s all along was not approaching a sign, the sign
    was matched to the wrong approach."""
    if c.get("first_control") != "stop":
        return True
    crossed = c.get("first_crossed_t", np.nan)
    low = c.get("first_min_speed_before_line", np.nan)
    if not (crossed == crossed) or crossed < MIN_APPROACH:
        return False
    return not (low == low and low >= NO_STOP_SPEED)


def derive(c: dict, f: dict) -> dict:
    """The verdict on the car that went first, from a conflict's measured numbers ``c`` (pet, angle, first,
    second_peak_decel, ...) and the annotator's facts ``f`` (holder, first_violated_control, reaction,
    reaction_cause). Returns {"first_aggressive": bool, "rule": str, "reason": str}."""
    first = c["first"]
    holder = f["holder"]
    if (str(f.get("basis", "")).startswith("stop_sign") and c.get("first_control") == "stop" and holder != first
            and c.get("first_min_speed_before_line", np.nan) >= NO_STOP_SPEED):
        holder = "unclear"  # F never slowed for the sign the right of way was read from: the sign is not F's
    if (str(f.get("basis", "")).startswith("left_turn") and holder != first
            and c.get("first_rule_control") == "unknown"):
        holder = "unclear"  # F's own light is not in the map: a protected left turn cannot be ruled out
    pet = float(c["pet"])
    merge = float(c["angle"]) < MERGE_ANGLE
    # side by side in opposite directions: the PET says nothing. So too where the two tracks never cross
    # (kind "closest") and head at least MERGE_ANGLE apart: two cars turning past each other
    passing = float(c["angle"]) >= OPPOSITE_ANGLE or (c.get("kind") == "closest" and not merge)
    close = (not passing) and pet < (HEADWAY if merge else PET_SERIOUS)
    reaction = reaction_of(c)
    attributable = f["reaction_cause"] == "first_car"
    hard = attributable and reaction in ("hard", "stopped")
    slowed = attributable and reaction in ("slowed", "hard", "stopped")
    violated = f["first_violated_control"] == "yes" and sign_violation_visible(c)
    peak = float(c.get("second_peak_decel_w", np.nan))
    facts = f"{reaction}/{f['reaction_cause']}, peak {peak:.1f} m/s^2, PET {pet:.1f} s"

    if holder == first:
        forced_near_stop = reaction == "stopped" or float(c.get("second_min_speed_w", np.inf)) < NEAR_STOP
        if hard and forced_near_stop and pet < PET_MIDDLE:
            return dict(first_aggressive=True, rule="with_right",
                        reason=f"{first.upper()} held the right of way but forced a hard reaction ({facts})")
        return dict(first_aggressive=False, rule="lawful", reason=f"{first.upper()} held the right of way ({facts})")
    # the other held it, or nobody did, or it is unclear
    if holder == "unclear":  # who had to yield is not known: only a forced hard reaction can be held against F
        if hard:
            return dict(first_aggressive=True, rule="clear", reason=f"{first.upper()} forced a hard reaction ({facts})")
        return dict(first_aggressive=False, rule="unclear", reason=f"right of way not established ({facts})")
    if close or hard:
        return dict(first_aggressive=True, rule="clear",
                    reason=f"{first.upper()} went first without the right of way ({holder}): "
                           f"{'closer than the limit' if close else 'forced a hard reaction'} ({facts})")
    at_light = c.get("first_control") == "light"  # a red is a violation wherever the paths lead; a rolled stop is not at a merge
    if (pet < PET_MIDDLE or slowed) and violated and (at_light or not merge) and not passing:
        return dict(first_aggressive=True, rule="middle",
                    reason=f"{first.upper()} went first without the right of way ({holder}) and broke its own "
                           f"control ({facts})")
    return dict(first_aggressive=False, rule="assertive",
                reason=f"{first.upper()} went first ({holder} held the right of way) with room to spare ({facts})")


def kappa(a, b) -> float:
    """Cohen's kappa of two label lists."""
    a, b = np.asarray(list(a)), np.asarray(list(b))
    labels = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == k)) * float(np.mean(b == k)) for k in labels)
    return 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)


def load_facts(pattern: str) -> pd.DataFrame:
    parts = [pd.read_csv(f) for f in sorted(glob.glob(pattern))]
    return pd.concat(parts).drop_duplicates("scene") if parts else pd.DataFrame()


def judge(conflicts: pd.DataFrame, facts: pd.DataFrame, overrides: pd.DataFrame = None) -> pd.DataFrame:
    d = conflicts.merge(facts, on="scene")
    d["reaction_n"] = [reaction_of(r) for r in d.to_dict("records")]
    unreviewed = (d.reaction == "none") & (d.reaction_n != "none")  # the numbers show one, the annotator saw none
    d["reaction_cause"] = np.where(unreviewed, "unreviewed", d.reaction_cause)
    if overrides is not None and len(overrides):
        cause = overrides.set_index("scene").reaction_cause
        d["reaction_cause"] = [cause.get(s, c) for s, c in zip(d.scene, d.reaction_cause)]
    verdicts = [derive(r, r) for r in d.to_dict("records")]
    d["first_aggressive"] = [v["first_aggressive"] for v in verdicts]
    d["rule_applied"] = [v["rule"] for v in verdicts]
    d["row_reason"] = [v["reason"] for v in verdicts]
    d["sdc_row_aggressive"] = ((d["first"] == "sdc") & d.first_aggressive).astype(int)
    d["ooi_row_aggressive"] = ((d["first"] == "ooi") & d.first_aggressive).astype(int)
    d["needs_review"] = (d.holder == "unclear") | (d.confidence == "low") | (d.reaction_cause == "unreviewed")
    return d


ROW_KINDS = {"clear": "took_row", "middle": "signal_or_sign", "with_right": "margin"}
LANE_KINDS = ("cut_in", "tailgate", "close_pass")  # the lane-level kinds the first labelling pass settled


def combine(lane: pd.DataFrame, row: pd.DataFrame, human: pd.DataFrame = None) -> pd.DataFrame:
    """One row per scene: a car is aggressive when either the lane-level pass (cut-in, tailgating, close
    pass: ``lane``, the first pass's labels, other kinds dropped) or the right-of-way pass (``row``, the
    judged conflicts) says so. Keeps each pass's kind and reason, and the right-of-way facts."""
    out = lane[["scene", "scenario_id", "sdc_id", "ooi_id"]].copy()
    row = row.set_index("scene")
    for who in ("sdc", "ooi"):
        lane_kind = np.where(lane[f"{who}_kind"].isin(LANE_KINDS), lane[f"{who}_kind"], "")
        lane_on = lane_kind != ""
        row_on = np.array([bool(row[f"{who}_row_aggressive"].get(s, 0)) for s in lane.scene])
        row_kind = np.array([ROW_KINDS.get(row.rule_applied.get(s, ""), "") if on else "" for s, on in zip(lane.scene, row_on)])
        out[f"{who}_aggressive"] = (lane_on | row_on).astype(int)
        out[f"{who}_kind"] = [lk or rk for lk, rk in zip(lane_kind, row_kind)]
        out[f"{who}_reason"] = [
            (lr if lk else rr if on else "") if (lk or on) else ""
            for lk, on, lr, rr in zip(lane_on, row_on, lane[f"{who}_reason"].fillna(""),
                                      [row.row_reason.get(s, "") for s in lane.scene])]
    for col in ("first", "pet", "angle", "holder", "basis", "first_violated_control", "reaction_n", "reaction_cause",
                "rule_applied", "confidence", "note"):
        out[f"conflict_{col}"] = [row[col].get(s, np.nan) if s in row.index else np.nan for s in lane.scene]
    out["lane_confidence"] = lane.confidence.values
    out["human_checked"] = ""
    if human is not None and len(human):
        out["human_note"] = ""
        for r in human.to_dict("records"):  # a person's verdict on a car replaces the derived one
            i = int(out.index[out.scene == r["scene"]][0])
            for who in ("sdc", "ooi"):
                v = r.get(f"{who}_aggressive")
                if v == v and v is not None and str(v) != "":
                    out.at[i, f"{who}_aggressive"] = int(v)
                    if int(v) == 0:
                        out.at[i, f"{who}_kind"] = ""
                        out.at[i, f"{who}_reason"] = ""
                    out.at[i, "human_checked"] += f"{who}={int(v)}; "
            out.at[i, "human_note"] = r.get("note", "")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--conflicts", required=True)
    p.add_argument("--facts", required=True, help="glob of annotator CSVs (scene, holder, ..., reaction_cause)")
    p.add_argument("--overrides", default=None, help="CSV (scene, reaction_cause) settling unreviewed reactions")
    p.add_argument("--lane-labels", default=None, help="first-pass labels CSV (scene, sdc_kind, ooi_kind, ...)")
    p.add_argument("--human-review", default=None, help="CSV (scene, sdc_aggressive, ooi_aggressive, note): verdicts "
                   "a person gave, replacing the derived ones; disagreements with the derivation are printed")
    p.add_argument("--duplicates", default=None, help="glob of the independent second annotations (agreement)")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    conflicts = pd.read_csv(a.conflicts)
    overrides = pd.read_csv(a.overrides) if a.overrides else None
    d = judge(conflicts, load_facts(a.facts), overrides)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "row_labels.csv", index=False)
    print(f"{len(d)} conflicts judged; first car aggressive in {int(d.first_aggressive.sum())} "
          f"({d.rule_applied.value_counts().to_dict()}); {int(d.needs_review.sum())} flagged for review")
    if a.lane_labels:
        human = pd.read_csv(a.human_review) if a.human_review else None
        final = combine(pd.read_csv(a.lane_labels), d, human)
        if human is not None:
            auto = combine(pd.read_csv(a.lane_labels), d).set_index("scene")
            for r in human.to_dict("records"):
                for who in ("sdc", "ooi"):
                    v = r.get(f"{who}_aggressive")
                    if v == v and str(v) != "" and int(v) != int(auto.at[r["scene"], f"{who}_aggressive"]):
                        print(f"  human differs from the derivation: scene {r['scene']} {who} human {int(v)}, "
                              f"derived {int(auto.at[r['scene'], f'{who}_aggressive'])}")
        final.to_csv(out / "labels.csv", index=False)
        print(f"final: SDC aggressive in {int(final.sdc_aggressive.sum())}, OOI in {int(final.ooi_aggressive.sum())}; "
              f"kinds {final.sdc_kind.value_counts().to_dict()} / {final.ooi_kind.value_counts().to_dict()}")
    if a.duplicates:
        dup = load_facts(a.duplicates)
        both = d.merge(dup, on="scene", suffixes=("", "_2"))
        two = judge(conflicts, dup, overrides).set_index("scene").first_aggressive
        both["agg_2"] = both.scene.map(two)
        rows = [("verdict (first car aggressive)", both.first_aggressive.astype(int), both.agg_2.astype(int))]
        for col in ("holder", "first_violated_control", "reaction_cause"):
            rows.append((col, both[col], both[f"{col}_2"]))
        table = pd.DataFrame([dict(item=n, n=len(x), agreement=float(np.mean(np.asarray(x) == np.asarray(y))),
                                   kappa=kappa(x, y)) for n, x, y in rows])
        table.to_csv(out / "agreement.csv", index=False)
        print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
