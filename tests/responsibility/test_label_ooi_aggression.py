"""The candidate rules of label_ooi_aggression.py on synthetic pairs: each
fires for the one who acted (the SDC is agent 0, its OOI partner agent 1)
and not for the one acted upon."""

import numpy as np
import pytest

from scripts.responsibility.label_ooi_aggression import box_gap, evidence
from scripts.responsibility.visualize_responsibility import box
from tests.responsibility.conftest import make_scene, track
from tests.responsibility.test_forced_yield import _pair


def fired(scene):
    row, _ = evidence(scene)
    return row["sdc_fired"], row["ooi_fired"]


def test_box_gap():
    a = box(np.array([0.0, 0.0]), 0.0, 4.0, 2.0)
    assert box_gap(a, box(np.array([5.0, 0.0]), 0.0, 4.0, 2.0)) == pytest.approx(1.0)  # end to end
    assert box_gap(a, box(np.array([0.0, 3.0]), 0.0, 4.0, 2.0)) == pytest.approx(1.0)  # side by side
    assert box_gap(a, box(np.array([3.0, 0.0]), 0.0, 4.0, 2.0)) == 0.0  # overlapping
    # a box turned 90 deg, its near side 1 m beyond a's end
    assert box_gap(a, box(np.array([4.0, 0.0]), np.pi / 2, 4.0, 2.0)) == pytest.approx(1.0)


def test_took_row_is_charged_to_the_one_who_went_first():
    # the SDC (stop sign) crosses ahead of the OOI (no control, the holder) 1 s before it arrives
    sdc, ooi = fired(_pair(40, 50))
    assert "took_row" in sdc and not ooi
    # the SDC crosses 2 s behind the holder: it is not aggressive. It never slows for its stop sign,
    # though, so the holder entered in front of a car that would have needed 5.2 m/s^2 to stop:
    # holding the right of way does not excuse that margin
    sdc, ooi = fired(_pair(60, 40))
    assert not sdc and "margin" in ooi


def _cut_in(gap: float):
    """The SDC changes lanes (3.5 m, steps 30-50) into the lane of the OOI, ``gap`` m ahead of its
    front bumper, both at 15 m/s; the clip ends 2 s after the change."""
    valid = np.arange(91) <= 70
    sdc = track((gap + 4.8, 3.5), (15.0, 0.0), valid=valid)
    lane = np.clip((np.arange(91) - 30) / 20, 0, 1)
    sdc["state"]["position"][:, 1] = 3.5 * (1 - lane)
    return make_scene({"0": sdc, "1": track((0.0, 0.0), (15.0, 0.0), valid=valid)}, map_features={}, lights={})


def test_cut_in_is_charged_to_the_one_who_changed_lanes():
    sdc, ooi = fired(_cut_in(4.0))  # 0.27 s headway left to the OOI
    assert "cut_in" in sdc and not ooi
    sdc, ooi = fired(_cut_in(40.0))  # 2.7 s: a lane change with room
    assert not sdc and not ooi


def _following(gap: float):
    """The SDC follows the OOI in its lane at 15 m/s, ``gap`` m between the bumpers."""
    return make_scene({"0": track((0.0, 0.0), (15.0, 0.0)), "1": track((gap + 4.8, 0.0), (15.0, 0.0))},
                      map_features={}, lights={})


def test_tailgating():
    sdc, ooi = fired(_following(6.0))  # 0.4 s for the whole clip
    assert "tailgate" in sdc and not ooi
    sdc, ooi = fired(_following(30.0))  # 2 s
    assert not sdc and not ooi
