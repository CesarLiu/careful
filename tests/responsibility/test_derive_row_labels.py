"""The verdicts derive_row_labels.py draws from recorded facts."""

import pytest

from scripts.responsibility.derive_row_labels import derive, kappa


def conflict(pet=2.5, angle=90.0, first="sdc", peak=0.5, drop=0.0, stopped=False):
    return dict(pet=pet, angle=angle, first=first, second_peak_decel_w=peak, second_drop=drop,
                second_stopped_from_5=stopped)


def facts(holder="ooi", violated="no", cause="none"):
    return dict(holder=holder, first_violated_control=violated, reaction_cause=cause)


def test_taking_the_right_of_way_closely_is_aggressive():
    v = derive(conflict(pet=1.0), facts())  # the SDC went first, the OOI held the right of way, PET 1 s
    assert v["first_aggressive"] and v["rule"] == "clear"


def test_a_gap_taken_with_room_to_spare_is_assertive_not_aggressive():
    v = derive(conflict(pet=2.5, peak=1.5, drop=2.0), facts(cause="first_car"))
    assert not v["first_aggressive"] and v["rule"] == "assertive"


def test_a_hard_reaction_forced_on_the_holder_is_aggressive_even_at_a_longer_pet():
    v = derive(conflict(pet=2.6, peak=4.0, drop=4.0), facts(cause="first_car"))
    assert v["first_aggressive"]


def test_a_reaction_with_another_cause_does_not_count():
    # the OOI braked hard, but for its own red light
    v = derive(conflict(pet=2.5, peak=4.0, drop=4.0), facts(cause="red_or_stop"))
    assert not v["first_aggressive"]


def test_the_middle_band_needs_a_broken_control():
    middle = conflict(pet=1.8)
    assert not derive(middle, facts(violated="no"))["first_aggressive"]
    v = derive(middle, facts(violated="yes"))
    assert v["first_aggressive"] and v["rule"] == "middle"


def test_holding_the_right_of_way_does_not_excuse_a_forced_hard_stop():
    stopped = conflict(pet=1.2, stopped=True)
    assert derive(stopped, facts(holder="sdc", cause="first_car"))["rule"] == "with_right"
    # but a close pass without a reaction is lawful
    assert derive(conflict(pet=1.2), facts(holder="sdc"))["rule"] == "lawful"


def test_a_merge_is_judged_by_the_headway_not_the_crossing_pet():
    merge = conflict(pet=1.2, angle=10.0)  # 1.2 s behind: above the 0.9 s tailgating limit
    assert not derive(merge, facts())["first_aggressive"]
    assert derive(conflict(pet=0.6, angle=10.0), facts())["first_aggressive"]


def test_two_cars_passing_in_opposite_directions_are_not_a_crossing():
    v = derive(conflict(pet=0.4, angle=179.0), facts(holder="none"))  # a close pass on a narrow street
    assert not v["first_aggressive"]
    assert derive(conflict(pet=2.0, angle=179.0, peak=4.0, drop=4.0), facts(holder="none", cause="first_car"))["first_aggressive"]


def test_two_cars_turning_past_each_other_are_not_a_crossing():
    # the tracks never cross (kind closest) and head 146 deg apart: a footprint overlap, not a conflict
    c = dict(conflict(pet=0.9, angle=146.0), kind="closest")
    assert not derive(c, facts(holder="ooi"))["first_aggressive"]
    assert derive(dict(c, kind="crossing"), facts(holder="ooi"))["first_aggressive"]


def test_a_stop_sign_violation_needs_a_record_that_could_show_the_stop():
    from scripts.responsibility.derive_row_labels import sign_violation_visible
    sign = dict(first_control="stop", first_crossed_t=4.0, first_min_speed_before_line=1.6)
    assert sign_violation_visible(sign)
    assert not sign_violation_visible(dict(sign, first_crossed_t=2.2))  # the record starts too close to the line
    assert not sign_violation_visible(dict(sign, first_min_speed_before_line=6.0))  # never slowed: wrong sign
    assert sign_violation_visible(dict(first_control="light", first_crossed_t=1.0))
    middle = dict(conflict(pet=1.8), **sign)
    assert derive(middle, facts(violated="yes"))["first_aggressive"]
    assert not derive(dict(middle, first_crossed_t=2.2), facts(violated="yes"))["first_aggressive"]


def test_an_unclear_right_of_way_is_not_held_against_the_first_car():
    close = conflict(pet=1.0)
    assert not derive(close, facts(holder="unclear"))["first_aggressive"]
    assert derive(dict(close, second_peak_decel_w=4.0, second_drop=4.0), facts(holder="unclear", cause="first_car"))["first_aggressive"]
    # the sign the priority came from was never slowed for: F was not facing it
    sign = dict(close, first_control="stop", first_crossed_t=3.8, first_min_speed_before_line=6.0)
    assert not derive(sign, dict(facts(holder="ooi"), basis="stop_sign"))["first_aggressive"]
    assert derive(dict(sign, first_min_speed_before_line=1.0), dict(facts(holder="ooi"), basis="stop_sign"))["first_aggressive"]


def test_a_left_turn_with_an_unmapped_light_may_be_protected():
    left = dict(conflict(pet=0.9), first_rule_control="unknown")
    permissive = dict(facts(holder="ooi"), basis="left_turn")  # the SDC went first; the OOI held priority
    assert not derive(left, permissive)["first_aggressive"]  # F's light is not in the map: not held against it
    assert derive(dict(left, first_rule_control="none"), permissive)["first_aggressive"]  # unsignalised: it yields


def test_the_middle_band_is_for_crossings_not_merges():
    merge = conflict(pet=1.2, angle=35.0)  # paths 35 deg apart run on together
    assert not derive(merge, facts(violated="yes"))["first_aggressive"]


def test_holding_the_right_of_way_costs_only_when_the_other_was_all_but_stopped():
    hard = conflict(pet=1.2, peak=4.0, drop=3.3)
    assert not derive(dict(hard, second_min_speed_w=5.5), facts(holder="sdc", cause="first_car"))["first_aggressive"]
    assert derive(dict(hard, second_min_speed_w=1.0), facts(holder="sdc", cause="first_car"))["first_aggressive"]


def test_the_reaction_is_measured_from_the_numbers():
    from scripts.responsibility.derive_row_labels import reaction_of
    assert reaction_of(conflict(peak=0.5, drop=0.4)) == "none"
    assert reaction_of(conflict(peak=1.0, drop=1.5)) == "slowed"
    assert reaction_of(conflict(peak=3.5, drop=3.0)) == "hard"
    assert reaction_of(conflict(peak=1.0, drop=6.0, stopped=True)) == "stopped"


def test_kappa():
    assert kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert kappa([1, 1, 1, 1], [1, 1, 1, 1]) == 1.0
