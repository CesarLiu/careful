"""The statistics and verdict of scripts/smart/compare_adapter_native.py
(the model-dependent part runs in CAT-K's environment)."""

import math

import numpy as np
import pytest

from scripts.smart.compare_adapter_native import comparisons, pair_stats, spearman, summarize, verdict


def test_spearman_and_pair_stats():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 3], [5, 5, 6, 7]) == pytest.approx(1.0)  # ties share a rank
    assert math.isnan(spearman([1, 2], [1, 2]))
    s = pair_stats([0.0, 1.0, 2.0, math.nan], [0.5, 1.5, 2.5, 3.0])
    assert s["n"] == 3 and s["mad"] == pytest.approx(0.5) and s["bias"] == pytest.approx(0.5)
    assert pair_stats([], [])["n"] == 0


def rows(n, noise, shift, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n)
    out = []
    for i in range(n):
        out.append({"scene": str(i % 3), "role": "sdc" if i % 2 else "adv", "neighbour_type": "VEHICLE",
                    "native_safety_0": truth[i] + rng.normal(0, noise),
                    "native_safety_1": truth[i] + rng.normal(0, noise),
                    "adapter_safety_0": truth[i] + shift + rng.normal(0, noise),
                    "adapter_safety_1": truth[i] + shift + rng.normal(0, noise),
                    "native_courtesy_0": abs(truth[i]), "native_courtesy_1": abs(truth[i]),
                    "native_courtesy_vehicles_0": abs(truth[i]),
                    "adapter_courtesy_0": abs(truth[i]), "adapter_courtesy_1": abs(truth[i]) + 1e-3})
    return out


def test_verdict_tells_noise_from_a_shift():
    same = comparisons(rows(400, 0.1, 0.0), "safety", 2)
    assert set(same) == {"native vs native (seed noise)", "adapter vs adapter (seed noise)", "native vs adapter"}
    assert verdict(same, 1.5) is True
    shifted = comparisons(rows(400, 0.1, 0.5), "safety", 2)
    assert verdict(shifted, 1.5) is False
    assert shifted["native vs adapter"]["bias"] == pytest.approx(0.5, abs=0.05)
    assert verdict(comparisons(rows(50, 0.1, 0.0), "safety", 1), 1.5) is None  # no yardstick


def test_summary_reports_every_metric_and_role():
    pairs = rows(60, 0.1, 0.0)
    pairs[0]["neighbour_type"] = "PEDESTRIAN"
    text = summarize(pairs, rows(30, 0.1, 0.0, seed=1), 2, 1.5,
                     {"neighbor_future": "hidden", "adapter_context": "step"})
    assert "## safety (m), per pair" in text and "## courtesy (nats), per window" in text
    assert "| sdc |" in text and "| adv |" in text
    assert "1 of 60 pairs have a non-vehicle neighbour" in text
    assert text.strip().splitlines()[-1].startswith("RESULT: safety CONSISTENT")


def test_unrelated_values_are_different_even_when_close():
    # the same small values, shuffled: the differences are as small as the noise, the ranks are not
    base = rows(400, 0.05, 0.0)
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(base))
    for r, j in zip(base, perm):
        r["adapter_safety_0"] = base[j]["native_safety_0"] + rng.normal(0, 0.05)
        r["adapter_safety_1"] = base[j]["native_safety_1"] + rng.normal(0, 0.05)
    stats = comparisons(base, "safety", 2)
    assert abs(stats["native vs adapter"]["spearman"]) < 0.2
    assert verdict(stats, 1.5) is False


def test_an_empty_smart_map_is_warned_about():
    from responsibility.predictors.smart.scene_data import PLACEHOLDER_MAP, check_map
    from tests.responsibility.conftest import make_scene, track

    scene = make_scene({"0": track((0, 0), (1, 0)), "1": track((5, 0), (1, 0))})
    placeholder = {"map_save": {"traj_pos": np.full((1, 3, 2), PLACEHOLDER_MAP)}}
    with pytest.warns(UserWarning, match="SMART sees no map"):
        assert check_map(scene, placeholder) is False
    assert check_map(scene, {"map_save": {"traj_pos": np.zeros((4, 3, 2))}}) is True
