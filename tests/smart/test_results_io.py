import json
import pickle

import pytest

from responsibility.predictors.smart.native.results_io import (
    atomic_pickle,
    check_run_config,
    load_observations,
    save_observations,
    scenario_files,
)


def test_atomic_pickle_leaves_no_temporary_files(tmp_path):
    target = tmp_path / "a.pkl"
    atomic_pickle([1, 2, 3], target)
    assert pickle.loads(target.read_bytes()) == [1, 2, 3]
    assert [p.name for p in tmp_path.iterdir()] == ["a.pkl"]


def test_run_config_is_recorded_then_enforced(tmp_path):
    cfg = {"responsibility": {"d_sat": 10.0}, "ckpt": "a.ckpt"}
    check_run_config(tmp_path, cfg)
    assert json.loads((tmp_path / "config.json").read_text()) == cfg

    check_run_config(tmp_path, dict(cfg))  # same settings: fine
    with pytest.raises(ValueError, match="responsibility"):
        check_run_config(tmp_path, {"responsibility": {"d_sat": 5.0}, "ckpt": "a.ckpt"})
    check_run_config(tmp_path, {"responsibility": {"d_sat": 5.0}, "ckpt": "a.ckpt"}, allow_mismatch=True)


def test_load_observations_from_file_and_directories(tmp_path):
    single = tmp_path / "all.pkl"
    save_observations(["x", "y"], str(single))
    assert load_observations(str(single)) == ["x", "y"]

    run = tmp_path / "run"
    (run / "obs").mkdir(parents=True)
    atomic_pickle(["a1", "a2"], run / "obs" / "scenA.pkl")
    atomic_pickle([], run / "obs" / "scenB.pkl")  # a finished scenario without observations
    atomic_pickle(["c1"], run / "obs" / "scenC.pkl")
    assert load_observations(str(run)) == ["a1", "a2", "c1"]

    flat = tmp_path / "flat"
    flat.mkdir()
    atomic_pickle(["f"], flat / "one.pkl")
    assert load_observations(str(flat)) == ["f"]


def test_scenario_files_are_sorted_like_the_dataset(tmp_path):
    for name in ("b.pkl", "a.pkl", "c.pkl"):
        (tmp_path / name).write_bytes(b"")
    assert [p.stem for p in scenario_files(str(tmp_path))] == ["a", "b", "c"]


def test_settings_added_later_match_at_their_default(tmp_path):
    check_run_config(tmp_path, {"responsibility": {"d_sat": 10.0}})  # written before "new" existed
    defaults = {"responsibility": {"d_sat": 10.0, "new": 1.0}}
    check_run_config(tmp_path, {"responsibility": {"d_sat": 10.0, "new": 1.0}}, defaults=defaults)
    with pytest.raises(ValueError, match=r"responsibility.new: \(absent\) -> 0.5"):
        check_run_config(tmp_path, {"responsibility": {"d_sat": 10.0, "new": 0.5}}, defaults=defaults)
    with pytest.raises(ValueError, match="responsibility.new"):  # no defaults given: strict
        check_run_config(tmp_path, {"responsibility": {"d_sat": 10.0, "new": 1.0}})


def test_results_pickled_by_catk4test_still_load():
    """catk4test pickled its observations as src.responsibility.dataset_builder.ResponsibilityObservation
    (and its HMMs as src.responsibility.hmm.GaussianHMM); results_io.load_pickle maps them here."""
    import io
    import pickle

    import torch

    from responsibility.predictors.smart.native.dataset_builder import ResponsibilityObservation
    from responsibility.predictors.smart.native.results_io import legacy_module, load_pickle

    obs = ResponsibilityObservation("s", 7, 2, 30, torch.tensor([0.5, 0.1]))
    old = pickle.dumps([obs], protocol=0).replace(b"responsibility.predictors.smart.native.dataset_builder",
                                                    b"src.responsibility.dataset_builder")
    assert b"src.responsibility.dataset_builder" in old
    back = load_pickle(io.BytesIO(old))
    assert isinstance(back[0], ResponsibilityObservation) and back[0].k == 30
    assert legacy_module("src.responsibility.hmm") == "responsibility.hmm"
    assert legacy_module("src.responsibility.metrics") == "responsibility.predictors.smart.native.metrics"
    assert legacy_module("src.smart.model.smart") == "catk.smart.model.smart"
    assert legacy_module("numpy.core.multiarray") == "numpy.core.multiarray"
