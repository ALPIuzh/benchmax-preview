"""Stored policies use the same reference and execution boundaries as job outputs."""

import json
from dataclasses import replace

import pytest

from benchmax.execution.jobs.reference import Reference
from benchmax.execution.specs import EntrySpec, TrainingSpec
from benchmax.storage import AmbiguousPolicyError, LocalStore, StorageError
from tests.artifact_helpers import write_policy_artifact
from tests.storage._helpers import _release, _training


def test_named_algorithm_selects_default_configuration_across_seeds(policy_store, tmp_path):
    default = TrainingSpec.create(game="rock_paper_scissors", algorithm="mappo", seed=7)
    custom = replace(default, total_frames=200)
    model_overrides = {"params": {"model": {"num_cells": [32]}}}
    custom_model = replace(default, algorithm=EntrySpec.create("algorithm", "mappo", model_overrides))
    specs = [default, replace(default, seed=8), custom, custom_model,
             replace(default, algorithm=EntrySpec("another-algorithm"))]
    for index, training in enumerate(specs):
        artifact = write_policy_artifact(tmp_path / str(index), training)
        policy_store.publish_policy(artifact, release=_release())

    references = policy_store.find_policies(default.game, algorithm="mappo")
    assert {reference.training.seed for reference in references} == {7, 8}
    assert all(reference.training.total_frames is None for reference in references)
    assert len(policy_store.find_policies(default.game)) == 5
    assert len(policy_store.find_policies(default.game, algorithm="mappo", training_config={"total_frames": 200})) == 1
    assert len(policy_store.find_policies(default.game, algorithm="mappo", seed=8)) == 1
    assert len(policy_store.find_policies(algorithm="mappo", training_config="any")) == 4
    assert len(policy_store.find_policies(default.game, algorithm="mappo", training_config="any", seed=7)) == 3
    assert policy_store.find_policies(algorithm="missing", training_config="any") == ()
    selected = policy_store.find_policies(default.game, algorithm="mappo", training_config=model_overrides)
    assert len(selected) == 1 and selected[0].training == custom_model
    with pytest.raises(ValueError, match="seed filter"):
        policy_store.find_policies(default.game, algorithm="mappo", training_config={"seed": 7})
    with pytest.raises(AmbiguousPolicyError):
        policy_store.load_policy(default.game, algorithm="mappo")


def test_stored_reference_round_trip_and_save_without_model_loading(policy_store, tmp_path, monkeypatch):
    from benchmax.execution.artifacts import policies

    if hasattr(policy_store, "token"):
        policy_store.token = "hf_reference_test_credential"
    artifact = write_policy_artifact(tmp_path / "policy", _training(seed=7))
    def no_backend_construction(*args, **kwargs):
        raise AssertionError("creating and decoding references must not construct backends")

    with monkeypatch.context() as patch:
        patch.setattr(type(policy_store), "__init__", no_backend_construction)
        reference = policy_store.publish_policy(artifact, release=_release())
        document = reference.to_document()
        restored = Reference.from_document(json.loads(json.dumps(document)))
    assert "hf_reference_test_credential" not in json.dumps(document) + repr(reference) + reference._repr_html_()
    assert restored == reference
    assert policy_store.publish_policy(reference) == reference
    assert restored.training.seed == 7
    assert "Content digest" in repr(restored)

    def unexpected(*args, **kwargs):
        raise AssertionError("saving must not construct a model")

    monkeypatch.setattr(policies, "load_policy", unexpected)
    saved = restored.save(tmp_path / "saved")
    assert saved.kind == "file"
    assert saved.path.read_bytes() == (reference.save(tmp_path / "second").path).read_bytes()
    with pytest.raises(FileExistsError):
        restored.save(tmp_path / "saved")


def test_missing_producer_is_not_invented_for_legacy_artifacts(tmp_path):
    store = LocalStore(tmp_path / "store")
    artifact = write_policy_artifact(tmp_path / "policy", _training(seed=7))
    with pytest.raises(StorageError, match="no producing release"):
        store.publish_policy(artifact)
    assert store.find() == ()


def test_stored_policy_reference_works_as_a_job_input(tmp_path, monkeypatch, tiny_trained_policy):
    game, policy, _ = tiny_trained_policy
    store = LocalStore(tmp_path / "store")
    reference = store.publish_policy(policy)
    monkeypatch.setenv("BENCHMAX_RUNS", str(tmp_path / "runs"))
    direct = game.evaluate(reference)
    job = game.evaluate(reference, submit=True)
    assert job.progress().dependencies == ()
    submitted = job.run().result()
    assert submitted["nash_conv"] == pytest.approx(direct["nash_conv"], abs=1e-5)

    run = tmp_path / "runs" / job.id
    assert json.loads((run / "spec.json").read_text())["policy"]["kind"] == "stored-policy"
    resolved = json.loads((run / "resolved-spec.json").read_text())
    saved = Reference.from_path(resolved["policy"]["uri"])
    assert saved.to_document()["digest"] == resolved["policy"]["digest"]
    import shutil

    shutil.rmtree(store.root)
    assert game.evaluate(saved)["nash_conv"] == pytest.approx(direct["nash_conv"], abs=1e-5)


def test_game_and_training_filters_are_independent(policy_store, tmp_path):
    from benchmax import load_game

    default = TrainingSpec.create(game="rock_paper_scissors", algorithm="mappo", seed=7)
    custom_game = EntrySpec.create("game", "rock_paper_scissors", {"max_episode_steps": 2})
    specs = [default, replace(default, total_frames=200), replace(default, game=custom_game),
             replace(default, game=custom_game, total_frames=200)]
    refs = [policy_store.publish_policy(write_policy_artifact(tmp_path / str(i), spec), release=_release())
            for i, spec in enumerate(specs)]
    assert policy_store.find_policies("rock_paper_scissors", algorithm="mappo") == (refs[0],)
    matching = policy_store.find_policies("rock_paper_scissors", algorithm="mappo", training_config="any")
    assert set(matching) == set(refs[:2])
    matching = policy_store.find_policies("rock_paper_scissors", algorithm="mappo", game_config="any")
    assert set(matching) == {refs[0], refs[2]}
    assert set(policy_store.find_policies("rock_paper_scissors", algorithm="mappo",
                                          game_config="any", training_config="any")) == set(refs)
    assert policy_store.find_policies("rock_paper_scissors", algorithm="mappo", game_config={"max_episode_steps": 2},
                                      training_config={"total_frames": 200}) == (refs[3],)
    configured_game = load_game("rock_paper_scissors", max_episode_steps=2)
    assert policy_store.find_policies(configured_game, algorithm="mappo") == (refs[2],)
    assert policy_store.find_policies("rock_paper_scissors", algorithm="mappo",
                                      game_config={"max_episode_steps": 1}) == (refs[0],)
    assert len(policy_store.find_policies("unregistered", game_config="any", training_config="any")) == 0

    with pytest.raises(ValueError, match="require a game name"):
        policy_store.find_policies(game_config={"max_episode_steps": 2})

    with pytest.raises(ValueError, match="require an algorithm"):
        policy_store.find_policies(training_config={"total_frames": 200})

    with pytest.raises(TypeError, match="configured game"):
        policy_store.find_policies(custom_game, game_config={"max_episode_steps": 1})


def test_stored_reference_checks_embedded_training_on_every_resolution(policy_store, tmp_path):
    original = policy_store.publish_policy(
        write_policy_artifact(tmp_path / "policy", _training(seed=7)), release=_release())
    document = original.to_document()
    document["training"]["seed"] = 99
    forged = Reference.from_document(document)
    assert forged.content_identity() == original.content_identity()
    report = policy_store.validate_policy(forged)
    assert not report.valid
    assert any("reference training" in error for error in report.errors)

    actions = [forged.result, lambda: forged.save(tmp_path / "saved"),
               lambda: forged._load_policy(expected_game=original.training.game),
               lambda: policy_store.publish_policy(forged)]
    for action in actions:
        with pytest.raises(StorageError, match="reference training"):
            action()
    assert not (tmp_path / "saved").exists()
    assert policy_store.find_policies() == (original,)


@pytest.mark.parametrize("requirements", [None, [], {"required_metrics": "score"},
    {"required_metrics": {"score": 1}}, {"expected_release": {}}, {"unknown": True}, {"max_tensor_bytes": True}])
def test_malformed_reference_requirements_fail_at_decode(tmp_path, requirements):
    document = {"schema_version": 1, "kind": "stored-policy", "output": "policy", "digest": "a" * 64,
                "training": _training(seed=7).to_document(),
                "store": {"kind": "local", "root": str(tmp_path / "store"), "requirements": requirements}}
    with pytest.raises(StorageError, match="requirements"):
        Reference.from_document(document)


@pytest.mark.parametrize("failure", ["competing-directory", "interrupted-save"])
def test_saving_a_stored_reference_preserves_other_directories_and_cleans_its_own(tmp_path, monkeypatch, failure):
    from pathlib import Path

    store = LocalStore(tmp_path / "store")
    reference = store.publish_policy(write_policy_artifact(tmp_path / "policy", _training(seed=7)), release=_release())
    destination = tmp_path / "saved"
    if failure == "competing-directory":
        original = Path.mkdir

        def competing_mkdir(path, *args, **kwargs):
            if path == destination and not path.exists():
                original(path)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", competing_mkdir)
        with pytest.raises(FileExistsError):
            reference.save(destination)
        assert destination.is_dir() and list(destination.iterdir()) == []
    else:
        def interrupted(store, record, root):
            (root / "partial").write_bytes(b"partial")
            raise OSError("interrupted save")

        monkeypatch.setattr(LocalStore, "_materialize_record", interrupted)
        with pytest.raises(OSError, match="interrupted save"):
            reference.save(destination)
        assert not destination.exists()
