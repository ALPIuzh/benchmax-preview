from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("benchmarl")

from benchmax import PolicyProfile, load_game
from benchmax.evaluation import MatrixEvaluator

from benchmax.execution.artifacts.policies import (  # isort: skip
    POLICY_ARTIFACT_SCHEMA_VERSION, ArtifactError, load_policy, write_policy)
from benchmax.execution.specs import SpecError, TrainingSpec, _ArtifactMetadata
from benchmax.integrations.benchmarl import artifact as policy_artifact
from benchmax.integrations.benchmarl import env as benchmarl_env
from benchmax.integrations.benchmarl import training as benchmarl_training
from benchmax.utils import json as json_utils
from tests.artifact_helpers import write_policy_artifact


@pytest.mark.parametrize("config_drift", [False, True])
def test_neural_policy_round_trips_without_pickling_python(tmp_path, tiny_trained_policy, monkeypatch, config_drift):
    game, policy, spec = tiny_trained_policy
    reference = write_policy(policy, spec, tmp_path / "policy")

    if config_drift:
        from benchmax import config

        read = config._read

        def changed(entry, label):
            value = read(entry, label)
            if label == "game/rock_paper_scissors":
                value["description"] = "Updated game description"
            return value

        monkeypatch.setattr(config, "_read", changed)
        with pytest.raises(SpecError, match="differs from the workload"):
            spec.game.validate_resolution("game")

    restored = load_policy(reference, expected_game=spec.game)

    restored_metrics = game.evaluate(restored)
    original_metrics = game.evaluate(policy)
    assert restored_metrics["exploitability"] == pytest.approx(original_metrics["exploitability"])
    assert restored_metrics["nash_conv"] == pytest.approx(original_metrics["nash_conv"])
    assert restored_metrics["player_improvements"] == pytest.approx(original_metrics["player_improvements"])
    manifest = json.loads((tmp_path / "policy" / "manifest.json").read_text())
    assert manifest["schema_version"] == POLICY_ARTIFACT_SCHEMA_VERSION
    assert manifest["format"] == policy_artifact.POLICY_FORMAT
    assert manifest["policy_interface"] == {
        "players": [
            {"player": 0, "observation_space": [1], "action_space": [0, 1, 2]},
            {"player": 1, "observation_space": [1], "action_space": [0, 1, 2]},
        ],
        "state_space": [7],
    }
    assert not any(path.suffix in {".pkl", ".pickle", ".pt", ".pth"} for path in tmp_path.rglob("*"))


@pytest.mark.parametrize("failure_at", [None, "state", "profile"])
def test_policy_hydration_always_closes_the_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str | None
) -> None:
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7, total_frames=1)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    state_error = RuntimeError("state failed") if failure_at == "state" else None
    experiment = SimpleNamespace(
        group_policies={f"player_{p}": SimpleNamespace(load_state_dict=Mock(side_effect=state_error)) for p in (0, 1)},
        close=Mock(),
    )
    make_profile = Mock(return_value=lambda *args: None,
                        side_effect=RuntimeError("profile failed") if failure_at == "profile" else None)

    monkeypatch.setattr(policy_artifact, "_load_tensor_states", lambda *args, **kwargs: {0: {}, 1: {}})
    monkeypatch.setattr(benchmarl_training, "_create_experiment", Mock(return_value=experiment))
    monkeypatch.setattr(benchmarl_env, "_BenchMARLPolicyProfile", make_profile)
    manifest = {
        "schema_version": 1,
        "kind": "policy-artifact",
        "format": policy_artifact.POLICY_FORMAT,
        "policy_interface": policy_artifact._PolicyInterface.from_game(load_game("rock_paper_scissors")).to_document(),
    }

    if failure_at is None:
        profile = policy_artifact.load_policy_document(manifest_path, manifest, spec, expected_game=spec.game)
        assert isinstance(profile, PolicyProfile)
    else:
        with pytest.raises(RuntimeError, match=f"{failure_at} failed"):
            policy_artifact.load_policy_document(manifest_path, manifest, spec, expected_game=spec.game)

    experiment.close.assert_called_once_with()
    experiment.group_policies["player_0"].load_state_dict.assert_called_once_with({}, strict=True)


def test_policy_artifact_format_is_validated_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    document = {"schema_version": POLICY_ARTIFACT_SCHEMA_VERSION, "kind": "policy-artifact", "format": "other"}
    path.write_text(json.dumps(document), encoding="utf-8")
    game = TrainingSpec.create(game="rock_paper_scissors", seed=7).game

    with pytest.raises(ArtifactError, match="unsupported policy artifact"):
        load_policy(str(path), expected_game=game)


def test_policy_loader_preflights_tensors_before_loading(tmp_path, monkeypatch):
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    manifest_path = write_policy_artifact(tmp_path / "policy", spec)
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_interface"] = policy_artifact._PolicyInterface.from_game(
        load_game("rock_paper_scissors")
    ).to_document()
    manifest_path.write_text(json.dumps(manifest))
    tensor_path = tmp_path / "policy" / "tensors.npz"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered")

    def unexpected_load(*args, **kwargs):
        raise AssertionError("np.load ran before artifact validation")

    monkeypatch.setattr(np, "load", unexpected_load)

    with pytest.raises(ArtifactError, match="tensor digest"):
        load_policy(str(manifest_path), expected_game=spec.game)


def test_policy_reuse_does_not_depend_on_the_configured_evaluator(tmp_path: Path, tiny_trained_policy) -> None:
    game, policy, spec = tiny_trained_policy
    reference = write_policy(policy, spec, tmp_path / "policy")
    game_without_evaluator = TrainingSpec.create(
        game="rock_paper_scissors", game_overrides={"evaluator": None}, seed=7
    ).game

    restored = load_policy(reference, expected_game=game_without_evaluator)
    restored_game = load_game("rock_paper_scissors", evaluator=None)
    restored_metrics = restored_game.evaluate(restored, evaluator=MatrixEvaluator())
    original_metrics = game.evaluate(policy)

    assert isinstance(restored, PolicyProfile)
    assert restored_metrics["nash_conv"] == pytest.approx(original_metrics["nash_conv"])
    assert restored_metrics["player_improvements"] == pytest.approx(original_metrics["player_improvements"])


def test_policy_rejects_a_different_game(tmp_path: Path, tiny_trained_policy) -> None:
    _, policy, spec = tiny_trained_policy
    reference = write_policy(policy, spec, tmp_path / "policy")

    with pytest.raises(ArtifactError, match="incompatible with game"):
        load_policy(
            _ArtifactMetadata(reference.uri, reference.digest),
            expected_game=TrainingSpec.create(game="kuhn_poker", seed=7).game,
        )


def test_policy_rejects_a_tampered_interface(tmp_path: Path, tiny_trained_policy) -> None:
    _, policy, spec = tiny_trained_policy
    reference = write_policy(policy, spec, tmp_path / "policy")
    manifest_path = Path(reference.uri)
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_interface"]["players"][0]["observation_space"] = [999]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactError, match="policy interface is incompatible"):
        load_policy(str(manifest_path), expected_game=spec.game)


def test_neural_profile_save_uses_existing_reference_lifecycle(tmp_path, tiny_trained_policy):
    game, policy, _ = tiny_trained_policy
    reference = policy.save(tmp_path / "policy")
    assert game.evaluate(reference) == game.evaluate(policy)
    assert game.evaluate(reference.result()) == game.evaluate(policy)


def test_neural_save_rejects_invalid_spaces_before_writing(tmp_path, monkeypatch, tiny_trained_policy):
    _, policy, _ = tiny_trained_policy
    implementation = policy._batched_policy_profile
    monkeypatch.setattr(implementation, "_interface", replace(implementation._interface, state_space=(0,)))

    with pytest.raises(ArtifactError, match="policy state_space"):
        policy.save(tmp_path / "invalid")

    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("failure_at", ["savez", "write_document"])
def test_failed_neural_save_removes_incomplete_output(tmp_path, monkeypatch, tiny_trained_policy, failure_at):
    _, policy, _ = tiny_trained_policy

    def fail(*args, **kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr(np if failure_at == "savez" else json_utils, failure_at, fail)
    with pytest.raises(OSError, match="disk write failed"):
        policy.save(tmp_path / "failed")

    assert not (tmp_path / "failed").exists()
