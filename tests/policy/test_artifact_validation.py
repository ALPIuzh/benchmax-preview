"""Data-only artifact validation is available without the training extra."""

import hashlib
import json

import numpy as np
import pytest

from benchmax.execution.artifacts.policies import ArtifactError, validate_policy_artifact
from benchmax.execution.specs import EntrySpec, TrainingSpec, _ArtifactMetadata
from tests.artifact_helpers import forbid_neural_imports, write_policy_artifact


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    training = TrainingSpec(EntrySpec("test-game"), EntrySpec("test-algorithm"), seed=7, device="cpu", total_frames=8)
    path = write_policy_artifact(tmp_path / "artifact", training)
    manifest = json.loads(path.read_text())
    manifest["state_values"] = [{"player": 0, "state_name": "shape", "type": "torch.Size", "value": [2]}]
    path.write_text(json.dumps(manifest))
    forbid_neural_imports(monkeypatch, "artifact inspection")

    def unexpected_array_load(*args, **kwargs):
        raise AssertionError("artifact inspection must not load tensor arrays")

    monkeypatch.setattr(np, "load", unexpected_array_load)
    return path, manifest, training


def test_validate_artifact_without_training_or_array_loading(artifact) -> None:
    path, manifest, training = artifact
    assert validate_policy_artifact(str(path)) == (path, manifest, training)


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param(("tensor_sha256",), "0" * 64, id="digest"),
        pytest.param(("tensors", 0, "dtype"), "|O", id="object-dtype"),
        pytest.param(("tensors", 0, "shape"), [3], id="npy-header-shape"),
        pytest.param(("policy_interface", "players", 0, "player"), 1, id="wrong-player"),
        pytest.param(("state_values", 0, "state_name"), "weight", id="duplicate-state-name"),
        pytest.param(("state_values",), {}, id="invalid-structural-index"),
        pytest.param(("state_values", 0, "type"), "object", id="unsupported-structural-type"),
        pytest.param(("state_values", 0, "value"), [True], id="invalid-structural-shape"),
        pytest.param(("tensor_file",), "../tensors.npz", id="path-traversal"),
        pytest.param(("tensor_file",), "..\\tensors.npz", id="windows-path-traversal"),
        pytest.param(("tensor_file",), "tensors\x00.npz", id="nul-in-filename"),
        pytest.param(("schema_version",), True, id="boolean-schema-version"),
    ],
)
def test_invalid_manifest_is_rejected_before_loading(artifact, field, value):
    path, manifest, _ = artifact
    target = manifest

    for key in field[:-1]:
        target = target[key]

    target[field[-1]] = value
    path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactError):
        validate_policy_artifact(str(path))


@pytest.mark.parametrize("budget", [1, 1.5], ids=["too-small", "noninteger"])
def test_tensor_budget_is_checked_before_array_loading(artifact, budget):
    path, _, _ = artifact

    with pytest.raises(ArtifactError, match="load limit|positive integer"):
        validate_policy_artifact(str(path), max_tensor_bytes=budget)


def test_malformed_archive_has_an_artifact_error(artifact):
    path, manifest, _ = artifact
    tensor = path.parent / manifest["tensor_file"]
    tensor.write_bytes(b"invalid archive")
    manifest["tensor_sha256"] = hashlib.sha256(tensor.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactError, match="archive"):
        validate_policy_artifact(str(path))


def test_validate_artifact_follows_training_result_reference(artifact) -> None:
    path, manifest, training = artifact
    result = path.parent / "result.json"
    reference = _ArtifactMetadata(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    result.write_text(json.dumps({"kind": "training-result", "policy": reference.to_document()}))
    resolved_path, resolved_manifest, resolved_training = validate_policy_artifact(str(result))
    assert resolved_path == path
    assert resolved_manifest == manifest
    assert resolved_training == training
