from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmax.execution.artifacts.policies import ArtifactError
from benchmax.execution.jobs.reference import Reference
from benchmax.storage import STRICT_REQUIREMENTS, LocalStore, PolicyValidationError, StorageError
from tests.artifact_helpers import write_policy_artifact
from tests.storage._helpers import _provenance, _release, _training


def test_publication_requires_the_document_to_declare_its_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="catalog kind"):
        LocalStore(tmp_path / "store").publish({"value": "missing kind"})


@pytest.mark.parametrize("failure,error,message", [
    ("format", ArtifactError, "unsupported policy artifact"),
    ("seed_scheme", ArtifactError, "seed_scheme"),
    ("dtype", PolicyValidationError, "dtype"),
])
def test_invalid_artifacts_are_rejected_without_publication(tmp_path, failure, error, message):
    store = LocalStore(tmp_path / "store")
    artifact = write_policy_artifact(tmp_path / "policy", _training(seed=7))
    document = json.loads(artifact.read_text())
    if failure == "format":
        document["format"] = "another-policy-format"
    elif failure == "seed_scheme":
        del document["training"]["seed_scheme"]
    else:
        document["tensors"][0]["dtype"] = "|O"
    artifact.write_text(json.dumps(document))

    report = store.validate_policy(artifact, release=_release())
    assert not report.valid
    assert any(message in reason for reason in report.errors)
    with pytest.raises(error, match=message):
        store.publish_policy(artifact, release=_release())
    assert store.find() == ()


def test_invalid_stored_release_stays_inside_the_storage_error_boundary(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "store")
    release = _release()
    training = _training(seed=7)
    manifest = write_policy_artifact(tmp_path / "policy", training)
    published = store.publish_policy(Reference.from_path(manifest), release=release)
    document = store.read_document(published.object)
    document["manifest"]["producer"]["release"] = {"kind": "library-release"}
    invalid = store.publish(document)

    report = store.validate_policy(invalid.object)
    assert not report.valid
    assert any("invalid policy release" in error for error in report.errors)


def test_strict_validation_reports_missing_fields_without_inventing_them(tmp_path: Path) -> None:
    policy_store = LocalStore(tmp_path / "store")
    policy_store.requirements = replace(STRICT_REQUIREMENTS, require_explicit_device=True, require_frame_budget=True)
    artifact = write_policy_artifact(tmp_path / "policy", _training(seed=7))
    document = json.loads(artifact.read_text())
    document["training"]["game"]["resolved"] = None
    document["training"]["algorithm"]["resolved"] = None
    document["training"]["device"] = "auto"
    document["training"]["total_frames"] = None
    artifact.write_text(json.dumps(document))
    report = policy_store.validate_policy(artifact, release=_release())
    assert not report.valid
    for field in ("game", "algorithm", "device", "total_frames", "provenance"):
        assert any(field in error for error in report.errors)


def test_custom_requirements_enforce_release_and_evaluation_evidence(tmp_path: Path) -> None:
    policy_store = LocalStore(tmp_path / "store")
    release = _release()
    training = _training(seed=7)
    artifact = write_policy_artifact(tmp_path / "policy", training)
    policy_store.requirements = replace(STRICT_REQUIREMENTS, required_metrics=("nash_conv",), expected_release=release)
    provenance = _provenance(release)
    wrong = {"workload_digest": "0" * 64, "metrics": {"other": 0.0}}
    report = policy_store.validate_policy(artifact, release=release, provenance=provenance, evaluation=wrong)
    assert any("workload_digest" in error for error in report.errors)
    assert any("nash_conv" in error for error in report.errors)
    tensor_digest = json.loads(artifact.read_text())["tensor_sha256"]
    evaluation = {"workload_digest": training.digest, "tensor_sha256": tensor_digest, "metrics": {"nash_conv": 0.1}}
    report = policy_store.validate_policy(artifact, release=release, provenance=provenance, evaluation=evaluation)
    assert report.valid, report.errors
    newer = replace(release, version="99.0.0")
    report = policy_store.validate_policy(artifact, release=newer, provenance=_provenance(newer), evaluation=evaluation)
    assert not report.valid
    assert any("library name or version differs" in error for error in report.errors)


@pytest.mark.parametrize(
    "field,value,message",
    [
        pytest.param("source_revision", "short", "source_revision", id="short-commit"),
        pytest.param("python_version", "3.11", "python_version", id="partial-python-version"),
        pytest.param("packages", {"benchmax": "latest"}, "exact versions", id="unversioned-package"),
        pytest.param("packages", {"another-library": "0.1.0"}, "missing", id="missing-packages"),
    ],
)
def test_strict_provenance_rejects_invalid_fields(tmp_path, field, value, message):
    store = LocalStore(tmp_path / "store", requirements=STRICT_REQUIREMENTS)
    release = _release()
    provenance = _provenance(release)
    provenance[field] = value
    artifact = write_policy_artifact(tmp_path / "policy", _training(seed=7))
    report = store.validate_policy(artifact, release=release, provenance=provenance)
    assert not report.valid
    assert any(message in error for error in report.errors), report.errors


def test_evaluation_evidence_must_describe_the_published_tensors(tmp_path):
    store = LocalStore(tmp_path / "store")
    training = _training(seed=7)
    artifact = write_policy_artifact(tmp_path / "policy", training)
    evaluation = {"workload_digest": training.digest, "tensor_sha256": "0" * 64, "metrics": {"nash_conv": 0.1}}
    report = store.validate_policy(artifact, release=_release(), evaluation=evaluation)
    assert report.errors == ("evaluation.tensor_sha256 must match the policy tensor archive",)


def test_receiving_store_owns_validation_requirements(tmp_path):
    source = LocalStore(tmp_path / "source")
    strict = LocalStore(tmp_path / "strict", requirements=STRICT_REQUIREMENTS)
    artifact = write_policy_artifact(tmp_path / "policy", _training(seed=7))
    reference = source.publish_policy(artifact, release=_release())
    report = strict.validate_policy(reference)
    assert report.requirements == STRICT_REQUIREMENTS.name
    assert not report.valid
    assert any("provenance" in error for error in report.errors)
    with pytest.raises(PolicyValidationError, match="provenance"):
        strict.publish_policy(reference)
    assert strict.find() == ()


def test_recorded_producer_cannot_be_relabelled_on_publication(tmp_path):
    source = LocalStore(tmp_path / "source")
    artifact = write_policy_artifact(tmp_path / "policy", _training(seed=7))
    reference = source.publish_policy(artifact, release=_release(), provenance=_provenance(_release()))
    with pytest.raises(StorageError, match="recorded producing release"):
        source.publish_policy(reference, release=replace(_release(), version="99.0.0"))
    with pytest.raises(StorageError, match="recorded producing provenance"):
        source.publish_policy(reference, provenance={"source_revision": "0" * 40})
