from __future__ import annotations  # noqa: I001

from dataclasses import replace
from pathlib import Path

import pytest

from benchmax.execution.artifacts.policies import resolve_policy_artifact
from benchmax.execution.jobs.reference import Reference
from benchmax.execution.specs import EntrySpec
from benchmax.storage import (
    STRICT_REQUIREMENTS, AmbiguousPolicyError, LocalStore, PolicyNotFoundError, PolicyValidationError,
)
from benchmax.storage._model import BlobReference, CatalogQuery, ObjectReference
from tests.artifact_helpers import write_policy_artifact
from tests.storage._helpers import _provenance, _release, _result_document, _training


def test_store_publishes_queries_and_materializes_content(policy_store, tmp_path: Path) -> None:
    store = policy_store
    weights = b"portable weights"
    blob = BlobReference.from_source(weights)
    document = _result_document(blob)

    published = store.publish(document, dimensions={"game": "game-1", "study": "study-1"}, blobs={blob: weights})

    assert published.object == ObjectReference.from_document(document)
    assert store.read_document(published.object) == document
    assert store.find(CatalogQuery(kind="result", dimensions={"game": "game-1"})) == (published,)
    assert store.find(CatalogQuery(kind="result", dimensions={"game": "other"})) == ()
    assert store.materialize_blob(blob, tmp_path / "weights.bin").read_bytes() == weights


def test_policy_publication_selection_and_saving(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "store")
    release = _release()
    training = _training(seed=7)
    manifest = write_policy_artifact(tmp_path / "policy", training)

    published = store.publish_policy(Reference.from_path(manifest), release=release)
    stored = store.read_document(published.object)

    selection = {"game": training.game, "algorithm": training.algorithm,
                 "training_config": {"total_frames": 100, "device": "cpu"}}
    assert stored["manifest"]["producer"]["release"] == release.to_document()
    assert store.find_policies(**selection, release=release) == (published,)
    assert store.find_policies(**selection, release=release, seed=8) == ()
    assert store.find_policies(training.game, algorithm=training.algorithm,
                               training_config={"total_frames": 200, "device": "cpu"}, release=release) == ()
    assert store.find_policies(**selection, release=replace(release, version="0.2.0")) == ()

    restored = published.save(tmp_path / "restored")
    restored_path, _, restored_training = resolve_policy_artifact(restored._artifact_metadata())
    assert restored_training == training
    assert (restored_path.parent / "tensors.npz").read_bytes() == (manifest.parent / "tensors.npz").read_bytes()


def test_policy_methods_select_by_game_algorithm_and_seed(policy_store, tmp_path: Path, monkeypatch) -> None:
    from benchmax.execution.artifacts import policies as artifacts
    from benchmax.policy import PolicyProfile

    training = _training(seed=7)
    first = policy_store.publish_policy(write_policy_artifact(tmp_path / "first", training), release=_release())
    second_artifact = write_policy_artifact(tmp_path / "second", _training(seed=8))
    second = policy_store.publish_policy(second_artifact, release=_release())
    matches = policy_store.find_policies(training.game, algorithm=training.algorithm, training_config="any")
    assert {entry.object for entry in matches} == {first.object, second.object}
    assert policy_store.find_policies(training.game, seed=8) == (second,)
    assert policy_store.find_policies(training.game, algorithm=EntrySpec("other"), training_config="any") == ()
    assert policy_store.find_policies(training.game, algorithm=training.algorithm,
                                      training_config={"total_frames": 200, "device": "cpu"}) == ()
    with pytest.raises(AmbiguousPolicyError, match="2 policies match"):
        policy_store.load_policy(training.game, algorithm=training.algorithm, training_config="any")
    with pytest.raises(PolicyNotFoundError):
        policy_store.load_policy(training.game, seed=99)
    loaded = PolicyProfile(lambda *args: {})
    seen = []

    def reconstruct(reference, *, expected_game, max_tensor_bytes, retain_source):
        assert not retain_source
        assert expected_game == training.game
        assert max_tensor_bytes == policy_store.requirements.max_tensor_bytes
        path, _, recorded = resolve_policy_artifact(reference)
        seen.append(recorded.seed)
        assert path.exists()
        return loaded

    monkeypatch.setattr(artifacts, "load_policy", reconstruct)
    assert policy_store.load_policy(training.game, seed=8) is loaded
    assert first.result() is loaded
    assert seen == [8, 7]


def test_strict_validation_is_shared_by_publish_load_and_materialize(policy_store, tmp_path: Path) -> None:
    training = _training(seed=7)
    artifact = write_policy_artifact(tmp_path / "policy", training)
    release = _release()
    published = policy_store.publish_policy(artifact, release=release)
    policy_store.requirements = STRICT_REQUIREMENTS
    report = policy_store.validate_policy(artifact, release=release)
    assert not report.valid
    assert any("provenance" in error for error in report.errors)
    with pytest.raises(PolicyValidationError, match="provenance"):
        policy_store.publish_policy(artifact, release=release)
    assert len(policy_store.find_policies()) == 1
    with pytest.raises(PolicyValidationError, match="provenance"):
        policy_store.load_policy(training.game)
    with pytest.raises(PolicyValidationError, match="provenance"):
        policy_store._save_reference(published.object, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    assert not policy_store.validate_policy(published.object).valid

    complete = policy_store.publish_policy(artifact, release=release, provenance=_provenance(release))
    assert policy_store.validate_policy(complete.object).valid
    restored = complete.save(tmp_path / "restored")
    assert restored.path.is_file()


def test_trained_policy_round_trip_through_store_methods(policy_store, tiny_trained_policy):
    game, policy, _ = tiny_trained_policy
    policy_store.requirements = STRICT_REQUIREMENTS
    assert policy_store.validate_policy(policy).valid
    published = policy_store.publish_policy(policy)
    assert policy_store.validate_policy(published).valid
    original_metrics = game.evaluate(policy)
    training = published.training
    selected = policy_store.load_policy(game, algorithm=training.algorithm, seed=training.seed,
                                        training_config={"total_frames": training.total_frames, "device": training.device})
    assert game.evaluate(selected)["nash_conv"] == pytest.approx(original_metrics["nash_conv"], abs=1e-5)
    evaluation = {"workload_digest": published.training.digest,
                  "tensor_sha256": policy_store.read_document(published.object)["tensor"]["digest"],
                  "metrics": {"nash_conv": original_metrics["nash_conv"]}}
    published = policy_store.publish_policy(published, evaluation=evaluation)
    restored = published.result()
    assert restored._portable_reference is None
    restored_metrics = game.evaluate(restored)
    assert restored_metrics["nash_conv"] == pytest.approx(original_metrics["nash_conv"], abs=1e-5)
    assert restored._artifact_annotations == {**policy._artifact_annotations, "evaluation": evaluation}
    # Republishing retains the producer, even after the loader's temporary directory is gone.
    repeated = policy_store.publish_policy(restored)
    assert repeated == published


def test_materialization_preserves_an_existing_destination(policy_store, tmp_path):
    blob = BlobReference.from_source(b"weights")
    policy_store.publish(_result_document(blob), blobs={blob: b"weights"})
    destination = tmp_path / "destination"
    destination.write_bytes(b"owned by caller")
    with pytest.raises(FileExistsError):
        policy_store.materialize_blob(blob, destination)
    assert destination.read_bytes() == b"owned by caller"


def test_policy_publication_checks_a_pinned_reference(tmp_path):
    from benchmax.execution.artifacts.policies import ArtifactError

    policy_store = LocalStore(tmp_path / "store")
    manifest = write_policy_artifact(tmp_path / "policy", _training(seed=7))
    reference = Reference.from_path(manifest)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ArtifactError, match="digest"):
        policy_store.publish_policy(reference, release=_release())


def test_policy_publication_rejects_metrics_references(tmp_path):
    from benchmax.execution.artifacts.results import WorkloadResult
    from benchmax.utils.json import write_document

    policy_store = LocalStore(tmp_path / "store")
    path = write_document(WorkloadResult.evaluation("0" * 64, {"score": 1.0}), tmp_path / "metrics.json")
    reference = Reference.from_path(path)
    with pytest.raises(TypeError, match="policy output"):
        policy_store.publish_policy(reference, release=_release())


def test_policy_publication_accepts_a_training_job_reference(tmp_path, monkeypatch):
    pytest.importorskip("benchmarl")
    from benchmax import load_game
    from benchmax.training import train
    from tests.training_helpers import tiny_training_params

    policy_store = LocalStore(tmp_path / "store")
    monkeypatch.setenv("BENCHMAX_RUNS", str(tmp_path / "runs"))
    job = train(load_game("rock_paper_scissors"), seed=7, total_frames=8, device="cpu", verbose=False,
                submit=True, params=tiny_training_params()).run()
    published = policy_store.publish_policy(job.reference())
    assert policy_store.validate_policy(published.object).valid
