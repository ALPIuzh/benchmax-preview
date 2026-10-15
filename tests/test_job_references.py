from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from benchmax import PolicyProfile, Reference, load_game
from benchmax.execution import Job, LocalTarget
from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.specs import EvaluationSpec, SpecError, TrainingSpec, _ArtifactMetadata, workload_from_document
from benchmax.execution.targets.local import LocalJobService
from benchmax.execution.targets.local.attempt import run_local_attempt
from benchmax.execution.targets.local.service import _write_state
from benchmax.execution.targets.service import JobError, JobFailedError, JobTimeoutError
from tests.artifact_helpers import write_policy_artifact


@pytest.fixture
def local(tmp_path, monkeypatch):
    target = LocalTarget(runs=str(tmp_path / "runs"))
    service = LocalJobService(target.runs, launcher=lambda *args: None)
    monkeypatch.setattr(LocalTarget, "service", lambda self: service)
    training = Job._from_workload(TrainingSpec.create(game="rock_paper_scissors", seed=7)).run(target)
    return target, service, training


def complete(local):
    target, _, training = local
    artifact = _ArtifactMetadata("/saved/manifest.json", "a" * 64)
    run_local_attempt(training.id, runs=target.runs,
                      executor=lambda spec, output: WorkloadResult.training(spec.digest, spec.seed, artifact))
    return artifact


def test_reference_and_evaluation_inspection_are_passive(local, monkeypatch):
    _, _, training = local
    game = load_game("rock_paper_scissors")

    def forbidden(*args, **kwargs):
        pytest.fail("reference creation/inspection must not perform I/O or wait")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", forbidden)
        guard.setattr(subprocess, "run", forbidden)
        guard.setattr(subprocess, "Popen", forbidden)
        guard.setattr(LocalTarget, "service", forbidden)
        guard.setattr("time.sleep", forbidden)
        reference = training.reference()
        evaluation = game.evaluate(reference, submit=True)
        document = evaluation.to_document()
        assert reference == Reference.from_document(reference.to_document())
        assert document["policy"] == reference.to_document()
        assert reference.output == "policy"

    assert training.state == "pending"
    assert evaluation.state == "created"


def test_reference_is_immutable_and_captures_target_settings(local):
    target, _, training = local
    reference = training.reference()
    reference.target["runs"] = "/different"
    document = reference.to_document()
    document["target"]["runs"] = "/changed"
    assert reference.target["runs"] == target.runs

    with pytest.raises(FrozenInstanceError):
        reference.job_id = "changed"

    assert hash(reference) == hash(training.reference())
    restored = Reference.from_document(json.loads(json.dumps(reference.to_document())))
    assert restored == reference


def test_artifact_metadata_is_not_a_public_reference_or_policy_argument(local):
    game = load_game("rock_paper_scissors")
    artifact = _ArtifactMetadata("/manifest.json", "a" * 64)

    with pytest.raises(TypeError):
        game.evaluate(artifact)

    with pytest.raises(TypeError):
        game.evaluate(artifact, submit=True).to_document()


def test_result_path_retains_local_policy_and_returns_loaded_value(local, monkeypatch, tmp_path):
    from benchmax.execution.artifacts import policies as artifacts

    target, _, training = local
    source = tmp_path / "original"
    manifest = write_policy_artifact(source, training._prepare())
    (source / "unrelated.txt").write_text("must not be copied")
    metadata = _ArtifactMetadata.from_path(manifest)
    run_local_attempt(training.id, runs=target.runs,
                      executor=lambda spec, output: WorkloadResult.training(spec.digest, spec.seed, metadata))
    profile = PolicyProfile(lambda batch: batch)
    loaded = []

    def load(ref, *, expected_game):
        loaded.append(ref)
        return profile

    monkeypatch.setattr(artifacts, "load_policy", load)
    destination = tmp_path / "retained"
    assert training.reference().result(path=destination) is profile
    assert {p.name for p in destination.iterdir()} == {"manifest.json", "tensors.npz"}
    assert loaded[-1].uri == str(destination / "manifest.json")
    assert loaded[-1].digest == metadata.digest
    assert asyncio.run(training.result_async(path=tmp_path / "async-retained")) is profile

    with pytest.raises(FileExistsError, match="already exists"):
        training.result(path=destination)

    assert len(loaded) == 2


def test_save_and_file_references_work_without_loading_or_reopening_a_job(local, monkeypatch, tmp_path):
    import shutil

    from benchmax.execution.artifacts import policies as artifacts

    target, _, training = local
    source = tmp_path / "source"
    manifest = write_policy_artifact(source, training._prepare())
    (source / "private.log").write_text("unrelated")
    metadata = _ArtifactMetadata.from_path(manifest)
    run_local_attempt(training.id, runs=target.runs,
                      executor=lambda spec, output: WorkloadResult.training(spec.digest, spec.seed, metadata))

    def forbidden(*args, **kwargs):
        pytest.fail("saving and file resolution must not load models or open jobs here")

    monkeypatch.setattr(artifacts, "load_policy", forbidden)
    saved = training.reference().save(path=tmp_path / "saved")
    assert saved.kind == "file" and saved.output == "policy"
    assert saved.path == tmp_path / "saved/manifest.json"
    assert set(saved.to_document()) == {"schema_version", "kind", "uri", "digest", "output"}
    assert {p.name for p in saved.path.parent.iterdir()} == {"manifest.json", "tensors.npz"}
    assert Reference[PolicyProfile].from_path(saved.path) == saved
    assert Reference[PolicyProfile].from_path(training.output).path == manifest

    shutil.rmtree(target.runs)
    shutil.rmtree(source)
    monkeypatch.setattr(Job, "open", forbidden)
    copied = saved.save(tmp_path / "copied")
    assert copied.content_identity() == saved.content_identity()
    game = load_game("rock_paper_scissors")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", forbidden)
        guard.setattr(Path, "resolve", forbidden)
        restored = Reference[PolicyProfile].from_document(saved.to_document())
        assert restored == saved
        assert "Local file" in repr(restored)
        assert "File identity" in restored._repr_html_()
        evaluation = game.evaluate(restored, submit=True)
        spec = evaluation.to_document()
        assert spec["policy"] == _ArtifactMetadata(str(saved.path), metadata.digest).to_document()

    profile = PolicyProfile(lambda batch: batch)
    loaded = []

    def load(ref, *, expected_game):
        loaded.append(ref.uri)
        return profile

    monkeypatch.setattr(artifacts, "load_policy", load)
    assert restored.result() is profile
    assert restored.result(path=tmp_path / "saved-and-loaded") is profile
    assert loaded == [str(saved.path), str(tmp_path / "saved-and-loaded/manifest.json")]
    monkeypatch.setattr(type(game.evaluator), "evaluate", lambda self, game, policy: {"ok": policy is profile})
    assert game.evaluate(restored) == {"ok": True}


def test_metrics_file_reference_survives_job_removal_and_detects_tampering(local, tmp_path, monkeypatch):
    import shutil

    from benchmax.execution.artifacts.policies import ArtifactError

    target, _, training = local
    complete(local)
    job = load_game("rock_paper_scissors").evaluate(training.reference(), submit=True).run(target)
    metrics = {"exploitability": 0.25}
    run_local_attempt(job.id, runs=target.runs,
                      executor=lambda spec, output: WorkloadResult.evaluation(spec.digest, metrics))
    destination = tmp_path / "retained-metrics"
    assert job.reference().result(path=destination) == metrics
    assert json.loads((destination / "result.json").read_text())["metrics"] == metrics
    saved = job.save(path=tmp_path / "metrics")
    assert saved.output == "metrics" and saved.kind == "file"
    shutil.rmtree(target.runs)
    monkeypatch.setattr(Job, "open", lambda *a, **kw: pytest.fail("file references must not open jobs"))
    restored = Reference[dict].from_document(saved.to_document())
    assert Reference[dict].from_path(saved.path) == restored
    assert restored.result(timeout=0) == metrics
    copied = restored.save(path=tmp_path / "metrics-copy")
    assert copied.path.read_bytes() == restored.path.read_bytes()
    assert restored.result(path=tmp_path / "metrics-load") == metrics

    for submit in (False, True):
        with pytest.raises(TypeError, match="evaluation requires a policy output"):
            load_game("rock_paper_scissors").evaluate(restored, submit=submit)

    with pytest.raises(FileExistsError):
        restored.save(path=tmp_path / "metrics")

    for method in (restored.result, lambda **kw: restored.save(path=tmp_path / "invalid-timeout", **kw)):
        with pytest.raises(ValueError, match="timeout"):
            method(timeout=-1)

    saved.path.write_text('{}')

    with pytest.raises(ArtifactError, match="digest"):
        restored.result()

    with pytest.raises(ArtifactError, match="digest"):
        restored.save(path=tmp_path / "tampered")


@pytest.mark.parametrize("field,value", [("uri", "relative.json"), ("uri", "ssh://host/file"), ("uri", None),
                                         ("digest", None), ("digest", "bad"),
                                         ("job_id", "extraneous"), ("kind", [])])
def test_invalid_file_reference_documents_are_rejected(tmp_path, field, value):
    document = {"schema_version": 1, "kind": "file", "output": "policy", "uri": str(tmp_path / "manifest.json"),
                "digest": "a" * 64}

    with pytest.raises(SpecError):
        Reference.from_document({**document, field: value})


def test_future_reference_is_distinct_and_roundtrips_in_workload(local, monkeypatch):
    _, _, training = local
    spec = EvaluationSpec(game=training._prepare().game, policy=training.reference())
    assert not isinstance(spec.policy, Reference)

    def forbidden(*args, **kwargs):
        pytest.fail("workload deserialization must not construct an executable reference")

    monkeypatch.setattr(Reference, "__init__", forbidden)
    restored = workload_from_document(json.loads(json.dumps(spec.to_document())))
    assert restored == spec and restored.digest == spec.digest
    assert restored.policy.to_document()["kind"] == "job-output"

    with pytest.raises(SpecError):
        _ArtifactMetadata.from_document(restored.policy.to_document())


def test_job_and_metrics_are_not_implicitly_converted_to_policies(local):
    target, _, training = local
    game = load_game("rock_paper_scissors")

    for submit in (False, True):
        with pytest.raises(TypeError, match=r"job.reference\(\)"):
            game.evaluate(training, submit=submit)

    evaluation = game.evaluate(training.reference(), submit=True).run(target)
    metrics_ref = evaluation.reference()
    assert metrics_ref.output == "metrics"

    for submit in (False, True):
        with pytest.raises(TypeError, match="evaluation requires a policy output"):
            game.evaluate(metrics_ref, submit=submit)

    with pytest.raises(SpecError, match="evaluation requires a policy output"):
        EvaluationSpec(game=training._prepare().game, policy=metrics_ref)


def test_reference_requires_submission():
    job = Job._from_workload(TrainingSpec.create(game="rock_paper_scissors", seed=7))

    with pytest.raises(JobError, match="has not been run"):
        job.reference()


@pytest.mark.parametrize("field,value", [("job_id", ""), ("output", []), ("output", "weights"),
                                         ("workload_digest", None), ("target", {"kind": []}),
                                         ("schema_version", 2), ("extra", True)])
def test_invalid_reference_documents_are_rejected(local, field, value):
    document = local[2].reference().to_document()
    document[field] = value

    with pytest.raises(SpecError):
        Reference.from_document(document)


def test_reference_result_reconnects_and_loads_policy_only_when_requested(local, monkeypatch):
    reference = local[2].reference()

    with pytest.raises(JobTimeoutError):
        reference.result(timeout=0)

    artifact = complete(local)
    profile = PolicyProfile(lambda batch: batch)
    loads = []

    def load(saved, *, expected_game):
        loads.append((saved, expected_game))
        return profile

    monkeypatch.setattr("benchmax.execution.artifacts.policies.load_policy", load)
    typed = Reference[PolicyProfile](reference.to_document())
    restored = Reference[PolicyProfile].from_document(typed.to_document())
    assert restored == reference
    assert restored.result(timeout=0) is profile
    assert loads == [(artifact, local[2]._prepare().game)]


def test_synchronous_evaluation_resolves_reference(local, monkeypatch):
    complete(local)
    profile = PolicyProfile(lambda batch: batch)
    monkeypatch.setattr("benchmax.execution.artifacts.policies.load_policy", lambda *args, **kwargs: profile)
    game = load_game("rock_paper_scissors")
    seen = []

    def evaluate(self, game, actual):
        seen.append(actual)
        return {"exploitability": 0.25}

    monkeypatch.setattr(type(game.evaluator), "evaluate", evaluate)
    assert game.evaluate(local[2].reference()) == {"exploitability": 0.25}
    assert seen == [profile]


def test_local_dependency_waits_before_computation_and_records_resolution(local, monkeypatch):
    target, _, training = local
    evaluation = load_game("rock_paper_scissors").evaluate(training.reference(), submit=True).run(target)
    from benchmax.execution.targets.local.service import JobHandle

    waiting, computed = threading.Event(), threading.Event()
    original = JobHandle.result

    def result(self, timeout=None):
        if self.id == training.id:
            waiting.set()

        return original(self, timeout)

    monkeypatch.setattr(JobHandle, "result", result)
    errors = []

    def execute(spec, output):
        assert isinstance(spec.policy, _ArtifactMetadata)
        assert spec.policy.digest == "a" * 64
        computed.set()
        return WorkloadResult.evaluation(spec.digest, {"exploitability": 0.25})

    def attempt():
        try:
            run_local_attempt(evaluation.id, runs=target.runs, executor=execute)
        except BaseException as error:  # noqa: BLE001 -- propagate worker assertions to the test thread
            errors.append(error)

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    assert waiting.wait(3)
    assert not computed.is_set()
    complete(local)
    thread.join(5)
    assert not thread.is_alive() and not errors and computed.is_set()
    assert evaluation.reference().result(timeout=0) == {"exploitability": 0.25}
    run = Path(target.runs) / evaluation.id
    assert json.loads((run / "spec.json").read_text())["policy"]["kind"] == "job-output"
    assert json.loads((run / "resolved-spec.json").read_text())["policy"]["digest"] == "a" * 64


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_local_dependency_failure_never_runs_evaluation(local, state):
    target, _, training = local
    game = load_game("rock_paper_scissors")
    evaluation = game.evaluate(training.reference(), submit=True).run(target)
    _write_state(Path(target.runs) / training.id, state)

    with pytest.raises(JobFailedError, match=state):
        run_local_attempt(evaluation.id, runs=target.runs, executor=lambda *args: pytest.fail("must not compute"))

    assert evaluation.state == "failed"

    with pytest.raises(JobFailedError, match=state):
        game.evaluate(training.reference(), submit=True).run(target)

    with pytest.raises(JobFailedError):
        training.reference().result(timeout=0)


def test_references_reject_wrong_store_and_producer_tampering(local):
    _, service, training = local
    reference = training.reference()
    wrong_store = Reference({**reference.to_document(), "target": {"kind": "local", "runs": "/unrelated"}})
    game = training._prepare().game

    with pytest.raises(JobError, match="same target endpoint"):
        service.submit(EvaluationSpec(game, wrong_store))

    tampered = Reference({**reference.to_document(), "workload_digest": "0" * 64})

    with pytest.raises(JobError, match="digest mismatch"):
        service.submit(EvaluationSpec(game, tampered))

    with pytest.raises(JobError, match="digest mismatch"):
        tampered.result(timeout=0)
