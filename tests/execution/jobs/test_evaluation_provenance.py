"""Evaluation exports retain scientific context independently of their job store."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmax import PolicyProfile, load_game
from benchmax.execution import Job, LocalTarget
from benchmax.execution.artifacts.policies import write_policy
from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.jobs.outputs import save_result
from benchmax.execution.jobs.workload import execute_workload
from benchmax.execution.specs import EvaluationSpec, TrainingSpec, _ArtifactMetadata
from benchmax.execution.targets.local import LocalJobService
from benchmax.execution.targets.local.attempt import run_local_attempt


@pytest.fixture
def policy(tmp_path):
    training = TrainingSpec.create(game="rock_paper_scissors", algorithm="cfr", seed=7)
    reference = write_policy(PolicyProfile.from_table({}, fallback="uniform"), training, tmp_path / "policy")
    manifest = json.loads(Path(reference.uri).read_text())
    manifest["producer"] = {"provenance": {"source_revision": "recorded-training-source"}}
    manifest["progress"] = {"iterations": 12}
    Path(reference.uri).write_text(json.dumps(manifest))
    return training, _ArtifactMetadata.from_path(reference.uri)


def test_synchronous_result_records_resolved_configuration_and_execution(tmp_path, policy):
    training, artifact = policy
    game = load_game("rock_paper_scissors", evaluator=None)
    output = tmp_path / "metrics.json"
    metrics = game.evaluate(artifact.uri, evaluator="matrix", output=output)
    result = WorkloadResult.from_document(json.loads(output.read_text()))

    assert result.metrics == metrics
    assert result.workload.digest == result.workload_digest
    assert result.workload.game.overrides["evaluator"] == "matrix"
    assert result.workload.game.resolved is not None
    assert result.workload.policy.digest == artifact.digest
    assert result.provenance["policy"]["training"] == training.to_document()
    assert result.provenance["policy"]["progress"] == {"iterations": 12}
    assert result.provenance["policy"]["producer"]["provenance"]["source_revision"] == "recorded-training-source"
    assert result.provenance["producer"]["provenance"]["packages"]["benchmax"]
    assert result.provenance["duration_seconds"] >= 0
    assert result.provenance["started_at"] <= result.provenance["finished_at"]


def test_job_dependency_export_survives_removal_of_job_store_in_fresh_process(tmp_path, monkeypatch, policy):
    training, artifact = policy
    target = LocalTarget(runs=str(tmp_path / "runs"))
    service = LocalJobService(target.runs, launcher=lambda *args: None)
    monkeypatch.setattr(LocalTarget, "service", lambda self: service)
    producer = Job._from_workload(training).run(target)
    run_local_attempt(producer.id, runs=target.runs,
                      executor=lambda spec, output: WorkloadResult.training(spec.digest, spec.seed, artifact))
    job = load_game("rock_paper_scissors").evaluate(producer.reference(), submit=True).run(target)
    def evaluate(spec, output):
        # Resolving the dependency must capture evidence before computation starts.
        shutil.rmtree(Path(target.runs) / producer.id)
        return execute_workload(spec, output)

    run_local_attempt(job.id, runs=target.runs, executor=evaluate)
    saved = job.save(path=tmp_path / "export")
    document = json.loads(saved.path.read_text())
    assert document["workload"]["policy"]["kind"] == "job-output"
    assert document["resolved_workload"]["policy"] == artifact.to_document()
    assert document["provenance"]["policy_job"]["job_id"] == producer.id
    assert document["provenance"]["policy_job"]["progress"] is None
    assert document["provenance"]["policy_job"]["finished_at"] is not None
    assert set(document["provenance"]["policy_job"]) == {"job_id", "state", "started_at", "finished_at", "progress"}
    shutil.rmtree(target.runs)
    shutil.rmtree(tmp_path / "policy")

    program = """
import json, sys
from benchmax import Reference
from benchmax.execution.artifacts.results import WorkloadResult
reference = Reference.from_path(sys.argv[1])
result = WorkloadResult.from_document(json.loads(reference.path.read_text()))
assert result.workload.game.name == 'rock_paper_scissors'
assert result.workload.game.resolved is not None
assert result.resolved_workload.policy.digest == sys.argv[2]
assert result.provenance['policy']['training']['seed'] == 7
assert result.provenance['policy']['progress'] == {'iterations': 12}
assert reference.result() == result.metrics
copy = reference.save(sys.argv[3])
assert copy.path.read_bytes() == reference.path.read_bytes()
"""
    process = subprocess.run([sys.executable, "-c", program, str(saved.path), artifact.digest, str(tmp_path / "copy")],
                             capture_output=True, text=True, timeout=30, check=False)
    assert process.returncode == 0, process.stdout + process.stderr


def test_unknown_training_provenance_is_explicit(tmp_path):
    policy = PolicyProfile.from_table({}, fallback="uniform")
    output = tmp_path / "metrics.json"
    load_game("rock_paper_scissors").evaluate(policy, output=output)
    provenance = WorkloadResult.from_document(json.loads(output.read_text())).provenance
    assert provenance["policy"] == {"training": None, "producer": None, "progress": None}


def test_legacy_export_adds_known_spec_without_inventing_execution_evidence(tmp_path, policy):
    _, artifact = policy
    workload = EvaluationSpec.create(game="rock_paper_scissors", policy=artifact)
    legacy = WorkloadResult.evaluation(workload.digest, {"score": 1.0})
    assert WorkloadResult.from_document(legacy.to_document()) == legacy
    saved = save_result(legacy, workload, tmp_path / "export", handle=None)
    result = WorkloadResult.from_document(json.loads(saved.path.read_text()))
    assert result.workload == workload
    assert result.provenance is None
