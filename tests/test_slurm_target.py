from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmax import Reference
from benchmax.execution import Job, SlurmTarget
from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.specs import EvaluationSpec, SpecError, TrainingSpec, _ArtifactMetadata
from benchmax.execution.targets.service import JobError, JobFailedError, JobTimeoutError
from benchmax.execution.targets.slurm import bridge
from benchmax.execution.targets.slurm.service import SlurmJobService
from benchmax.execution.targets.slurm.transfer import unpack_policy


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    target = SlurmTarget(python=sys.executable, workspace=str(tmp_path), runs=str(tmp_path / "scratch"),
                         artifacts=str(tmp_path / "retained"), account="team", poll_seconds=0.001)
    commands = []
    state = {"value": "PENDING", "id": "123"}

    def run(command, **kwargs):
        commands.append((command, kwargs))

        if command[0] == "sbatch":
            return SimpleNamespace(stdout=state["id"] + ";cluster\n", stderr="", returncode=0)

        if command[0] == "sacct":
            return SimpleNamespace(stdout=f"{state['id']}|{state['value']}|0:0\n", stderr="", returncode=0)

        if command[0] == "scancel":
            state["value"] = "CANCELLED"
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(bridge.subprocess, "run", run)
    monkeypatch.setattr("platform.platform", lambda: "test-login")

    class DirectBridgeService(SlurmJobService):
        def _request(self, operation, **fields):
            return bridge.dispatch({"schema_version": 1, "operation": operation, "target": self.target.to_document(),
                                    "source_digest": bridge.package_fingerprint(), **fields})

    service = DirectBridgeService(target)
    monkeypatch.setattr(SlurmTarget, "service", lambda self: DirectBridgeService(self))
    return SimpleNamespace(target=target, service=service, commands=commands, state=state)


def evaluation():
    return EvaluationSpec.create(game="rock_paper_scissors", policy=_ArtifactMetadata("/policy/manifest.json", "0" * 64))


def finish(scheduler, handle, monkeypatch):
    from benchmax.execution.jobs import workload as workload_module

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(workload_module, "execute_workload",
                        lambda spec, output: WorkloadResult.evaluation(spec.digest, {"exploitability": 0.0}))
    bridge.run_attempt(Path(scheduler.target.runs) / handle.id)
    scheduler.state["value"] = "COMPLETED"


def test_submission_runs_only_inside_allocation_and_persists_settings(scheduler, monkeypatch, record_property):
    monkeypatch.setenv("SBATCH_GPUS", "8")
    monkeypatch.setenv("SBATCH_MEM_PER_NODE", "100G")
    handle = scheduler.service.submit(evaluation(), output="metrics.json")
    command, options = scheduler.commands[0]
    run = Path(scheduler.target.runs) / handle.id
    assert command == [
        "sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=1024M",
        "--time=00:10:00", "--no-requeue", f"--job-name=benchmax-{handle.id}",
        f"--output={run / 'attempt.log'}", f"--chdir={scheduler.target.workspace}", "--account=team",
        str(run / "attempt.sh"),
    ]
    assert options["check"] is True and options["timeout"] == 60
    assert not options.get("shell", False)
    assert not any(k.startswith("SBATCH_") for k in options["env"])
    record = json.loads((run / "job.json").read_text())
    assert record["target"] == scheduler.target.to_document()
    assert record["output"] == str(Path(scheduler.target.artifacts) / "metrics.json")
    script = (run / "attempt.sh").read_text()
    assert script.splitlines() == [
        "#!/bin/sh", "set -eu", "umask 0007", "export PYTHONHASHSEED=0", "export PYTHONNOUSERSITE=1",
        "unset PYTHONHOME PYTHONPATH", f"cd {shlex.quote(scheduler.target.workspace)}",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        f"exec {shlex.quote(sys.executable)} -m benchmax.execution.targets.slurm.bridge attempt {shlex.quote(str(run))}",
    ]
    record_property("sbatch_argv", json.dumps(command))
    record_property("batch_script", script)

    with pytest.raises(JobError, match="allocation"):
        bridge.run_attempt(run)

    finish(scheduler, handle, monkeypatch)
    assert Path(record["output"]).is_file()
    assert scheduler.service.open(handle.id).result()["metrics"]["exploitability"] == 0.0


def test_job_reconnects_without_submitter_and_returns_metrics(scheduler, monkeypatch):
    job = Job._from_workload(evaluation(), output="final.json").run(scheduler.target)
    job_id = job.id
    finish(scheduler, job._handle, monkeypatch)
    del job
    reopened = Job.open(job_id, target=scheduler.target)
    assert reopened.output == Path(scheduler.target.artifacts) / "final.json"
    assert reopened.result() == {"exploitability": 0.0}
    assert asyncio.run(reopened.result_async(timeout=0)) == {"exploitability": 0.0}
    assert sum(command[0] == "sbatch" for command, _ in scheduler.commands) == 1


@pytest.mark.parametrize("state", ["TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "COMPLETED"])
def test_scheduler_terminal_state_without_result_fails_usefully(scheduler, state):
    handle = scheduler.service.submit(evaluation())
    scheduler.state["value"] = state
    reopened = scheduler.service.open(handle.id)

    with pytest.raises(JobFailedError, match=state):
        reopened.result()


def test_cancel_scopes_exact_scheduler_id(scheduler):
    job = Job._from_workload(evaluation()).run(scheduler.target)
    job.cancel()
    assert scheduler.commands[-1][0] == ["scancel", "123"]

    with pytest.raises(JobFailedError, match="CANCELLED"):
        Job.open(job.id, target=scheduler.target).result()


def test_result_wait_timeout_and_async_cancellation(scheduler):
    handle = scheduler.service.submit(evaluation())

    with pytest.raises(JobTimeoutError):
        handle.result(timeout=0.005)

    with pytest.raises(JobTimeoutError):
        asyncio.run(handle.result_async(timeout=0))

    async def cancel_wait():
        task = asyncio.create_task(handle.result_async())
        await asyncio.sleep(0.005)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_wait())
    assert not any(command[0] == "scancel" for command, _ in scheduler.commands)


def test_status_queries_are_cached(scheduler):
    service = type(scheduler.service)(replace(scheduler.target, poll_seconds=300))
    handle = service.submit(evaluation())

    for _ in range(20):
        assert handle.state == "pending"

    assert len(scheduler.commands) == 1


@pytest.mark.parametrize("operation", ["submit", "audit"])
def test_strict_source_rejects_different_installations_before_sbatch(scheduler, operation):
    target = replace(scheduler.target, strict_source=True)

    with pytest.raises(JobError, match="source differs"):
        bridge.dispatch({"schema_version": 1, "operation": operation, "target": target.to_document(),
                         "workload": evaluation().to_document(), "source_digest": "changed"})

    assert not scheduler.commands


@pytest.mark.parametrize("operation", ["submit", "audit"])
def test_compatible_workload_accepts_different_sources_and_records_them(scheduler, operation):
    job_id = "evaluation-0000000000000001"
    result = bridge.dispatch({"schema_version": 1, "operation": operation, "target": scheduler.target.to_document(),
                              "id": job_id, "workload": evaluation().to_document(), "source_digest": "a" * 64})

    if operation == "submit":
        record = json.loads((Path(scheduler.target.runs) / job_id / "job.json").read_text())
        assert record["source_digest"] == "a" * 64
        assert record["target_source_digest"] == bridge.package_fingerprint()
        assert result["state"] == "pending"
    else:
        machine = result["evidence"]["machine"]
        assert machine["caller_source_digest"] == "a" * 64
        assert machine["source_digest"] == bridge.package_fingerprint()
        assert not scheduler.commands


@pytest.mark.parametrize("strict_source", [False, True])
def test_source_changes_while_queued_require_strict_mode_to_fail(scheduler, monkeypatch, strict_source):
    service = type(scheduler.service)(replace(scheduler.target, strict_source=strict_source))
    handle = service.submit(evaluation())
    monkeypatch.setattr(bridge, "package_fingerprint", lambda: "b" * 64)

    if strict_source:
        with pytest.raises(JobError, match="source has drifted"):
            finish(scheduler, handle, monkeypatch)

        assert service.open(handle.id).state == "failed"
    else:
        finish(scheduler, handle, monkeypatch)
        assert service.open(handle.id).state == "succeeded"

    record = json.loads((Path(scheduler.target.runs) / handle.id / "job.json").read_text())
    assert record["execution_source_digest"] == "b" * 64


@pytest.mark.parametrize("operation", ["submit", "audit", "attempt"])
def test_selected_configuration_drift_still_fails(scheduler, monkeypatch, operation):
    from benchmax import registry

    workload = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    handle = scheduler.service.submit(workload) if operation == "attempt" else None
    monkeypatch.setattr(registry._GAMES, "resolve", lambda *a, **kw: SimpleNamespace(configuration={"changed": True}))
    monkeypatch.setattr("benchmax.execution.jobs.workload.execute_workload", lambda *a: pytest.fail("must not compute"))

    with pytest.raises(SpecError, match="resolved game .* differs from the workload"):
        if handle is not None:
            monkeypatch.setenv("SLURM_JOB_ID", "123")
            bridge.run_attempt(Path(scheduler.target.runs) / handle.id)
        else:
            bridge.dispatch({"schema_version": 1, "operation": operation, "target": scheduler.target.to_document(),
                             "workload": workload.to_document(), "source_digest": "a" * 64})

    if handle is None:
        assert not scheduler.commands
    else:
        assert json.loads((Path(scheduler.target.runs) / handle.id / "state.json").read_text())["state"] == "failed"


def test_workload_tampering_still_fails_without_strict_source(scheduler, monkeypatch):
    handle = scheduler.service.submit(evaluation())
    run = Path(scheduler.target.runs) / handle.id
    document = json.loads((run / "spec.json").read_text())
    document["game"]["resolved"]["max_episode_steps"] = 2
    (run / "spec.json").write_text(json.dumps(document))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr("benchmax.execution.jobs.workload.execute_workload", lambda *a: pytest.fail("must not compute"))

    with pytest.raises(JobError, match="workload has drifted"):
        bridge.run_attempt(run)


def test_attempt_failure_and_tampered_results_are_reported(scheduler, monkeypatch):
    from benchmax.execution.jobs import workload as workload_module

    handle = scheduler.service.submit(evaluation())
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    def fail(*args):
        raise ValueError("bad policy")

    monkeypatch.setattr(workload_module, "execute_workload", fail)

    with pytest.raises(ValueError, match="bad policy"):
        bridge.run_attempt(Path(scheduler.target.runs) / handle.id)

    with pytest.raises(JobFailedError, match="bad policy"):
        scheduler.service.open(handle.id).result()

    second = scheduler.service.submit(evaluation())
    finish(scheduler, second, monkeypatch)
    (Path(scheduler.target.runs) / second.id / "result.json").write_text("{}")

    with pytest.raises(JobError, match="digest mismatch"):
        scheduler.service.open(second.id)


@pytest.mark.parametrize("reply", ["timeout", "unrecognized"])
def test_unknown_submission_is_not_retried_and_an_accepted_attempt_can_finish(scheduler, monkeypatch, reply):
    calls = []

    def timed_out(command, **kwargs):
        calls.append(command)
        if reply == "timeout":
            raise subprocess.TimeoutExpired(command, 1)

        return SimpleNamespace(stdout="unrecognized", stderr="", returncode=0)

    monkeypatch.setattr(bridge.subprocess, "run", timed_out)

    with pytest.raises(JobError, match="submission outcome unknown|submission may have succeeded"):
        scheduler.service.submit(evaluation())

    assert len(calls) == 1
    run = next(Path(scheduler.target.runs).glob("evaluation-*"))
    assert json.loads((run / "state.json").read_text())["state"] == "submission_unknown"

    with pytest.raises(JobError, match="submission outcome unknown"):
        scheduler.service.open(run.name).result()

    finish(scheduler, scheduler.service.open(run.name), monkeypatch)
    assert scheduler.service.open(run.name).result() == WorkloadResult.evaluation(
        evaluation().digest, {"exploitability": 0.0}).to_document()
    assert json.loads((run / "scheduler.json").read_text()) == {"id": "123"}
    assert not any((Path(scheduler.target.runs) / ".output-reservations").iterdir())


@pytest.mark.parametrize("state", ["running", "succeeded", "failed", "cancelled"])
def test_late_submission_uncertainty_preserves_worker_state(tmp_path, state):
    bridge._write_state(tmp_path, state)
    original = (tmp_path / "state.json").read_bytes()

    bridge._submission_unknown(tmp_path, "late timeout")

    assert (tmp_path / "state.json").read_bytes() == original


def test_script_write_failure_releases_output_before_submission(scheduler, monkeypatch):
    original = Path.write_text

    def write(path, *args, **kwargs):
        if path.name == "attempt.sh":
            raise OSError("disk full")

        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(JobError, match="disk full"):
        scheduler.service.submit(evaluation())

    run = next(Path(scheduler.target.runs).glob("evaluation-*"))
    assert json.loads((run / "state.json").read_text())["state"] == "failed"
    assert not any((Path(scheduler.target.runs) / ".output-reservations").iterdir())
    assert not scheduler.commands


def test_remote_paths_and_ssh_command_never_use_caller_shell(tmp_path, monkeypatch, record_property):
    target = SlurmTarget(python="/env with spaces/bin/python", workspace="/project", runs="/scratch/runs",
                         host="cluster", user="alice", control_path="/tmp/shared-%C")
    service = SlurmJobService(target)
    command = service._command()
    assert command == [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
        "-o", "ClearAllForwardings=yes", "-p", "22", "-S", "/tmp/shared-%C", "-l", "alice", "cluster",
        "'/env with spaces/bin/python' -m benchmax.execution.targets.slurm.bridge",
    ]
    assert shlex.split(command[-1]) == [target.python, "-m", "benchmax.execution.targets.slurm.bridge"]
    policy = service._reference("/policies/a;$(touch nope).json", "0" * 64)
    spec = EvaluationSpec.create(game="rock_paper_scissors", policy=policy)
    assert service._workload(spec).policy.uri == "/policies/a;$(touch nope).json"
    assert service._workload(spec).digest == spec.digest

    with pytest.raises(JobError, match="another target endpoint"):
        SlurmJobService(replace(target, host="other"))._workload(spec)

    with pytest.raises(ValueError, match="cannot read policy document"):
        service._workload(evaluation())

    calls = []
    output = "metrics;$(touch nope).json"

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        request = json.loads(kwargs["input"])
        value = {"id": request["id"], "state": "pending", "workload": request["workload"],
                 "output": "/project/" + request["output"]}
        return SimpleNamespace(stdout=json.dumps({"ok": True, "value": value}), stderr="", returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", capture)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("unexpected real subprocess"))
    handle = service.submit(spec, output=output)
    assert len(calls) == 1
    actual, options = calls[0]
    assert actual == command
    assert options == {"input": options["input"], "text": True, "capture_output": True, "timeout": 70,
                       "check": False}
    request = json.loads(options["input"])
    assert request == {"schema_version": 1, "operation": "submit", "target": target.to_document(),
                       "source_digest": bridge.package_fingerprint(), "id": handle.id,
                       "workload": service._workload(spec).to_document(), "output": output}
    # Shell metacharacters remain JSON data, never part of the SSH command.
    assert request["workload"]["policy"]["uri"] == "/policies/a;$(touch nope).json"
    assert not list(tmp_path.iterdir())
    record_property("ssh_argv", json.dumps(actual))
    record_property("ssh_stdin", options["input"])


def test_audit_reports_storage_and_does_not_submit(scheduler):
    report = scheduler.service.audit(evaluation())
    assert report.estimates["storage"]["runs"] == scheduler.target.runs
    assert report.machine["allocation"]["gpus"] == 0
    assert "login" in report.machine["scope"]
    assert not scheduler.commands


def test_training_artifact_stays_remote_until_explicit_download(scheduler, monkeypatch, tmp_path):
    from benchmax.execution.jobs import workload as workload_module
    from benchmax.seeding import derive_seed
    from tests.artifact_helpers import write_policy_artifact

    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    source = tmp_path / "remote-policy"
    manifest = write_policy_artifact(source, spec)
    sidecar = source / "tensors.npz"
    reference = _ArtifactMetadata.from_path(manifest)
    monkeypatch.setattr(workload_module, "execute_workload",
                        lambda workload, output: WorkloadResult.training(workload.digest, workload.seed, reference))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("PYTHONHASHSEED", str(derive_seed(spec.seed, "python-hash", scheme=spec.seed_scheme)))
    job = Job._from_workload(spec).run(scheduler.target)
    bridge.run_attempt(Path(scheduler.target.runs) / job.id)
    job = Job.open(job.id, target=scheduler.target)
    remote = job.reference()
    assert remote.output == "policy" and remote.job_id == job.id
    # Reuse the reference as an evaluation input without copying tensor data.
    reused = scheduler.service._workload(EvaluationSpec.create(game="rock_paper_scissors", policy=remote))
    assert reused.policy.to_document() == remote.to_document()
    from benchmax import PolicyProfile

    profile = PolicyProfile(lambda batch: batch)
    monkeypatch.setattr("benchmax.execution.artifacts.policies.load_policy", lambda *a, **kw: pytest.fail("save must not load"))
    saved = remote.save(path=tmp_path / "save-only")
    assert saved.kind == "file" and saved.path == tmp_path / "save-only/manifest.json"
    monkeypatch.setattr("benchmax.execution.artifacts.policies.load_policy", lambda *args, **kwargs: profile)
    assert job.result(path=tmp_path / "download") is profile
    assert (tmp_path / "download/manifest.json").read_bytes() == manifest.read_bytes()
    assert (tmp_path / "download" / "tensors.npz").read_bytes() == sidecar.read_bytes()


def finish_training(scheduler, training, monkeypatch):
    from benchmax.execution.jobs import workload as workload_module
    from benchmax.seeding import derive_seed

    spec = training._prepare()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("PYTHONHASHSEED", str(derive_seed(spec.seed, "python-hash", scheme=spec.seed_scheme)))
    artifact = _ArtifactMetadata("/saved/manifest.json", "a" * 64)
    monkeypatch.setattr(workload_module, "execute_workload",
                        lambda spec, output: WorkloadResult.training(spec.digest, spec.seed, artifact))
    bridge.run_attempt(Path(scheduler.target.runs) / training.id)
    return artifact


def test_future_policy_creates_exact_slurm_dependency_and_resolves_in_worker(scheduler, monkeypatch, record_property):
    from benchmax import load_game
    from benchmax.execution.jobs import workload as workload_module

    training = Job._from_workload(TrainingSpec.create(game="rock_paper_scissors", seed=7)).run(scheduler.target)
    reference = training.reference()
    scheduler.state["id"] = "124"
    # Resource requests can differ while both targets use the same job store.
    target = replace(scheduler.target, cpus=2, memory_mb=2048)
    evaluation_job = load_game("rock_paper_scissors").evaluate(reference, submit=True).run(target)
    assert len(scheduler.commands) == 2
    command = scheduler.commands[-1][0]
    assert "--dependency=afterok:123" in command
    assert "--kill-on-invalid-dep=yes" in command
    assert "--cpus-per-task=2" in command
    record_property("dependent_sbatch_argv", json.dumps(command))
    artifact = finish_training(scheduler, training, monkeypatch)

    def execute(spec, output):
        assert spec.policy == artifact
        return WorkloadResult.evaluation(spec.digest, {"exploitability": 0.25})

    monkeypatch.setattr(workload_module, "execute_workload", execute)
    bridge.run_attempt(Path(target.runs) / evaluation_job.id)
    reopened = Job.open(evaluation_job.id, target=target)
    assert reopened.result(timeout=0) == {"exploitability": 0.25}
    assert reopened.reference().result(timeout=0) == reopened.result(timeout=0)
    run = Path(target.runs) / evaluation_job.id
    assert json.loads((run / "spec.json").read_text())["policy"] == reference.to_document()
    assert json.loads((run / "resolved-spec.json").read_text())["policy"] == artifact.to_document()
    # No login-node policy loading, dependency polling, or transfers.
    assert all(command[0] == "sbatch" for command, _ in scheduler.commands)


def test_completed_producer_needs_no_scheduler_dependency(scheduler, monkeypatch):
    training = Job._from_workload(TrainingSpec.create(game="rock_paper_scissors", seed=7)).run(scheduler.target)
    finish_training(scheduler, training, monkeypatch)
    scheduler.commands.clear()
    scheduler.service.submit(EvaluationSpec(training._prepare().game, training.reference()))
    assert len(scheduler.commands) == 1
    assert not any(arg.startswith("--dependency") for arg in scheduler.commands[0][0])


@pytest.mark.parametrize("state", ["failed", "cancelled", "submission_unknown"])
def test_bad_slurm_dependency_rejected_before_submission(scheduler, state):
    training = Job._from_workload(TrainingSpec.create(game="rock_paper_scissors", seed=7)).run(scheduler.target)
    from benchmax.execution.targets.local.service import _write_state

    _write_state(Path(scheduler.target.runs) / training.id, state)
    scheduler.commands.clear()

    with pytest.raises(JobError, match=state):
        scheduler.service.submit(EvaluationSpec(training._prepare().game, training.reference()))

    assert not scheduler.commands


def test_dependency_failure_after_submission_prevents_worker_computation(scheduler, monkeypatch):
    from benchmax.execution.targets.local.service import _write_state

    training = Job._from_workload(TrainingSpec.create(game="rock_paper_scissors", seed=7)).run(scheduler.target)
    consumer = scheduler.service.submit(EvaluationSpec(training._prepare().game, training.reference()))
    _write_state(Path(scheduler.target.runs) / training.id, "failed")
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    monkeypatch.setattr("benchmax.execution.jobs.workload.execute_workload", lambda *args: pytest.fail("must not compute"))

    with pytest.raises(JobError, match="dependency.*failed"):
        bridge.run_attempt(Path(scheduler.target.runs) / consumer.id)

    assert scheduler.service.open(consumer.id).state == "failed"


def test_dependency_rejects_tampering_and_cross_endpoint_without_submission(scheduler):
    training = Job._from_workload(TrainingSpec.create(game="rock_paper_scissors", seed=7)).run(scheduler.target)
    reference = training.reference()
    tampered = Reference({**reference.to_document(), "workload_digest": "0" * 64})
    scheduler.commands.clear()

    with pytest.raises(JobError, match="digest mismatch"):
        scheduler.service.submit(EvaluationSpec(training._prepare().game, tampered))

    other = SlurmJobService(replace(scheduler.target, host="other-cluster"))

    with pytest.raises(JobError, match="same target endpoint"):
        other.submit(EvaluationSpec(training._prepare().game, reference))

    run = Path(scheduler.target.runs) / training.id
    (run / "scheduler.json").write_text('{"id": "123; dangerous"}')

    with pytest.raises(JobError, match="scheduler ID"):
        scheduler.service.submit(EvaluationSpec(training._prepare().game, reference))

    assert not scheduler.commands


def test_future_reference_stays_in_ssh_json_without_client_resolution(monkeypatch, record_property):
    target = SlurmTarget(python="/env/bin/python", workspace="/project", runs="/scratch/runs",
                         host="cluster", user="alice", control_path="/tmp/shared-%C")
    training = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    reference = Reference({"schema_version": 1, "kind": "job-output", "job_id": "training-0000000000000001",
                           "target": target.to_document(), "output": "policy", "workload_digest": training.digest})
    calls = []

    def capture(command, **kwargs):
        request = json.loads(kwargs["input"])
        calls.append(request)
        assert command == SlurmJobService(target)._command()
        assert request["operation"] == "submit"
        assert request["workload"]["policy"] == reference.to_document()
        value = {"id": request["id"], "state": "pending", "workload": request["workload"],
                 "output": "/project/result.json"}
        return SimpleNamespace(stdout=json.dumps({"ok": True, "value": value}), stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", capture)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("unexpected subprocess"))
    SlurmJobService(target).submit(EvaluationSpec(training.game, reference))
    assert len(calls) == 1
    record_property("future_reference_ssh_request", json.dumps(calls[0]))


def test_download_rejects_tampering_traversal_and_oversize(tmp_path):
    import base64

    raw = b"manifest"
    digest = hashlib.sha256(raw).hexdigest()
    bundle = {"manifest": "manifest.json", "digest": digest,
              "files": {"manifest.json": base64.b64encode(raw).decode()}}

    with pytest.raises(ValueError, match="requested artifact"):
        unpack_policy(bundle, tmp_path / "wrong", digest="0" * 64, limit=100)

    bundle["files"]["../escape"] = ""

    with pytest.raises(ValueError, match="file name"):
        unpack_policy(bundle, tmp_path / "bad", digest=digest, limit=100)

    del bundle["files"]["../escape"]

    with pytest.raises(ValueError, match="transfer limit"):
        unpack_policy(bundle, tmp_path / "large", digest=digest, limit=1)


def test_submitting_from_new_process_uses_control_protocol(tmp_path):
    # A separate local Python process exercises the real JSON bridge without SSH or Slurm.
    target = SlurmTarget(python=sys.executable, workspace=str(tmp_path), runs=str(tmp_path / "runs"))
    report = SlurmJobService(target).audit(evaluation())
    assert report.machine["python"] == sys.executable
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("corruption", ["tensors", "extra-file"])
def test_failed_download_leaves_destination_available_for_retry(tmp_path, corruption):
    import base64

    from benchmax.execution.targets.slurm.transfer import pack_policy
    from tests.artifact_helpers import write_policy_artifact

    source = write_policy_artifact(tmp_path / "source", TrainingSpec.create(game="rock_paper_scissors", seed=7))
    reference = _ArtifactMetadata.from_path(source)
    bundle = pack_policy(reference, limit=1024**2)
    valid = dict(bundle["files"])
    bundle["files"]["tensors.npz" if corruption == "tensors" else "extra"] = base64.b64encode(b"invalid").decode()
    destination = tmp_path / "downloaded"

    with pytest.raises(ValueError):
        unpack_policy(bundle, destination, digest=reference.digest, limit=1024**2)

    assert not destination.exists() and not list(tmp_path.glob(".download-*"))
    bundle["files"] = valid
    unpack_policy(bundle, destination, digest=reference.digest, limit=1024**2)
    with pytest.raises(FileExistsError):
        unpack_policy(bundle, destination, digest=reference.digest, limit=1024**2)

    assert (destination / source.name).read_bytes() == source.read_bytes()


def test_pending_jobs_cannot_claim_the_same_output(scheduler):
    first = scheduler.service.submit(evaluation(), output="shared.json")

    with pytest.raises(JobError, match="reserved"):
        scheduler.service.submit(evaluation(), output="shared.json")

    scheduler.state["value"] = "TIMEOUT"
    assert scheduler.service.open(first.id).state == "failed"
    scheduler.state["value"] = "PENDING"
    assert scheduler.service.submit(evaluation(), output="shared.json").id != first.id


def test_tampered_scheduler_id_cannot_turn_cancellation_into_a_broad_command(scheduler):
    job = Job._from_workload(evaluation()).run(scheduler.target)
    (Path(scheduler.target.runs) / job.id / "scheduler.json").write_text('{"id":"--user=someone"}')

    with pytest.raises(JobError, match="invalid persisted Slurm scheduler ID"):
        job.cancel()

    assert not any(command[0] == "scancel" for command, _ in scheduler.commands)


@pytest.mark.parametrize("input_kind", ["manifest", "reference", "profile", "training-result"])
def test_local_policy_is_staged_for_ssh_evaluation(scheduler, tmp_path, monkeypatch, tiny_trained_policy, input_kind):
    import shutil

    from benchmax.execution.artifacts.policies import write_policy

    monkeypatch.setattr("benchmax.execution.jobs.workload.capture_producer",
                        lambda: {"provenance": {"source_revision": "test-execution-host"}})
    game, policy, training = tiny_trained_policy
    expected = game.evaluate(policy)
    local = tmp_path / "client-policy"
    artifact = write_policy(policy, training, local)
    result_path = local / "training.json"
    result_path.write_text(json.dumps(WorkloadResult.training(training.digest, training.seed, artifact).to_document()))
    target = replace(scheduler.target, host="cluster")
    operations = []
    original = type(scheduler.service)._request

    def request(self, operation, **fields):
        operations.append(operation)
        return original(self, operation, **fields)

    monkeypatch.setattr(type(scheduler.service), "_request", request)
    monkeypatch.setattr(policy, "_portable_reference", artifact)
    source = {"manifest": artifact.uri, "reference": Reference.from_path(artifact.uri), "profile": policy,
              "training-result": result_path}[input_kind]
    job = game.evaluate(source, submit=True)
    digest = job._prepare().digest
    report = job.audit(target)
    assert report.workload.policy.uri == artifact.uri
    cache = Path(target.artifacts) / "policy-cache"
    assert report.estimates["transfers"]["policy_bytes"] > 0
    assert not cache.exists()
    assert "upload" not in operations
    assert not scheduler.commands

    job.run(target)
    staged = Path(job._handle.workload.policy.uri)
    assert staged.is_file() and cache in staged.parents
    assert job._handle.workload.digest == digest
    assert operations.count("upload") == 1
    monkeypatch.setattr("benchmax.execution.targets.slurm.transfer.pack_policy",
                        lambda *args, **kwargs: pytest.fail("cached policy must not be packed again"))
    second = game.evaluate(source, submit=True).run(target)
    assert second._handle.workload.policy.uri == str(staged)
    assert operations.count("upload") == 1
    assert job.audit(target).estimates["transfers"]["policy_bytes"] == 0

    shutil.rmtree(local)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    bridge.run_attempt(Path(target.runs) / job.id)
    monkeypatch.setattr("benchmax.execution.jobs.workload.capture_producer",
                        lambda: pytest.fail("export must preserve the execution host's provenance"))
    scheduler.state["value"] = "COMPLETED"
    result = Job.open(job.id, target=target).result()
    assert result["exploitability"] == pytest.approx(expected["exploitability"])
    assert job.result(path=tmp_path / "saved-metrics") == result
    saved = json.loads((tmp_path / "saved-metrics/result.json").read_text())
    assert saved["workload"]["policy"]["uri"] == str(staged)
    assert saved["provenance"]["producer"]["provenance"]["source_revision"] == "test-execution-host"


def test_policy_upload_failure_never_publishes_partial_cache(tmp_path):
    import base64

    from benchmax.execution.targets.slurm.transfer import cached_policy, pack_policy
    from tests.artifact_helpers import write_policy_artifact

    path = write_policy_artifact(tmp_path / "client", TrainingSpec.create(game="rock_paper_scissors", seed=7))
    artifact = _ArtifactMetadata.from_path(path)
    bundle = pack_policy(artifact, limit=1024**2)
    root = tmp_path / "remote"
    original = bundle["files"]["tensors.npz"]
    bundle["files"]["tensors.npz"] = base64.b64encode(b"corrupt").decode()
    with pytest.raises(ValueError):
        cached_policy(root, artifact.digest, path.name, limit=1024**2, bundle=bundle)

    assert not (root / artifact.digest).exists()
    assert not list(root.glob(".upload-*"))
    bundle["files"]["tensors.npz"] = original
    with pytest.raises(ValueError, match="limit"):
        cached_policy(root, artifact.digest, path.name, limit=1, bundle=bundle)

    assert not (root / artifact.digest).exists()
    result = cached_policy(root, artifact.digest, path.name, limit=1024**2, bundle=bundle)
    assert result["cached"]
    assert Path(result["policy"]["uri"]).is_file()
    with pytest.raises(ValueError, match="limit"):
        cached_policy(root, artifact.digest, path.name, limit=1)


@pytest.mark.parametrize("digest,manifest", [("../escape", "manifest.json"), ("0" * 64, "../manifest.json")])
def test_policy_cache_rejects_invalid_identity(tmp_path, digest, manifest):
    from benchmax.execution.targets.slurm.transfer import cached_policy

    with pytest.raises(ValueError):
        cached_policy(tmp_path / "remote", digest, manifest, limit=1024)

    assert not list(tmp_path.iterdir())


def test_policy_staging_uses_json_bridge_without_importing_torch(tmp_path, monkeypatch):
    from tests.artifact_helpers import write_policy_artifact

    path = write_policy_artifact(tmp_path / "client", TrainingSpec.create(game="rock_paper_scissors", seed=7))
    target = SlurmTarget(python=sys.executable, host="cluster", workspace=str(tmp_path), runs=str(tmp_path / "remote"))
    service = SlurmJobService(target)
    program = (
        "import sys; from benchmax.execution.targets.slurm.bridge import main; "
        "main(); assert 'torch' not in sys.modules"
    )
    monkeypatch.setattr(service, "_command", lambda: [sys.executable, "-c", program])
    spec = EvaluationSpec.create(game="rock_paper_scissors", policy=str(path))
    staged = service._workload(spec)
    assert staged.digest == spec.digest
    assert Path(staged.policy.uri).read_bytes() == path.read_bytes()
    assert Path(target.runs) in Path(staged.policy.uri).parents
    assert service._workload(spec) == staged
