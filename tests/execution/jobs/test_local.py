from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.specs import EvaluationSpec, TrainingSpec, _ArtifactMetadata
from benchmax.execution.targets.local import LocalJobService
from benchmax.execution.targets.local import service as execution_local
from benchmax.execution.targets.service import JobError, JobFailedError, JobTimeoutError
from tests.execution.jobs._helpers import _execute, _result


def test_local_submission_persists_before_launching_and_uses_the_cli(tmp_path: Path) -> None:
    launches: list[tuple[list[str], dict[str, str]]] = []

    def launch(command: list[str], environment: dict[str, str], log: Path) -> None:
        assert json.loads((log.parent / "spec.json").read_text())["seed"] == 7
        assert json.loads((log.parent / "state.json").read_text())["state"] == "pending"
        assert json.loads((log.parent / "job.json").read_text())["output"] == str(output.resolve())
        launches.append((command, environment))

    service = LocalJobService(tmp_path / "runs", launcher=launch)
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    output = tmp_path / "policy.json"

    job = service.submit(spec, output=output)

    command, environment = launches[0]
    run = tmp_path / "runs" / job.id
    assert json.loads((run / "spec.json").read_text())["seed"] == 7
    assert json.loads((run / "state.json").read_text())["state"] == "pending"
    assert command[1:] == ["-m", "benchmax.cli", "_attempt", job.id]
    job_record = json.loads((run / "job.json").read_text())
    assert job_record["output"] == str(output.resolve())
    assert str(output.resolve()) not in command
    assert environment["BENCHMAX_RUNS"] == str((tmp_path / "runs").resolve())
    assert environment["PYTHONHASHSEED"] == "1291131919"


@pytest.mark.parametrize("operation,expected_seed", [("training", "1291131919"), ("evaluation", "0")])
def test_worker_hashes_replay_independently_of_caller_environment(tmp_path, monkeypatch, operation, expected_seed):
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    environments = []
    service = LocalJobService(tmp_path / "runs", launcher=lambda command, environment, log: environments.append(environment))
    training = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    workload = (training if operation == "training"
                else EvaluationSpec(training.game, _ArtifactMetadata("policy.json", "0" * 64)))

    service.submit(workload)
    service.submit(workload)

    assert [environment["PYTHONHASHSEED"] for environment in environments] == [expected_seed, expected_seed]
    assert os.environ["PYTHONHASHSEED"] == "random"
    command = [sys.executable, "-c", "print(hash('benchmax-worker-policy'))"]
    hashes = [subprocess.check_output(command, env=environment, text=True) for environment in environments]
    assert hashes[0] == hashes[1]

def test_a_durable_handle_can_be_reopened_after_success(local_service, tmp_path: Path) -> None:
    service = local_service
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    job = service.submit(spec)

    _execute(service, job.id, executor=lambda workload, output: _result(workload))
    reopened = LocalJobService(tmp_path / "runs", launcher=lambda *args: None).open(job.id)

    assert reopened.result(timeout=0.1)["kind"] == "training-result"
    assert reopened.id == job.id


def test_reopened_result_is_revalidated_against_the_persisted_workload(local_service, tmp_path: Path) -> None:
    service = local_service
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    job = service.submit(spec)
    _execute(service, job.id, executor=lambda workload, output: _result(workload))
    result_path = job.run_directory / "result.json"
    result = json.loads(result_path.read_text())
    result["seed"] = 8
    result_path.write_text(json.dumps(result), encoding="utf-8")
    state_path = job.run_directory / "state.json"
    state = json.loads(state_path.read_text())
    state["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(JobError, match="seed does not match"):
        service.open(job.id).result(timeout=0.1)


def test_launcher_records_process_birth_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution_local, "_process_identity", lambda pid: f"birth:{pid}")
    service = LocalJobService(tmp_path / "runs", launcher=lambda *args: 4242)

    job = service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7))

    launcher = json.loads((job.run_directory / "launcher.json").read_text())
    assert launcher["status"] == "launched"
    assert launcher["pid"] == 4242
    assert launcher["process_identity"] == "birth:4242"


def test_attempt_failure_is_persisted_and_explained(local_service, tmp_path: Path) -> None:
    service = local_service
    job = service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7))

    def fail(*args: object) -> dict[str, object]:
        raise RuntimeError("accelerator disappeared")

    with pytest.raises(RuntimeError, match="accelerator disappeared"):
        _execute(service, job.id, executor=fail)

    with pytest.raises(JobFailedError, match="accelerator disappeared"):
        job.result(timeout=0.1)

    state = json.loads((tmp_path / "runs" / job.id / "state.json").read_text())
    assert state["state"] == "failed"


@pytest.mark.parametrize(
    "wrong_result, message",
    (
        (
            lambda workload: WorkloadResult.evaluation(workload.digest, {"score": 1.0}),
            "training attempt returned an evaluation result",
        ),
        (
            lambda workload: WorkloadResult.training(
                workload.digest, workload.seed + 1, _ArtifactMetadata("policy/manifest.json", "0" * 64)
            ),
            "seed does not match",
        ),
    ),
)
def test_training_result_is_bound_to_its_workload_variant_and_seed(
    tmp_path: Path, wrong_result: Callable[[TrainingSpec], WorkloadResult], message: str
) -> None:
    service = LocalJobService(tmp_path / message.split()[0], launcher=lambda *args: None)
    job = service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7))

    with pytest.raises(JobError, match=message):
        _execute(service, job.id, executor=lambda workload, output: wrong_result(workload))


def test_evaluation_attempt_rejects_a_training_result(local_service, tmp_path: Path) -> None:
    service = local_service
    spec = EvaluationSpec.create(game="rock_paper_scissors", policy=_ArtifactMetadata("policy/manifest.json", "0" * 64))
    job = service.submit(spec)

    with pytest.raises(JobError, match="evaluation attempt returned a training result"):
        _execute(
            service,
            job.id,
            executor=lambda workload, output: WorkloadResult.training(
                workload.digest, 7, _ArtifactMetadata("policy/manifest.json", "0" * 64)
            ),
        )


def test_unknown_birth_identity_is_not_proof_that_a_process_died(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution_local, "_process_identity", lambda pid: None)

    assert execution_local._process_alive(os.getpid(), "recorded-birth")


def test_unverified_non_child_pid_is_never_signalled(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[tuple[int, int]] = []

    def not_a_child(pid: int, options: int) -> tuple[int, int]:
        del pid, options
        raise ChildProcessError

    monkeypatch.setattr(execution_local, "_process_alive", lambda *args: True)
    monkeypatch.setattr(execution_local, "_process_identity", lambda pid: None)
    monkeypatch.setattr(execution_local.os, "waitpid", not_a_child)
    monkeypatch.setattr(execution_local.os, "kill", lambda pid, signal_number: signals.append((pid, signal_number)))

    stopped = execution_local._terminate_launched_process(4242, "birth:4242")

    assert not stopped
    assert signals == []


def test_launch_record_failure_terminates_the_untracked_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write_document = execution_local.write_document

    def fail_final_launcher_record(document, path, **kwargs):
        value = dict(document) if isinstance(document, Mapping) else document.to_document()
        if Path(path).name == "launcher.json" and value.get("status") == "launched":
            raise OSError("disk full")

        return real_write_document(document, path, **kwargs)

    monkeypatch.setattr(execution_local, "write_document", fail_final_launcher_record)
    monkeypatch.setattr(execution_local, "_terminate_launched_process", lambda pid, identity: True)
    service = LocalJobService(tmp_path / "runs", launcher=lambda *args: 4242)
    output = tmp_path / "result.json"

    with pytest.raises(JobError, match="process metadata could not be persisted"):
        service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7), output=output)

    states = list((tmp_path / "runs").glob("*/state.json"))
    assert json.loads(states[0].read_text())["state"] == "failed"
    assert not any((tmp_path / "runs" / ".output-reservations").iterdir())


def test_dead_attempt_is_reconciled_to_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = LocalJobService(tmp_path / "runs", launcher=lambda *args: 4242)
    output = tmp_path / "result.json"
    job = service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7), output=output)
    monkeypatch.setattr(execution_local, "_process_alive", lambda pid, identity=None: False)

    assert job.state == "failed"

    with pytest.raises(JobFailedError, match="exited without a valid result"):
        job.result(timeout=1)

    assert not any((tmp_path / "runs" / ".output-reservations").iterdir())


def test_result_wait_timeout_and_async_cancellation(local_service, tmp_path: Path) -> None:
    service = local_service
    submitted = service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7))
    handle = type(submitted)(submitted.id, submitted._runs, _poll_seconds=0.001)

    with pytest.raises(JobTimeoutError):
        handle.result(timeout=0)

    with pytest.raises(JobTimeoutError):
        asyncio.run(handle.result_async(timeout=0.005))

    async def cancel_wait() -> None:
        task = asyncio.create_task(handle.result_async())
        await asyncio.sleep(0.005)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_wait())


def test_async_readiness_check_does_not_block_the_event_loop(local_service, monkeypatch):
    handle = local_service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7))
    checking, release = threading.Event(), threading.Event()

    def ready(self):
        checking.set()
        assert release.wait(timeout=2), "event loop could not release the readiness check"
        return {"finished": True}

    monkeypatch.setattr(type(handle), "_result_if_ready", ready)

    async def wait():
        task = asyncio.create_task(handle.result_async())
        try:
            assert await asyncio.to_thread(checking.wait, 1)
        finally:
            release.set()

        assert await task == {"finished": True}

    asyncio.run(wait())


@pytest.mark.parametrize("dedicated", [True, False])
def test_launch_recovery_stops_a_verified_private_group(dedicated, monkeypatch):
    signals = []
    monkeypatch.setattr(execution_local, "_process_alive", lambda *args: not signals)
    monkeypatch.setattr(execution_local, "_may_signal_launched_process", lambda *args: True)
    monkeypatch.setattr(execution_local.os, "getpgid", lambda pid: pid if dedicated else 2)
    monkeypatch.setattr(execution_local.os, "getsid", lambda pid: pid if dedicated else 2)
    monkeypatch.setattr(execution_local.os, "killpg", lambda pid, number: signals.append(("group", pid, number)))
    monkeypatch.setattr(execution_local.os, "kill", lambda pid, number: signals.append(("process", pid, number)))

    assert execution_local._terminate_launched_process(4242, "verified")
    assert signals == [("group", 4242, execution_local.signal.SIGKILL) if dedicated
                       else ("process", 4242, execution_local.signal.SIGTERM)]


def test_canonical_result_tampering_is_detected(local_service, tmp_path: Path) -> None:
    service = local_service
    job = service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7))
    _execute(service, job.id, executor=lambda workload, destination: _result(workload))
    (job.run_directory / "result.json").write_text("{}", encoding="utf-8")

    with pytest.raises(JobFailedError, match="failed verification"):
        job.result(timeout=0.1)


def test_job_ids_cannot_escape_the_run_store(local_service, tmp_path: Path) -> None:
    service = local_service

    with pytest.raises(JobError, match="invalid job ID"):
        service.open("../../another-directory")
