from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from benchmax.execution.specs import TrainingSpec
from benchmax.execution.targets.local import LocalJobService
from benchmax.execution.targets.local import service as execution_local
from benchmax.execution.targets.service import JobError, JobFailedError
from tests.execution.jobs._helpers import _execute, _result


def test_invalid_persisted_spec_releases_its_output_reservation(local_service, tmp_path: Path) -> None:
    service = local_service
    output = tmp_path / "result.json"
    job = service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7), output=output)
    spec_path = job.run_directory / "spec.json"
    document = json.loads(spec_path.read_text())
    document["seed"] = 8
    spec_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(JobFailedError, match="does not match"):
        _execute(service, job.id, executor=lambda *args: {})

    assert not any((tmp_path / "runs" / ".output-reservations").iterdir())


def test_launcher_failure_is_terminal_and_releases_the_output(tmp_path: Path) -> None:
    def fail_launch(*args: object) -> None:
        raise OSError("process table full")

    service = LocalJobService(tmp_path / "runs", launcher=fail_launch)
    output = tmp_path / "result.json"

    with pytest.raises(OSError, match="process table full"):
        service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7), output=output)

    states = list((tmp_path / "runs").glob("*/state.json"))
    assert len(states) == 1
    state = json.loads(states[0].read_text())
    assert state["state"] == "failed"
    assert "process table full" in state["error"]
    assert not any((tmp_path / "runs" / ".output-reservations").iterdir())


def test_two_pending_jobs_cannot_claim_the_same_output(local_service, tmp_path: Path) -> None:
    service = local_service
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    output = tmp_path / "result.json"

    service.submit(spec, output=output)

    with pytest.raises(JobError, match="already reserved"):
        service.submit(spec, output=output)


def test_terminal_owner_reservation_is_reclaimed_after_a_crash(local_service, tmp_path: Path) -> None:
    service = local_service
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    output = tmp_path / "result.json"
    abandoned = service.submit(spec, output=output)
    state_path = abandoned.run_directory / "state.json"
    state = json.loads(state_path.read_text())
    state.update({"state": "failed", "error": "simulated crash after state write"})
    state_path.write_text(json.dumps(state), encoding="utf-8")

    replacement = service.submit(spec, output=output)

    assert replacement.id != abandoned.id


def test_dead_owner_is_reconciled_before_reusing_its_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = LocalJobService(tmp_path / "runs", launcher=lambda *args: 4242)
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    output = tmp_path / "result.json"
    abandoned = service.submit(spec, output=output)
    monkeypatch.setattr(execution_local, "_process_alive", lambda pid, identity=None: False)

    replacement = service.submit(spec, output=output)

    assert abandoned.state == "failed"
    assert replacement.id != abandoned.id


def test_crashed_submitter_is_reconciled_before_the_attempt_launches(
    local_service, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = local_service
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    output = tmp_path / "result.json"
    abandoned = service.submit(spec, output=output)
    (abandoned.run_directory / "launcher.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "launching",
                "pid": None,
                "owner_pid": 4242,
                "owner_process_identity": "birth:4242",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(execution_local, "_process_alive", lambda pid, identity=None: False)

    replacement = service.submit(spec, output=output)

    assert abandoned.state == "failed"
    assert replacement.id != abandoned.id


def test_unreadable_owner_state_never_authorizes_reservation_takeover(local_service, tmp_path: Path) -> None:
    service = local_service
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    output = tmp_path / "result.json"
    owner = service.submit(spec, output=output)
    (owner.run_directory / "state.json").write_text("not JSON", encoding="utf-8")

    with pytest.raises(JobError, match="already reserved"):
        service.submit(spec, output=output)


def test_concurrent_submission_cannot_steal_an_initializing_claim(tmp_path: Path) -> None:
    launching = threading.Event()
    finish_launch = threading.Event()

    def launch(*args: object) -> None:
        launching.set()
        assert finish_launch.wait(timeout=2)

    service = LocalJobService(tmp_path / "runs", launcher=launch)
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    output = tmp_path / "result.json"
    failures: list[BaseException] = []

    def submit_first() -> None:
        try:
            service.submit(spec, output=output)
        except (JobError, OSError, AssertionError) as error:
            failures.append(error)

    thread = threading.Thread(target=submit_first)
    thread.start()
    assert launching.wait(timeout=2)

    try:
        with pytest.raises(JobError, match="already reserved"):
            service.submit(spec, output=output)
    finally:
        finish_launch.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert not failures


def test_attempt_never_overwrites_an_output_created_while_it_ran(local_service, tmp_path: Path) -> None:
    service = local_service
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    output = tmp_path / "result.json"
    job = service.submit(spec, output=output)
    output.write_text("owned by another process", encoding="utf-8")

    with pytest.raises(JobError, match="created while the job ran"):
        _execute(service, job.id, executor=lambda workload, destination: _result(workload))

    assert output.read_text(encoding="utf-8") == "owned by another process"
    assert job.state == "failed"
    assert not any((tmp_path / "runs" / ".output-reservations").iterdir())


def test_success_publishes_once_and_releases_its_reservation(local_service, tmp_path: Path) -> None:
    service = local_service
    output = tmp_path / "result.json"
    job = service.submit(TrainingSpec.create(game="rock_paper_scissors", seed=7), output=output)

    _execute(service, job.id, executor=lambda workload, destination: _result(workload))

    assert json.loads(output.read_text())["kind"] == "training-result"
    assert not any((tmp_path / "runs" / ".output-reservations").iterdir())
