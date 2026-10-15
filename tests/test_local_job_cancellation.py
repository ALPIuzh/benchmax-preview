from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmax import load_game
from benchmax.execution import Job, LocalTarget
from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.specs import TrainingSpec, _ArtifactMetadata
from benchmax.execution.targets.local import LocalJobService, cancellation, service
from benchmax.execution.targets.local.attempt import run_local_attempt
from benchmax.execution.targets.service import JobError, JobFailedError


def wait_for(predicate, *, timeout=5):
    deadline = time.monotonic() + timeout

    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("local attempt did not reach the expected state")

        time.sleep(0.02)


def process_running(pid):
    # A killed child may remain a zombie until its new parent reaps it on Linux.
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], text=True, capture_output=True, check=False)
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


@pytest.fixture
def local(tmp_path, monkeypatch):
    target = LocalTarget(runs=str(tmp_path / "runs"))
    jobs = LocalJobService(target.runs, launcher=lambda *a: None)
    monkeypatch.setattr(LocalTarget, "service", lambda self: jobs)
    return SimpleNamespace(target=target, jobs=jobs, runs=Path(target.runs), root=tmp_path)


def training(local, *, output=None):
    return Job._from_workload(TrainingSpec.create(game="rock_paper_scissors", seed=7), output=output).run(local.target)


def test_pending_cancellation_prevents_late_execution_and_releases_output(local):
    output = local.root / "policy.json"
    job = training(local, output=output)
    reopened = Job.open(job.id, target=local.target)
    reopened.cancel()
    assert job.state == "cancelled" and job.progress().state == "cancelled"
    assert not output.exists()

    with pytest.raises(JobFailedError, match="cancelled"):
        job.result(timeout=0)

    with pytest.raises(JobFailedError, match="cancelled"):
        run_local_attempt(job.id, runs=local.runs, executor=lambda *a: pytest.fail("cancelled job computed"))

    assert training(local, output=output).id != job.id
    # A second cancellation cannot release the replacement job's reservation.
    reopened.cancel()

    with pytest.raises(JobError, match="already reserved"):
        training(local, output=output)


@pytest.fixture
def processes(local):
    program = '''import json
import os
import pathlib
import sys
import time

from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.specs import _ArtifactMetadata
from benchmax.execution.targets.local.attempt import run_local_attempt
from benchmax.execution.worker import LocalWorkerRuntime, use_evaluation_runtime
from tests.execution.worker._kernels import process_identity, sleep_in_worker

job_id = sys.argv[1]
run = pathlib.Path(os.environ["BENCHMAX_RUNS"]) / job_id
mode = sys.argv[2]
if mode == "pending":
    (run / "ready.json").write_text(json.dumps({"pid": os.getpid()}))
    time.sleep(30)

def execute(spec, output):
    if mode == "worker":
        with LocalWorkerRuntime() as runtime, use_evaluation_runtime(runtime):
            worker = process_identity()
            (run / "ready.json").write_text(json.dumps({"pid": os.getpid(), "worker": worker}))
            sleep_in_worker(30)
    else:
        (run / "ready.json").write_text(json.dumps({"pid": os.getpid()}))
        while not (run / "release").exists():
            time.sleep(0.02)
    if spec.kind == "training":
        return WorkloadResult.training(spec.digest, spec.seed, _ArtifactMetadata("/unused.json", "a" * 64))
    return WorkloadResult.evaluation(spec.digest, {"ok": True})

run_local_attempt(job_id, executor=execute)
'''
    children = []
    mode = ["running"]

    def launch(command, environment, log):
        arguments = [sys.executable, "-c", program, command[-1], mode[0]]

        with log.open("wb") as stream:
            process = subprocess.Popen(arguments, env=environment,
                                       stdout=stream, stderr=stream, stdin=subprocess.DEVNULL, start_new_session=True)

        children.append(process)
        return process.pid

    local.jobs._launcher = launch
    yield SimpleNamespace(children=children, mode=mode)

    for process in children:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)

        process.wait(timeout=5)


@pytest.mark.parametrize("mode", ["pending", "running", "worker"])
def test_cancellation_stops_only_the_selected_process_group_and_survives_reopening(local, processes, mode):
    processes.mode[0] = mode
    output = local.root / "cancelled.json"
    job = training(local, output=output)
    run = local.runs / job.id
    wait_for(lambda: (run / "ready.json").exists())
    record = json.loads((run / "ready.json").read_text())
    if "worker" in record:
        assert os.getpgid(record["worker"]) == record["pid"]

    other = training(local)
    other_pid = processes.children[-1].pid
    # Cancellation from a separate Python process proves no live Job/Popen object is required.
    command = ("from benchmax.execution import Job, LocalTarget; import sys; "
               "Job.open(sys.argv[1], target=LocalTarget(runs=sys.argv[2])).cancel()")
    subprocess.run([sys.executable, "-c", command, job.id, str(local.runs)], check=True, timeout=10)
    assert job.state == "cancelled"
    assert Job.open(job.id, target=local.target).state == "cancelled"

    for pid in record.values():
        wait_for(lambda pid=pid: not process_running(pid))

    assert process_running(other_pid) and other.state in {"pending", "running"}
    assert not output.exists() and not (run / "result.json").exists()

    with pytest.raises(JobFailedError, match="cancelled"):
        job.reference().result(timeout=0)

    job.cancel()


def test_cancelling_a_waiting_evaluation_does_not_cancel_its_producer(local, processes):
    producer = training(local)
    evaluation = load_game("rock_paper_scissors").evaluate(producer.reference(), submit=True).run(local.target)
    wait_for(lambda: producer.state == evaluation.state == "running")
    assert evaluation.progress().phase == "waiting_for_policy"
    evaluation.cancel()
    assert evaluation.state == "cancelled" and producer.state == "running"
    assert not (local.runs / evaluation.id / "ready.json").exists()
    (local.runs / producer.id / "release").touch()
    assert producer._handle.result(timeout=5)["kind"] == "training-result"


def test_cancelling_training_fails_a_waiting_evaluation_without_computation(local, processes):
    producer = training(local)
    evaluation = load_game("rock_paper_scissors").evaluate(producer.reference(), submit=True).run(local.target)
    wait_for(lambda: producer.state == evaluation.state == "running")
    producer.cancel()

    with pytest.raises(JobFailedError, match="cancelled"):
        evaluation.result(timeout=5)

    assert evaluation.state == "failed"
    assert not (local.runs / evaluation.id / "ready.json").exists()


@pytest.mark.parametrize("problem", ["missing_identity", "reused_pid", "shared_group", "self_pid", "missing_pid"])
def test_cancellation_never_signals_an_unverified_process(local, monkeypatch, problem):
    job = training(local)
    run = local.runs / job.id
    values = {"pid": 4242, "process_identity": "recorded"}

    if problem == "missing_identity":
        values["process_identity"] = None

    if problem == "self_pid":
        values["pid"] = os.getpid()

    if problem == "missing_pid":
        values.pop("pid")

    service._write_state(run, "running", **values)
    monkeypatch.setattr(cancellation.os, "getpgid", lambda pid: 999 if problem == "shared_group" else pid)
    monkeypatch.setattr(cancellation.os, "getsid", lambda pid: pid)
    monkeypatch.setattr(cancellation, "_process_identity", lambda pid: "new" if problem == "reused_pid" else "recorded")
    monkeypatch.setattr(cancellation.os, "killpg", lambda *a: pytest.fail("must not signal an unverified process"))

    with pytest.raises(JobError, match="cannot safely cancel"):
        job._handle.cancel()

    assert service._read_json(run / "state.json")["state"] == "running"
    assert not (run / "cancel.json").exists()


def test_cancellation_wins_over_result_publication_after_a_request(local):
    job = training(local)
    run = local.runs / job.id

    def execute(spec, output):
        (run / "cancel.json").write_text('{"requested_at":"now"}')
        return WorkloadResult.training(spec.digest, spec.seed, _ArtifactMetadata("/unused.json", "a" * 64))

    with pytest.raises(JobFailedError, match="cancelled"):
        run_local_attempt(job.id, runs=local.runs, executor=execute)

    assert job.state == "cancelled" and not (run / "result.json").exists()


def test_completion_wins_over_a_later_cancel(local, monkeypatch):
    job = training(local)
    run_local_attempt(job.id, runs=local.runs, executor=lambda spec, output: WorkloadResult.training(
        spec.digest, spec.seed, _ArtifactMetadata("/unused.json", "a" * 64)))
    original = (local.runs / job.id / "state.json").read_bytes()
    monkeypatch.setattr(cancellation.os, "killpg", lambda *a: pytest.fail("completed job must not be signalled"))
    job._handle.cancel()
    assert (local.runs / job.id / "state.json").read_bytes() == original


def test_signal_failure_and_stop_timeout_keep_the_output_reserved_until_retry(local, monkeypatch):
    import fcntl

    output = local.root / "reserved.json"
    job = training(local, output=output)
    run = local.runs / job.id
    monkeypatch.setattr(cancellation, "_process", lambda *a: (4242, "verified"))

    def denied(*args):
        raise PermissionError("signal denied")

    monkeypatch.setattr(cancellation.os, "killpg", denied)

    with pytest.raises(JobError, match="could not stop"):
        job.cancel()

    assert job.state == "pending"

    with pytest.raises(JobError, match="already reserved"):
        training(local, output=output)

    monkeypatch.setattr(cancellation.os, "killpg", lambda *a: None)
    monkeypatch.setattr(cancellation, "_STOP_TIMEOUT_SECONDS", 0)

    with (run / "attempt.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        with pytest.raises(JobError, match="has not stopped"):
            job.cancel()

        assert job.state == "pending"

        with pytest.raises(JobError, match="already reserved"):
            training(local, output=output)

    job.cancel()
    assert job.state == "cancelled"
    assert training(local, output=output).id != job.id


def test_reconnection_recovers_cancellation_after_the_canceller_disappears(local, monkeypatch):
    job = training(local, output=local.root / "result.json")
    run = local.runs / job.id
    workload = job._prepare()
    result = WorkloadResult.training(workload.digest, workload.seed, _ArtifactMetadata("/unused.json", "a" * 64))
    (run / "result.json").write_text(json.dumps(result.to_document()))
    (run / "cancel.json").write_text('{"requested_at":"now"}')
    service._write_state(run, "running", pid=4242, process_identity="old")
    monkeypatch.setattr(service, "_process_alive", lambda *a: False)
    assert Job.open(job.id, target=local.target).state == "cancelled"
    assert not job.output.exists()
    assert not any((local.runs / ".output-reservations").iterdir())


def test_result_publication_and_cancellation_are_serialized(local, monkeypatch):
    job = training(local)
    publishing, release, cancelling, returned = (threading.Event() for _ in range(4))
    errors = []

    def publish(*args, **kwargs):
        publishing.set()
        assert release.wait(5)

    def attempt():
        try:
            run_local_attempt(job.id, runs=local.runs, executor=lambda spec, output: WorkloadResult.training(
                spec.digest, spec.seed, _ArtifactMetadata("/unused.json", "a" * 64)))
        except BaseException as error:  # noqa: BLE001 -- surface worker assertions in the test
            errors.append(error)

    def cancel():
        cancelling.set()

        try:
            job._handle.cancel()
        except BaseException as error:  # noqa: BLE001 -- surface worker assertions in the test
            errors.append(error)
        finally:
            returned.set()

    monkeypatch.setattr("benchmax.execution.targets.local.attempt._publish_result", publish)
    monkeypatch.setattr(cancellation.os, "killpg", lambda *a: pytest.fail("completed job must not be signalled"))
    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    assert publishing.wait(5)
    canceller = threading.Thread(target=cancel, daemon=True)
    canceller.start()

    try:
        assert cancelling.wait(5) and not returned.wait(0.1)
    finally:
        release.set()
        worker.join(5)
        canceller.join(5)

    assert not worker.is_alive() and not canceller.is_alive() and not errors
    assert job.state == "succeeded"


def test_cancellation_during_launch_never_signals_submitter_or_loses_terminal_state(local, monkeypatch):
    started, release = threading.Event(), threading.Event()
    errors = []

    def launch(command, environment, log):
        started.set()
        assert release.wait(5)
        raise OSError("launch failed after cancellation")

    def submit():
        try:
            training(local, output=local.root / "reserved.json")
        except OSError as error:
            errors.append(error)

    local.jobs._launcher = launch
    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    assert started.wait(5)
    monkeypatch.setattr(cancellation.os, "killpg", lambda *a: pytest.fail("must not kill the submitting process"))
    run = next(local.runs.glob("training-*"))

    try:
        local.jobs.open(run.name).cancel()
    finally:
        release.set()
        thread.join(5)

    assert not thread.is_alive() and len(errors) == 1
    assert service._read_json(run / "state.json")["state"] == "cancelled"
    assert not any((local.runs / ".output-reservations").iterdir())
