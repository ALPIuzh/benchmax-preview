from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmax.execution import Job, LocalTarget, SlurmTarget
from benchmax.execution.specs import TrainingSpec
from benchmax.execution.targets.service import JobError, JobSubmissionUnknownError
from benchmax.execution.targets.slurm.service import SlurmJobService


def unsubmitted():
    return Job._from_workload(TrainingSpec.create(game="rock_paper_scissors", seed=7))


def test_cancel_unsubmitted_is_passive_and_does_not_prepare_workload(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unsubmitted cancellation must not perform I/O or prepare work")

    job = Job._create(forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr("benchmax.execution.jobs.job.resolve_job_service", forbidden)
    assert job.cancel() is None
    assert job.cancel() is None
    assert job.state == "created" and job.id is None


def test_cancel_after_definite_rejection_is_noop_and_job_can_be_retried(monkeypatch, tmp_path):
    job = unsubmitted()
    requests = []
    handle = SimpleNamespace(id="training-0000000000000001", state="pending")

    def submit(*args, **kwargs):
        requests.append("submit")

        if len(requests) == 1:
            raise JobError("installed BenchMAX source differs from the caller")

        return handle

    monkeypatch.setattr("benchmax.execution.jobs.job.resolve_job_service", lambda target: SimpleNamespace(submit=submit))
    target = LocalTarget(runs=str(tmp_path))

    with pytest.raises(JobError, match="source differs"):
        job.run(target)

    assert job.cancel() is None
    assert job.progress().state == "created"
    assert requests == ["submit"]
    assert job.run(target) is job
    assert job.id == handle.id and requests == ["submit", "submit"]


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled"])
def test_cancel_terminal_job_does_not_call_adapter_cancel(state):
    job = unsubmitted()
    job._handle = SimpleNamespace(state=state, cancel=lambda: pytest.fail("terminal job must not be cancelled again"))
    assert job.cancel() is None


def test_active_job_cancellation_is_forwarded_and_real_errors_survive():
    job = unsubmitted()
    calls = []

    def cancel():
        calls.append("cancel")
        raise JobError("SSH unavailable")

    job._handle = SimpleNamespace(state="running", cancel=cancel)

    with pytest.raises(JobError, match="SSH unavailable"):
        job.cancel()

    assert calls == ["cancel"]
    job._handle = SimpleNamespace(state="running")

    with pytest.raises(JobError, match="does not support cancellation"):
        job.cancel()


def test_repeated_cancellation_does_not_repeat_an_accepted_request():
    job = unsubmitted()
    calls = []
    job._handle = SimpleNamespace(state="running", cancel=lambda: calls.append("cancel"))
    assert job.cancel() is None
    assert job.cancel() is None
    assert calls == ["cancel"]


def test_bridge_preserves_submission_uncertainty_in_error_response(monkeypatch, capsys):
    from benchmax.execution.targets.slurm import bridge

    def dispatch(request):
        raise JobSubmissionUnknownError("sbatch response lost", job_id="training-0000000000000001")

    monkeypatch.setattr(bridge, "dispatch", dispatch)
    monkeypatch.setattr("sys.argv", ["bridge"])
    monkeypatch.setattr("sys.stdin", StringIO("{}"))

    with pytest.raises(SystemExit):
        bridge.main()

    response = json.loads(capsys.readouterr().out)
    assert response == {"ok": False, "error": "sbatch response lost", "submission_unknown": True}


@pytest.fixture
def ssh_submission(monkeypatch):
    target = SlurmTarget(python="/env/bin/python", workspace="/work", runs="/runs", host="cluster.example")
    responses = []
    requests = []

    def run(command, **kwargs):
        assert command[0] == "ssh"
        requests.append(json.loads(kwargs["input"]))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response

        return SimpleNamespace(returncode=1 if isinstance(response, dict) and response.get("ok") is False else 0,
                               stdout=response if isinstance(response, str) else json.dumps(response), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(SlurmTarget, "service", lambda self: SlurmJobService(self))
    return SimpleNamespace(target=target, responses=responses, requests=requests)


def test_source_mismatch_reply_allows_noop_cancel(ssh_submission):
    ssh = ssh_submission
    ssh.responses.append({"ok": False, "error": "installed BenchMAX source differs from the caller"})
    job = unsubmitted()

    with pytest.raises(JobError, match="source differs"):
        job.run(ssh.target)

    assert job.cancel() is None
    assert job.state == "created" and job.id is None
    assert len(ssh.requests) == 1


@pytest.mark.parametrize("response", [
    subprocess.TimeoutExpired("ssh", 10), OSError("connection interrupted"), "not JSON", [], {},
    {"ok": True, "value": None}, {"ok": True, "value": {}},
    {"ok": False, "error": "sbatch timed out", "submission_unknown": True},
])
def test_uncertain_submission_is_retained_and_cannot_be_silently_cancelled_or_retried(ssh_submission, response):
    ssh = ssh_submission
    ssh.responses.append(response)
    job = unsubmitted()

    with pytest.raises(JobSubmissionUnknownError) as submitted:
        job.run(ssh.target)

    assert job.state == "submission_unknown"
    assert job.id == submitted.value.job_id == ssh.requests[0]["id"]
    progress = job.progress()
    assert progress.state == "submission_unknown" and progress.phase == "unknown"

    for action in (job.cancel, lambda: job.run(ssh.target)):
        with pytest.raises(JobSubmissionUnknownError) as repeated:
            action()

        assert repeated.value is submitted.value

    assert len(ssh.requests) == 1
