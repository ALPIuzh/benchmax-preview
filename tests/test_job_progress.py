from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmax.execution import Job, LocalTarget, SlurmTarget
from benchmax.execution.jobs.progress import JobProgress
from benchmax.execution.specs import EvaluationSpec, TrainingSpec
from benchmax.execution.targets.local.service import LocalJobService
from benchmax.execution.targets.service import JobError, _write_state
from benchmax.execution.targets.slurm import bridge
from benchmax.execution.targets.slurm.service import SlurmJobHandle, SlurmJobService


@pytest.fixture
def local_jobs(tmp_path, monkeypatch):
    target = LocalTarget(runs=str(tmp_path / "runs"))
    service = LocalJobService(target.runs, launcher=lambda *args: None)
    monkeypatch.setattr(LocalTarget, "service", lambda self: service)
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7, total_frames=100)
    training = Job._from_workload(spec).run(target)
    evaluation = Job._from_workload(EvaluationSpec(spec.game, training.reference())).run(target)
    return SimpleNamespace(target=target, training=training, evaluation=evaluation, runs=Path(target.runs))


def test_unsubmitted_progress_does_not_follow_remote_references(local_jobs, monkeypatch):
    spec = EvaluationSpec(local_jobs.training._prepare().game, local_jobs.training.reference())
    evaluation = Job._from_workload(spec)

    def forbidden(*args, **kwargs):
        pytest.fail("unsubmitted inspection and rendering must be passive")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    progress = evaluation.progress()
    assert progress.state == "created"
    assert progress.dependencies[0].state == "unknown"
    assert "Not queried" in str(progress)
    assert "not submitted" in progress._repr_html_()


@pytest.mark.parametrize("producer_state", ["pending", "running", "failed", "cancelled", "succeeded"])
def test_local_progress_explains_dependency_lifecycle(local_jobs, producer_state):
    jobs = local_jobs
    _write_state(jobs.runs / jobs.training.id, producer_state)
    before = {path: path.read_bytes() for path in jobs.runs.rglob("*") if path.is_file()}
    progress = jobs.evaluation.progress()
    after = {path: path.read_bytes() for path in jobs.runs.rglob("*") if path.is_file()}
    assert before == after  # Progress reads metadata; it never reconciles or mutates a job.
    assert progress.state == "pending"
    assert progress.dependencies[0].job_id == jobs.training.id
    assert progress.dependencies[0].state == producer_state

    if producer_state in {"pending", "running"}:
        assert progress.phase == "waiting_for_policy"
        assert jobs.training.id in progress.reason
    elif producer_state in {"failed", "cancelled"}:
        assert progress.phase == "dependency_failed"
    else:
        assert progress.phase == "queued"
        assert progress.dependencies[0].fraction == 1.0


def test_running_local_wrapper_is_shown_as_waiting_for_policy(local_jobs):
    _write_state(local_jobs.runs / local_jobs.evaluation.id, "running", started_at="2026-09-10T12:00:00+00:00")
    progress = local_jobs.evaluation.progress()
    assert progress.state == "running" and progress.phase == "waiting_for_policy"
    assert progress.elapsed_seconds is not None
    assert progress.eta_seconds is None


def test_snapshot_is_static_immutable_serializable_and_renders_in_jupyter(local_jobs, monkeypatch):
    progress = local_jobs.evaluation.progress()
    DisplayFormatter = pytest.importorskip("IPython.core.formatters").DisplayFormatter

    formatter = DisplayFormatter()

    def forbidden(*args, **kwargs):
        pytest.fail("rendering a fetched snapshot must not perform I/O or wait")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", forbidden)
        guard.setattr(subprocess, "run", forbidden)
        guard.setattr("time.sleep", forbidden)
        mime, _ = formatter.format(progress)
        assert "text/html" in mime and "text/plain" in mime
        assert local_jobs.training.id in mime["text/html"]
        assert local_jobs.evaluation.id in mime["text/plain"]
        assert "Depends on" in mime["text/html"]
        assert "Checked:" in mime["text/html"]
        restored = JobProgress.from_document(json.loads(json.dumps(progress.to_document())))
        assert restored == progress

    with pytest.raises(FrozenInstanceError):
        progress.state = "running"

    _write_state(local_jobs.runs / local_jobs.training.id, "running")
    assert progress.dependencies[0].state == "pending"
    assert local_jobs.evaluation.progress().dependencies[0].state == "running"


def test_html_escapes_untrusted_job_text():
    progress = JobProgress("<script>bad()</script>", "training", "failed", "failed", "now",
                           reason='<img src=x onerror="bad()">')
    html = progress._repr_html_()
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html


def test_missing_or_tampered_dependency_is_shown_as_unknown(local_jobs):
    (local_jobs.runs / local_jobs.training.id / "spec.json").unlink()
    progress = local_jobs.evaluation.progress()
    assert progress.dependencies[0].state == "unknown"
    assert "unavailable" in progress.dependencies[0].reason
    assert progress.phase == "waiting_for_policy"


@pytest.fixture
def slurm_jobs(tmp_path, monkeypatch):
    target = SlurmTarget(python=sys.executable, workspace=str(tmp_path), runs=str(tmp_path / "runs"),
                         host="cluster.example", user="tester", poll_seconds=300)
    commands, requests = [], []
    queue = {"123": ("PENDING", "Resources", "Unknown"), "124": ("PENDING", "Dependency", "Unknown")}
    accounting = {"123": "PENDING", "124": "PENDING"}
    submitted = []

    def run(command, **kwargs):
        commands.append(command)

        if command[0] == "sbatch":
            sid = str(123 + len(submitted))
            submitted.append(sid)
            return SimpleNamespace(stdout=sid, stderr="", returncode=0)

        if command[0] == "sacct":
            ids = command[command.index("-j") + 1].split(",")
            rows = [f"{sid}|{accounting[sid]}|0:0|Unknown|Unknown|0" for sid in ids if sid in accounting]
            return SimpleNamespace(stdout="\n".join(rows), stderr="", returncode=0)

        if command[0] == "squeue":
            ids = next(arg.split("=", 1)[1].split(",") for arg in command if arg.startswith("--jobs="))
            rows = ["|".join((sid, *queue[sid])) for sid in ids if sid in queue]
            return SimpleNamespace(stdout="\n".join(rows), stderr="", returncode=0)

        pytest.fail(f"unexpected command {command}")

    monkeypatch.setattr(subprocess, "run", run)

    class DirectService(SlurmJobService):
        def _request(self, operation, **fields):
            requests.append(operation)
            return bridge.dispatch({"schema_version": 1, "operation": operation,
                                    "target": self.target.to_document(),
                                    "source_digest": bridge.package_fingerprint(), **fields})

    monkeypatch.setattr(SlurmTarget, "service", lambda self: DirectService(self))
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7, total_frames=100)
    training = Job._from_workload(spec).run(target)
    evaluation = Job._from_workload(EvaluationSpec(spec.game, training.reference())).run(target)
    commands.clear()
    requests.clear()
    return SimpleNamespace(target=target, training=training, evaluation=evaluation, commands=commands,
                           requests=requests, queue=queue, accounting=accounting, runs=Path(target.runs))


def test_slurm_progress_batches_exact_ids_in_one_control_request(slurm_jobs):
    jobs = slurm_jobs
    progress = jobs.evaluation.progress()
    assert jobs.requests == ["progress"]
    assert jobs.commands == [
        ["sacct", "-n", "-P", "-j", "123,124", "--format=JobIDRaw,State%30,ExitCode,Start,End,ElapsedRaw"],
        ["squeue", "--noheader", "--jobs=123,124", "--format=%i|%T|%r|%S"],
    ]
    assert progress.phase == "waiting_for_policy"
    assert progress.dependencies[0].phase == "queued"
    assert progress.dependencies[0].reason == "Resources"
    assert progress.dependencies[0].scheduler_id == "123"
    assert progress.eta_seconds is None


def test_slurm_progress_shows_running_upstream_and_later_resource_queue(slurm_jobs):
    jobs = slurm_jobs
    jobs.queue["123"] = ("RUNNING", "None", "2026-09-10T12:00:00")
    progress = jobs.evaluation.progress()
    assert progress.phase == "waiting_for_policy" and progress.dependencies[0].phase == "running"
    _write_state(jobs.runs / jobs.training.id, "succeeded")
    jobs.queue["124"] = ("PENDING", "Resources", "2026-09-10T13:00:00")
    jobs.commands.clear()
    progress = jobs.evaluation.progress()
    assert progress.phase == "queued" and progress.reason == "Resources"
    assert progress.dependencies[0].state == "succeeded"
    assert progress.expected_start == "2026-09-10T13:00:00" and progress.eta_seconds is None
    assert jobs.commands[0][4] == "124"  # Retain completed dependency, without querying it again.


def test_slurm_progress_reports_failed_dependency_without_loading_results(slurm_jobs, monkeypatch):
    jobs = slurm_jobs
    jobs.accounting["123"] = "FAILED"
    jobs.queue.pop("123")
    monkeypatch.setattr(bridge, "_snapshot", lambda *args, **kwargs: pytest.fail("do not load or reconcile results"))
    progress = jobs.evaluation.progress()
    assert progress.phase == "dependency_failed"
    assert progress.dependencies[0].state == "failed"
    jobs.accounting["124"] = "CANCELLED"
    jobs.queue.pop("124")
    progress = jobs.evaluation.progress()
    assert progress.state == "cancelled" and "dependency failed" in progress.reason


def test_scheduler_completion_does_not_claim_an_unpublished_result(slurm_jobs):
    jobs = slurm_jobs
    jobs.accounting["123"] = "COMPLETED"
    jobs.queue.pop("123")
    progress = jobs.training.progress()
    assert progress.state != "succeeded"
    assert progress.phase == "awaiting_result"
    assert "not yet been confirmed" in progress.reason


@pytest.mark.parametrize("state", ["OUT_OF_MEMORY", "NODE_FAIL", "TIMEOUT", "CANCELLED by 1000"])
def test_accounting_terminal_states_are_reported_without_queue_polling(slurm_jobs, state):
    jobs = slurm_jobs
    jobs.accounting["123"] = state
    progress = jobs.training.progress()
    expected = "cancelled" if state.startswith("CANCELLED") else "failed"
    assert progress.state == expected
    assert len(jobs.commands) == 1 and jobs.commands[0][0] == "sacct"


def test_scheduler_failures_and_missing_records_are_visible(slurm_jobs, monkeypatch):
    monkeypatch.setattr(bridge, "_run", Mock(side_effect=JobError("offline")))
    progress = slurm_jobs.evaluation.progress()
    assert progress.state == "pending"
    assert "last saved" in progress.warnings[0]
    assert "offline" in " ".join(progress.warnings)


def test_invalid_scheduler_id_never_reaches_a_command(slurm_jobs):
    jobs = slurm_jobs
    (jobs.runs / jobs.evaluation.id / "scheduler.json").write_text('{"id": "123; other"}')

    with pytest.raises(JobError, match="scheduler ID"):
        jobs.evaluation.progress()

    assert not jobs.commands


def test_ssh_progress_uses_existing_transport_and_never_submits(monkeypatch, tmp_path):
    target = SlurmTarget(python="/env/bin/python", workspace="/work", runs="/runs", host="cluster.example")
    spec = TrainingSpec.create(game="rock_paper_scissors", seed=7)
    service = SlurmJobService(target)
    job = Job._from_workload(spec)
    job._target = target
    job._handle = SlurmJobHandle(service, {"id": "training-0000000000000001", "workload": spec.to_document(),
                                          "state": "pending", "output": "/result.json"})
    calls = []

    def run(command, **kwargs):
        request = json.loads(kwargs["input"])
        calls.append((command, request))
        value = JobProgress(job.id, "training", "pending", "queued", "now", reason="Resources").to_document()
        return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "value": value}), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert job.progress().reason == "Resources"
    assert len(calls) == 1 and calls[0][0][0] == "ssh"
    assert calls[0][1]["operation"] == "progress"
    assert calls[0][1]["id"] == job.id
