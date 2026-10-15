from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmax.cli import jobs
from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.specs import EntrySpec, EvaluationSpec, _ArtifactMetadata
from benchmax.execution.targets.local import LocalJobService
from benchmax.execution.targets.local.attempt import run_local_attempt
from tests.cli._helpers import run_cli

_ID = "evaluation-0000000000000001"



def test_job_command_shows_status_and_its_output(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = SimpleNamespace(id=_ID, state="running", output=Path("evaluation.json").resolve())
    open_job = Mock(return_value=opened)
    monkeypatch.setattr(jobs.Job, "open", open_job)

    code, output, _ = run_cli(capsys, "job", _ID, "--target", "gpu-node")

    assert code == 0
    open_job.assert_called_once_with(_ID, target="gpu-node")
    assert _ID in output
    assert "running" in output
    assert "gpu-node" in output
    assert "evaluation.json" in output


def test_job_help_is_signature_derived(capsys: pytest.CaptureFixture[str]) -> None:
    code, output, _ = run_cli(capsys, "job", "--help")

    assert code == 0
    assert "Durable job ID printed by --submit" in output
    assert "--wait" in output
    assert "--target" in output


@pytest.mark.parametrize("as_json", [False, True])
def test_job_wait_reconnects_to_a_real_persisted_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, as_json: bool
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("BENCHMAX_RUNS", str(runs))
    service = LocalJobService(runs, launcher=lambda command, environment, log: None)
    workload = EvaluationSpec(EntrySpec("rock_paper_scissors"), _ArtifactMetadata("policy.json", "1" * 64))
    submitted = service.submit(workload)
    run_local_attempt(
        submitted.id,
        runs=service._runs,
        executor=lambda spec, output: WorkloadResult.evaluation(spec.digest, {"exploitability": 0.5}),
    )

    code, output, _ = run_cli(capsys, "job", submitted.id, "--wait", *(["--json"] if as_json else []))

    document = json.loads(output)
    assert code == 0
    assert document["kind"] == "evaluation-result"
    assert document["workload_digest"] == workload.digest
    assert document["metrics"] == {"exploitability": 0.5}
