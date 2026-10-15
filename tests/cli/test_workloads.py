from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmax import cli
from benchmax.cli import workloads
from benchmax.execution.audit.resources import AuditReport
from benchmax.execution.jobs import job as job_module
from benchmax.execution.specs import EvaluationSpec, TrainingSpec, _ArtifactMetadata
from tests.cli._helpers import run_cli


def _report(spec: TrainingSpec | EvaluationSpec) -> AuditReport:
    return AuditReport(
        workload=spec, target="local", budget_seconds=45.0, duration_seconds=0.1,
        requirements={"host_memory": "ordinary"}, estimates={}, machine={"accelerators": []},
        findings=("No unusual resource requirement was found.",)
    )


@pytest.mark.parametrize("as_json", [False, True])
def test_train_audit_writes_a_reusable_audited_spec(
    as_json: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "training.json"

    monkeypatch.setattr(job_module, "resolve_job_service", lambda target: SimpleNamespace(audit=_report))

    code, out, _ = run_cli(
        capsys, "train", "--game", "rock_paper_scissors", "--seed", "7", "--audit", "--output", str(output),
        *(["--json"] if as_json else [])
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert document["kind"] == "audit"
    assert document["workload"]["kind"] == "training"
    assert document["workload"]["seed"] == 7

    if as_json:
        assert json.loads(out) == document
    else:
        assert str(output) in out


def test_evaluate_uses_the_same_mode_and_io_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evaluation.json"
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(job_module, "resolve_job_service", lambda target: SimpleNamespace(audit=_report))

    code, _, _ = run_cli(
        capsys, "evaluate", "--game", "rock_paper_scissors", "--policy", str(policy), "--audit", "--output", str(output)
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert document["workload"]["kind"] == "evaluation"


def test_input_cannot_be_mixed_with_domain_arguments(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "training.json"
    input_path.write_text(
        json.dumps(TrainingSpec.create(game="rock_paper_scissors", seed=7).to_document()), encoding="utf-8"
    )

    with pytest.raises(SystemExit):
        run_cli(capsys, "train", "--input", str(input_path), "--game", "kuhn_poker", "--audit")


def test_submit_and_audit_are_mutually_exclusive(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        run_cli(capsys, "train", "--game", "rock_paper_scissors", "--submit", "--audit")


def test_private_attempt_command_is_hidden_from_root_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["--help"])

    assert code == 0
    assert "_attempt" not in capsys.readouterr().out


def test_input_kind_mismatch_is_a_cli_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "evaluation.json"
    input_path.write_text(
        json.dumps(
            EvaluationSpec.create(
                game="rock_paper_scissors", policy=_ArtifactMetadata("policy.json", "0" * 64)
            ).to_document()
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        run_cli(capsys, "train", "--input", str(input_path), "--audit")


def test_submit_prints_the_job_id_without_waiting(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    submit = Mock(return_value=SimpleNamespace(id="job-7"))
    monkeypatch.setattr(job_module, "resolve_job_service", lambda target: SimpleNamespace(submit=submit))

    code, out, _ = run_cli(
        capsys, "train", "--game", "rock_paper_scissors", "--seed", "7", "--submit", "--output", "result.json"
    )

    assert code == 0
    assert out.strip() == "job-7"
    submit.assert_called_once()
    assert submit.call_args.args[0].digest == TrainingSpec.create(game="rock_paper_scissors", seed=7).digest


def test_workload_help_is_derived_from_the_typed_command_signature(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["train", "--help"])

    output = capsys.readouterr().out
    parameters = inspect.signature(workloads.train).parameters
    assert code == 0
    assert "game" in parameters
    assert "total_frames" in parameters
    assert "--total-frames" in output
    assert "Positive training frame budget" in output
    assert "Train a policy profile" in output


def test_unknown_target_is_a_clean_cli_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["train", "--game", "rock_paper_scissors", "--audit", "--target", "missing://example"])

    captured = capsys.readouterr()
    assert code == 1
    assert "unknown target scheme 'missing'" in captured.err
    assert "Traceback" not in captured.err


def test_targets_lists_installed_adapters_without_inventing_their_syntax(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmax.execution.targets import service as service_module

    class EntryPoint:
        name = "example"

        @staticmethod
        def load():
            raise AssertionError("listing targets must not import an adapter")

    monkeypatch.setattr(service_module, "entry_points", lambda **kwargs: [EntryPoint()])

    code, output, _ = run_cli(capsys, "targets")

    assert code == 0
    assert "local" in output
    assert "example" in output
    assert "://" not in output
