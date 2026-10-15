from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmax import PolicyProfile, load_game
from benchmax.execution import AdapterTarget, Job, LocalTarget, Target
from benchmax.execution.artifacts.results import WorkloadResult
from benchmax.execution.specs import EntrySpec, EvaluationSpec, TrainingSpec, _ArtifactMetadata
from benchmax.execution.targets.service import JobError, _JobDeadline
from benchmax.training import train
from tests.execution.jobs._helpers import _Service


def test_job_cannot_be_constructed_from_private_execution_payloads() -> None:
    with pytest.raises(TypeError, match="Jobs are created by"):
        Job()


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_job_deadline_rejects_invalid_timeouts(timeout):
    with pytest.raises(ValueError, match="timeout"):
        _JobDeadline("job-1", timeout)


def test_training_and_evaluation_return_the_same_deferred_job_class(tmp_path: Path) -> None:
    game = load_game("rock_paper_scissors")
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")

    training = train(game, seed=7, submit=True, output=tmp_path / "training.json")
    evaluation = game.evaluate(policy, submit=True, output=tmp_path / "evaluation.json")

    assert type(training) is Job
    assert type(evaluation) is Job
    assert training.state == "created"
    assert evaluation.state == "created"
    assert training.id is None
    assert evaluation.id is None
    assert training.output == (tmp_path / "training.json").resolve()
    assert evaluation.output == (tmp_path / "evaluation.json").resolve()


def test_relative_output_is_owned_by_the_calling_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    job = train(load_game("rock_paper_scissors"), seed=7, submit=True, output="training.json")

    assert job.output == tmp_path / "training.json"


def test_run_binds_one_target_and_result_returns_the_domain_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmax.execution.jobs import job as job_module

    service = _Service()
    monkeypatch.setattr(job_module, "resolve_job_service", lambda target: service)
    game = load_game("rock_paper_scissors")
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    output = tmp_path / "evaluation.json"
    job = game.evaluate(policy, submit=True, output=output)

    assert job.run(target="local") is job
    assert job.run(target="local") is job
    assert job.id == "evaluation-0000000000000001"
    assert job.state == "succeeded"
    assert len(service.submissions) == 1
    assert service.submissions[0][1] == output
    expected = {"exploitability": 0.25, "custom_player_metric": {0: 1.0, 1: -1.0}}
    assert job.result() == expected
    assert asyncio.run(job.result_async()) == expected

    reopened = Job.open(job.id, target="local")
    assert type(reopened) is Job
    assert reopened.result() == expected

    with pytest.raises(JobError, match="already runs on target"):
        job.run(target="gpu-node")


def test_target_is_a_shortcut_for_running_the_created_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmax.execution.jobs import job as job_module

    service = _Service()
    targets: list[Target] = []

    def resolve(target: Target) -> _Service:
        targets.append(target)
        return service

    monkeypatch.setattr(job_module, "resolve_job_service", resolve)
    game = load_game("rock_paper_scissors")
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")

    training = train(game, seed=7, submit=True, target="test://gpu")
    evaluation = game.evaluate(policy, submit=True, target="test://cpu")

    assert training.state == "succeeded"
    assert evaluation.state == "succeeded"
    assert targets == [AdapterTarget("test://gpu"), AdapterTarget("test://cpu")]
    assert len(service.submissions) == 2


def test_target_requires_job_mode(tmp_path: Path) -> None:
    game = load_game("rock_paper_scissors")
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="target selects where a submitted job runs"):
        train(game, target="local")

    with pytest.raises(ValueError, match="target selects where a submitted job runs"):
        game.evaluate(policy, target="local")


def test_unstarted_job_can_be_audited_but_not_awaited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmax.execution.jobs import job as job_module

    service = _Service()
    monkeypatch.setattr(job_module, "resolve_job_service", lambda target: service)
    game = load_game("rock_paper_scissors")
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    job = game.evaluate(policy, submit=True)

    report = job.audit(target="local")

    assert report.workload_digest
    assert len(service.audits) == 1

    with pytest.raises(JobError, match="has not been run"):
        job.result()


def test_job_preserves_the_exact_loaded_game_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmax.execution.jobs import job as job_module

    service = _Service()
    monkeypatch.setattr(job_module, "resolve_job_service", lambda target: service)
    game = load_game("rock_paper_scissors", max_episode_steps=2)
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")

    game.evaluate(policy, submit=True).run()

    workload, _ = service.submissions[0]
    assert isinstance(workload, EvaluationSpec)
    assert workload.game.overrides == {"max_episode_steps": 2}
    assert workload.game.resolved is not None
    assert workload.game.resolved["max_episode_steps"] == 2


def test_submitted_named_evaluator_is_part_of_the_portable_game_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmax.execution.jobs import job as job_module

    service = _Service()
    monkeypatch.setattr(job_module, "resolve_job_service", lambda target: service)
    game = load_game("rock_paper_scissors", evaluator=None)
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")

    game.evaluate(policy, evaluator="matrix", submit=True).run()

    workload, _ = service.submissions[0]
    assert isinstance(workload, EvaluationSpec)
    assert workload.game.overrides == {"evaluator": "matrix"}
    assert workload.game.resolved is not None
    assert workload.game.resolved["evaluator"] == {"_target_": "benchmax.evaluation.MatrixEvaluator"}
    assert game.evaluator is None


def test_synchronous_evaluation_remains_a_plain_metrics_call() -> None:
    game = load_game("rock_paper_scissors")

    def uniform(player, observations, action_space, legal_action_mask):
        del player, observations, action_space
        probabilities = legal_action_mask.astype(float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    metrics = game.evaluate(PolicyProfile(uniform))

    assert isinstance(metrics, dict)
    assert "exploitability" in metrics


def test_target_uri_is_resolved_by_an_installed_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmax.execution.targets import service as service_module

    service = _Service()
    targets: list[str] = []

    class EntryPoint:
        name = "test"

        @staticmethod
        def load():
            def factory(target: str) -> _Service:
                targets.append(target)
                return service

            return factory

    monkeypatch.setattr(
        service_module, "entry_points", lambda *, group: [EntryPoint()] if group == "benchmax.execution_targets" else []
    )
    game = load_game("rock_paper_scissors")
    job = train(game, seed=7, submit=True)

    job.run(target="test://gpu-node")

    assert targets == ["test://gpu-node"]
    assert len(service.submissions) == 1
    workload, _ = service.submissions[0]
    assert isinstance(workload, TrainingSpec)
    assert workload.game.resolved is not None
    assert workload.algorithm.resolved is not None


def test_broken_target_adapter_has_a_clean_execution_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmax.execution.targets import service as service_module

    class EntryPoint:
        name = "broken"

        @staticmethod
        def load():
            raise ImportError("adapter dependency is missing")

    monkeypatch.setattr(service_module, "entry_points", lambda **kwargs: [EntryPoint()])

    with pytest.raises(JobError, match="cannot load target adapter 'broken'"):
        service_module.resolve_job_service("broken://example")


def test_remote_adapter_can_stage_inputs_without_exposing_remote_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmax.execution.specs import _ArtifactMetadata
    from benchmax.execution.targets import service as service_module

    class RemoteService(_Service):
        staged: EvaluationSpec | None = None

        def submit(self, workload, *, output=None):
            assert isinstance(workload, EvaluationSpec)
            local_policy = Path(workload.policy.uri)
            assert local_policy.read_bytes() == b"{}"
            self.staged = EvaluationSpec(
                workload.game, _ArtifactMetadata(f"artifact://sha256/{workload.policy.digest}", workload.policy.digest)
            )
            assert self.staged.digest == workload.digest
            return super().submit(workload, output=output)

    remote = RemoteService()

    class EntryPoint:
        name = "test"

        @staticmethod
        def load():
            return lambda target: remote

    monkeypatch.setattr(service_module, "entry_points", lambda **kwargs: [EntryPoint()])
    policy = tmp_path / "policy.json"
    policy.write_bytes(b"{}")
    game = load_game("rock_paper_scissors")
    job = game.evaluate(policy, submit=True)

    job.run(target="test://isolated-node")
    policy.unlink()

    assert remote.staged is not None
    assert remote.staged.policy.uri.startswith("artifact://")
    assert job.result()["exploitability"] == 0.25


@pytest.mark.parametrize("policy", [None, 42, {}, [], object()])
def test_evaluate_rejects_unsupported_policy_types_at_entry(policy) -> None:
    game = load_game("rock_paper_scissors")

    for submit in (False, True):
        with pytest.raises(TypeError, match="policy must be a PolicyProfile"):
            game.evaluate(policy, submit=submit)


@pytest.mark.parametrize("as_string", [False, True])
def test_evaluation_pins_a_path_lazily_in_its_creation_directory(tmp_path, monkeypatch, as_string) -> None:
    from benchmax.execution.jobs import submission

    monkeypatch.chdir(tmp_path)
    pin = Mock(wraps=submission.pin_policy)
    monkeypatch.setattr(submission, "pin_policy", pin)
    policy = tmp_path / "policy.json"
    source = policy.name if as_string else Path(policy.name)
    job = load_game("rock_paper_scissors").evaluate(source, submit=True, output="evaluation.json")
    assert job.state == "created"
    assert not policy.exists()
    pin.assert_not_called()
    policy.write_text("{}")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / policy.name).write_text('{"wrong": true}')
    monkeypatch.chdir(elsewhere)
    document = job.to_document()
    assert document["policy"]["uri"] == str(policy.resolve())
    assert document["policy"]["digest"]
    pin.assert_called_once_with(policy, output=tmp_path / "evaluation.json",
                                game=EntrySpec.from_document(document["game"], label="game"))


def test_deferred_sweep_captures_each_nested_configuration() -> None:
    game = load_game("rock_paper_scissors")
    settings = {"model": {"num_cells": [8]}}
    jobs = []
    for width in (8, 16):
        settings["model"]["num_cells"][0] = width
        jobs.append(train(game, params=settings, seed=7, submit=True))

    settings["model"]["num_cells"].append(32)
    assert [job.to_document()["algorithm"]["overrides"]["params"]["model"]["num_cells"]
            for job in jobs] == [[8], [16]]


@pytest.mark.parametrize("invalid", ["digest", "operation", "seed"])
def test_adapter_results_are_validated_before_loading(invalid, monkeypatch) -> None:
    service = _Service()
    monkeypatch.setattr(LocalTarget, "service", lambda self: service)
    game = load_game("rock_paper_scissors")
    job = train(game, seed=7, submit=True, target=LocalTarget())
    workload, handle = job._prepare(), service.handles[job.id]
    result = (WorkloadResult.evaluation(workload.digest, {}) if invalid == "operation" else
              WorkloadResult.training("0" * 64 if invalid == "digest" else workload.digest,
                                      8 if invalid == "seed" else 7, _ArtifactMetadata("/policy.json", "0" * 64)))
    monkeypatch.setattr(handle, "result", lambda timeout=None: result.to_document())
    loader = Mock(side_effect=AssertionError("invalid result reached loader"))
    monkeypatch.setattr(handle, "load_policy", loader, raising=False)

    for load in (job.result, job.reference().result, lambda: game.evaluate(job.reference(), submit=False),
                 lambda: asyncio.run(job.result_async())):
        with pytest.raises(JobError, match="does not match|returned an"):
            load()

    loader.assert_not_called()
