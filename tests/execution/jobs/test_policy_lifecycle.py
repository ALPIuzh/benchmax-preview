from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.training_helpers import tiny_training_params

pytest.importorskip("benchmarl")

from benchmax import load_algorithm, load_game
from benchmax.evaluation import IncompatibleEvaluatorError, MatrixEvaluator
from benchmax.execution import Job
from benchmax.execution.artifacts.policies import load_policy
from benchmax.execution.artifacts.results import result_from_document
from benchmax.execution.specs import TrainingSpec
from benchmax.execution.targets.service import JobError
from benchmax.training import train


def test_trained_policy_becomes_a_job_artifact_without_user_scaffolding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_trained_policy
) -> None:
    game, policy, _ = tiny_trained_policy
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("BENCHMAX_ARTIFACTS", str(artifacts))

    job = game.evaluate(policy, submit=True)
    repeated = game.evaluate(policy, submit=True)

    assert type(job) is Job
    assert type(repeated) is Job
    assert job.state == "created"
    assert not artifacts.exists()

    job.audit()
    repeated.audit()

    manifests = list(artifacts.glob("policy-*/manifest.json"))
    assert len(manifests) == 1


@pytest.mark.parametrize("discount", [None, 0.9])
def test_named_evaluator_output_preserves_the_callers_game(
    tmp_path: Path, tiny_trained_policy, discount: float | None,
) -> None:
    _, policy, _ = tiny_trained_policy
    game = load_game("rock_paper_scissors", evaluator=None)
    game.discount = discount
    output = tmp_path / "evaluation.json"

    if discount is None:
        expected = game.evaluate(policy, evaluator="matrix")
        metrics = game.evaluate(policy, evaluator="matrix", output=output)
        persisted = result_from_document(json.loads(output.read_text()))
        assert persisted.metrics == metrics == expected
    else:
        for destination in (None, output):
            with pytest.raises(IncompatibleEvaluatorError, match="undiscounted episodic"):
                game.evaluate(policy, evaluator="matrix", output=destination)

        assert not output.exists()

    assert game.evaluator is None
    assert game.discount == discount


def test_named_evaluator_can_be_submitted_without_reconfiguring_the_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_trained_policy
) -> None:
    _, policy, _ = tiny_trained_policy
    monkeypatch.setenv("BENCHMAX_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BENCHMAX_RUNS", str(tmp_path / "runs"))
    game = load_game("rock_paper_scissors", evaluator=None)
    expected = game.evaluate(policy, evaluator="matrix")

    job = game.evaluate(policy, evaluator="matrix", submit=True)
    entry = job._prepare().game
    assert entry.overrides["evaluator"] == "matrix"
    entry.validate_resolution("game")
    metrics = job.run().result(timeout=30)

    assert metrics == expected
    assert game.evaluator is None


def test_synchronous_training_publishes_a_reusable_caller_owned_artifact(tmp_path: Path) -> None:
    game = load_game("rock_paper_scissors", evaluator=None)
    output = tmp_path / "training.json"
    settings = tiny_training_params()
    algorithm = load_algorithm("mappo", total_frames=8, device="cpu", verbose=False, params=settings)
    with pytest.warns(UserWarning, match="no evaluator is configured"):
        profile = train(game, algorithm, seed=7, output=output)

    game.evaluator = MatrixEvaluator()

    document = json.loads(output.read_text())
    manifest = Path(document["policy"]["uri"])
    spec = TrainingSpec.create(
        game="rock_paper_scissors", seed=7, total_frames=8, algorithm_overrides={"params": settings}
    )
    restored = load_policy(str(output), expected_game=spec.game)
    expected = game.evaluate(profile)

    assert document["kind"] == "training-result"
    assert manifest.parent.parent == tmp_path
    assert game.evaluate(restored) == expected
    assert game.evaluate(output) == expected


def test_arbitrary_callable_is_not_silently_serialized_for_a_job() -> None:
    game = load_game("rock_paper_scissors")

    def uniform(player, observations, action_space, legal_action_mask):
        del player, observations, action_space
        probabilities = legal_action_mask.astype(float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    job = game.evaluate(uniform, submit=True)

    with pytest.raises(JobError, match="provide a policy manifest"):
        job.audit()
