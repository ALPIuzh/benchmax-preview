import warnings

import numpy as np
import pytest

from benchmax import load_game
from benchmax.core import Game
from benchmax.evaluation import (
    DiscountedNashGapEvaluator,
    IncompatibleEvaluatorError,
    MatrixEvaluator,
    require_discounted,
    require_undiscounted,
)
from benchmax.execution import JobError
from benchmax.games.matrix import TwoPlayerZeroSumMatrixGame
from benchmax.games.tabular import Gridworld


class RecordingEvaluator:
    def __init__(self) -> None:
        self.validated: list[str] = []
        self.evaluations = 0

    def validate(self, game: Game) -> None:
        self.validated.append(game.name)

    def evaluate(self, game, policy_profile):
        self.evaluations += 1
        return {}


class RefusingEvaluator:
    def validate(self, game: Game) -> None:
        raise IncompatibleEvaluatorError("nope")

    def evaluate(self, game, policy_profile):
        return {}


def test_custom_evaluators_are_called_without_validate_introspection() -> None:
    evaluator = RecordingEvaluator()
    game = Gridworld(evaluator=evaluator, name="probe")
    game.evaluate(lambda p, o, a, m: m.astype(float) / m.sum(axis=1, keepdims=True))
    game.evaluate(lambda p, o, a, m: m.astype(float) / m.sum(axis=1, keepdims=True))

    assert evaluator.validated == []
    assert evaluator.evaluations == 2
    assert game.evaluator is evaluator


def test_mutating_a_compatibility_value_cannot_bypass_validation() -> None:
    game = load_game("rock_paper_scissors")
    assert isinstance(game.evaluator, MatrixEvaluator)
    game.discount = 0.9

    with pytest.raises(IncompatibleEvaluatorError, match="undiscounted episodic"):
        game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1])


def test_training_does_not_introspect_custom_evaluators() -> None:
    from benchmax.training import train

    game = load_game("gridworld_3x3")
    game.evaluator = RefusingEvaluator()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        job = train(game, submit=True)

    assert job.state == "created"


def test_a_missing_evaluator_warns_before_training() -> None:
    from benchmax.training import train

    game = load_game("gridworld_3x3", evaluator=None)
    assert game.evaluator is None

    with pytest.warns(UserWarning, match="training will proceed"):
        train(game, submit=True)

    with pytest.raises(NotImplementedError, match="no evaluator"):
        game.evaluate(lambda p, o, a, m: np.ones_like(m, dtype=float) / m.shape[1])


def test_a_directly_assigned_evaluator_is_synchronous_only() -> None:
    game = load_game("rock_paper_scissors", evaluator=None)
    game.evaluator = MatrixEvaluator()

    with pytest.raises(JobError, match="load the game with evaluator="):
        game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1], submit=True)


@pytest.mark.parametrize("evaluator", [MatrixEvaluator(), "matrix"], ids=["instance", "registered-name"])
def test_evaluate_accepts_an_evaluator_override_without_mutating_the_game(evaluator) -> None:
    game = load_game("rock_paper_scissors", evaluator=None)

    metrics = game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1], evaluator=evaluator)

    assert metrics["exploitability"] == pytest.approx(0.0)
    assert game.evaluator is None


def test_a_named_evaluator_is_validated_against_the_game() -> None:
    game = load_game("rock_paper_scissors", evaluator=None)

    with pytest.raises(IncompatibleEvaluatorError, match="tabular Markov games"):
        game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1], evaluator="discounted_nash_gap")


def test_a_named_evaluator_uses_the_callers_game_for_synchronous_evaluation() -> None:
    game = load_game("rock_paper_scissors", evaluator=None)
    game.discount = 0.9

    with pytest.raises(IncompatibleEvaluatorError, match="undiscounted episodic"):
        game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1], evaluator="matrix")


def test_a_named_evaluator_accepts_a_custom_game() -> None:
    game = TwoPlayerZeroSumMatrixGame(name="custom", payoffs=np.zeros((1, 1)))

    metrics = game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1], evaluator="matrix")

    assert metrics["exploitability"] == pytest.approx(0.0)


def test_a_one_off_evaluator_is_not_implicitly_serialized() -> None:
    game = load_game("rock_paper_scissors", evaluator=None)

    with pytest.raises(ValueError, match="only be used for synchronous"):
        game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1], evaluator=MatrixEvaluator(), submit=True)


def test_the_discounted_evaluator_refuses_an_undiscounted_game() -> None:
    class Undiscounted(Gridworld):
        def __init__(self) -> None:
            super().__init__(evaluator=None)
            self.discount = None

    game = Undiscounted()
    game.evaluator = DiscountedNashGapEvaluator()

    with pytest.raises(IncompatibleEvaluatorError, match="declares no discount"):
        game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1])


def test_require_helpers_state_which_convention_is_wrong() -> None:
    discounted = Gridworld(evaluator=DiscountedNashGapEvaluator())
    episodic = load_game("rock_paper_scissors")

    require_discounted(discounted, "discounted_nash_gap")
    require_undiscounted(episodic, "nash_conv")

    with pytest.raises(IncompatibleEvaluatorError, match="undiscounted episodic"):
        require_undiscounted(discounted, "nash_conv")

    with pytest.raises(IncompatibleEvaluatorError, match="no discount factor"):
        require_discounted(episodic, "discounted_nash_gap")


def test_custom_evaluators_only_need_evaluate() -> None:
    class Bare:
        def evaluate(self, game, policy_profile):  # pragma: no cover
            return {}

    game = load_game("rock_paper_scissors")
    game.evaluator = Bare()

    assert game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1]) == {}


@pytest.mark.parametrize(
    ("game", "evaluator", "requirement"),
    [
        ("gridworld_3x3", "nash_conv", "OpenSpielGame"),
        ("rock_paper_scissors", "discounted_nash_gap", "tabular Markov games"),
    ],
)
def test_incompatible_registered_evaluators_fail_when_evaluation_starts(
    game: str, evaluator: str, requirement: str
) -> None:
    configured = load_game(game, evaluator=evaluator)

    with pytest.raises(IncompatibleEvaluatorError, match=requirement):
        configured.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1])
