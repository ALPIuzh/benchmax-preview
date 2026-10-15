import pickle

import numpy as np
import pytest

from benchmax import IncompatibleEvaluatorError, load_game
from benchmax.config import FixedParameterError
from benchmax.evaluation import MatrixEvaluator
from benchmax.games.matrix import MatrixGame, TwoPlayerZeroSumMatrixGame


def test_general_sum_matrix_game_needs_only_payoff_tensors_for_dynamics() -> None:
    game = MatrixGame(
        name="coordination", rewards={0: np.array([[2.0, 0.0], [0.0, 1.0]]), 1: np.array([[1.0, 0.0], [0.0, 2.0]])}
    )

    env = game.new_env()
    time_step = env.reset()
    assert time_step.players_to_act == (0, 1)
    assert time_step.legal_actions == {0: (0, 1), 1: (0, 1)}

    time_step = env.step({0: 1, 1: 1})
    assert time_step.rewards == {0: 1.0, 1: 2.0}
    assert time_step.terminated is True


def test_payoff_axes_follow_sorted_player_ids() -> None:
    game = MatrixGame(name="nonconsecutive_players", rewards={7: np.zeros((2, 3)), 2: np.zeros((2, 3))})

    assert game.players == (2, 7)
    assert game.action_spaces == {2: (0, 1), 7: (0, 1, 2)}
    assert game.is_fully_observable is True
    assert game.is_simultaneous is True


def test_matrix_game_exposes_its_simultaneous_decision_as_an_exact_tree() -> None:
    game = MatrixGame(
        name="coordination", rewards={2: np.array([[2.0, 0.0], [0.0, 1.0]]), 7: np.array([[1.0, 0.0], [0.0, 2.0]])},
        action_spaces={2: (3, 5), 7: (11, 13)}
    )

    root = game.new_tree()
    assert root.returns() is None
    assert root.chance_outcomes() == ()
    assert root.players_to_act() == (2, 7)
    assert root.observation(2) == (0.0,)
    assert root.legal_actions(7) == (11, 13)

    terminal = root.child({2: 5, 7: 13})
    assert terminal.players_to_act() == ()
    assert terminal.returns() == {2: 1.0, 7: 2.0}


def test_one_matrix_evaluator_serves_another_zero_sum_matrix_game() -> None:
    game = TwoPlayerZeroSumMatrixGame(
        name="matching_pennies", payoffs=np.array([[1.0, -1.0], [-1.0, 1.0]]), evaluator=MatrixEvaluator()
    )

    def uniform(player, observations, action_space, legal_action_mask):
        del player, observations, action_space
        return legal_action_mask.astype(float) / 2.0

    def first_action(player, observations, action_space, legal_action_mask):
        del player, observations, action_space
        probabilities = np.zeros_like(legal_action_mask, dtype=float)
        probabilities[:, 0] = 1.0
        return probabilities

    assert game.evaluate(uniform)["exploitability"] == pytest.approx(0.0)
    assert game.evaluate(first_action)["exploitability"] == pytest.approx(1.0)


def test_matrix_evaluator_reports_general_sum_deviation_metrics() -> None:
    game = MatrixGame(
        name="coordination", rewards={0: np.array([[2.0, 0.0], [0.0, 1.0]]), 1: np.array([[1.0, 0.0], [0.0, 2.0]])},
        evaluator=MatrixEvaluator()
    )

    metrics = game.evaluate(lambda player, observations, actions, mask: mask.astype(float) / 2.0)

    assert metrics["expected_returns"] == pytest.approx({0: 0.75, 1: 0.75})
    assert metrics["player_improvements"] == pytest.approx({0: 0.25, 1: 0.25})
    assert metrics["nash_gap"] == pytest.approx(0.25)
    assert metrics["nash_conv"] == pytest.approx(0.5)
    assert "exploitability" not in metrics


def test_matrix_evaluator_supports_more_than_two_players() -> None:
    shape = (2, 2, 2)
    game = MatrixGame(
        name="three_player_dominance",
        rewards={
            0: np.broadcast_to(np.array([0.0, 1.0])[:, None, None], shape),
            1: np.broadcast_to(np.array([0.0, 1.0])[None, :, None], shape),
            2: np.broadcast_to(np.array([0.0, 1.0])[None, None, :], shape),
        },
        evaluator=MatrixEvaluator(),
    )

    metrics = game.evaluate(lambda player, observations, actions, mask: mask.astype(float) / 2.0)

    assert metrics["expected_returns"] == pytest.approx({0: 0.5, 1: 0.5, 2: 0.5})
    assert metrics["player_improvements"] == pytest.approx({0: 0.5, 1: 0.5, 2: 0.5})
    assert metrics["nash_gap"] == pytest.approx(0.5)
    assert metrics["nash_conv"] == pytest.approx(1.5)
    assert "exploitability" not in metrics


def test_matrix_payoffs_are_immutable_copies() -> None:
    source = np.array([[1.0, -1.0], [-1.0, 1.0]])
    game = TwoPlayerZeroSumMatrixGame(name="matching_pennies", payoffs=source)

    source[0, 0] = 99.0
    assert game.payoffs[0, 0] == 1.0
    assert np.array_equal(game.rewards[1], -game.rewards[0])

    with pytest.raises(ValueError, match="read-only"):
        game.payoffs[0, 0] = 99.0

    with pytest.raises(ValueError):
        game.payoffs.setflags(write=True)

    detached_rewards = game.rewards
    detached_rewards[0] = np.zeros((2, 2))  # type: ignore[index]
    assert game.payoffs[0, 0] == 1.0

    with pytest.raises(AttributeError):
        game.payoffs = np.zeros((2, 2))  # type: ignore[misc]


def test_pickled_matrix_game_restores_immutable_payoffs() -> None:
    game = TwoPlayerZeroSumMatrixGame(name="matching_pennies", payoffs=np.array([[1.0, -1.0], [-1.0, 1.0]]))

    restored = pickle.loads(pickle.dumps(game))

    assert np.array_equal(restored.rewards[1], -restored.payoffs)

    with pytest.raises(ValueError, match="read-only"):
        restored.payoffs[0, 0] = 99.0

    with pytest.raises(ValueError):
        restored.payoffs.setflags(write=True)


def test_matrix_evaluator_rejects_a_non_matrix_game() -> None:
    from benchmax.games.tabular import Gridworld

    game = Gridworld(evaluator=MatrixEvaluator())

    with pytest.raises(IncompatibleEvaluatorError, match="finite matrix game"):
        game.evaluate(lambda player, observations, actions, mask: mask.astype(float) / mask.sum(axis=1, keepdims=True))


def test_matrix_shapes_define_one_action_axis_per_player() -> None:
    with pytest.raises(ValueError, match="one action axis per player"):
        MatrixGame(name="malformed", rewards={0: np.zeros((2, 2)), 1: np.zeros((2, 2)), 2: np.zeros((2, 2))})


def test_direct_matrix_games_do_not_implicitly_compose_an_evaluator() -> None:
    assert MatrixGame(name="matrix", rewards={0: np.zeros((1,))}).evaluator is None
    assert TwoPlayerZeroSumMatrixGame(name="matrix", payoffs=np.zeros((1, 1))).evaluator is None


def test_registered_matrix_games_inherit_the_matrix_evaluator_configuration() -> None:
    game = load_game("rock_paper_scissors")

    assert isinstance(game, TwoPlayerZeroSumMatrixGame)
    assert np.array_equal(game.payoffs, [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    assert isinstance(game.evaluator, MatrixEvaluator)


def test_registered_matrix_game_payoffs_cannot_be_overridden() -> None:
    with pytest.raises(FixedParameterError, match="payoffs"):
        load_game("rock_paper_scissors", payoffs=[[0.0]])
