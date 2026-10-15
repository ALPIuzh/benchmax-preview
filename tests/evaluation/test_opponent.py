import numpy as np
import pytest

from benchmax import load_game
from benchmax.evaluation import IncompatibleEvaluatorError, OpponentEvaluator


class LowestAction:
    name = "lowest"

    def __init__(self, *, is_perfect=False, value=None):
        self.is_perfect, self.value = is_perfect, value
        self.opened = self.closed = 0

    def open(self, game):
        self.opened += 1
        return self

    def act(self, envs, rngs):
        return [min(env.spiel_state().legal_actions()) for env in envs]

    def game_value(self):
        return self.value

    def close(self):
        self.closed += 1


def lowest(player, observations, actions, mask):
    probabilities = np.zeros(mask.shape)
    probabilities[np.arange(len(mask)), mask.argmax(axis=1)] = 1
    return probabilities


def uniform(player, observations, actions, mask):
    return mask / mask.sum(axis=1, keepdims=True)


def test_matches_are_per_seat_and_close_the_session():
    opponent = LowestAction()
    result = OpponentEvaluator(opponent=opponent, episodes=3).evaluate(load_game("openspiel.tic_tac_toe"), lowest)
    assert result["expected_returns"] == {0: 1, 1: -1}
    assert result["win_rates"] == {0: 1, 1: 0}
    assert result["draw_rates"] == {0: 0, 1: 0}
    assert result["loss_rates"] == {0: 0, 1: 1}
    assert result["return_standard_errors"] == {0: 0, 1: 0}
    assert "solver_exploitability" not in result
    assert (opponent.opened, opponent.closed) == (1, 1)


@pytest.mark.parametrize("perfect,value,has_bound", [(False, {0: 0, 1: 0}, False), (True, None, False),
                                                     (True, {0: 0, 1: 0}, True)])
def test_bounds_require_both_guarantees(perfect, value, has_bound):
    # Test double exercises the metric contract, not a claim about LowestAction.
    opponent = LowestAction(is_perfect=perfect, value=value)
    result = OpponentEvaluator(opponent=opponent, seats=[1], episodes=3).evaluate(
        load_game("openspiel.tic_tac_toe"), lowest
    )
    assert ("solver_exploitability" in result) is has_bound
    if has_bound:
        assert result["solver_exploitability"] == 0.5
        assert result["solver_exploitability_standard_error"] == 0


def test_seed_is_independent_of_batch_size():
    game = load_game("openspiel.tic_tac_toe")
    def run(seed, batch_size):
        evaluator = OpponentEvaluator(opponent=LowestAction(), episodes=20, seed=seed, batch_size=batch_size)
        return evaluator.evaluate(game, uniform)
    assert run(1, 1) == run(1, 16)
    assert run(1, 1) != run(2, 1)


def test_policy_failure_releases_opponent():
    opponent = LowestAction()
    def broken(*args):
        raise RuntimeError("policy failed")
    with pytest.raises(RuntimeError, match="policy failed"):
        OpponentEvaluator(opponent=opponent, episodes=2).evaluate(load_game("openspiel.tic_tac_toe"), broken)
    assert opponent.closed == 1


@pytest.mark.parametrize("settings", [{"seats": [9]}, {"seats": [0, 0]}, {"episodes": 1}, {"batch_size": 0}])
def test_invalid_settings_do_not_open_opponent(settings):
    opponent = LowestAction()
    with pytest.raises(ValueError):
        OpponentEvaluator(opponent=opponent, **settings).evaluate(load_game("openspiel.tic_tac_toe"), uniform)
    assert opponent.opened == 0


def test_incompatible_game_does_not_open_opponent():
    opponent = LowestAction()
    with pytest.raises(IncompatibleEvaluatorError, match="sequential two-player"):
        OpponentEvaluator(opponent=opponent).evaluate(load_game("rock_paper_scissors"), uniform)
    assert opponent.opened == 0


@pytest.mark.parametrize("value", [{0: 0}, {0: float("nan"), 1: 0}, {0: 1, 1: 1}])
def test_invalid_exact_values_are_rejected_and_session_closed(value):
    opponent = LowestAction(is_perfect=True, value=value)
    with pytest.raises(ValueError):
        OpponentEvaluator(opponent=opponent).evaluate(load_game("openspiel.tic_tac_toe"), uniform)
    assert opponent.closed == 1
