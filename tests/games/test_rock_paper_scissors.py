from collections.abc import Sequence
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from benchmax import PolicyProfile, load_game
from benchmax.games import TwoPlayerZeroSumMatrixGame


def _rps() -> TwoPlayerZeroSumMatrixGame:
    game = load_game("rock_paper_scissors")
    assert isinstance(game, TwoPlayerZeroSumMatrixGame)
    return game


@pytest.mark.parametrize(
    ("first", "second", "payoff"), [(0, 0, 0.0), (1, 2, -1.0), (2, 1, 1.0)], ids=["draw", "loss", "win"]
)
def test_actions_produce_terminal_payoffs(first: int, second: int, payoff: float) -> None:
    env = _rps().new_env()
    env.reset()
    time_step = env.step({0: first, 1: second})
    assert time_step.observations == {}
    assert time_step.rewards == {0: payoff, 1: -payoff}
    assert time_step.players_to_act == ()
    assert time_step.legal_actions == {}
    assert time_step.terminated is True
    assert time_step.truncated is False
    assert time_step.info == {}


def test_reset_exposes_one_simultaneous_decision() -> None:
    time_step = _rps().new_env().reset()

    assert time_step.observations == {0: (0.0,), 1: (0.0,)}
    assert time_step.rewards == {0: 0.0, 1: 0.0}
    assert time_step.players_to_act == (0, 1)
    assert time_step.legal_actions == {0: (0, 1, 2), 1: (0, 1, 2)}
    assert time_step.terminated is False
    assert time_step.truncated is False
    assert time_step.info == {}

    with pytest.raises(FrozenInstanceError):
        time_step.terminated = True


def test_uniform_policy_profile_has_zero_exploitability() -> None:
    def uniform(player: int, observation: tuple[float, ...], legal_actions: Sequence[int]) -> dict[int, float]:
        del player, observation
        return {action: 1.0 / len(legal_actions) for action in legal_actions}

    game = _rps()
    results = game.evaluate(PolicyProfile.from_scalar_callable(uniform))

    assert results["player_improvements"] == pytest.approx({0: 0.0, 1: 0.0})
    assert results["nash_conv"] == pytest.approx(0.0)
    assert results["exploitability"] == pytest.approx(0.0)


def test_deterministic_policy_profile_is_exploitable() -> None:
    def always_rock(player: int, observation: tuple[float, ...], legal_actions: Sequence[int]) -> dict[int, float]:
        del player, observation
        return {action: float(action == 0) for action in legal_actions}

    game = _rps()
    results = game.evaluate(PolicyProfile.from_scalar_callable(always_rock))

    assert results["player_improvements"] == pytest.approx({0: 1.0, 1: 1.0})
    assert results["nash_conv"] == pytest.approx(2.0)
    assert results["exploitability"] == pytest.approx(1.0)


def test_equivalent_scalar_and_batched_profiles_produce_identical_results() -> None:
    def scalar(player: int, observation: tuple[float, ...], legal_actions: Sequence[int]) -> dict[int, float]:
        del observation
        weights = (1.0, 2.0, 3.0) if player == 0 else (3.0, 1.0, 2.0)
        total = sum(weights[action] for action in legal_actions)
        return {action: weights[action] / total for action in legal_actions}

    def batched(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        assert observations.shape == (1, 1)
        assert action_space == (0, 1, 2)
        weights = np.array((1.0, 2.0, 3.0) if player == 0 else (3.0, 1.0, 2.0))
        probabilities = legal_action_mask * weights
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    game = _rps()

    assert game.evaluate(batched) == game.evaluate(
        PolicyProfile.from_scalar_callable(scalar)
    )


def test_batched_policy_profile_probabilities_are_validated() -> None:
    def invalid_policy_profile(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        del player, observations, action_space
        return legal_action_mask.astype(float) * 0.3

    with pytest.raises(ValueError, match="must sum to one"):
        _rps().evaluate(invalid_policy_profile)
