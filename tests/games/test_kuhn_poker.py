from collections.abc import Sequence
from typing import Any

import numpy as np
import pyspiel
import pytest
from open_spiel.python.algorithms import cfr

from benchmax import PolicyProfile, load_game


def test_episode_handles_chance_and_reaches_termination() -> None:
    env = load_game("kuhn_poker").new_env()
    time_step = env.reset(seed=7)

    assert time_step.players_to_act == (0,)
    assert time_step.observations.keys() == {0}
    assert time_step.legal_actions == {0: (0, 1)}
    assert len(time_step.observations[0]) == 11
    assert time_step.rewards == {0: 0.0, 1: 0.0}
    assert time_step.terminated is False
    assert time_step.truncated is False
    assert time_step.info == {}

    steps = 0

    while not time_step.terminated:
        (player,) = time_step.players_to_act
        assert time_step.observations.keys() == {player}
        assert time_step.legal_actions.keys() == {player}
        legal_actions = time_step.legal_actions[player]
        assert legal_actions == (0, 1)
        time_step = env.step({player: legal_actions[0]})
        assert time_step.rewards.keys() == {0, 1}
        assert not time_step.truncated
        assert time_step.info == {}
        steps += 1
        assert steps <= 3

    assert time_step.observations == {}
    assert time_step.players_to_act == ()
    assert time_step.legal_actions == {}
    assert time_step.rewards[0] == -time_step.rewards[1]


def test_nonterminal_step_describes_the_next_sequential_player() -> None:
    env = load_game("kuhn_poker").new_env()
    time_step = env.reset(seed=7)

    next_time_step = env.step({0: time_step.legal_actions[0][0]})

    assert next_time_step.players_to_act == (1,)
    assert next_time_step.observations.keys() == {1}
    assert next_time_step.legal_actions == {1: (0, 1)}
    assert next_time_step.rewards == {0: 0.0, 1: 0.0}
    assert next_time_step.terminated is False
    assert next_time_step.truncated is False
    assert next_time_step.info == {}


def test_openspiel_equilibrium_policy_profile_has_low_exploitability() -> None:
    spiel_game = pyspiel.load_game("kuhn_poker")
    solver = cfr.CFRSolver(spiel_game)

    for _ in range(1_000):
        solver.evaluate_and_update_policy()

    average_policy = solver.average_policy()

    table: dict[tuple[int, tuple[float, ...]], dict[int, float]] = {}

    def collect(state: Any) -> None:
        if state.is_terminal():
            return

        if state.is_chance_node():
            actions = [action for action, _ in state.chance_outcomes()]
        else:
            player = int(state.current_player())
            observation = tuple(state.information_state_tensor(player))
            distribution = {
                int(action): float(probability)
                for action, probability in average_policy.action_probabilities(
                    state
                ).items()
            }
            previous = table.setdefault((player, observation), distribution)
            assert previous == pytest.approx(distribution)
            actions = state.legal_actions(player)

        for action in actions:
            collect(state.child(action))

    collect(spiel_game.new_initial_state())

    def equilibrium_policy_profile(
        player: int, observation: tuple[float, ...], legal_actions: Sequence[int]
    ) -> dict[int, float]:
        del legal_actions
        return table[(player, observation)]

    results = load_game("kuhn_poker").evaluate(
        PolicyProfile.from_scalar_callable(equilibrium_policy_profile)
    )

    assert results["nash_conv"] < 0.004
    assert results["exploitability"] < 0.002


def test_kuhn_evaluation_validates_policy_profile_probabilities() -> None:
    def invalid_batched_policy(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        del player, observations, action_space
        return np.zeros_like(legal_action_mask, dtype=float)

    with pytest.raises(ValueError, match="must sum to one"):
        load_game("kuhn_poker").evaluate(invalid_batched_policy)


def test_equivalent_scalar_and_batched_profiles_produce_identical_results() -> None:
    batch_sizes: dict[int, int] = {}

    def scalar(player: int, observation: tuple[float, ...], legal_actions: Sequence[int]) -> dict[int, float]:
        del player, observation
        return {action: 1.0 / len(legal_actions) for action in legal_actions}

    def batched(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        assert observations.shape[1:] == (11,)
        assert action_space == (0, 1)
        batch_sizes[player] = observations.shape[0]
        probabilities = legal_action_mask.astype(float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    game = load_game("kuhn_poker")

    assert game.evaluate(batched) == game.evaluate(
        PolicyProfile.from_scalar_callable(scalar)
    )
    assert batch_sizes.keys() == {0, 1}
    assert all(batch_size > 1 for batch_size in batch_sizes.values())
