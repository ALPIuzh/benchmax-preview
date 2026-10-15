from collections.abc import Sequence
from typing import Any

import pyspiel
import pytest
from open_spiel.python import policy as spiel_policy
from open_spiel.python.algorithms import expected_game_score, exploitability

from benchmax import PolicyProfile, load_game
from benchmax.integrations.openspiel.evaluation import _BenchmaxToOpenSpielPolicyAdapter


def _uniform_policy(player: int, observation: tuple[float, ...], legal_actions: Sequence[int]) -> dict[int, float]:
    del player, observation
    probability = 1.0 / len(legal_actions)
    return {action: probability for action in legal_actions}


def test_information_state_does_not_expose_opponents_card() -> None:
    root = load_game("kuhn_poker").new_tree()

    jack_against_queen = root.chance_child(0).chance_child(1)
    jack_against_king = root.chance_child(0).chance_child(2)
    queen_against_jack = root.chance_child(1).chance_child(0)

    assert jack_against_queen.observation(0) == jack_against_king.observation(0)
    assert jack_against_queen.observation(0) != queen_against_jack.observation(0)
    assert jack_against_queen.information_state_key(0) == jack_against_king.information_state_key(0)


@pytest.mark.parametrize(
    "name,params",
    [("kuhn_poker", {}), ("leduc_poker", {}), ("connect_four", {"rows": 2, "columns": 3, "x_in_row": 3})],
)
def test_exact_evaluation_matches_openspiel(name, params) -> None:
    game = load_game(f"openspiel.{name}", params=params, evaluator="nash_conv")
    result = game.evaluate(PolicyProfile.from_scalar_callable(_uniform_policy))
    reference = spiel_policy.UniformRandomPolicy(game.spiel_game)
    expected = exploitability.nash_conv(game.spiel_game, reference, return_only_nash_conv=False, use_cpp_br=False)
    returns = expected_game_score.policy_value(game.spiel_game.new_initial_state(), [reference, reference])

    assert result["nash_conv"] == pytest.approx(expected.nash_conv)
    assert result["player_improvements"] == pytest.approx(dict(enumerate(expected.player_improvements)))
    assert result["expected_returns"] == pytest.approx(dict(enumerate(returns)))


def test_adapter_encodes_the_acting_players_observation() -> None:
    game = pyspiel.load_game("kuhn_poker")
    state = game.new_initial_state().child(0).child(1)
    encoded: list[tuple[Any, int]] = []

    def encode_observation(state_to_encode: Any, player: int) -> tuple[float, ...]:
        encoded.append((state_to_encode, player))
        return (42.0,)

    queried: list[tuple[int, tuple[float, ...], tuple[int, ...]]] = []

    def policy_profile(player: int, observation: tuple[float, ...], legal_actions: Sequence[int]) -> dict[int, float]:
        queried.append((player, observation, tuple(legal_actions)))
        return _uniform_policy(player, observation, legal_actions)

    adapter = _BenchmaxToOpenSpielPolicyAdapter(game, policy_profile, encode_observation)

    assert adapter.action_probabilities(state) == {0: 0.5, 1: 0.5}
    assert encoded == [(state, 0)]
    assert queried == [(0, (42.0,), (0, 1))]


def test_adapter_requires_a_player_at_simultaneous_states() -> None:
    game = pyspiel.load_game("matrix_rps")
    state = game.new_initial_state()
    adapter = _BenchmaxToOpenSpielPolicyAdapter(game, _uniform_policy, lambda state, player: (float(player),))

    assert state.is_simultaneous_node()

    with pytest.raises(ValueError, match="player_id is required.*simultaneous node"):
        adapter.action_probabilities(state)


def test_adapter_queries_each_player_at_simultaneous_states() -> None:
    game = pyspiel.load_game("matrix_rps")
    state = game.new_initial_state()
    encoded: list[tuple[Any, int]] = []
    queried: list[tuple[int, tuple[float, ...], tuple[int, ...]]] = []

    def encode_observation(state_to_encode: Any, player: int) -> tuple[float, ...]:
        encoded.append((state_to_encode, player))
        return (float(player),)

    def player_specific_policy(
        player: int, observation: tuple[float, ...], legal_actions: Sequence[int]
    ) -> dict[int, float]:
        queried.append((player, observation, tuple(legal_actions)))
        return {action: float(action == player) for action in legal_actions}

    adapter = _BenchmaxToOpenSpielPolicyAdapter(game, player_specific_policy, encode_observation)

    assert adapter.action_probabilities(state, player_id=0) == {0: 1.0, 1: 0.0, 2: 0.0}
    assert adapter.action_probabilities(state, player_id=1) == {0: 0.0, 1: 1.0, 2: 0.0}
    assert encoded == [(state, 0), (state, 1)]
    assert queried == [(0, (0.0,), (0, 1, 2)), (1, (1.0,), (0, 1, 2))]


def test_adapter_rejects_simultaneous_player_identifier() -> None:
    game = pyspiel.load_game("matrix_rps")
    state = game.new_initial_state()
    queried = False

    def policy_profile(player: int, observation: tuple[float, ...], legal_actions: Sequence[int]) -> dict[int, float]:
        nonlocal queried
        queried = True
        return _uniform_policy(player, observation, legal_actions)

    adapter = _BenchmaxToOpenSpielPolicyAdapter(game, policy_profile, lambda state, player: (float(player),))

    with pytest.raises(ValueError, match="invalid player_id -2"):
        adapter.action_probabilities(state, player_id=int(pyspiel.PlayerId.SIMULTANEOUS))

    assert not queried


def test_adapter_returns_empty_distribution_for_terminal_states() -> None:
    game = pyspiel.load_game("kuhn_poker")
    terminal_state = game.new_initial_state().child(0).child(1).child(0).child(0)

    def fail_if_encoded(state: Any, player: int) -> tuple[float, ...]:
        del state, player
        raise AssertionError("terminal states must not be encoded")

    adapter = _BenchmaxToOpenSpielPolicyAdapter(game, _uniform_policy, fail_if_encoded)

    assert terminal_state.is_terminal()
    assert adapter.action_probabilities(terminal_state) == {}


def test_adapter_rejects_chance_nodes_and_nonacting_players() -> None:
    game = pyspiel.load_game("kuhn_poker")
    adapter = _BenchmaxToOpenSpielPolicyAdapter(game, _uniform_policy, lambda state, player: (float(player),))

    with pytest.raises(ValueError, match="chance node"):
        adapter.action_probabilities(game.new_initial_state())

    player_zero_state = game.new_initial_state().child(0).child(1)

    with pytest.raises(ValueError, match="player 1 while player 0 acts"):
        adapter.action_probabilities(player_zero_state, player_id=1)


def test_adapter_validates_policy_profile_distribution() -> None:
    game = pyspiel.load_game("kuhn_poker")
    state = game.new_initial_state().child(0).child(1)

    def incomplete_policy(
        player: int, observation: tuple[float, ...], legal_actions: Sequence[int]
    ) -> dict[int, float]:
        del player, observation
        return {legal_actions[0]: 1.0}

    adapter = _BenchmaxToOpenSpielPolicyAdapter(game, incomplete_policy, lambda state, player: (float(player),))

    with pytest.raises(ValueError, match="omitted legal actions"):
        adapter.action_probabilities(state)
