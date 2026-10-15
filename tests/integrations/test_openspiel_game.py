import numpy as np
import pytest

from benchmax import load_game
from benchmax.integrations.openspiel import OpenSpielGame
from benchmax.integrations.openspiel.evaluation import OpenSpielNashConvEvaluator


def _uniform(
    player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
) -> np.ndarray:
    del player, observations, action_space
    probabilities = legal_action_mask.astype(float)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def test_openspiel_game_is_concrete_once_an_evaluator_is_composed() -> None:
    game = OpenSpielGame(name="kuhn", spiel_name="kuhn_poker", evaluator=OpenSpielNashConvEvaluator())

    assert game.name == "kuhn"
    assert game.observation_spaces == {0: (11,), 1: (11,)}
    assert game.state_space == (22,)


def test_connect_four_uses_board_observations_through_a_complete_episode() -> None:
    game = load_game("openspiel.connect_four")
    env = game.new_env()
    time_step = env.reset(seed=7)
    reference = game.spiel_game.new_initial_state()
    tree = game.new_tree()

    assert game.observation_spaces == {0: (126,), 1: (126,)}
    assert game.state_space == (252,)

    for action in (0, 1, 0, 1, 0, 1, 0):
        player = reference.current_player()
        expected = tuple(reference.observation_tensor(player))
        assert time_step.players_to_act == (player,)
        assert time_step.observations == {player: expected}
        assert tree.observation(player) == expected
        assert time_step.legal_actions[player] == tuple(reference.legal_actions())
        assert env.state() == tuple(value for p in game.players for value in reference.observation_tensor(p))
        time_step = env.step({player: action})
        reference.apply_action(action)
        tree = tree.child({player: action})

    assert time_step.terminated
    assert time_step.observations == {}
    assert time_step.rewards == tree.returns() == {0: 1.0, 1: -1.0}
    assert len(env.state()) == game.state_space[0]


def test_partial_observability_does_not_fall_back_to_a_history_free_observation() -> None:
    with pytest.raises(ValueError, match="requires an information-state tensor"):
        OpenSpielGame(spiel_name="blackjack")


@pytest.mark.parametrize(
    "name,observable,simultaneous",
    [("connect_four", True, False), ("kuhn_poker", False, False),
     ("goofspiel", True, True), ("matrix_rps", True, True)],
)
def test_observability_and_dynamics_are_independent(name, observable, simultaneous) -> None:
    game = OpenSpielGame(spiel_name=name)
    assert game.is_fully_observable is observable
    assert game.is_simultaneous is simultaneous


def test_simultaneous_environment_validates_and_applies_the_whole_joint_action() -> None:
    game = load_game("openspiel.matrix_rps")
    env = game.new_env()
    initial = env.reset(seed=7)
    assert initial.players_to_act == (0, 1)
    assert initial.observations.keys() == {0, 1}

    with pytest.raises(ValueError, match="exactly players"):
        env.step({0: 0})

    with pytest.raises(ValueError, match="illegal action"):
        env.step({0: 0, 1: 3})

    result = env.step({1: 2, 0: 0})
    terminal = game.new_tree().child({1: 2, 0: 0})
    assert result.terminated
    assert result.players_to_act == ()
    assert result.rewards == terminal.returns() == {0: 1.0, 1: -1.0}


def test_observation_only_game_remains_playable_without_exact_tree_keys() -> None:
    game = load_game("openspiel.catch")
    initial = game.new_env().reset(seed=7)
    assert initial.players_to_act == (0,)
    assert len(initial.observations[0]) == game.observation_spaces[0][0]

    with pytest.raises(NotImplementedError, match="information-state strings"):
        game.new_tree()


def test_a_missing_evaluator_fails_only_when_evaluation_is_requested() -> None:
    game = OpenSpielGame(name="kuhn", spiel_name="kuhn_poker")

    with pytest.raises(NotImplementedError, match="no evaluator"):
        game.evaluate(_uniform)


def test_the_openspiel_name_defaults_to_the_benchmax_name() -> None:
    # a configuration whose BenchMAX name is already OpenSpiel's needs only one
    # of the two, so a loadable game can be named without repeating it
    game = OpenSpielGame(name="kuhn_poker")

    assert game.spiel_name == "kuhn_poker"
    assert game.name == "kuhn_poker"


def test_an_openspiel_game_needs_one_of_the_two_names() -> None:
    with pytest.raises(ValueError, match="spiel_name or a name"):
        OpenSpielGame(params={"board_size": 4})


def test_an_openspiel_environment_rejects_an_unseeded_first_reset() -> None:
    game = OpenSpielGame(name="kuhn", spiel_name="kuhn_poker", evaluator=OpenSpielNashConvEvaluator())

    with pytest.raises(RuntimeError, match="first reset requires an explicit seed"):
        game.new_env().reset()


def test_a_seeded_openspiel_stream_continues_across_resets() -> None:
    game = OpenSpielGame(name="kuhn", spiel_name="kuhn_poker", evaluator=OpenSpielNashConvEvaluator())
    first = game.new_env()
    second = game.new_env()

    first.reset(seed=7)
    second.reset(seed=7)
    next_first = first.reset()
    next_second = second.reset()

    for player in next_first.players_to_act:
        np.testing.assert_array_equal(next_first.observations[player], next_second.observations[player])
