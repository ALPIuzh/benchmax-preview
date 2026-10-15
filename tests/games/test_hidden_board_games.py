import pytest

from benchmax import load_game


@pytest.fixture(params=[
    ("dark_hex_3x3", 162, 0, (0, 1, 3, 2, 6)),
    ("abrupt_dark_hex_3x3", 162, 1, (0, 1, 3, 2, 6)),
    ("phantom_tic_tac_toe", 108, 0, (0, 3, 1, 4, 2)),
    ("abrupt_phantom_tic_tac_toe", 108, 1, (0, 3, 1, 4, 2)),
], ids=lambda case: case[0])
def variant(request):
    return request.param


def play_actions(env, actions):
    time_step = env.reset(seed=0)

    for action in actions:
        (player,) = time_step.players_to_act
        time_step = env.step({player: action})

    return time_step


def test_registry_games_reset(variant) -> None:
    name, observation_size, _, _ = variant
    game = load_game(name)
    time_step = game.new_env().reset(seed=0)

    assert game.players == (0, 1)
    assert game.observation_spaces == {0: (observation_size,), 1: (observation_size,)}
    assert game.action_spaces == {0: tuple(range(9)), 1: tuple(range(9))}
    assert time_step.players_to_act == (0,)
    assert time_step.legal_actions == {0: tuple(range(9))}
    assert len(time_step.observations[0]) == observation_size
    assert isinstance(time_step.observations[0], tuple)
    assert all(isinstance(value, float) for value in time_step.observations[0])


def test_collision_rule(variant) -> None:
    name, _, expected_player, _ = variant
    time_step = play_actions(load_game(name).new_env(), (0, 1, 1))

    assert time_step.players_to_act == (expected_player,)


def test_winning_sequence_terminates_with_win_rewards(variant) -> None:
    name, _, _, winning_actions = variant
    time_step = play_actions(load_game(name).new_env(), winning_actions)

    assert time_step.terminated is True
    assert time_step.truncated is False
    assert time_step.rewards == {0: 1.0, 1: -1.0}
