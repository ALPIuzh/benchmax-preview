import math

import pytest

from benchmax import list_available_games, load_game

_STATE_SIZES = {
    "connect_four": 252,
    "abrupt_dark_hex_3x3": 324,
    "abrupt_phantom_tic_tac_toe": 216,
    "dark_hex_3x3": 324,
    "gridworld_3x3": 72,
    "kuhn_poker": 22,
    "liars_dice_2d5f": 66,
    "phantom_tic_tac_toe": 216,
    "rock_paper_scissors": 7,
}


@pytest.mark.parametrize(("game_name", "state_size"), _STATE_SIZES.items())
def test_every_game_has_independent_environments_with_fixed_global_state(game_name: str, state_size: int) -> None:
    game = load_game(game_name)
    env = game.new_env()
    other_game = load_game(game_name)
    sibling, other_env = game.new_env(), other_game.new_env()

    assert other_game is not game
    assert env is not sibling
    assert game.state_space == (state_size,)

    with pytest.raises(RuntimeError, match="reset must be called before state"):
        env.state()

    time_step = env.reset(seed=7)
    initial_state = env.state()
    assert sibling.reset(seed=7) == other_env.reset(seed=7) == time_step
    assert sibling.state() == other_env.state() == initial_state

    assert isinstance(initial_state, tuple)
    assert len(initial_state) == state_size
    assert all(isinstance(value, float) for value in initial_state)

    actions = {
        player: time_step.legal_actions[player][0]
        for player in time_step.players_to_act
    }
    time_step = env.step(actions)
    assert env.state() != initial_state
    assert len(env.state()) == state_size
    assert sibling.state() == other_env.state() == initial_state
    assert sibling.step(actions) == other_env.step(actions) == time_step

    # A game with a discounted objective has no terminal state, so its rollout
    # is bounded here rather than by the game.
    steps = 1

    while not time_step.terminated and steps < 100:
        actions = {
            player: time_step.legal_actions[player][0]
            for player in time_step.players_to_act
        }
        time_step = env.step(actions)
        steps += 1

    if game.discount is None:
        assert time_step.terminated, f"{game_name} did not terminate within 100 steps"

    assert len(env.state()) == state_size


def test_state_cases_cover_the_complete_registry() -> None:
    listed = {name for name in list_available_games() if "." not in name}
    assert set(_STATE_SIZES) == listed


def test_openspiel_global_state_contains_each_acting_players_observation() -> None:
    game = load_game("kuhn_poker")
    env = game.new_env()
    time_step = env.reset(seed=7)

    for _ in range(2):
        state = env.state()
        offsets = {}
        offset = 0

        for player in game.players:
            offsets[player] = offset
            offset += math.prod(game.observation_spaces[player])

        for player, observation in time_step.observations.items():
            start = offsets[player]
            stop = start + len(observation)
            assert state[start:stop] == observation

        if time_step.terminated:
            break

        actions = {
            player: time_step.legal_actions[player][0]
            for player in time_step.players_to_act
        }
        time_step = env.step(actions)
