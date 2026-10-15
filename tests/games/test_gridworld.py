import warnings

import numpy as np
import pytest

from benchmax import load_game
from benchmax.evaluation import DiscountedNashGapEvaluator
from benchmax.games.tabular import Gridworld, TabularMarkovGame

GAMMA = 0.9


@pytest.fixture
def game() -> Gridworld:
    return load_game("gridworld_3x3")


def gridworld(**params) -> Gridworld:
    """A gridworld built straight from parameters, without a configuration."""
    return Gridworld(params=params, evaluator=DiscountedNashGapEvaluator())


def uniform_profile(player, observations, action_space, legal_action_mask):
    probabilities = legal_action_mask.astype(float)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def tabular_profile(game: Gridworld, policies: dict[int, np.ndarray]):
    def profile(player, observations, action_space, legal_action_mask):
        return policies[player][game.state_indices(observations)]

    return profile


def test_declares_the_expected_spaces(game: Gridworld) -> None:
    assert game.players == (0, 1)
    assert game.num_states == 72
    assert game.action_spaces == {0: (0, 1, 2, 3), 1: (0, 1, 2, 3)}
    assert game.observation_spaces == {0: (72,), 1: (72,)}
    assert game.discount == GAMMA
    assert game.is_fully_observable is True
    assert game.is_simultaneous is True


def test_episodes_never_end(game: Gridworld) -> None:
    env = game.new_env()
    time_step = env.reset(seed=0)

    for _ in range(50):
        assert not time_step.terminated
        assert not time_step.truncated
        assert time_step.players_to_act == (0, 1)
        time_step = env.step({0: 1, 1: 1})

    assert not time_step.terminated


def test_starts_where_the_configuration_says(game: Gridworld) -> None:
    assert game.goal_cell == 2
    start = int(np.argmax(game.initial_state_distribution))
    assert game.states[start] == (1 * 3 + 0, 2 * 3 + 1)

    env = game.new_env()
    observation = env.reset(seed=0).observations[0]
    assert game.state_indices(np.asarray([observation]))[0] == start


def test_state_indices_round_trip(game: Gridworld) -> None:
    sources = {name: getattr(game, name).copy() for name in (
        "transitions", "initial_state_distribution", "observation_table"
    )}
    rewards = {player: reward.copy() for player, reward in game.rewards.items()}
    copied = TabularMarkovGame(name="copy", rewards=rewards, discount=GAMMA, **sources)

    for name, source in sources.items():
        source.fill(99.0)
        np.testing.assert_array_equal(getattr(copied, name), getattr(game, name))
        assert getattr(copied, name).flags.writeable

    for player, source in rewards.items():
        source.fill(99.0)
        np.testing.assert_array_equal(copied.rewards[player], game.rewards[player])

    indices = copied.state_indices(copied.observation_table)
    assert np.array_equal(indices, np.arange(game.num_states))


def test_state_indices_rejects_foreign_observations(game: Gridworld) -> None:
    with pytest.raises(KeyError, match="does not belong to"):
        game.state_indices(np.full((1, 72), 0.5))


def test_a_wall_keeps_an_agent_in_place(game: Gridworld) -> None:
    state = game.states.index((0, 4))
    successor = int(np.argmax(game.transitions[state, 2, 0]))
    assert game.states[successor][0] == 0


def test_the_goal_absorbs(game: Gridworld) -> None:
    state = next(s for s, (first, _) in enumerate(game.states) if first == 2)

    for action in range(4):
        successor = int(np.argmax(game.transitions[state, action, 0]))
        assert game.states[successor][0] == 2


@pytest.mark.parametrize("positions", [(3, 5), (3, 4)], ids=["collision", "swap"])
def test_agents_that_would_collide_or_swap_both_stay(game: Gridworld, positions) -> None:
    state = game.states.index(positions)
    successor = int(np.argmax(game.transitions[state, 1, 0]))
    assert game.states[successor] == positions


def test_payoffs_are_zero_sum_and_paid_at_the_goal(game: Gridworld) -> None:
    reaching = [
        (state, action_0, action_1)
        for state in range(game.num_states)
        for action_0 in range(4)
        for action_1 in range(4)
        if game.states[int(np.argmax(game.transitions[state, action_0, action_1]))][0]
        == game.goal_cell
    ]
    assert reaching
    assert all(game.payoffs[key] == 1.0 for key in reaching)
    assert np.array_equal(game.rewards[1], -game.rewards[0])


def test_a_best_responding_player_cannot_improve(game: Gridworld) -> None:
    rng = np.random.default_rng(1)
    policy_0 = rng.random((72, 4))
    policy_0 /= policy_0.sum(axis=1, keepdims=True)

    rewards = -(game.payoffs * policy_0[:, :, None]).sum(axis=1)
    transitions = (game.transitions * policy_0[:, :, None, None]).sum(axis=1)
    values = np.zeros(72)

    for _ in range(10_000):
        updated = np.max(rewards + GAMMA * transitions.dot(values), axis=1)

        if np.max(np.abs(updated - values)) < 1e-12:
            values = updated
            break

        values = updated

    policy_1 = np.zeros((72, 4))
    policy_1[
        np.arange(72), np.argmax(rewards + GAMMA * transitions.dot(values), axis=1)
    ] = 1.0

    result = game.evaluate(tabular_profile(game, {0: policy_0, 1: policy_1}))
    assert result["player_improvements"][1] == pytest.approx(0.0, abs=1e-5)


def test_improvements_are_never_negative(game: Gridworld) -> None:
    rng = np.random.default_rng(0)

    for _ in range(5):
        policies = {}

        for player in (0, 1):
            policy = rng.random((72, 4))
            policies[player] = policy / policy.sum(axis=1, keepdims=True)

        result = game.evaluate(tabular_profile(game, policies))
        assert min(result["player_improvements"].values()) >= -1e-6
        assert result["nash_gap"] == max(result["player_improvements"].values())
        assert result["nash_conv"] == pytest.approx(
            sum(result["player_improvements"].values())
        )


def test_uniform_play_is_exploitable(game: Gridworld) -> None:
    result = game.evaluate(uniform_profile)
    assert result["nash_gap"] > 1.0


def test_rejects_a_start_off_the_grid() -> None:
    with pytest.raises(ValueError, match="off the grid"):
        gridworld(goal=(0, 2), initial_state=((0, 0), (9, 9)))


def test_rejects_both_agents_starting_on_one_cell() -> None:
    with pytest.raises(ValueError, match="both agents on one cell"):
        gridworld(goal=(0, 2), initial_state=((1, 1), (1, 1)))


def test_rejects_a_parameter_the_game_does_not_have() -> None:
    with pytest.raises(ValueError, match="not a gridworld parameter"):
        gridworld(rowz=4)


def test_a_stated_layout_ignores_the_seed() -> None:
    layouts = {
        gridworld(
            goal=(0, 2), initial_state=((1, 0), (2, 1)), seed=seed
        ).goal_cell
        for seed in (0, 1, 2, None)
    }
    assert layouts == {2}


@pytest.mark.parametrize("params", [{}, {"seed": 7}], ids=["default-seed", "explicit-seed"])
def test_seeded_layouts_are_reproducible(params) -> None:
    first, second = gridworld(**params), gridworld(**params)
    assert 0 <= first.goal_cell < 9
    assert first.goal_cell == second.goal_cell
    assert np.array_equal(first.initial_state_distribution, second.initial_state_distribution)


def test_drawing_from_fresh_entropy_must_be_asked_for_and_warns() -> None:
    with pytest.warns(UserWarning, match="seed=None"):
        gridworld(seed=None)


def test_different_seeds_draw_different_layouts() -> None:
    layouts = {
        (game.goal_cell, int(np.argmax(game.initial_state_distribution)))
        for game in (gridworld(seed=seed) for seed in range(8))
    }
    assert len(layouts) > 1


def test_a_seeded_layout_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        gridworld(seed=3)
        gridworld()


def test_a_given_layout_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        gridworld(goal=(0, 2), initial_state=((1, 0), (2, 1)))


def test_a_drawn_layout_is_still_a_valid_game() -> None:
    drawn = gridworld(seed=11)
    assert np.allclose(drawn.transitions.sum(axis=-1), 1.0)
    assert np.isclose(drawn.initial_state_distribution.sum(), 1.0)
    assert drawn.evaluate(uniform_profile)["nash_gap"] >= 0.0


def test_params_record_a_drawn_layout_well_enough_to_ask_for_it_again() -> None:
    drawn = gridworld(seed=11)
    rebuilt = gridworld(**drawn.params)
    assert rebuilt.goal_cell == drawn.goal_cell
    assert np.array_equal(
        rebuilt.initial_state_distribution, drawn.initial_state_distribution
    )


def test_params_record_a_stated_layout_over_the_drawn_one() -> None:
    partly_drawn = gridworld(goal=(0, 2), seed=11)
    assert partly_drawn.params["goal"] == (0, 2)
    assert (
        partly_drawn.params["initial_state"]
        == gridworld(seed=11).params["initial_state"]
    )


def test_params_record_only_what_the_instance_needed() -> None:
    assert gridworld(goal=(0, 2), initial_state=((1, 0), (2, 1))).params == {
        "goal": (0, 2),
        "initial_state": ((1, 0), (2, 1)),
    }


def test_a_grid_too_small_to_draw_a_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="at least three cells"):
        gridworld(rows=1, columns=2, seed=0)


def test_state_index_of_matches_the_state_table(game: Gridworld) -> None:
    assert game.states[game.state_index_of((1, 0), (2, 1))] == (3, 7)


def test_state_index_of_rejects_impossible_positions(game: Gridworld) -> None:
    with pytest.raises(ValueError, match="off the grid"):
        game.state_index_of((0, 0), (9, 9))

    with pytest.raises(ValueError, match="cannot occupy one cell"):
        game.state_index_of((1, 1), (1, 1))


def test_tabular_strategies_have_an_explicit_policy_adapter(game: Gridworld) -> None:
    uniform = {player: np.full((72, 4), 0.25) for player in game.players}
    profile = game.tabular_profile(uniform)

    assert game.evaluate(profile) == game.evaluate(tabular_profile(game, uniform))


def test_tabular_profile_rejects_a_malformed_strategy(game: Gridworld) -> None:
    with pytest.raises(ValueError, match="no strategy given for players"):
        game.tabular_profile({0: np.full((72, 4), 0.25)})

    with pytest.raises(ValueError, match="must have shape"):
        game.tabular_profile(
            {player: np.full((72, 3), 1 / 3) for player in game.players}
        )


def test_fresh_layout_entropy_can_be_forbidden() -> None:
    from benchmax.seeding import EntropyDisabledError, forbid_entropy

    with (
        forbid_entropy(),
        pytest.warns(UserWarning, match="fresh entropy"),
        pytest.raises(EntropyDisabledError, match="layout requested fresh entropy"),
    ):
        gridworld(seed=None)


@pytest.mark.parametrize("changes", [
    {"transitions": np.array(1.0)},
    {"transitions": np.ones((2, 1, 3))},
    {"transitions": np.empty((2, 0, 2)), "rewards": {0: np.empty((2, 0))}},
    {"transitions": np.array([[[1.2, -0.2]], [[0.0, 1.0]]])},
    {"initial_state_distribution": np.array([1.2, -0.2])},
    {"initial_state_distribution": np.array([0.0, 0.0])},
    {"transitions": np.full((2, 1, 2), np.nan)},
    {"rewards": {0: np.full((2, 1), np.inf)}},
    {"observation_table": np.array([[0.0], [np.nan]])},
    {"observation_table": np.empty((2, 0))},
    {"discount": None},
])
def test_tabular_model_rejects_invalid_data_at_construction(changes) -> None:
    arguments = {
        "transitions": np.array([[[1.0, 0.0]], [[0.0, 1.0]]]), "rewards": {0: np.zeros((2, 1))},
        "initial_state_distribution": np.array([1.0, 0.0]), "discount": 0.9
    }
    with pytest.raises(ValueError):
        TabularMarkovGame(name="invalid", **(arguments | changes))
