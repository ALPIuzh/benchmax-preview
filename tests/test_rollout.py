"""Public array collection and its shared simulation path."""

import random
from dataclasses import replace

import numpy as np
import pytest

from benchmax import InvalidPolicyError, PolicyProfile, load_game
from benchmax.games.matrix import MatrixGame
from benchmax.seeding import derive_seed


def uniform(player, observations, action_space, mask):
    return mask / mask.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("name,max_steps", [
    ("rock_paper_scissors", np.int64(1)), ("kuhn_poker", None), ("phantom_tic_tac_toe", None),
    ("gridworld_3x3", 5), ("gridworld_3x3", np.array([1, 4, 2, 7, 3, 6, 1])),
])
def test_reproducible_arrays_match_replayed_environments(name, max_steps):
    game = load_game(name)
    options = {"episodes": 7, "seed": 7, "max_steps": max_steps}
    returns, data = game.rollout(uniform, batch_size=3, collect_data=True, **options)
    returns_only, empty = game.rollout(uniform, batch_size=3, **options)
    np.testing.assert_array_equal(returns, returns_only)
    assert empty == {}
    for batch_size in (1, 16):
        again_returns, again = game.rollout(PolicyProfile(uniform), batch_size=batch_size, collect_data=True, **options)
        np.testing.assert_array_equal(returns, again_returns)
        for player in game.players:
            first, second = data[player], again[player]
            first_order = np.lexsort((first["step_ids"], first["episode_ids"]))
            second_order = np.lexsort((second["step_ids"], second["episode_ids"]))
            for key in first:
                np.testing.assert_array_equal(first[key][first_order], second[key][second_order])

    limits = np.broadcast_to(game.max_episode_steps if max_steps is None else max_steps, (7,))
    for episode in range(7):
        env = game.new_env()
        step = env.reset(seed=derive_seed(7, "env", episode))
        t = 0
        expected_returns = np.zeros(len(game.players))
        while t < limits[episode] and not step.terminated:
            actions = {}
            for player in step.players_to_act:
                rows = data[player]
                selected = np.flatnonzero((rows["episode_ids"] == episode) & (rows["step_ids"] == t))
                assert len(selected) == 1
                row = selected[0]
                np.testing.assert_array_equal(rows["observations"][row], step.observations[player])
                np.testing.assert_array_equal(rows["legal_action_mask"][row],
                                              [a in step.legal_actions[player] for a in game.action_spaces[player]])
                actions[player] = int(rows["actions"][row])
                assert actions[player] in step.legal_actions[player]

            step = env.step(actions)
            expected_returns += [step.rewards[p] for p in game.players]
            t += 1
        np.testing.assert_array_equal(returns[episode], expected_returns)
        assert not step.truncated
        if game.discount is not None:
            for player in game.players:
                np.testing.assert_array_equal(data[player]["step_ids"][data[player]["episode_ids"] == episode],
                                              np.arange(limits[episode]))


def test_noncontiguous_ids_and_batched_inference():
    game = MatrixGame(name="ids", rewards={2: np.ones((2, 2)), 7: np.full((2, 2), -2.0)},
                      action_spaces={2: (5, 3), 7: (13, 11)})
    calls = []

    def policy(player, observations, action_space, mask):
        calls.append((player, len(observations), action_space))
        return np.tile([0.0, 1.0], (len(observations), 1))

    _, data = game.rollout(policy, episodes=5, seed=7, batch_size=3, collect_data=True)
    assert calls == [(2, 3, (5, 3)), (7, 3, (13, 11)), (2, 2, (5, 3)), (7, 2, (13, 11))]
    np.testing.assert_array_equal(data[2]["actions"], [3] * 5)
    np.testing.assert_array_equal(data[7]["actions"], [11] * 5)
    returns, uncollected = game.rollout(policy, episodes=5, seed=7, batch_size=3)
    np.testing.assert_array_equal(returns, [[1, -2]] * 5)
    assert uncollected == {}


def test_complete_episode_mode_rejects_cutoff():
    game = load_game("kuhn_poker")

    def check(player, observations, actions, mask):
        return np.tile([1.0, 0.0], (len(observations), 1))

    _, data = game.rollout(check, episodes=2, seed=7, max_steps=[2, 5], collect_data=True, require_complete=True)
    assert sum(len(rows["actions"]) for rows in data.values()) == 4
    for limits in (1, [1, 5]):
        with pytest.raises(ValueError, match="complete episodes"):
            game.rollout(check, episodes=2, seed=7, max_steps=limits, require_complete=True)

    with pytest.raises(ValueError, match="complete episodes"):
        load_game("gridworld_3x3").rollout(uniform, seed=7, max_steps=5, require_complete=True)


@pytest.mark.parametrize("at_reset", [False, True])
def test_complete_episode_mode_rejects_environment_truncation(monkeypatch, at_reset):
    game = load_game("kuhn_poker")
    original = game.new_env

    def new_env():
        env = original()
        method = env.reset if at_reset else env.step

        def truncated(*args, **kwargs):
            return replace(method(*args, **kwargs), observations={}, legal_actions={}, players_to_act=(),
                           terminated=False, truncated=True)

        if at_reset:
            env.reset = truncated
        else:
            env.step = truncated
        return env

    monkeypatch.setattr(game, "new_env", new_env)
    returns, data = game.rollout(uniform, episodes=3, seed=7, collect_data=True)
    assert returns.shape == (3, len(game.players))
    assert sum(len(rows["actions"]) for rows in data.values()) == (0 if at_reset else 3)
    with pytest.raises(ValueError, match="complete episodes"):
        game.rollout(uniform, episodes=3, seed=7, require_complete=True)


def test_observations_survive_environment_buffer_reuse(monkeypatch):
    game = load_game("rock_paper_scissors")
    env = game.new_env()
    original_reset, original_step = env.reset, env.step
    observation = np.array([0.0])

    def reset(seed):
        return replace(original_reset(seed=seed), observations={p: observation for p in game.players})

    def step(actions):
        observation[:] = 99
        return original_step(actions)

    env.reset, env.step = reset, step
    monkeypatch.setattr(game, "new_env", lambda: env)
    _, data = game.rollout(uniform, seed=7, collect_data=True)
    for player in game.players:
        np.testing.assert_array_equal(data[player]["observations"], [[0.0]])


@pytest.mark.parametrize("max_steps", [None, []])
def test_empty_arrays_have_policy_shapes(max_steps):
    game = load_game("kuhn_poker")
    _, data = game.rollout(uniform, episodes=0, seed=7, max_steps=max_steps, collect_data=True)
    for player in game.players:
        assert data[player]["observations"].shape == (0, *game.observation_spaces[player])
        assert data[player]["legal_action_mask"].shape == (0, len(game.action_spaces[player]))
        assert data[player]["legal_action_mask"].dtype == bool
        assert data[player]["actions"].shape == (0,)
        assert data[player]["actions"].dtype.kind == "i"


def test_invalid_policy_is_rejected():
    with pytest.raises(InvalidPolicyError):
        load_game("rock_paper_scissors").rollout(lambda p, o, a, m: np.zeros(m.shape), seed=7)


@pytest.mark.parametrize("kwargs,error", [
    ({"episodes": -1}, ValueError), ({"episodes": True}, TypeError), ({"episodes": 1.5}, TypeError),
    ({"batch_size": 0}, ValueError), ({"batch_size": False}, TypeError),
    ({"max_steps": 0}, ValueError), ({"max_steps": 2.5}, TypeError),
    ({"seed": -1}, ValueError), ({"seed": True}, TypeError),
    ({"max_steps": [1, 2]}, ValueError), ({"max_steps": [[1]]}, ValueError),
    ({"max_steps": [0]}, ValueError), ({"max_steps": [True]}, TypeError),
    ({"max_steps": [2.5]}, TypeError), ({"max_steps": ["2"]}, TypeError),
])
def test_invalid_arguments(kwargs, error):
    with pytest.raises(error):
        load_game("rock_paper_scissors").rollout(uniform, **({"seed": 7} | kwargs))


def test_global_rng_isolation():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    load_game("rock_paper_scissors").rollout(uniform, episodes=3, seed=7)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    np.testing.assert_array_equal(numpy_state[1], after[1])
