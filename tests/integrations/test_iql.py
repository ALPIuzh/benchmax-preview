"""Decision replay, legal Q policies, and their existing artifact/evaluator paths."""

import numpy as np
import pytest

pytest.importorskip("benchmarl")
import torch
from tensordict import TensorDict

from benchmax import Game, GameInstance, TimeStep, load_game
from benchmax.execution.artifacts.policies import load_policy
from benchmax.integrations.benchmarl.iql import _DecisionTransitions
from benchmax.training import train_iql


def _params(**overrides):
    return {
        "experiment": {
            "off_policy_collected_frames_per_batch": 8,
            "off_policy_n_envs_per_worker": 2,
            "off_policy_n_optimizer_steps": 1,
            "off_policy_train_batch_size": 4,
            "off_policy_memory_size": 32,
            **overrides,
        },
        "model": {"num_cells": [8]},
    }


def _batch(active, next_active, rewards, terminated=None):
    count = len(active)
    terminated = torch.tensor(terminated or [False] * count).reshape(1, count, 1)
    return TensorDict({
        "player_0": TensorDict({
            "observation": torch.tensor([[0., float(a)] for a in active]).reshape(1, count, 1, 2),
            "action_mask": torch.ones((1, count, 1, 2), dtype=torch.bool),
            "action": torch.ones((1, count, 1), dtype=torch.int64),
        }, batch_size=(1, count, 1)),
        "next": TensorDict({
            "player_0": TensorDict({
                "observation": torch.tensor([[0., float(a)] for a in next_active]).reshape(1, count, 1, 2),
                "action_mask": torch.ones((1, count, 1, 2), dtype=torch.bool),
                "reward": torch.tensor(rewards, dtype=torch.float32).reshape(1, count, 1, 1),
            }, batch_size=(1, count, 1)),
            "done": terminated,
            "terminated": terminated.clone(),
        }, batch_size=(1, count)),
    }, batch_size=(1, count))


def test_replay_carries_decisions_across_batches_and_preserves_discounted_targets():
    from torchrl.objectives.value import TD0Estimator

    replay = _DecisionTransitions(0.5)
    assert replay(_batch([1, 0], [0, 0], [1, 2]), "player_0") is None
    result = replay(_batch([0, 1], [1, 0], [4, 8], [False, True]), "player_0")
    # The first action waits three original steps: 1 + .5*2 + .25*4.
    assert result["next", "player_0", "reward"].flatten().tolist() == [3, 8]
    assert result["steps_to_next_obs"].flatten().tolist() == [3, 1]
    assert result["player_0", "observation"][..., -1].bool().all()
    td0 = TD0Estimator(gamma=0.5, value_network=None)
    td0.set_keys(reward=("player_0", "reward"), done=("player_0", "done"), terminated=("player_0", "terminated"))
    targets = td0.value_estimate(result, next_value=torch.full((2, 1, 1), 16.))
    torch.testing.assert_close(targets.flatten(), torch.tensor([5., 8.]))
    truncated = _batch([1], [1], [2])
    truncated["next", "done"].fill_(True)
    result = replay(truncated, "player_0")
    target = td0.value_estimate(result, next_value=torch.full((1, 1, 1), 16.))
    assert target.item() == 10  # A rollout cutoff still bootstraps its discounted tail.


@pytest.mark.parametrize("name", [
    "rock_paper_scissors", "gridworld_3x3", "kuhn_poker", "dark_hex_3x3", "abrupt_dark_hex_3x3",
    "phantom_tic_tac_toe", "abrupt_phantom_tic_tac_toe", "liars_dice_2d5f",
])
def test_iql_training_returns_legal_deterministic_policies_with_random_warmup(name):
    game = load_game(name)
    profile = train_iql(
        game, total_frames=16, seed=7, device="cpu", verbose=False,
        params=_params(off_policy_init_random_frames=16),
    )
    env = game.new_env()
    state = env.reset(seed=3)
    for player in state.players_to_act:
        actions = game.action_spaces[player]
        mask = np.array([[action in state.legal_actions[player] for action in actions]])
        inputs = np.array([state.observations[player]])
        random_state = torch.get_rng_state().clone()
        probabilities = profile(player, inputs, actions, mask)
        np.testing.assert_array_equal(probabilities.sum(-1), 1)
        assert np.all((probabilities == 0) | (probabilities == 1))
        assert np.all(probabilities[~mask] == 0)
        torch.testing.assert_close(torch.get_rng_state(), random_state, rtol=0, atol=0)
    if game.discount is None:
        for seed in range(4):
            state = env.reset(seed=seed)
            for _ in range(game.max_episode_steps):
                if state.terminated:
                    break
                joint = {}
                for player in state.players_to_act:
                    actions = game.action_spaces[player]
                    mask = np.array([[action in state.legal_actions[player] for action in actions]])
                    probabilities = profile(player, np.array([state.observations[player]]), actions, mask)
                    joint[player] = actions[int(probabilities[0].argmax())]
                state = env.step(joint)
                assert np.isfinite(list(state.rewards.values())).all()
            assert state.terminated
    else:
        assert np.isfinite(game.evaluate(profile)["exploitability"])


def test_iql_policy_artifact_round_trip_preserves_greedy_actions(tmp_path):
    game = load_game("kuhn_poker")
    path = tmp_path / "training.json"
    profile = train_iql(game, total_frames=16, seed=7, device="cpu", verbose=False, params=_params(), output=path)
    restored = load_policy(str(path), expected_game=profile._portable_training.game)
    assert game.evaluate(restored) == game.evaluate(profile)


class _DelayedGame(Game):
    def __init__(self):
        super().__init__(
            name="delayed", players=(0, 1), observation_spaces={0: (1,), 1: (1,)},
            action_spaces={0: (0, 1), 1: (0,)}, is_fully_observable=True, is_simultaneous=False,
            max_episode_steps=4,
        )

    def new_env(self):
        return _DelayedInstance(self)


class _DelayedInstance(GameInstance):
    def reset(self, seed=None):
        self.turn, self.choice = 0, 0
        return self._state()

    def step(self, actions):
        if self.turn == 0:
            self.choice = actions[0]
        self.turn += 1
        return self._state()

    def _state(self):
        done = self.turn == 4
        player = 0 if self.turn == 0 else 1
        reward = float(2 * self.choice - 1) if done else 0.
        return TimeStep(
            {} if done else {player: (float(self.turn),)}, {0: reward, 1: -reward},
            () if done else (player,), {} if done else {player: self.game.action_spaces[player]}, done, False, {},
        )


def test_iql_learns_a_reward_delivered_only_on_an_opponents_later_turn():
    game = _DelayedGame()
    response = train_iql(
        game, total_frames=512, seed=7, device="cpu", verbose=False,
        params=_params(
            lr=0.02, off_policy_collected_frames_per_batch=32, off_policy_n_envs_per_worker=4,
            off_policy_n_optimizer_steps=4, off_policy_train_batch_size=16,
        ),
    )
    np.testing.assert_array_equal(response(0, np.array([[0.]]), (0, 1), np.array([[True, True]])), [[0., 1.]])
