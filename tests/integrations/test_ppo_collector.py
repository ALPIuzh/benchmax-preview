"""Compare buffered PPO rollouts with the upstream collector, field by field."""

from dataclasses import replace

import pytest

pytest.importorskip("benchmarl")

import torch

from benchmax import PolicyProfile, load_game
from benchmax.integrations.benchmarl.collector import _PPOCollector
from benchmax.integrations.benchmarl.training import _create_experiment
from benchmax.training.algorithm import load_training_function

GAMES = [
    "rock_paper_scissors",
    "kuhn_poker",
    "openspiel.connect_four",
    "openspiel.matrix_rps",
    "phantom_tic_tac_toe",
    "abrupt_phantom_tic_tac_toe",
    "dark_hex_3x3",
    "abrupt_dark_hex_3x3",
    "liars_dice_2d5f",
    "gridworld_3x3",
]


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def compare_collectors(
    game, algorithm, share, tmp_path, *, frames=128, truncate=False, model=None, buffered=True,
    response_player=None, opponents=None,
):
    params = {
        "model": {"num_cells": [16, 16]},
        "experiment": {
            "on_policy_collected_frames_per_batch": frames,
            "on_policy_n_envs_per_worker": 4,
            "share_policy_params": share,
        },
    }
    function = load_training_function(algorithm, total_frames=3 * frames, params=params, cuda_graphs=False)
    for key, value in (model or {}).items():
        setattr(function.model, key, value)
    experiment = _create_experiment(
        game, function, seed=17, device="cpu", output_dir=str(tmp_path),
        response_player=response_player, opponents=opponents,
    )
    optimized = _PPOCollector(
        experiment.env_func,
        experiment.policy,
        device="cpu",
        storing_device="cpu",
        frames_per_batch=frames,
        total_frames=3 * frames,
        init_random_frames=0,
    )
    try:
        reference = experiment.collector
        assert optimized._buffered is buffered
        if response_player is not None:
            assert experiment.group_map == {f"player_{response_player}": [f"player_{response_player}"]}
            assert experiment.config.gamma == (1.0 if game.discount is None else game.discount)
        reference.set_seed(42)
        optimized.set_seed(42)
        if truncate:
            # Exercise reset/bootstrap semantics independently of the built-in
            # games, some of which intentionally never truncate.
            for collector in (reference, optimized):
                for env in collector.env.base_env._envs:
                    step = env.step
                    env.step = lambda actions, step=step: replace(step(actions), terminated=False, truncated=True)
        left_iter, right_iter = iter(reference), iter(optimized)
        first = snapshot = None
        for index in range(3):
            torch.manual_seed(123 + index)
            left = next(left_iter)
            rng = torch.get_rng_state()
            torch.manual_seed(123 + index)
            right = next(right_iter)
            torch.testing.assert_close(torch.get_rng_state(), rng)
            assert left.batch_size == right.batch_size
            assert left.names == right.names
            assert set(left.keys(True, True)) == set(right.keys(True, True))
            for key in left.keys(True, True):
                torch.testing.assert_close(left[key], right[key], rtol=1e-6, atol=1e-7, msg=str(key))
            if truncate:
                assert right["next", "truncated"].all()
                assert not right["next", "terminated"].any()
            if response_player is not None and game.discount is not None:
                assert right["next", "truncated"].any()
                assert not right["next", "terminated"].any()
                assert (right["next", "step_count"][right["next", "truncated"]] == game.max_episode_steps).all()
            for group in experiment.group_map:
                targets = [
                    experiment.algorithm.process_batch(
                        group, batch.exclude(*experiment._get_excluded_keys(group)).clone()
                    )
                    for batch in (left, right)
                ]
                for key in ("advantage", "value_target"):
                    torch.testing.assert_close(targets[0][group, key], targets[1][group, key])
            if first is None:
                first, snapshot = right, right.clone()
            else:
                for key in first.keys(True, True):
                    torch.testing.assert_close(first[key], snapshot[key], rtol=0, atol=0)
            # Stateful inference networks must observe policy synchronization.
            with torch.no_grad():
                for parameter in experiment.policy.parameters():
                    parameter.add_(0.01)
            reference.update_policy_weights_()
            optimized.update_policy_weights_()
    finally:
        optimized.shutdown()
        experiment.close()


@pytest.mark.parametrize("game_name,algorithm,share", [
    (game, algorithm, share) for game in GAMES
    for algorithm, share in (
        [("ippo", True), ("ippo", False), ("mappo", True), ("mappo", False)]
        if game in {"rock_paper_scissors", "kuhn_poker", "gridworld_3x3"} else [("mappo", False)]
    )
])
def test_buffered_rollouts_match_upstream(game_name, algorithm, share, tmp_path):
    compare_collectors(load_game(game_name), algorithm, share, tmp_path)


@pytest.mark.parametrize("game_name", ["rock_paper_scissors", "kuhn_poker", "gridworld_3x3"])
@pytest.mark.parametrize("player", [0, 1])
def test_response_rollouts_keep_opponents_frozen_and_match_upstream(game_name, player, tmp_path):
    game = load_game(game_name)
    game.max_episode_steps = 3
    calls = []

    def opponent(p, obs, actions, mask):
        assert p != player
        calls.append((p, len(obs)))
        weights = mask * list(range(1, len(actions) + 1))
        return weights / weights.sum(axis=1, keepdims=True)

    compare_collectors(
        game, "mappo", False, tmp_path, response_player=player, opponents=PolicyProfile(opponent),
        buffered=game.discount is None,
    )
    assert calls


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_truncated_rows_keep_bootstrap_observations(algorithm, tmp_path):
    compare_collectors(load_game("gridworld_3x3"), algorithm, False, tmp_path, frames=32, truncate=True)


def test_reordered_actions_and_single_step_rollouts(tmp_path):
    game = load_game("rock_paper_scissors")
    game.action_spaces = {player: (2, 1, 0) for player in game.players}
    compare_collectors(game, "mappo", True, tmp_path, frames=4)


def test_layer_norm_and_relu_keep_the_same_inference(tmp_path):
    compare_collectors(
        load_game("kuhn_poker"),
        "mappo",
        False,
        tmp_path,
        model={
            "norm_class": torch.nn.LayerNorm,
            "norm_kwargs": {"normalized_shape": 16},
            "activation_class": torch.nn.ReLU,
        },
    )


def test_additional_environment_transforms_use_upstream_rollout(monkeypatch, tmp_path):
    from torchrl.envs import StepCounter

    from benchmax.integrations.benchmarl.env import _BenchmaxTask

    monkeypatch.setattr(_BenchmaxTask, "get_env_transforms", lambda self, env: [StepCounter(max_steps=3)])
    compare_collectors(load_game("gridworld_3x3"), "mappo", True, tmp_path, buffered=False)
