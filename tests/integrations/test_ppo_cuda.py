"""Numerical and lifecycle checks for the optional CUDA PPO update path."""

import gc
import tempfile

import pytest

pytest.importorskip("benchmarl")

import torch
from benchmarl.experiment import Experiment

from benchmax import PolicyProfile, load_game
from benchmax.integrations.benchmarl.ppo import _can_use_cuda_graphs, _PPOExperiment
from benchmax.integrations.benchmarl.training import _create_experiment
from benchmax.training.algorithm import load_training_function


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph execution needs a GPU")
@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
@pytest.mark.parametrize(
    "clip,loss_type,adam",
    [
        ({"clip_grad_norm": True, "clip_grad_val": 0.3}, "l2", {}),
        ({"clip_grad_norm": False, "clip_grad_val": 0.2}, "smooth_l1", {"amsgrad": True}),
        ({"clip_grad_val": None}, "l1", {"foreach": False}),
    ],
)
def test_graphed_updates_match_upstream_with_changing_data_and_short_batches(algorithm, clip, loss_type, adam):
    game = load_game("kuhn_poker")
    params = {
        "model": {"num_cells": [16, 16]},
        "algorithm": {"entropy_coef": 0.03, "critic_coef": 0.7, "loss_critic_type": loss_type},
        "experiment": {
            "on_policy_collected_frames_per_batch": 32,
            "on_policy_n_envs_per_worker": 4,
            "on_policy_minibatch_size": 8,
            "on_policy_n_minibatch_iters": 3,
            "adam_extra_kwargs": adam,
            **clip,
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        experiments = []
        try:
            for graphs in (False, True):
                function = load_training_function(
                    algorithm, device="cuda", total_frames=32, params=params, cuda_graphs=graphs
                )
                experiments.append(_create_experiment(game, function, seed=17, device="cuda", output_dir=directory))
            reference, optimized = experiments
            assert type(reference) is Experiment
            assert type(optimized) is _PPOExperiment
            batch = next(iter(reference.collector))
            for group in reference.group_map:
                data = batch.exclude(*reference._get_excluded_keys(group)).to("cuda")
                data = reference.algorithm.process_batch(group, data).reshape(-1)
                # Short batches before AND after capture exercise the eager path
                # without changing the captured graph's gradient buffers.
                sizes = [8, 8, 5, 8, 8, 8, 3, 8, 8]
                for index, size in enumerate(sizes):
                    sample = data[index : index + size].clone()
                    reference.replay_buffers[group].sample = lambda sample=sample: sample.clone()
                    optimized.replay_buffers[group].sample = lambda sample=sample: sample.clone()
                    expected = reference._optimizer_loop(group)
                    actual = optimized._optimizer_loop(group)
                    for key in list(expected.keys()):
                        torch.testing.assert_close(actual[key], expected[key], rtol=3e-4, atol=3e-5, msg=key)
                    for name in reference.optimizers[group]:
                        before = reference.optimizers[group][name].state_dict()
                        after = optimized.optimizers[group][name].state_dict()
                        for param_id in before["state"]:
                            for key, value in before["state"][param_id].items():
                                torch.testing.assert_close(
                                    after["state"][param_id][key].cpu(), value.cpu(), rtol=3e-4, atol=3e-6
                                )
                        for p, q in zip(
                            reference.optimizers[group][name].param_groups[0]["params"],
                            optimized.optimizers[group][name].param_groups[0]["params"],
                            strict=True,
                        ):
                            torch.testing.assert_close(p, q, rtol=3e-4, atol=3e-6)
                assert optimized._updates[group].graph is not None
                for optimizer in optimized.optimizers[group].values():
                    for state in optimizer.state.values():
                        assert state["step"].item() == len(sizes)
        finally:
            for experiment in experiments:
                experiment.close()


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_cpu_uses_upstream_experiment(algorithm):
    from tests.training_helpers import tiny_training_params

    function = load_training_function(algorithm, total_frames=8, params=tiny_training_params())
    with tempfile.TemporaryDirectory() as directory:
        experiment = _create_experiment(
            load_game("rock_paper_scissors"), function, seed=7, device="cpu", output_dir=directory
        )
        try:
            assert type(experiment) is Experiment
        finally:
            experiment.close()


@pytest.mark.parametrize("unsupported", ["optimizer", "differentiable", "activation", "norm", "algorithm"])
def test_custom_configurations_keep_the_upstream_path(unsupported):
    function = load_training_function("ippo")
    algorithm, config, model = function.algorithm, function.experiment, function.model
    config.train_device = "cuda"
    assert _can_use_cuda_graphs(algorithm, config, model)
    if unsupported == "optimizer":
        config.adam_extra_kwargs = {"capturable": False}
    elif unsupported == "differentiable":
        config.adam_extra_kwargs = {"differentiable": True}
    elif unsupported == "activation":
        model.activation_class = torch.nn.Dropout
    elif unsupported == "norm":
        model.norm_class = torch.nn.BatchNorm1d
    else:
        from benchmax.integrations.benchmarl.provider import algorithm_config_class

        algorithm = algorithm_config_class("iql").get_from_yaml()
    assert not _can_use_cuda_graphs(algorithm, config, model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph execution needs a GPU")
@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_cuda_training_matches_eager_and_round_trips_policy(algorithm, tmp_path):
    from benchmax.execution.artifacts.policies import load_policy, write_policy
    from benchmax.training import train

    game = load_game("kuhn_poker")
    params = {
        "experiment": {
            "on_policy_collected_frames_per_batch": 64,
            "on_policy_n_envs_per_worker": 4,
            "on_policy_n_minibatch_iters": 3,
            "on_policy_minibatch_size": 8,
        },
        "algorithm": {"entropy_coef": 0.02},
        "model": {"num_cells": [16, 16]},
    }
    policies = [
        train(
            game, algorithm, total_frames=128, seed=7, device="cuda", verbose=False, params=params, cuda_graphs=graphs
        )
        for graphs in (False, True, True)
    ]
    expected = policies[0]._batched_policy_profile._state_dicts()
    for policy in policies[1:]:
        actual = policy._batched_policy_profile._state_dicts()
        for player in game.players:
            for name, value in expected[player].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[player][name], value, rtol=3e-4, atol=3e-6)
    policy = policies[-1]
    spec = policy._portable_training
    reference = write_policy(policy, spec, tmp_path / "policy")
    restored = load_policy(reference, expected_game=spec.game)
    expected_metrics = game.evaluate(policy)
    actual_metrics = game.evaluate(restored)
    assert actual_metrics["nash_conv"] == pytest.approx(expected_metrics["nash_conv"], abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph execution needs a GPU")
@pytest.mark.parametrize("game_name", ["kuhn_poker", "gridworld_3x3"])
def test_cuda_response_training_matches_eager(game_name):
    from tests.training_helpers import tiny_training_params

    game = load_game(game_name)
    game.max_episode_steps = 3
    opponents = PolicyProfile(lambda p, obs, actions, mask: mask / mask.sum(axis=1, keepdims=True))
    states = []
    for graphs in (False, True):
        trainer = load_training_function(
            "ippo", device="cuda", total_frames=24, verbose=False, params=tiny_training_params(), cuda_graphs=graphs,
        )
        checkpoints = []
        response = trainer.best_response(
            game, opponents, 1, seed=7, checkpoint_frames=(8, 24),
            on_checkpoint=lambda policy, frames, checkpoints=checkpoints: checkpoints.append((policy, frames)),
        )
        assert [frames for _, frames in checkpoints] == [8, 24]
        state = response._batched_policy_profile._state_dicts()
        assert set(state) == {1}
        states.append(state[1])
    for name, value in states[0].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(states[1][name], value, rtol=3e-4, atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph execution needs a GPU")
@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_indexed_replay_keeps_sampler_and_refreshed_storage(algorithm, tmp_path):
    from benchmax.integrations.benchmarl.env import _ResponseCheckpoints

    params = {
        "model": {"num_cells": [16, 16]},
        "experiment": {
            "on_policy_collected_frames_per_batch": 32,
            "on_policy_n_envs_per_worker": 4,
            "on_policy_minibatch_size": 7,
        },
    }
    experiments = []
    try:
        game = load_game("kuhn_poker")
        for index in range(2):
            function = load_training_function(algorithm, device="cuda", total_frames=96, params=params)
            experiments.append(_create_experiment(
                game, function, seed=17, device="cuda", output_dir=str(tmp_path),
                callbacks=[_ResponseCheckpoints(game, 0, (96,), lambda policy, frames: None)] if index else None,
            ))
        reference, candidate = experiments
        assert candidate._indexed_groups == set(candidate.group_map)
        reference._indexed_groups.clear()
        for batch in reference.collector:
            for group in reference.group_map:
                data = batch.exclude(*reference._get_excluded_keys(group)).to("cuda")
                data = reference.algorithm.process_batch(group, data).reshape(-1)
                for experiment in experiments:
                    experiment.replay_buffers[group].extend(data.clone())
                for _ in range(10):
                    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                    expected = reference._optimizer_loop(group)
                    next_cpu, next_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state(cuda_rng)
                    actual = candidate._optimizer_loop(group)
                    torch.testing.assert_close(torch.get_rng_state(), next_cpu)
                    torch.testing.assert_close(torch.cuda.get_rng_state(), next_cuda)
                    for key in list(expected.keys()):
                        torch.testing.assert_close(actual[key], expected[key], rtol=3e-4, atol=3e-5, msg=key)
                for name in reference.optimizers[group]:
                    left, right = reference.optimizers[group][name], candidate.optimizers[group][name]
                    for p, q in zip(left.param_groups[0]["params"], right.param_groups[0]["params"], strict=True):
                        torch.testing.assert_close(p, q, rtol=3e-4, atol=3e-6)
                        for key in left.state[p]:
                            torch.testing.assert_close(left.state[p][key], right.state[q][key], rtol=3e-4, atol=3e-6)
        assert all(update.graph is not None for update in candidate._indexed_updates.values())
        assert all(state["step"].item() == 30 for optimizers in candidate.optimizers.values()
                   for optimizer in optimizers.values() for state in optimizer.state.values())
    finally:
        for experiment in experiments:
            experiment.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 training requires CUDA")
def test_tf32_request_is_scoped_and_survives_policy_serialization(tmp_path):
    from benchmax.execution.artifacts.policies import load_policy, write_policy
    from benchmax.training import train

    previous = torch.backends.cuda.matmul.fp32_precision
    params = {
        "model": {"num_cells": [32, 32]},
        "experiment": {
            "on_policy_collected_frames_per_batch": 64,
            "on_policy_n_envs_per_worker": 4,
            "on_policy_n_minibatch_iters": 3,
            "on_policy_minibatch_size": 8,
        },
    }
    game = load_game("kuhn_poker")
    policy = train(game, "mappo", total_frames=128, seed=7, device="cuda", verbose=False,
                   params=params, matmul_precision="tf32")
    assert torch.backends.cuda.matmul.fp32_precision == previous
    spec = policy._portable_training
    assert spec.algorithm.overrides["matmul_precision"] == "tf32"
    reference = write_policy(policy, spec, tmp_path / "tf32-policy")
    restored = load_policy(reference, expected_game=spec.game)
    expected, actual = game.evaluate(policy), game.evaluate(restored)
    assert actual["nash_conv"] == pytest.approx(expected["nash_conv"], abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capture requires a GPU")
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_capture_defers_automatic_gc_and_restores_it(enabled, fail):
    from tensordict import TensorDict

    from benchmax.integrations.benchmarl.ppo import _CudaGraphUpdate

    previous = gc.isenabled()
    observations = []

    def update(batch):
        capturing = torch.cuda.is_current_stream_capturing()
        observations.append((capturing, gc.isenabled()))
        if capturing and fail:
            raise RuntimeError("test capture failure")
        return batch + 1

    try:
        (gc.enable if enabled else gc.disable)()
        batch = TensorDict({"x": torch.ones(3, device="cuda")}, batch_size=[3], device="cuda")
        graph = _CudaGraphUpdate(update, batch)
        for _ in range(3):
            graph(batch)
        if fail:
            with pytest.raises(RuntimeError, match="test capture failure"):
                graph(batch)
        else:
            torch.testing.assert_close(graph(batch)["x"], batch["x"] + 1)
        assert observations == [(False, enabled)] * 3 + [(True, False)]
        assert gc.isenabled() == enabled
    finally:
        (gc.enable if previous else gc.disable)()
