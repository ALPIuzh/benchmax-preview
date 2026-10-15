from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("benchmarl")

import torch
from torchrl.envs.utils import check_env_specs

from benchmax import load_game
from benchmax.execution.worker import InProcessRuntime, use_evaluation_runtime
from benchmax.integrations.benchmarl.env import _BatchedBenchmaxEnv, _BenchMARLPolicyProfile, _BenchmaxTask
from benchmax.integrations.exp_a_spiel import kernel as exp_a_spiel_kernel


@pytest.mark.parametrize("game_name", ["rock_paper_scissors", "kuhn_poker", "openspiel.connect_four"])
def test_benchmarl_adapter_matches_torchrl_specs(game_name: str) -> None:
    env = _BatchedBenchmaxEnv(load_game(game_name), num_envs=3, seed=7, device="cpu")

    check_env_specs(env)

    assert env.batch_size == (3,)
    env.close()


def test_benchmarl_adapter_uses_passed_game_configuration() -> None:
    game = load_game("rock_paper_scissors")
    game.name = "configured_rock_paper_scissors"
    game.action_spaces = {player: (2, 1, 0) for player in game.players}
    env = _BatchedBenchmaxEnv(game, num_envs=1, seed=7, device="cpu")

    tensordict = env.reset()
    tensordict["player_0", "action"] = torch.tensor([[0]])
    tensordict["player_1", "action"] = torch.tensor([[1]])
    result = env.step(tensordict)

    assert env._game is game
    assert result["next", "player_0", "reward"].item() == 1.0
    assert result["next", "player_1", "reward"].item() == -1.0
    env.close()


def test_benchmarl_task_creates_one_batched_environment() -> None:
    task = _BenchmaxTask(load_game("kuhn_poker"), max_episode_steps=3)
    factory = task.get_env_fun(3, continuous_actions=False, seed=7, device="cpu")
    env = factory()
    tensordict = env.reset()

    assert isinstance(env, _BatchedBenchmaxEnv)
    assert env.batch_size == (3,)
    assert len(env._envs) == 3
    assert tensordict.batch_size == (3,)
    assert tensordict["player_0"].batch_size == (3, 1)
    assert tensordict["player_0", "observation"].shape == (3, 1, 12)
    assert tensordict["player_0", "action_mask"].shape == (3, 1, 2)
    assert tensordict["done"].shape == (3, 1)
    assert task.observation_spec(env).shape == ()
    assert task.action_spec(env).shape == ()
    env.close()


def test_batched_environment_seeds_rows_reproducibly() -> None:
    game = load_game("kuhn_poker")
    first = _BatchedBenchmaxEnv(game, num_envs=8, seed=7, device="cpu")
    second = _BatchedBenchmaxEnv(game, num_envs=8, seed=7, device="cpu")

    first.set_seed(7)
    second.set_seed(7)
    assert first._pending_seeds == second._pending_seeds
    assert len(set(first._pending_seeds)) == 8

    first_reset = first.reset()
    second_reset = second.reset()

    for key in first_reset.keys(include_nested=True, leaves_only=True):
        torch.testing.assert_close(first_reset[key], second_reset[key])

    first.close()
    second.close()


def test_batched_environment_honors_partial_step_and_reset() -> None:
    env = _BatchedBenchmaxEnv(load_game("kuhn_poker"), num_envs=2, seed=7, device="cpu")
    tensordict = env.reset()
    before_step = tuple(env._time_steps)

    for player in env._game.players:
        tensordict[f"player_{player}", "action"] = torch.zeros((2, 1), dtype=torch.int64)

    tensordict["_step"] = torch.tensor([[True], [False]])

    stepped = env.step(tensordict)

    assert env._time_steps[0] is not before_step[0]
    assert env._time_steps[1] is before_step[1]
    torch.testing.assert_close(stepped["next", "player_0", "observation"][1], tensordict["player_0", "observation"][1])
    assert stepped["next", "player_0", "reward"][1].item() == 0.0

    saved_step = {
        key: value.clone()
        for key, value in stepped["next"].items(include_nested=True, leaves_only=True)
    }
    reset_input = stepped["next"].clone()
    reset_input["_reset"] = torch.tensor([[True], [False]])
    before_reset = tuple(env._time_steps)

    reset = env.reset(reset_input)

    assert env._time_steps[0] is not before_reset[0]
    assert env._time_steps[1] is before_reset[1]
    assert not reset["done"][0].item()

    for key, expected in saved_step.items():
        torch.testing.assert_close(stepped["next"][key], expected)

    env.close()


@pytest.mark.parametrize("game_name", ["rock_paper_scissors", "kuhn_poker", "gridworld_3x3"])
def test_benchmarl_adapter_exposes_global_state_for_each_environment_type(
    game_name: str,
) -> None:
    env = _BatchedBenchmaxEnv(load_game(game_name), num_envs=2, seed=7, device="cpu", use_state=True)

    tensordict = env.reset()

    assert env._game.state_space is not None
    assert tensordict["state"].shape == (2, *env._game.state_space)
    assert env.observation_spec["state"].is_in(tensordict["state"])
    env.close()


class _UniformMaskedPolicy(torch.nn.Module):
    def __init__(self, group: str) -> None:
        super().__init__()
        self._group = group

    def get_dist(self, tensordict: object) -> SimpleNamespace:
        mask = tensordict[self._group, "action_mask"]  # type: ignore[index]
        probabilities = mask.to(dtype=torch.float32)
        probabilities /= probabilities.sum(dim=-1, keepdim=True)
        return SimpleNamespace(probs=probabilities)


class _FakeExperiment:
    def __init__(self) -> None:
        self.group_policies = {
            "player_0": _UniformMaskedPolicy("player_0"),
            "player_1": _UniformMaskedPolicy("player_1"),
        }


def test_exported_profile_supports_exp_a_spiel_batched_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 3

    class FakeEvaluator:
        def num_information_states(self, player: int) -> int:
            del player
            return rows

        def policy_input_batch(self, player: int, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
            observations = np.zeros((stop - start, 108), dtype=np.float32)
            mask = np.zeros((stop - start, 9), dtype=bool)
            mask[:, 0] = True
            mask[:, player + 1] = True
            return observations, mask

        def evaluate(self, strategy_0: np.ndarray, strategy_1: np.ndarray) -> tuple[float, float, float]:
            assert strategy_0.shape == (rows, 9)
            assert strategy_1.shape == (rows, 9)
            np.testing.assert_allclose(strategy_0.sum(axis=1), 1.0)
            np.testing.assert_allclose(strategy_1.sum(axis=1), 1.0)
            return 0.0, 0.0, 0.0

    def fake_traverser(traverser_name: str) -> FakeEvaluator:
        assert traverser_name == "PtttEvaluator"
        return FakeEvaluator()

    monkeypatch.setattr(exp_a_spiel_kernel, "_prepared_traverser", fake_traverser)
    game = load_game("phantom_tic_tac_toe")
    profile = _BenchMARLPolicyProfile(_FakeExperiment(), game)  # type: ignore[arg-type]

    with use_evaluation_runtime(InProcessRuntime()):
        results = game.evaluate(profile)

    assert results["exploitability"] == 0.0
