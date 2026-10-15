"""Exercise response training through the existing BenchMARL adapter."""

import numpy as np
import pytest

pytest.importorskip("benchmarl")
import torch

from benchmax import PolicyProfile, load_game
from benchmax.execution.specs import EvaluationSpec
from benchmax.integrations.benchmarl.env import _BatchedBenchmaxEnv
from benchmax.seeding import forbid_entropy
from tests.training_helpers import tiny_training_params


def test_frozen_opponent_inference_is_batched_and_respects_partial_steps():
    game = load_game("rock_paper_scissors")
    calls = []

    def opponent(player, obs, actions, mask):
        calls.append((player, len(obs)))
        return np.broadcast_to([1.0, 0.0, 0.0], mask.shape)

    env = _BatchedBenchmaxEnv(
        game,
        num_envs=3,
        seed=7,
        device="cpu",
        response_player=1,
        opponents=PolicyProfile(opponent),
    )
    try:
        data = env.reset()
        assert env.group_map == {"player_1": ["player_1"]}
        data["player_1", "action"] = torch.ones((3, 1), dtype=torch.int64)
        data["_step"] = torch.tensor([[True], [False], [True]])
        step = env.step(data)["next"]
        assert calls == [(0, 2)]
        assert step["player_1", "reward"].flatten().tolist() == [1.0, 0.0, 1.0]
        assert step["done"].flatten().tolist() == [True, False, True]
    finally:
        env.close()


@pytest.mark.parametrize("name", [
    "rock_paper_scissors", "kuhn_poker", "phantom_tic_tac_toe", "abrupt_phantom_tic_tac_toe",
    "dark_hex_3x3", "abrupt_dark_hex_3x3", "liars_dice_2d5f", "gridworld_3x3",
])
def test_registered_evaluator_trains_both_sides_with_a_fixed_callable_and_portable_config(name, tmp_path):
    config = {
        "name": "approximate_exploitability",
        "total_frames": 24,
        "restarts": 1,
        "episodes": 8,
        "validation_episodes": 4,
        "seed": 19,
        "device": "cpu",
        "cuda_graphs": False,
        "params": tiny_training_params(),
    }
    game = load_game(name, evaluator=config)
    reference = PolicyProfile(lambda p, obs, actions, mask: mask / mask.sum(axis=1, keepdims=True))
    with forbid_entropy():
        result = game.evaluate(reference)
    assert result["training_frames_per_restart"] == 24
    assert result["actual_training_frames"] == 48
    assert result["evaluated_checkpoints_per_player"] == {p: 3 for p in game.players}
    assert set(game.evaluator.responses) == set(game.players)
    assert np.isfinite(result["approximate_exploitability"])
    assert result["return_estimation"] == ("exact_model" if game.discount is not None else "monte_carlo")
    # The existing configuration/spec mechanism carries the complete evaluation
    # budget and seeds; no new job type is necessary.
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}")
    spec = EvaluationSpec.create(game=name, game_overrides={"evaluator": config}, policy=str(policy_path))
    restored = EvaluationSpec.from_document(spec.to_document())
    assert restored.game.resolved["evaluator"]["seed"] == 19
    assert restored.game.resolved["evaluator"]["params"] == game.evaluator.params
    for group, overrides in tiny_training_params().items():
        for key, value in overrides.items():
            assert game.evaluator.params[group][key] == value
    assert restored.game.resolved["evaluator"]["cuda_graphs"] is False
    from benchmax.integrations.benchmarl.evidence import resolve

    assert resolve(restored).requirements["accelerator"] == "none"


def test_checkpoint_selection_retains_an_early_greedy_response_when_training_regresses(monkeypatch):
    from benchmax import config
    from benchmax.evaluation import ApproximateExploitabilityEvaluator, DiscountedNashGapEvaluator
    from benchmax.games.tabular.game import TabularZeroSumMarkovGame
    from benchmax.integrations.benchmarl.training import BenchMARLTrainingFunction

    game = TabularZeroSumMarkovGame(
        name="continuing", transitions=np.ones((1, 2, 2, 1)), payoffs=np.array([[[5, 3], [8, 6]]]),
        initial_state_distribution=np.ones(1), discount=0.5,
        evaluator=DiscountedNashGapEvaluator(),
    )
    reference = PolicyProfile(lambda p, obs, actions, mask: np.broadcast_to([1., 0.], mask.shape))

    original_aliases = config._aliases

    def aliases(group):
        entries = original_aliases(group)
        if group == "algorithm":
            entries["response_preset"] = {
                "_extends_": "ippo", "_params_": {"experiment": {"lr": 0.00123}},
            }
        return entries

    monkeypatch.setattr(config, "_aliases", aliases)

    def train(self, game, opponents, player, *, seed, checkpoint_frames, on_checkpoint):
        assert self.experiment.lr == 0.00123
        assert checkpoint_frames == (8, 16)
        for frames, probabilities in ((8, [0.2, 0.8]), (16, [0.8, 0.2])):
            def response(p, obs, actions, mask, probabilities=probabilities):
                assert p == player
                return np.broadcast_to(probabilities, mask.shape)

            policy = PolicyProfile(response)
            on_checkpoint(policy, frames)
        return policy

    monkeypatch.setattr(BenchMARLTrainingFunction, "best_response", train)
    evaluator = ApproximateExploitabilityEvaluator(algorithm="response_preset", total_frames=16, checkpoints=2)
    result = game.evaluate(reference, evaluator=evaluator)
    assert result["selected_response_frames"] == {0: 8, 1: 8}
    assert result["response_action_modes"] == {0: "greedy", 1: "greedy"}
    assert result["approximate_exploitability"] == 5
    assert result["approximate_exploitability"] == pytest.approx(game.evaluate(reference)["exploitability"])
    assert result["response_returns"] == {0: 16, 1: -6}


def test_submitted_approximation_matches_direct_evaluation(tmp_path, monkeypatch, tiny_trained_policy):
    from benchmax.execution import Job
    from benchmax.execution.artifacts.policies import write_policy

    monkeypatch.setenv("BENCHMAX_RUNS", str(tmp_path / "runs"))
    _, policy, training = tiny_trained_policy
    write_policy(policy, training, tmp_path / "policy")
    manifest = tmp_path / "policy" / "manifest.json"
    game = load_game(
        "rock_paper_scissors",
        evaluator={
            "name": "approximate_exploitability",
            "total_frames": 8,
            "restarts": 1,
            "episodes": 16,
            "validation_episodes": 8,
            "seed": 23,
            "device": "cpu",
            "params": tiny_training_params(),
        },
    )
    expected = game.evaluate(manifest, output=tmp_path / "direct.json")
    job = game.evaluate(manifest, submit=True, output=tmp_path / "submitted.json").run()
    actual = job.result(timeout=45)
    assert actual == expected
    assert Job.open(job.id).result() == actual
