"""Compare sampled unilateral deviations with independent exact calculations."""

import numpy as np
import pytest

from benchmax import PolicyProfile, load_game
from benchmax.evaluation import ApproximateExploitabilityEvaluator, MatrixEvaluator
from benchmax.games.matrix import TwoPlayerZeroSumMatrixGame
from benchmax.games.tabular.game import TabularZeroSumMarkovGame
from benchmax.seeding import forbid_entropy


def fixed(column, player=None):
    def policy(p, observations, actions, mask):
        if player is not None:
            assert p == player  # A response must never replace its opponent.
        probabilities = np.zeros_like(mask, dtype=float)
        probabilities[:, column] = 1
        return probabilities

    return PolicyProfile(policy)


def test_supplied_responses_recover_exact_gains_without_training_or_traversal():
    game = TwoPlayerZeroSumMatrixGame(
        name="asymmetric",
        payoffs=np.array([[5, 3], [8, 6]]),
        evaluator=MatrixEvaluator(),
    )
    evaluator = ApproximateExploitabilityEvaluator(
        response_profiles={p: fixed(1, p) for p in game.players},
        episodes=8,
        validation_episodes=4,
        per_player_metrics=True,
    )
    exact = game.evaluate(fixed(0))
    with forbid_entropy():
        result = game.evaluate(fixed(0), evaluator=evaluator)
    assert result["player_improvements"] == exact["player_improvements"] == {0: 3, 1: 2}
    assert result["approximate_exploitability"] == exact["exploitability"] == 2.5
    assert result["approximate_nash_gap"] == 3
    assert result["player_improvement_standard_errors"] == {0: 0, 1: 0}
    assert "exploitability" not in result


def test_intermediate_openspiel_rewards_are_counted_once():
    game = load_game(
        "openspiel.repeated_game",
        params={"stage_game": {"name": "matrix_rps"}, "num_repetitions": 3},
    )
    env = game.new_env()
    assert env.reset(seed=7).rewards == {0: 0.0, 1: 0.0}
    for _ in range(3):
        step = env.step({0: 1, 1: 0})
        assert step.rewards == {0: 1.0, 1: -1.0}
    assert step.terminated
    assert env.reset().rewards == {0: 0.0, 1: 0.0}

    evaluator = ApproximateExploitabilityEvaluator(
        response_profiles={p: fixed(1, p) for p in game.players},
        episodes=8,
        validation_episodes=4,
    )
    result = game.evaluate(fixed(0), evaluator=evaluator)
    assert result["approximate_exploitability"] == 3
    assert result["raw_exploitability_standard_error"] == 0


def test_reference_fallback_excludes_strictly_worse_responses():
    game = load_game("rock_paper_scissors")
    evaluator = ApproximateExploitabilityEvaluator(
        response_profiles={p: fixed(2, p) for p in game.players},
        episodes=8,
        validation_episodes=4,
    )
    result = game.evaluate(fixed(0), evaluator=evaluator)
    assert result["response_policy_sources"] == {0: "reference", 1: "reference"}
    assert result["approximate_exploitability"] == 0
    assert result["raw_exploitability_standard_error"] == 0
    assert game.evaluate(fixed(0))["exploitability"] == 1  # Zero found does not imply an equilibrium.


def test_continuing_returns_include_the_discounted_tail_without_rollouts(monkeypatch):
    game = TabularZeroSumMarkovGame(
        name="continuing",
        transitions=np.ones((1, 2, 2, 1)),
        payoffs=np.array([[[0, -2], [3, 1]]]),
        initial_state_distribution=np.ones(1),
        discount=0.5,
        max_episode_steps=1,
    )

    def no_rollout():
        pytest.fail("tabular return measurement must use the model, not a truncated rollout")

    monkeypatch.setattr(game, "new_env", no_rollout)
    evaluator = ApproximateExploitabilityEvaluator(
        response_profiles={p: fixed(1, p) for p in game.players}, per_player_metrics=True,
    )
    with forbid_entropy():
        result = game.evaluate(fixed(0), evaluator=evaluator)
    assert result["player_improvements"] == {0: 6, 1: 4}
    assert result["approximate_exploitability"] == 5
    assert result["return_estimation"] == "exact_model"
    assert result["discount"] == 0.5
    assert result["player_improvement_standard_errors"] == {0: 0, 1: 0}
    assert result["raw_exploitability_standard_error"] == 0
    assert result["episodes_per_profile"] == result["validation_episodes_per_candidate"] == 0


def test_kuhn_sampled_best_responses_match_exact_with_batch_independent_seeds():
    from open_spiel.python.algorithms.best_response import BestResponsePolicy
    from open_spiel.python.policy import UniformRandomPolicy

    game = load_game("kuhn_poker")
    best = {p: BestResponsePolicy(game.spiel_game, p, UniformRandomPolicy(game.spiel_game)) for p in game.players}
    tables = {p: {} for p in game.players}

    def visit(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            actions = [a for a, _ in state.chance_outcomes()]
        else:
            p = state.current_player()
            probabilities = best[p].action_probabilities(state)
            tables[p][tuple(state.information_state_tensor(p))] = [
                probabilities.get(a, 0) for a in game.action_spaces[p]
            ]
            actions = state.legal_actions()
        for a in actions:
            visit(state.child(a))

    visit(game.spiel_game.new_initial_state())
    reference = PolicyProfile(lambda p, obs, actions, mask: mask / mask.sum(axis=1, keepdims=True))
    response = PolicyProfile(lambda p, obs, actions, mask: np.asarray([tables[p][tuple(row)] for row in obs]))
    exact = game.evaluate(reference)
    settings = {
        "response_profiles": {p: response for p in game.players},
        "episodes": 4096,
        "validation_episodes": 512,
        "seed": 17,
        "per_player_metrics": True,
    }
    with forbid_entropy():
        first = game.evaluate(reference, evaluator=ApproximateExploitabilityEvaluator(**settings, batch_size=64))
        second = game.evaluate(reference, evaluator=ApproximateExploitabilityEvaluator(**settings, batch_size=127))
    assert first == second
    for p in game.players:
        assert first["raw_player_improvements"][p] == pytest.approx(
            exact["player_improvements"][p],
            abs=4 * first["player_improvement_standard_errors"][p],
        )


def test_aggregate_uses_only_two_final_matchups_and_preserves_negative_response_returns(monkeypatch):
    game = TwoPlayerZeroSumMatrixGame(name="shifted", payoffs=np.array([[5, 3], [8, 6]]))
    evaluator = ApproximateExploitabilityEvaluator(
        response_profiles={p: fixed(1, p) for p in game.players}, episodes=8, validation_episodes=4,
    )
    calls = []
    original = evaluator._returns

    def returns(game, profile, episodes, *stream):
        calls.append(stream)
        return original(game, profile, episodes, *stream)

    monkeypatch.setattr(evaluator, "_returns", returns)
    result = game.evaluate(fixed(0), evaluator=evaluator)
    assert [stream for stream in calls if stream[0] == "test"] == [("test", 0), ("test", 1)]
    assert result["response_returns"] == {0: 8, 1: -3}
    assert result["approximate_exploitability"] == 2.5
    assert result["raw_approximate_exploitability"] == 2.5
    assert "expected_returns" not in result
    assert "player_improvements" not in result
    assert "approximate_nash_gap" not in result


def test_rollout_cutoff_cannot_be_reported_as_zero_exploitability():
    game = load_game("kuhn_poker")
    game.max_episode_steps = 1
    evaluator = ApproximateExploitabilityEvaluator(response_profiles={p: fixed(0) for p in game.players})
    with pytest.raises(ValueError, match="terminate within max_episode_steps"):
        game.evaluate(fixed(0), evaluator=evaluator)


def test_rollouts_validate_batched_callables_that_are_not_policy_profiles():
    from benchmax import InvalidPolicyError
    from benchmax._rollout import sample_actions
    from benchmax.seeding import python_generator

    game = load_game("rock_paper_scissors")
    step = game.new_env().reset(seed=0)

    def negative_weights(player, observations, actions, mask):
        return np.tile([-1.0, 1.0, 1.0], (len(observations), 1))

    with pytest.raises(InvalidPolicyError, match="policy profile for player 0"):
        sample_actions(game, negative_weights, [step], [python_generator(0)])
