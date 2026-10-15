"""Essential numerical and integration checks, with native code isolated."""

import pytest

from tests.process_helpers import assert_python_succeeds


def _compare_openspiel(deterministic):
    import numpy as np
    import pyspiel
    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import best_response, expected_game_score

    from benchmax import PolicyProfile
    from benchmax.execution.worker import PolicyProfileOracle
    from benchmax.integrations.exp_a_spiel import _exp_a_spiel
    from benchmax.integrations.exp_a_spiel.kernel import evaluate_exact_with_oracle

    # Two dice exercise the unequal probabilities of sorted private hands.
    game = pyspiel.load_game("liars_dice", {"numdice": 2, "dice_sides": 2})
    actions = tuple(range(game.num_distinct_actions()))

    def probabilities(player, observations, actions, mask):
        # Depend on private dice and the full bid history, with different seats.
        codes = observations @ np.arange(1, observations.shape[1] + 1)
        weights = (1 + (codes[:, None] * (np.arange(len(actions)) + 3) + 7 * player) % 19) * mask
        if deterministic:
            chosen = weights.argmax(axis=1)
            weights[:] = 0
            weights[np.arange(len(mask)), chosen] = 1
        return weights / weights.sum(axis=1, keepdims=True)

    class ReferencePolicy(policy_lib.Policy):
        def action_probabilities(self, state, player_id=None):
            player = state.current_player()
            obs = np.array([state.information_state_tensor(player)])
            legal = state.legal_actions()
            mask = np.zeros((1, len(actions)), dtype=bool)
            mask[0, legal] = True
            probs = probabilities(player, obs, actions, mask)[0]
            return {a: probs[a] for a in legal}

    reference = ReferencePolicy(game, [0, 1])
    result = evaluate_exact_with_oracle(
        _exp_a_spiel.Ld2d2fEvaluator(), PolicyProfileOracle(PolicyProfile(probabilities)), {0: actions, 1: actions}
    )
    root = game.new_initial_state()
    expected = expected_game_score.policy_value(root, [reference, reference])
    for player in (0, 1):
        br = best_response.BestResponsePolicy(game, player, reference).value(root)
        assert result["expected_returns"][player] == pytest.approx(expected[player], abs=1e-12, rel=0)
        assert result["player_improvements"][player] == pytest.approx(br - expected[player], abs=1e-12, rel=0)


@pytest.mark.parametrize("deterministic", [False, True], ids=["mixed", "zero-reach-branches"])
def test_expected_value_and_both_best_responses_match_openspiel(deterministic):
    assert_python_succeeds(
        "from tests.integrations.test_liars_dice_evaluation import _compare_openspiel\n"
        f"_compare_openspiel({deterministic!r})"
    )


def test_policy_inputs_match_openspiel():
    assert_python_succeeds(
        """
        import numpy as np
        import pyspiel
        from benchmax.integrations.exp_a_spiel import _exp_a_spiel

        native = _exp_a_spiel.Ld2d2fEvaluator()
        game = pyspiel.load_game('liars_dice', {'numdice': 2, 'dice_sides': 2})
        for player in (0, 1):
            rows = native.num_information_states(player)
            for start in range(0, rows, 17):
                obs, mask = native.policy_input_batch(player, start, min(start + 17, rows))
                for observation, legal in zip(obs, mask):
                    dice = observation[2:6].reshape(2, 2).argmax(axis=1)
                    rolls = list(dice) + [0, 0] if player == 0 else [0, 0] + list(dice)
                    state = game.new_initial_state()
                    for roll in rolls:
                        state.apply_action(int(roll))
                    for bid in np.flatnonzero(observation[6:-1]):
                        state.apply_action(int(bid))
                    assert state.current_player() == player
                    np.testing.assert_array_equal(state.information_state_tensor(player), observation)
                    assert state.legal_actions() == list(np.flatnonzero(legal))
        """
    )


def test_packed_strategy_validation():
    assert_python_succeeds(
        """
        import numpy as np
        import pytest
        from benchmax.integrations.exp_a_spiel import _exp_a_spiel

        native = _exp_a_spiel.Ld2d2fEvaluator()
        strategies = []
        for player in (0, 1):
            strategy = native.empty_strategy(player)
            rows = native.num_information_states(player)
            for start in range(0, rows, 17):
                _, mask = native.policy_input_batch(player, start, min(start + 17, rows))
                native.pack_policy_batch(player, start, mask / mask.sum(axis=1, keepdims=True), strategy)
            strategies.append(strategy)
        assert np.isfinite(native.evaluate(*strategies)).all()
        with pytest.raises(ValueError, match='shape'):
            native.evaluate(strategies[0][:-1], strategies[1])
        strategies[0][0] = np.nan
        with pytest.raises(ValueError, match='valid strategy'):
            native.evaluate(*strategies)
        illegal = np.zeros((1, 9))
        illegal[0, -1] = 1 # Calling at the root is forbidden.
        with pytest.raises(ValueError, match='valid strategy'):
            native.pack_policy_batch(0, 0, illegal, strategies[0])
        """
    )


def test_rejects_incompatible_bidding_rule():
    from benchmax import load_game
    from benchmax.evaluation import IncompatibleEvaluatorError

    game = load_game("liars_dice_2d5f", params={"bidding_rule": "reset-quantity"})
    with pytest.raises(IncompatibleEvaluatorError, match="bidding_rule"):
        game.evaluate(lambda p, o, a, m: m / m.sum(axis=1, keepdims=True))


def test_full_game_uniform_evaluation():
    assert_python_succeeds(
        """
        import pytest
        from benchmax import load_game
        from benchmax.execution.worker import close_default_evaluation_runtime

        game = load_game('liars_dice_2d5f')
        result = game.evaluate(lambda p, o, a, m: m / m.sum(axis=1, keepdims=True))
        # Baseline independently validated using exact-rational binomial dice
        # probabilities and a last-bid DP (uniform bids reveal no private dice).
        assert result['expected_returns'] == pytest.approx({0: -0.014, 1: 0.014}, abs=1e-12, rel=0)
        assert result['player_improvements'] == pytest.approx(
            {0: 0.9083142187408474, 1: 0.8728810429166719}, abs=1e-12, rel=0,
        )
        assert result['exploitability'] == pytest.approx(0.8905976308287596, abs=1e-12, rel=0)
        close_default_evaluation_runtime()
        """,
        timeout=120,
    )
