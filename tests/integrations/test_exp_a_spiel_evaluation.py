import sys
from collections.abc import Mapping

import numpy as np
import pytest

from benchmax import PolicyProfile, load_game
from benchmax.evaluation import IncompatibleEvaluatorError
from benchmax.execution.worker import InProcessRuntime, PolicyProfileOracle, use_evaluation_runtime
from benchmax.integrations.exp_a_spiel import ExpASpielEvaluator
from benchmax.integrations.exp_a_spiel import kernel as exp_a_spiel_kernel
from benchmax.integrations.exp_a_spiel.kernel import evaluate_exact_with_oracle


class _FakeExactEvaluator:
    def __init__(self) -> None:
        self.rows = {0: 4099, 1: 4101}
        self.evaluated_strategies: tuple[np.ndarray, np.ndarray] | None = None

    def num_information_states(self, player: int) -> int:
        return self.rows[player]

    def policy_input_batch(self, player: int, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        row_ids = np.arange(start, stop)
        observations = np.column_stack((np.full(stop - start, player), row_ids)).astype(np.float64)
        legal_action_mask = np.zeros((stop - start, 9), dtype=bool)
        legal_action_mask[np.arange(stop - start), (row_ids + player) % 9] = True
        return observations, legal_action_mask

    def evaluate(self, strategy_0: np.ndarray, strategy_1: np.ndarray) -> tuple[np.float64, np.float64, np.float64]:
        self.evaluated_strategies = (strategy_0.copy(), strategy_1.copy())
        return np.float64(1.25), np.float64(0.75), np.float64(0.5)


def test_exact_helper_batches_policies_and_returns_portable_evidence() -> None:
    evaluator = _FakeExactEvaluator()
    action_spaces = {0: tuple(range(9)), 1: tuple(range(10, 19))}
    calls: list[tuple[int, np.ndarray, tuple[int, ...], np.ndarray]] = []

    def policy(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        calls.append((player, observations.copy(), action_space, legal_action_mask.copy()))
        return legal_action_mask.astype(np.float64)

    result = evaluate_exact_with_oracle(evaluator, PolicyProfileOracle(PolicyProfile(policy)), action_spaces)

    assert [(player, len(observations)) for player, observations, _, _ in calls] == [
        (0, 4096),
        (0, 3),
        (1, 4096),
        (1, 5),
    ]

    for player, observations, action_space, legal_action_mask in calls:
        row_ids = observations[:, 1].astype(int)
        assert np.all(observations[:, 0] == player)
        assert action_space is action_spaces[player]
        expected_mask = np.zeros_like(legal_action_mask)
        expected_mask[np.arange(len(row_ids)), (row_ids + player) % 9] = True
        np.testing.assert_array_equal(legal_action_mask, expected_mask)

    assert evaluator.evaluated_strategies is not None

    for player, strategy in enumerate(evaluator.evaluated_strategies):
        expected = np.zeros((evaluator.rows[player], 9), dtype=np.float64)
        row_ids = np.arange(evaluator.rows[player])
        expected[row_ids, (row_ids + player) % 9] = 1.0
        assert strategy.dtype == np.float64
        assert strategy.flags.c_contiguous
        np.testing.assert_array_equal(strategy, expected)

    assert result == {"expected_returns": {0: 1.25, 1: -1.25}, "player_improvements": {0: 0.5, 1: 0.75}}
    assert all(
        type(value) is float
        for metrics in result.values()
        for value in (metrics.values() if isinstance(metrics, Mapping) else (metrics,))
    )


@pytest.mark.parametrize(
    ("name", "exact_evaluator_name", "observation_size"),
    (
        ("phantom_tic_tac_toe", "PtttEvaluator", 108),
        ("abrupt_phantom_tic_tac_toe", "AbruptPtttEvaluator", 108),
        ("dark_hex_3x3", "DhEvaluator", 162),
        ("abrupt_dark_hex_3x3", "AbruptDhEvaluator", 162),
    ),
)
def test_game_rejects_scalar_adapter_before_selecting_exact_evaluator(
    monkeypatch: pytest.MonkeyPatch, name: str, exact_evaluator_name: str, observation_size: int
) -> None:
    constructed: list[str] = []

    def fake_evaluator(name: str) -> object:
        class FakeEvaluator:
            def __init__(self) -> None:
                constructed.append(name)

            def num_information_states(self, player: int) -> int:
                del player
                return 0

            def evaluate(self, strategy_0: np.ndarray, strategy_1: np.ndarray) -> tuple[float, float, float]:
                assert strategy_0.shape == (0, 9)
                assert strategy_1.shape == (0, 9)
                return 0.0, 0.0, 0.0

        return FakeEvaluator()

    monkeypatch.setattr(exp_a_spiel_kernel, "_prepared_traverser", fake_evaluator)

    game = load_game(name)
    assert game.observation_spaces == {0: (observation_size,), 1: (observation_size,)}

    def scalar_policy(player: int, observation: tuple[float, ...], legal_actions: tuple[int, ...]) -> dict[int, float]:
        del player, observation, legal_actions
        return {}

    with pytest.raises(TypeError) as error:
        game.evaluate(PolicyProfile.from_scalar_callable(scalar_policy))

    message = str(error.value)
    assert "exp-a-spiel evaluation" in message
    assert "PolicyProfile.from_scalar_callable()" in message
    assert "once per information state" in message
    assert "directly batched policy profile" in message
    assert constructed == []

    def batched_policy(
        player: int, observations: np.ndarray, action_space: tuple[int, ...], legal_action_mask: np.ndarray
    ) -> np.ndarray:
        del player, observations, action_space
        return legal_action_mask.astype(np.float64)

    with use_evaluation_runtime(InProcessRuntime()):
        game.evaluate(batched_policy)

    assert constructed == [exact_evaluator_name]


@pytest.mark.parametrize(
    ("game_name", "traverser", "requirement"),
    (
        ("kuhn_poker", "PtttEvaluator", "requires OpenSpiel game 'phantom_ttt'"),
        (
            "phantom_tic_tac_toe",
            "AbruptPtttEvaluator",
            r"gameversion='classical' \(requires 'abrupt'\)",
        ),
        (
            "abrupt_phantom_tic_tac_toe",
            "PtttEvaluator",
            r"gameversion='abrupt' \(requires 'classical'\)",
        ),
        (
            "dark_hex_3x3",
            "AbruptDhEvaluator",
            r"gameversion='cdh' \(requires 'adh'\)",
        ),
        (
            "abrupt_dark_hex_3x3",
            "DhEvaluator",
            r"gameversion='adh' \(requires 'cdh'\)",
        ),
    ),
)
def test_native_evaluator_rejects_wrong_game_semantics_when_evaluation_starts(
    game_name: str,
    traverser: str,
    requirement: str,
) -> None:
    native_module = "benchmax.integrations.exp_a_spiel._exp_a_spiel"
    existing_native_module = sys.modules.get(native_module)
    game = load_game(
        game_name,
        evaluator={"_use_": "exp_a_spiel", "traverser": traverser},
    )

    with pytest.raises(IncompatibleEvaluatorError, match=requirement):
        game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1])

    assert sys.modules.get(native_module) is existing_native_module


@pytest.mark.parametrize(
    ("params", "requirement"),
    (
        ({"obstype": "reveal-numturns"}, "obstype='reveal-numturns'"),
        ({"board_size": 4}, "board_size=4"),
    ),
)
def test_native_dark_hex_rejects_observation_and_board_variants(
    params: dict[str, str | int],
    requirement: str,
) -> None:
    game = load_game("openspiel.dark_hex", params=params)

    with pytest.raises(IncompatibleEvaluatorError, match=requirement):
        game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1])


def test_native_contract_rejects_unknown_and_unrepresentable_traversers() -> None:
    unknown = load_game(
        "dark_hex_3x3",
        evaluator={"_use_": "exp_a_spiel", "traverser": "NotATraverser"},
    )
    with pytest.raises(IncompatibleEvaluatorError, match="unknown exp-a-spiel"):
        unknown.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1])

    evaluator = ExpASpielEvaluator("CornerDhEvaluator")
    game = load_game("dark_hex_3x3", evaluator=None)
    game.evaluator = evaluator
    with pytest.raises(IncompatibleEvaluatorError, match="two occupied corners"):
        game.evaluate(lambda p, o, a, m: m.astype(float) / m.shape[1])
