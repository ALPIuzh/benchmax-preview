"""Exact unilateral-deviation metrics for finite matrix games."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from benchmax.evaluation._base import IncompatibleEvaluatorError, require_undiscounted
from benchmax.evaluation.metrics import deviation_metrics, zero_sum_deviation_metrics
from benchmax.games.matrix.game import MatrixGame, TwoPlayerZeroSumMatrixGame

if TYPE_CHECKING:
    from benchmax.core import Game
    from benchmax.policy import PolicyProfile
    from benchmax.types import Player


class MatrixEvaluator:
    """Exact unilateral-deviation metrics for any finite matrix game."""

    def _validate_game(self, game: Game) -> None:
        """Reject games that do not expose finite matrix payoffs."""

        if not isinstance(game, MatrixGame):
            raise IncompatibleEvaluatorError(
                f"matrix evaluator requires a finite matrix game; got {game.name!r} ({type(game).__name__})"
            )

        require_undiscounted(game, "matrix")

    def evaluate(self, game: MatrixGame, policy_profile: PolicyProfile) -> dict[str, Any]:
        """Compute exact unilateral improvements for a fixed strategy profile.

        Raises:
            IncompatibleEvaluatorError: If the game is not a finite,
                undiscounted matrix game.
        """

        self._validate_game(game)
        policies = _query_policy(game, policy_profile)
        expected_returns: dict[Player, float] = {}
        improvements: dict[Player, float] = {}

        for player_index, player in enumerate(game.players):
            action_values = _action_values(game, player_index, policies)
            expected_return = float(action_values @ policies[player])
            expected_returns[player] = expected_return
            improvements[player] = max(0.0, float(np.max(action_values)) - expected_return)

        if isinstance(game, TwoPlayerZeroSumMatrixGame):
            return zero_sum_deviation_metrics(improvements, expected_returns=expected_returns)

        return deviation_metrics(improvements, expected_returns=expected_returns)


def _action_values(game: MatrixGame, player_index: int, policies: dict[Player, np.ndarray]) -> np.ndarray:
    """Return one player's payoff for each pure action against its opponents."""

    player = game.players[player_index]
    values = game.rewards[player]

    # Contract higher axes first so each remaining original axis keeps its
    # index until it is visited. The selected player's action axis is retained.
    for opponent_index in range(len(game.players) - 1, -1, -1):
        if opponent_index == player_index:
            continue

        opponent = game.players[opponent_index]
        values = np.tensordot(values, policies[opponent], axes=(opponent_index, 0))

    return np.asarray(values, dtype=float)


def _query_policy(game: MatrixGame, policy_profile: PolicyProfile) -> dict[Player, np.ndarray]:
    observations: np.ndarray = np.zeros((1, 1), dtype=float)
    policies: dict[Player, np.ndarray] = {}

    for player in game.players:
        action_space = game.action_spaces[player]
        legal_action_mask: np.ndarray = np.ones((1, len(action_space)), dtype=bool)
        policies[player] = policy_profile(player, observations, action_space, legal_action_mask)[0]

    return policies
