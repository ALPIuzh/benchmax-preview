"""Exact exploitability for zero-sum tabular Markov games.

The profile value is a linear solve and each best response is value iteration,
so the result is exact up to the solver's tolerance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmax.evaluation._base import IncompatibleEvaluatorError, require_discounted
from benchmax.evaluation.metrics import zero_sum_deviation_metrics
from benchmax.games.tabular.game import TabularZeroSumMarkovGame
from benchmax.games.tabular.solver import best_response_value, profile_value

if TYPE_CHECKING:
    from benchmax.core import Game
    from benchmax.policy import PolicyProfile


class DiscountedNashGapEvaluator:
    """Exact discounted exploitability for a zero-sum tabular Markov game.

    Each player's strategy is queried once, as a single batch covering every
    state. The profile value is computed by a linear solve and each best
    response by value iteration. Runtime and memory grow with the state and
    action spaces, and best responses are subject to the solver's tolerances.

    Two conventions for the same quantity are reported. ``nash_gap`` is the
    larger of the two players' improvements, which is the Nash gap as the
    Markov-game literature states it. ``nash_conv`` is their sum and
    ``exploitability`` its half, matching what every other BenchMAX evaluator
    returns.
    """

    def _validate_game(self, game: Game) -> None:
        """Reject games this evaluator cannot measure.

        Raises:
            IncompatibleEvaluatorError: If the game is not a zero-sum tabular
                Markov game, or declares no discount factor.
        """

        if not isinstance(game, TabularZeroSumMarkovGame):
            raise IncompatibleEvaluatorError(
                "discounted_nash_gap evaluates zero-sum tabular Markov games, "
                f"which {game.name!r} ({type(game).__name__}) is not; it needs "
                "an explicit transition tensor and a single Nash value"
            )

        require_discounted(game, "discounted_nash_gap")

    def evaluate(self, game: TabularZeroSumMarkovGame, policy_profile: PolicyProfile) -> dict[str, Any]:
        """Compute exact deviation evidence for a fixed policy profile.

        Args:
            game: A zero-sum tabular Markov game.
            policy_profile: Fixed strategies for both players.

        Returns:
            The fixed profile's returns and each player's unilateral
            best-response improvement.

        Raises:
            IncompatibleEvaluatorError: If the game is unsuitable.
            InvalidPolicyError: If either player's distributions are invalid.
        """

        self._validate_game(game)
        policies = game._policy_tables(policy_profile)
        policy_0, policy_1 = policies[0], policies[1]

        value = profile_value(
            game.transitions, game.payoffs, policy_0, policy_1, game.initial_state_distribution, game.discount
        )

        # Player 0 maximizes its own payoff against a fixed opponent; player 1
        # maximizes the negated payoff, so both reduce to the same MDP solve.
        rewards_0 = (game.payoffs * policy_1[:, None, :]).sum(axis=2)
        transitions_0 = (game.transitions * policy_1[:, None, :, None]).sum(axis=2)
        rewards_1 = -(game.payoffs * policy_0[:, :, None]).sum(axis=1)
        transitions_1 = (game.transitions * policy_0[:, :, None, None]).sum(axis=1)

        best_0 = best_response_value(rewards_0, transitions_0, game.initial_state_distribution, game.discount)
        best_1 = best_response_value(rewards_1, transitions_1, game.initial_state_distribution, game.discount)

        improvements = {0: best_0 - value, 1: best_1 + value}
        return zero_sum_deviation_metrics(improvements, expected_returns={0: value, 1: -value})
