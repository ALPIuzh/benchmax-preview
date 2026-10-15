"""Exploitability evaluation for OpenSpiel-backed games."""

import contextlib
import io
from collections.abc import Callable
from typing import Any

with contextlib.redirect_stdout(io.StringIO()):
    import pyspiel
    from open_spiel.python import policy as open_spiel_policy
    from open_spiel.python.algorithms import expected_game_score, exploitability

from benchmax.core import Game
from benchmax.evaluation import (
    IncompatibleEvaluatorError,
    require_undiscounted,
)
from benchmax.evaluation.metrics import zero_sum_deviation_metrics
from benchmax.integrations.openspiel.game import OpenSpielGame
from benchmax.policy import (
    PolicyProfile,
    ScalarPolicyProfile,
    _validate_scalar_distribution,
)
from benchmax.types import Observation, Player


class OpenSpielNashConvEvaluator:
    """Exact exploitability via OpenSpiel's ``nash_conv``.

    Applies to any two-player zero-sum OpenSpiel game with perfect recall whose
    tree is small enough to traverse exhaustively. The policy profile is
    materialized once per player over every reachable information state, so it
    is queried in batches rather than one state at a time.
    """

    def _validate_game(self, game: Game) -> None:
        """Check value-level assumptions of OpenSpiel's Nash convergence."""

        if not isinstance(game, OpenSpielGame):
            raise IncompatibleEvaluatorError(
                f"nash_conv requires an OpenSpielGame; got {game.name!r} ({type(game).__name__})"
            )

        require_undiscounted(game, "nash_conv")
        game_type = game.spiel_game.get_type()

        if game.spiel_game.num_players() != 2:
            raise IncompatibleEvaluatorError(
                f"nash_conv currently serves two-player games; {game.name!r} "
                f"has {game.spiel_game.num_players()} players"
            )

        if game_type.utility != pyspiel.GameType.Utility.ZERO_SUM:
            raise IncompatibleEvaluatorError(
                f"nash_conv reports zero-sum exploitability, but {game.name!r} "
                f"declares {game_type.utility.name.lower()} utility"
            )

        if game.is_simultaneous:
            raise IncompatibleEvaluatorError(
                "nash_conv currently requires sequential OpenSpiel dynamics; "
                f"{game.name!r} declares {game_type.dynamics.name.lower()} dynamics"
            )

        if not game_type.provides_information_state_string:
            raise IncompatibleEvaluatorError(
                f"nash_conv requires information-state strings, which {game.name!r} does not provide"
            )

    def evaluate(self, game: OpenSpielGame, policy_profile: PolicyProfile) -> dict[str, Any]:
        """Compute exact deviation evidence for a fixed policy profile.

        Args:
            game: An OpenSpiel-backed game exposing ``spiel_game``.
            policy_profile: Fixed strategies for both players, expressed as
                dense batches over the supplied action space.

        Returns:
            Each player's unilateral best-response improvement in the
            canonical two-player zero-sum result contract.

        Raises:
            IncompatibleEvaluatorError: If the game is unsuitable.
            InvalidPolicyError: If the policy profile returns an invalid action
                distribution at any reachable information state.
        """

        self._validate_game(game)
        tabular_policy = policy_profile.tabulate(_collect_reachable_queries(game), game.action_spaces)
        adapter = _BenchmaxToOpenSpielPolicyAdapter(game.spiel_game, tabular_policy, game._encode_observation)
        result = exploitability.nash_conv(game.spiel_game, adapter, return_only_nash_conv=False, use_cpp_br=False)
        improvements = {player: float(result.player_improvements[player]) for player in game.players}
        expected_returns = expected_game_score.policy_value(
            game.spiel_game.new_initial_state(), [adapter for _ in game.players]
        )
        return zero_sum_deviation_metrics(
            improvements, expected_returns={player: float(expected_returns[player]) for player in game.players}
        )


def _collect_reachable_queries(game: OpenSpielGame) -> dict[Player, tuple[tuple[Observation, tuple[int, ...]], ...]]:
    """Enumerate every reachable information state, per player.

    Walks the complete game tree, resolving chance nodes by visiting all of
    their outcomes, and deduplicates the ``(observation, legal_actions)`` pairs
    each player can face. The result is what
    :meth:`benchmax.policy.PolicyProfile.tabulate` needs to materialize a
    batched profile for an evaluator that queries one state at a time.
    """

    queries: dict[Player, dict[tuple[Observation, tuple[int, ...]], None]] = {player: {} for player in game.players}

    def collect(state: Any) -> None:
        if state.is_terminal():
            return

        if state.is_chance_node():
            actions = tuple(int(action) for action, _ in state.chance_outcomes())
        else:
            player = int(state.current_player())
            legal_actions = tuple(int(action) for action in state.legal_actions(player))
            observation = game._encode_observation(state, player)
            queries[player][(observation, legal_actions)] = None
            actions = legal_actions

        for action in actions:
            collect(state.child(action))

    collect(game.spiel_game.new_initial_state())

    return {player: tuple(player_queries) for player, player_queries in queries.items()}


class _BenchmaxToOpenSpielPolicyAdapter(open_spiel_policy.Policy):  # type: ignore[misc]
    """Expose a scalar BenchMAX policy through OpenSpiel's policy interface."""

    def __init__(
        self, game: Any, policy_profile: ScalarPolicyProfile, encode_observation: Callable[[Any, Player], Observation]
    ) -> None:
        super().__init__(game, list(range(game.num_players())))
        self._policy_profile = policy_profile
        self._encode_observation = encode_observation

    def action_probabilities(self, state: Any, player_id: int | None = None) -> dict[int, float]:
        if state.is_terminal():
            return {}

        if state.is_chance_node():
            raise ValueError("a policy profile cannot act at a chance node")

        if state.is_simultaneous_node():
            if player_id is None:
                raise ValueError("player_id is required when querying a simultaneous node")

            player = int(player_id)
            if player not in self.player_ids:
                raise ValueError(
                    f"invalid player_id {player_id} at a simultaneous node; expected one of {self.player_ids}"
                )
        else:
            player = int(state.current_player())
            if player_id is not None and int(player_id) != player:
                raise ValueError(f"policy profile queried for player {player_id} while player {player} acts")

        legal_actions = tuple(int(action) for action in state.legal_actions(player))
        observation = self._encode_observation(state, player)
        return _validate_scalar_distribution(
            self._policy_profile(player, observation, legal_actions), legal_actions,
            context=f"policy profile for player {player}"
        )
