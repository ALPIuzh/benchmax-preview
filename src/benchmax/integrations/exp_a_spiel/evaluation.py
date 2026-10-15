"""Exact-evaluation bridge for the exp-a-spiel-compatible traversers."""

from typing import Any

from benchmax.core import Game
from benchmax.evaluation import IncompatibleEvaluatorError, require_undiscounted
from benchmax.evaluation.metrics import zero_sum_deviation_metrics
from benchmax.execution.worker import PolicyProfileOracle
from benchmax.integrations.exp_a_spiel.kernel import _traverser_type, describe_traverser_kernel, evaluate_exact_kernel
from benchmax.integrations.openspiel.game import OpenSpielGame
from benchmax.policy import PolicyProfile, _ScalarAdaptedPolicyProfile


class ExpASpielEvaluator:
    """Exact exploitability via an exp-a-spiel-compatible native traverser.

    Handles Liar's dice and the phantom board games, whose trees are too large for
    OpenSpiel's ``nash_conv``. Exact evaluation retains its traverser in an
    isolated worker. Direct CFR may separately request an in-process traverser
    through :meth:`payoff_operator`.
    """

    def __init__(self, traverser: str) -> None:
        """Record which worker-local traverser to prepare on first evaluation.

        Args:
            traverser: Name of a zero-argument traverser class in the native
                binding, such as ``"DhEvaluator"``.
        """

        self._traverser_name = traverser
        self._traverser: Any | None = None
        self._traverser_contract: dict[str, Any] | None = None

    def _validate_game(self, game: Game) -> None:
        """Check the game against semantics declared by the native traverser."""

        if not isinstance(game, OpenSpielGame):
            raise IncompatibleEvaluatorError(
                "exp-a-spiel requires an OpenSpielGame with the matching "
                f"observation encoding; got {game.name!r} ({type(game).__name__})"
            )
        require_undiscounted(game, "exp_a_spiel")

        contract = self._contract()
        compatible_game = contract["compatible_game"]
        expected_name = compatible_game["spiel_name"]
        if expected_name is None:
            reason = compatible_game.get("reason", "no compatible OpenSpiel game")
            raise IncompatibleEvaluatorError(f"{self._traverser_name} cannot evaluate an OpenSpielGame: {reason}")

        actual_name = game.spiel_game.get_type().short_name
        if actual_name != expected_name:
            raise IncompatibleEvaluatorError(
                f"{self._traverser_name} requires OpenSpiel game {expected_name!r}; {game.name!r} loads {actual_name!r}"
            )

        expected_parameters = compatible_game["parameters"]
        actual_parameters = game.spiel_game.get_parameters()
        mismatches = {
            name: (expected, actual_parameters.get(name))
            for name, expected in expected_parameters.items()
            if actual_parameters.get(name) != expected
        }
        if actual_name == "dark_hex" and all(
            actual_parameters.get(dimension) == expected_parameters[dimension] for dimension in ("num_rows", "num_cols")
        ):
            # OpenSpiel resolves the dimensions using board_size only as a fallback.
            mismatches.pop("board_size", None)
        if mismatches:
            details = ", ".join(
                f"{name}={actual!r} (requires {expected!r})" for name, (expected, actual) in sorted(mismatches.items())
            )
            raise IncompatibleEvaluatorError(f"{self._traverser_name} is not compatible with {game.name!r}: {details}")

        expected_actions = tuple(range(contract["num_actions"]))
        if game.players != (0, 1) or any(game.action_spaces[player] != expected_actions for player in game.players):
            raise IncompatibleEvaluatorError(
                "exp-a-spiel traversers currently require players (0, 1) "
                f"with actions {expected_actions!r}; {game.name!r} exposes "
                f"players={game.players!r} and action spaces={game.action_spaces!r}"
            )

        information_state_size = contract["information_state_size"]
        expected_observation_space = (information_state_size,)
        if any(game.observation_spaces[player] != expected_observation_space for player in game.players):
            raise IncompatibleEvaluatorError(
                f"{self._traverser_name} emits {information_state_size}-value "
                f"information states, but {game.name!r} exposes "
                f"{game.observation_spaces!r}"
            )

    def evaluate(self, game: Game, policy_profile: PolicyProfile) -> dict[str, Any]:
        """Compute exact metrics in the isolated evaluation runtime.

        Raises:
            IncompatibleEvaluatorError: If the game does not match the native
                traverser's declared semantics.
            TypeError: If ``policy_profile`` came from
                :meth:`PolicyProfile.from_scalar_callable`, which exact
                traversal cannot afford.
        """

        self._validate_game(game)
        _reject_scalar_adapted_policy_profile(policy_profile)
        evidence = evaluate_exact_kernel(self._traverser_name, PolicyProfileOracle(policy_profile), game.action_spaces)
        return zero_sum_deviation_metrics(
            evidence["player_improvements"], expected_returns=evidence["expected_returns"]
        )

    def payoff_operator(self, game: Game) -> Any:
        """Return the compact payoff operator when it matches the game."""

        if not self._supports(game):
            return None

        if not hasattr(_traverser_type(self._traverser_name), "cfr_treeplex"):
            return None

        from benchmax.integrations.exp_a_spiel.payoff import (
            _ExpASpielPayoffOperator,
        )

        return _ExpASpielPayoffOperator(game, self._get_traverser())

    def _supports(self, game: Game) -> bool:
        expected = _traverser_type(self._traverser_name).compatible_game
        spiel_game = getattr(game, "spiel_game", None)
        return bool(
            isinstance(expected, dict)
            and expected.get("spiel_name") is not None
            and getattr(game, "spiel_name", None) == expected["spiel_name"]
            and spiel_game is not None
            and spiel_game.get_parameters() == expected["parameters"]
        )

    def _get_traverser(self) -> Any:
        """Import the native binding for direct CFR and construct its traverser.

        Exact evaluation uses the isolated worker instead. Direct CFR is a
        native, non-Torch path and deliberately loads the binding in the caller,
        so it must not be mixed with BenchMARL training in the same process.
        """
        if self._traverser is None:
            self._traverser = _traverser_type(self._traverser_name)()

        return self._traverser

    def _contract(self) -> dict[str, Any]:
        """Load registered class metadata without constructing the traverser."""

        if self._traverser_contract is None:
            contract = describe_traverser_kernel(self._traverser_name)
            error = contract.get("error")
            if isinstance(error, str):
                raise IncompatibleEvaluatorError(error)
            compatible_game = contract.get("compatible_game")
            information_state_size = contract.get("information_state_size")
            num_actions = contract.get("num_actions")
            if (
                not isinstance(compatible_game, dict)
                or not isinstance(information_state_size, int)
                or not isinstance(num_actions, int)
            ):
                raise RuntimeError(f"invalid compatibility metadata for {self._traverser_name!r}")
            self._traverser_contract = {
                "compatible_game": compatible_game,
                "information_state_size": information_state_size,
                "num_actions": num_actions,
            }
        return self._traverser_contract


def _reject_scalar_adapted_policy_profile(policy_profile: PolicyProfile) -> None:
    """Reject the row-wise scalar adapter, too slow for exact traversal."""

    if isinstance(policy_profile, _ScalarAdaptedPolicyProfile):
        raise TypeError(
            "exp-a-spiel evaluation requires a directly batched policy "
            "profile; policy profiles returned by "
            "PolicyProfile.from_scalar_callable() are not supported because "
            "they invoke the scalar policy once per information state"
        )
