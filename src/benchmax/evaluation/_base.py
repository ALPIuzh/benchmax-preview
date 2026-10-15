"""Core evaluator contracts and objective-compatibility checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from benchmax.core import Game
    from benchmax.policy import PolicyProfile


_GameT_contra = TypeVar("_GameT_contra", contravariant=True)


class IncompatibleEvaluatorError(TypeError):
    """Raised when an evaluator cannot serve the game passed to it.

    BenchMAX's evaluators raise this when evaluation starts if the game does
    not provide the semantics or data they require.
    """


@runtime_checkable
class Evaluator(Protocol[_GameT_contra]):
    """Contract implemented by evaluation backends.

    Evaluators are composed onto a game rather than inherited by it, so the
    same game can be evaluated by different methods and the same method can
    serve many games. Which evaluator a game uses is part of its configuration.

    An evaluator may be expensive to construct. Any such cost belongs inside
    the evaluator and must be deferred until :meth:`evaluate` is first called,
    so that merely loading a game stays cheap.

    Compatibility checks belong to the implementation of :meth:`evaluate`.
    """

    def evaluate(self, game: _GameT_contra, policy_profile: PolicyProfile) -> dict[str, Any]:
        """Return metrics for a fixed policy profile.

        Args:
            game: The game being evaluated. Evaluators may read whatever the
                game exposes. OpenSpiel evaluators require ``spiel_game``.
            policy_profile: Fixed batched strategies for every player.

        Returns:
            Metrics dictionary.
        """

        ...


def require_undiscounted(game: Game, evaluator: str) -> None:
    """Reject a discounted game for an evaluator that sums rewards undiscounted.

    Raises:
        IncompatibleEvaluatorError: If ``game`` declares a discount factor.
    """

    if game.discount is not None:
        raise IncompatibleEvaluatorError(
            f"{evaluator} computes undiscounted episodic returns, but "
            f"{game.name!r} declares discount={game.discount!r}; its "
            "equilibrium is defined by the discounted objective"
        )


def require_discounted(game: Game, evaluator: str) -> None:
    """Reject an undiscounted game for an evaluator that needs a discount factor.

    Raises:
        IncompatibleEvaluatorError: If ``game`` declares no discount factor.
    """

    if game.discount is None:
        raise IncompatibleEvaluatorError(
            f"{evaluator} computes discounted value, but {game.name!r} "
            "declares no discount factor; its objective is the undiscounted "
            "episodic return"
        )
