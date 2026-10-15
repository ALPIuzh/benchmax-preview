"""Core interfaces for BenchMAX games."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, overload

from benchmax.execution.jobs.submission import evaluation_job
from benchmax.execution.operation import executable
from benchmax.types import (
    Action,
    ActionSpace,
    Info,
    Observation,
    Observations,
    ObservationSpace,
    Player,
    Rewards,
)

if TYPE_CHECKING:
    import numpy as np

    from benchmax.evaluation import Evaluator
    from benchmax.execution.jobs.job import Job
    from benchmax.execution.jobs.policy_inputs import PolicyInput
    from benchmax.execution.targets.base import Target
    from benchmax.policy import PolicyProfileLike


@dataclass(frozen=True, slots=True)
class TimeStep:
    """Snapshot of one environment decision point.

    ``observations`` and ``legal_actions`` contain exactly the players in
    ``players_to_act``, and every acting player has at least one legal action.
    ``rewards`` contains exactly every player declared by the game and reports
    the immediate rewards from the transition that produced this snapshot.
    ``terminated`` and ``truncated`` are global episode flags, and ``info`` is
    one global dictionary.

    When either episode flag is true, ``observations``, ``legal_actions``, and
    ``players_to_act`` are empty. Chance events are resolved internally before
    a time step is returned.
    """

    observations: Observations
    rewards: Rewards
    players_to_act: tuple[Player, ...]
    legal_actions: dict[Player, tuple[Action, ...]]
    terminated: bool
    truncated: bool
    info: Info


class Game(ABC):
    """Episode-independent description and evaluation interface for a game.

    A game defines its players and their observation and action spaces. Mutable
    episode state lives in :class:`GameInstance` objects created by
    :meth:`new_env`, so separate instances can be used independently.
    :meth:`evaluate` evaluates a complete strategy profile using a composed
    evaluator. A game without one remains playable and trainable.

    Attributes:
        name: Stable name identifying the game.
        players: All valid player identifiers in the game.
        is_fully_observable: Whether every player can observe the full current
            game state. This does not imply that policy inputs retain history.
        is_simultaneous: Whether the game's dynamics include simultaneous
            choices. False means exclusively sequential decisions. Simultaneous
            games can still be fully observable before players choose actions.
        observation_spaces: Observation shape for every player.
        action_spaces: Full action space for every player. The actions legal at
            a particular decision point are provided by
            :attr:`TimeStep.legal_actions`.
        state_space: Optional shape of the global training state used by
            centralized critics. Policies never receive this state.
        max_episode_steps: Upper bound on episode length, used to bound training
            rollouts. For games that never terminate it is only a rollout budget.
        discount: Discount factor of the game's objective, or ``None`` when the
            objective is the undiscounted episodic return. A value here means
            the game is evaluated on discounted return, which is what makes an
            objective well defined for a game whose episodes never end.
        description: Human-readable account of the rules, observation encoding,
            and rewards. For configured games this comes from the game's
            configuration file rather than from a class docstring.
        provenance: Which library implements the game dynamics. ``"benchmax"``
            for games written here, and providers override it.
        evaluator: The composed evaluator, or ``None`` when this game is not
            currently evaluable. Evaluators are independent of the dynamics
            and do not contribute to the game's identity.
    """

    provenance: str = "benchmax"

    def __init__(
        self, *, name: str, players: tuple[Player, ...], observation_spaces: Mapping[Player, ObservationSpace],
        action_spaces: Mapping[Player, ActionSpace], is_fully_observable: bool, is_simultaneous: bool,
        state_space: ObservationSpace | None = None, max_episode_steps: int = 1000, discount: float | None = None,
        description: str = "", evaluator: Evaluator[Self] | None = None
    ) -> None:
        self.name = name
        self.players = players
        self.observation_spaces = dict(observation_spaces)
        self.action_spaces = dict(action_spaces)
        self.is_fully_observable = is_fully_observable
        self.is_simultaneous = is_simultaneous
        self.state_space = state_space
        self.max_episode_steps = max_episode_steps

        if discount is not None and not 0.0 <= discount < 1.0:
            raise ValueError(
                f"discount must lie in [0, 1); got {discount!r}. Use None for "
                "an undiscounted episodic objective rather than a discount of "
                "one, whose value does not converge when episodes never end"
            )

        self.discount = discount
        self.description = description
        self.evaluator = evaluator

    @abstractmethod
    def new_env(self) -> GameInstance:
        """Create a new, independent environment for this game.

        The returned instance does not share mutable episode state with other
        instances. It must be initialized with :meth:`GameInstance.reset`
        before :meth:`GameInstance.step` is called.
        """

        raise NotImplementedError

    def new_tree(self) -> GameTreeNode:
        """Return the root of an exact extensive-form game tree.

        Tree-based algorithms branch from this immutable view instead of the
        sampled, mutable environment returned by :meth:`new_env`. Games that
        support exact traversal override this method.
        """

        raise NotImplementedError(f"{self.name!r} does not expose an exact game tree")

    def rollout(
        self, policy_profile: PolicyProfileLike, *, episodes: int = 1, seed: int, batch_size: int = 32,
        max_steps: int | Sequence[int] | np.ndarray | None = None, collect_data: bool = False,
        require_complete: bool = False
    ) -> tuple[np.ndarray, dict[Player, dict[str, np.ndarray]]]:
        """Sample episodes, optionally collecting decision arrays for each player.

        Each episode stops at termination, environment truncation, or its step
        budget. Only acting players' observations and legal actions reach the
        policy.

        Args:
            policy_profile: A PolicyProfile or batched callable, as with
                :meth:`evaluate`.
            episodes: Number of episodes to sample. May be zero.
            seed: Root seed in ``[0, 2**31)``. Each episode derives independent
                environment and action RNG streams from it.
            batch_size: Positive number of episodes run together. For a fixed
                policy, it does not change the sampled decisions, only the row
                order of ``data``.
            max_steps: Step budget, ``self.max_episode_steps`` by default. A
                scalar applies to every episode, and a sequence of length
                ``episodes`` sets one positive budget per episode.
            collect_data: Whether to return each player's decision arrays.
                They are held in memory.
            require_complete: Raise if any episode is truncated or reaches its
                step budget.

        Returns:
            ``(returns, data)``. ``returns`` has shape
            ``(episodes, len(self.players))`` and sums each episode's sampled
            rewards in player order, without discounting or a bootstrap value.
            ``data`` is empty unless ``collect_data=True``. Then it maps each
            player to ``observations`` ``(N, *observation_spaces[player])``,
            ``actions`` ``(N,)``, a boolean ``legal_action_mask``
            ``(N, len(action_spaces[player]))``, and integer ``episode_ids`` and
            ``step_ids``. Rows follow collection order and contain only that
            player's turns.

        Raises:
            TypeError: If ``episodes``, ``batch_size``, ``max_steps``, or
                ``seed`` is not an integer.
            ValueError: If one of them is out of range, or if
                ``require_complete=True`` and an episode is incomplete.

        Example::

            returns, data = game.rollout(expert, episodes=1000, seed=7, collect_data=True)
            observations = data[0]["observations"]
            actions = data[0]["actions"]
        """
        from benchmax._rollout import rollout

        return rollout(
            self, policy_profile, episodes=episodes, seed=seed, batch_size=batch_size,
            max_steps=max_steps, collect_data=collect_data, require_complete=require_complete
        )

    @overload
    def evaluate(
        self, policy_profile: PolicyInput, *, evaluator: str | None = None, submit: Literal[False] = False,
        target: None = None, output: str | os.PathLike[str] | None = None
    ) -> dict[str, Any]: ...

    @overload
    def evaluate(
        self, policy_profile: PolicyInput, *, evaluator: Evaluator, submit: Literal[False] = False, target: None = None,
        output: None = None
    ) -> dict[str, Any]: ...

    @overload
    def evaluate(
        self, policy_profile: PolicyInput, *, evaluator: str | None = None, submit: Literal[True],
        target: str | Target | None = None, output: str | os.PathLike[str] | None = None
    ) -> Job[dict[str, Any]]: ...

    @overload
    def evaluate(
        self, policy_profile: PolicyInput, *, evaluator: Evaluator | str | None = None, submit: bool,
        target: str | Target | None = None, output: str | os.PathLike[str] | None = None
    ) -> dict[str, Any] | Job[dict[str, Any]]: ...

    @executable(job=evaluation_job)
    def evaluate(
        self, policy_profile: PolicyInput, *, evaluator: Evaluator | str | None = None,
        output: str | os.PathLike[str] | None = None
    ) -> dict[str, Any]:
        """Evaluate one fixed strategy profile with shared equilibrium metrics.

        Evaluation may query ``policy_profile`` at arbitrary reachable
        information states, including states that were not encountered during
        training, and may do so in any order. It must represent the same fixed
        strategy profile for the entire evaluation. Evaluation always supplies
        batches of observations and legal-action masks.

        A bare callable is always interpreted as a batched policy profile and
        is wrapped in a :class:`~benchmax.policy.PolicyProfile`. Scalar
        policies must first be converted with
        :meth:`~benchmax.policy.PolicyProfile.from_scalar_callable`.

        Args:
            policy_profile: Fixed batched strategies for every player, as a
                :class:`~benchmax.policy.PolicyProfile` or a batched callable.
                A safe policy artifact path is loaded before synchronous
                evaluation or pinned into a submitted job. A policy Reference
                resolves synchronously or records an explicit job dependency.
            evaluator: Registered evaluator name, or an evaluator object for a
                synchronous call without durable output. Either form overrides
                the game's configured default without modifying the game. A
                registered name is portable and can therefore be submitted.
            submit: True creates a job, and False runs synchronously.
            target: Execution target to submit to immediately in job mode.
            output: Optional durable evaluation result. In job mode this is
                the target's primary output path.

        Returns:
            A global metrics dictionary, or a job with ``submit=True``.
        """

        from benchmax.execution.jobs.policy_inputs import resolve_policy, validate_evaluation_input

        evaluator_name = validate_evaluation_input(policy_profile, evaluator=evaluator, portable=output is not None)

        if evaluator_name is not None:
            from benchmax.registry import _EVALUATORS

            selected_evaluator = _EVALUATORS.load(evaluator_name)
        else:
            selected_evaluator = evaluator if evaluator is not None else self.evaluator

        profile = resolve_policy(policy_profile, game=self)

        if selected_evaluator is None:
            raise NotImplementedError(f"{self.name!r} is playable but has no evaluator configured")

        if output is not None:
            from benchmax.execution.jobs.synchronous import evaluate_and_save

            return evaluate_and_save(self, profile, output, evaluator=selected_evaluator, evaluator_name=evaluator_name)

        return selected_evaluator.evaluate(self, profile)


def _validate_joint_actions(actions: Mapping[Player, Action], legal: Mapping[Player, Collection[Action]]) -> None:
    """Require exactly one legal action from each player in ``legal``."""

    expected = set(legal)
    if set(actions) != expected:
        raise ValueError(f"actions must contain exactly players {expected!r}")

    for player, action in actions.items():
        if action not in legal[player]:
            raise ValueError(f"illegal action {action} for player {player}")


class GameInstance(ABC):
    """Mutable environment for running independent episodes of a game.

    An instance holds at most one active episode and can be reused by calling
    :meth:`reset`, which starts a new episode independent of the previous one.
    A newly created instance is not ready for :meth:`step` until it has been
    reset.

    :meth:`reset` and :meth:`step` each return one :class:`TimeStep` containing
    everything needed to choose the next actions. Chance events are resolved
    internally by the environment and are never exposed as players that
    require actions.

    Attributes:
        game: The episode-independent game definition for this environment.
    """

    def __init__(self, game: Game) -> None:
        self.game = game

    def state(self) -> Observation:
        """Return the current global state for centralized training.

        Exposes a privileged, global view of the current environment
        for centralized-critic algorithms such as MAPPO. The state may contain
        privileged information and must never be supplied to an actor/policy.

        Returns:
            A global observation of the current environment state.
        """
        raise NotImplementedError(f"{self.game.name!r} does not expose global state")

    @abstractmethod
    def reset(self, seed: int | None = None) -> TimeStep:
        """Start a new, independent episode and return its first time step.

        Any current or completed episode state is discarded. ``seed``, when
        provided, controls the environment's randomness for the new episode,
        including internally resolved chance events. The returned time step
        describes the first player decision.

        For the built-in games, its rewards contain zero for every player,
        both episode flags are false, and ``info`` is empty.

        Args:
            seed: Optional seed for environment randomness. Environments that
                own a random stream require a seed on their first reset,
                and omitting it on later resets continues that stream.

        Returns:
            A snapshot of the initial decision point.
        """

        raise NotImplementedError

    @abstractmethod
    def step(self, actions: Mapping[Player, Action]) -> TimeStep:
        """Advance the active episode by one environment step.

        ``actions`` must contain exactly one legal action for every player in
        the preceding :class:`TimeStep`'s ``players_to_act``, with no missing or
        additional player keys. The environment resolves any subsequent chance
        events internally before it returns.

        The returned rewards are the immediate rewards from this transition.
        Its observations, acting players, and legal actions describe the next
        decision point.

        Args:
            actions: One action for each and only the players currently acting.

        Returns:
            A snapshot of the resulting decision point.

        Raises:
            RuntimeError: If :meth:`reset` has not been called, or the current
                episode has already ended.
            ValueError: If the action mapping has the wrong players or contains
                an illegal action.
        """

        raise NotImplementedError


class GameTreeNode(ABC):
    """One immutable node in an exact extensive-form game tree.

    Exactly one of three cases applies: ``returns`` is non-``None`` at a
    terminal node, ``chance_outcomes`` is nonempty at a chance node, and
    ``players_to_act`` is nonempty at a decision node. Multiple acting players
    represent a simultaneous decision.
    """

    @abstractmethod
    def returns(self) -> Rewards | None:
        """Return terminal utilities, or ``None`` before termination."""

        raise NotImplementedError

    @abstractmethod
    def chance_outcomes(self) -> tuple[tuple[Action, float], ...]:
        """Return every chance action and its probability at a chance node."""

        raise NotImplementedError

    @abstractmethod
    def players_to_act(self) -> tuple[Player, ...]:
        """Return the players acting at this decision node."""

        raise NotImplementedError

    @abstractmethod
    def observation(self, player: Player) -> Observation:
        """Return the acting player's policy input, matching the environment."""

        raise NotImplementedError

    def information_state_key(self, player: Player) -> Hashable:
        """Identify the player's perfect-recall information set within this tree.

        Defaults to the observation. Override when policy inputs omit history:
        a perfect-information board can be reached by different move sequences.
        Hidden information must never distinguish keys within an information set.
        """
        return self.observation(player)

    @abstractmethod
    def legal_actions(self, player: Player) -> tuple[Action, ...]:
        """Return the acting player's legal actions."""

        raise NotImplementedError

    @abstractmethod
    def chance_child(self, action: Action) -> GameTreeNode:
        """Return the child reached by one chance action."""

        raise NotImplementedError

    @abstractmethod
    def child(self, actions: Mapping[Player, Action]) -> GameTreeNode:
        """Return the child reached by one sequential or joint decision."""

        raise NotImplementedError
