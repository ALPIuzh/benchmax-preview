"""BenchMAX works with batched policies by default so models can process multiple observations in one call. A
``PolicyProfile`` is a batched joint policy: it supplies a strategy for every player and returns action probabilities
for a batch of observations for the requested player.

Accepted policy types
---------------------

``PolicyProfileLike`` is the type alias ``PolicyProfile | BatchedCallable``: either a wrapped profile or a plain
function with the signature ``policy(player, observations, action_space, legal_action_mask)``. It returns an array of
probabilities shaped like the mask, with each row summing to one and illegal actions assigned zero probability. See the
:doc:`policy interface </getting-started/quickstart>` for the arguments.

``ScalarPolicyProfile`` instead handles one observation at a time: ``policy(player, observation, legal_actions)``
returns a mapping from legal actions to probabilities. Convert it with ``PolicyProfile.from_scalar_callable(policy)``
before evaluation.

Creating and loading policies
-----------------------------

- **Training:** :doc:`training algorithms </reference/training>` return policy profiles.
- **User-defined:** pass a batched callable, wrap it with ``PolicyProfile(...)``, or build a tabular profile with
  ``PolicyProfile.from_table(...)``.
- **Pretrained:** use :doc:`load_expert(game) </reference/experts>`.
- **Storage:** use :doc:`store.load_policy(...) </reference/storage>` or load a policy reference with
  ``reference.result()``.
- **Local files:** load a saved BenchMAX artifact with :doc:`Reference.from_path(path).result()
  </reference/references>`. For another checkpoint format, load the model with its own library and adapt it to the
  callable interface.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from numbers import Number
from os import PathLike
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np

from benchmax.types import Action, ActionSpace, Observation, Player

if TYPE_CHECKING:
    from benchmax.core import Game
    from benchmax.execution.jobs.reference import Reference
    from benchmax.execution.specs import EntrySpec, TrainingSpec, _ArtifactMetadata

_PROBABILITY_TOLERANCE = 1e-6


class InvalidPolicyError(ValueError):
    """Raised when a policy profile violates the BenchMAX policy contract."""


class ScalarPolicyProfile(Protocol):
    """Scalar callable strategy profile containing every player's strategy.

    A policy profile must represent one fixed strategy profile for an entire
    evaluation. Repeated equivalent queries during that evaluation must return
    equivalent probability distributions. Scalar policy profiles receive one
    immutable, flat observation and one ordered, nonempty sequence of legal
    action IDs at a time.

    A callable of this shape is not accepted by :meth:`Game.evaluate` directly,
    because it cannot be told apart from a batched callable without inspecting
    its signature. Convert it explicitly with
    :meth:`PolicyProfile.from_scalar_callable`.
    """

    def __call__(
        self, player: Player, observation: Observation, legal_actions: Sequence[Action], /
    ) -> Mapping[Action, float]:
        """Return the requested player's distribution over legal actions.

        The returned mapping must contain every legal action exactly once,
        including legal actions assigned zero probability. It must not contain
        illegal actions.

        Args:
            player: One Python integer identifying the queried player.
            observation: One immutable, flat observation tuple containing only
                information available to that player.
            legal_actions: Ordered, nonempty sequence of the actual action IDs
                legal at this information state, without duplicates.

        Returns:
            A probability mapping containing every legal action exactly once
            and no illegal actions.
        """

        ...


def _validate_scalar_distribution(
    probabilities: Mapping[Action, float], legal_actions: Sequence[Action], *, context: str
) -> dict[Action, float]:
    """Validate and copy a probability distribution over legal actions.

    The returned dictionary contains the legal actions in their supplied order
    and converts each numeric probability to ``float``.

    Args:
        probabilities: Candidate mapping from actions to probabilities.
        legal_actions: Nonempty sequence of unique legal actions.
        context: Description included in every validation error.

    Raises:
        InvalidPolicyError: If either the legal actions or candidate
            distribution violates the policy contract.
    """

    if not isinstance(probabilities, Mapping):
        raise InvalidPolicyError(f"{context} must return a mapping")

    legal = tuple(legal_actions)
    if not legal:
        raise InvalidPolicyError(f"{context} received an empty legal-action sequence")

    legal_set = set(legal)

    if len(legal_set) != len(legal):
        duplicates = sorted(
            action for action in legal_set if legal.count(action) > 1
        )
        raise InvalidPolicyError(
            f"{context} received duplicate legal actions: {duplicates!r}"
        )

    supplied_actions = set(probabilities)
    missing = sorted(legal_set - supplied_actions)
    illegal = sorted(supplied_actions - legal_set)

    if missing or illegal:
        problems = []

        if missing:
            problems.append(f"omitted legal actions: {missing!r}")

        if illegal:
            problems.append(f"returned illegal actions: {illegal!r}")

        raise InvalidPolicyError(f"{context} {'; '.join(problems)}")

    distribution: dict[Action, float] = {}

    for action in legal:
        value = probabilities[action]
        if not isinstance(value, Number):
            raise InvalidPolicyError(
                f"{context} returned a non-numeric probability "
                f"for action {action!r}: {value!r}"
            )

        try:
            probability = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidPolicyError(
                f"{context} returned a non-numeric probability "
                f"for action {action!r}: {value!r}"
            ) from error

        if not math.isfinite(probability):
            raise InvalidPolicyError(
                f"{context} returned a non-finite probability "
                f"for action {action!r}: {probability!r}"
            )

        if probability < 0.0:
            raise InvalidPolicyError(
                f"{context} returned a negative probability "
                f"for action {action!r}: {probability!r}"
            )

        distribution[action] = probability

    try:
        total = math.fsum(distribution.values())
    except OverflowError:
        total = math.inf

    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE):
        raise InvalidPolicyError(
            f"{context} probabilities must sum to one; got total {total!r}"
        )

    return distribution


def _validate_batched_distribution(
    probabilities: object, action_space: ActionSpace, legal_action_mask: np.ndarray, *, context: str
) -> np.ndarray:
    """Validate a dense batch of probability distributions.

    Illegal-action probabilities are required to be exactly zero. Row sums use
    the same absolute tolerance as scalar distributions. The returned array is
    floating point, but the supplied policy output and mask are never mutated.

    Args:
        probabilities: Candidate NumPy-compatible numeric probability matrix.
        action_space: Ordered action IDs corresponding to matrix columns.
        legal_action_mask: Boolean matrix identifying each row's legal actions.
        context: Description included in every validation error.

    Returns:
        The validated probability matrix as a floating-point NumPy array.

    Raises:
        InvalidPolicyError: If the candidate matrix violates the batched policy
            contract.
    """

    mask = np.asarray(legal_action_mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.bool_):
        raise InvalidPolicyError(
            f"{context} received a legal-action mask that is not a "
            "two-dimensional boolean array"
        )

    if mask.shape[1] != len(action_space):
        raise InvalidPolicyError(
            f"{context} received legal-action mask shape {mask.shape!r}; "
            f"expected {len(action_space)} action columns"
        )

    try:
        candidate = np.asarray(probabilities)
    except (TypeError, ValueError) as error:
        raise InvalidPolicyError(
            f"{context} must return a two-dimensional numeric array"
        ) from error

    if candidate.ndim != 2:
        raise InvalidPolicyError(
            f"{context} must return a two-dimensional numeric array; "
            f"got {candidate.ndim} dimensions"
        )

    if candidate.dtype.kind not in "iuf":
        raise InvalidPolicyError(
            f"{context} must return a two-dimensional numeric array; "
            f"got dtype {candidate.dtype!r}"
        )

    if candidate.shape != mask.shape:
        raise InvalidPolicyError(
            f"{context} returned shape {candidate.shape!r}; "
            f"expected {mask.shape!r}"
        )

    values = candidate.astype(float, copy=False)
    non_finite = np.argwhere(~np.isfinite(values))

    if non_finite.size:
        row, column = (int(index) for index in non_finite[0])
        raise InvalidPolicyError(
            f"{context} returned a non-finite probability in row {row} "
            f"for action {action_space[column]!r}: {values[row, column]!r}"
        )

    negative = np.argwhere(values < 0.0)

    if negative.size:
        row, column = (int(index) for index in negative[0])
        raise InvalidPolicyError(
            f"{context} returned a negative probability in row {row} "
            f"for action {action_space[column]!r}: {values[row, column]!r}"
        )

    illegal_nonzero = np.argwhere(~mask & (values != 0.0))

    if illegal_nonzero.size:
        row, column = (int(index) for index in illegal_nonzero[0])
        raise InvalidPolicyError(
            f"{context} returned a nonzero probability for illegal action "
            f"{action_space[column]!r} in row {row}: {values[row, column]!r}"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        totals = np.sum(values, axis=1)

    invalid_totals = np.flatnonzero(
        ~np.isclose(totals, 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE)
    )

    if invalid_totals.size:
        row = int(invalid_totals[0])
        raise InvalidPolicyError(
            f"{context} probabilities in row {row} must sum to one; "
            f"got total {float(totals[row])!r}"
        )

    return values


BatchedCallable: TypeAlias = Callable[[Player, np.ndarray, ActionSpace, np.ndarray], np.ndarray]


class PolicyProfile:
    """Batched callable strategy profile containing every player's strategy.

    A policy profile must represent one fixed strategy profile for an entire
    evaluation. Repeated equivalent queries during that evaluation must return
    equivalent probability distributions. Batched policy profiles receive a
    floating-point observation array with shape
    ``(batch_size, *game.observation_spaces[player])`` and a boolean legal-
    action mask with shape ``(batch_size, len(action_space))``.

    This is the single policy type BenchMAX evaluates. Wrap an already-batched
    callable directly, or convert a :class:`ScalarPolicyProfile` with
    :meth:`from_scalar_callable`::

        profile = PolicyProfile(batched_callable)
        profile = PolicyProfile.from_scalar_callable(scalar_callable)
    """

    def __init__(self, batched_policy_profile: BatchedCallable) -> None:
        """Wrap an already-batched callable.

        Args:
            batched_policy_profile: Callable following the calling convention
                documented by :meth:`__call__`. A scalar callable passed here
                is not detected and fails once it is called with four
                arguments. Use :meth:`from_scalar_callable` instead.

        Raises:
            TypeError: If ``batched_policy_profile`` is not callable.
        """

        if not callable(batched_policy_profile):
            raise TypeError(
                "a policy profile must be callable; got "
                f"{type(batched_policy_profile).__name__}"
            )

        self._batched_policy_profile = batched_policy_profile
        self._artifact_annotations: dict[str, object] = {}
        self._portable_training: TrainingSpec | None = None
        self._portable_game: EntrySpec | None = None
        self._portable_reference: _ArtifactMetadata | None = None
        self._portable_lock = threading.Lock()

    @classmethod
    def from_scalar_callable(cls, scalar_policy_profile: ScalarPolicyProfile) -> PolicyProfile:
        """Adapt a scalar callable to the batched policy contract.

        The returned profile loops over each batch row, flattens its
        observation to an immutable tuple of floats, and calls
        ``scalar_policy_profile`` once with that row's legal actions in
        ``action_space`` order. Scalar distributions are validated and
        copied into a dense floating-point array
        whose columns correspond to ``action_space``, with illegal-action columns
        left at zero.

        This adapter is intended for compatibility and small evaluators. Exact
        exp-a-spiel evaluation rejects the profiles it returns, because they
        invoke Python once per information state.

        Args:
            scalar_policy_profile: Fixed scalar strategies for every player.

        Returns:
            A policy profile producing an array with the same shape as the
            supplied legal-action mask.

        Raises:
            TypeError: If ``scalar_policy_profile`` is not callable.
        """

        return _ScalarAdaptedPolicyProfile(scalar_policy_profile)

    @classmethod
    def from_table(
        cls, table: Mapping[tuple[Player, Observation, tuple[Action, ...]], Mapping[Action, float]], *,
        fallback: str = "error"
    ) -> PolicyProfile:
        """Copy and validate a finite policy table.

        Keys are ``(player, flattened_observation, legal_actions)`` and values map
        action IDs to probabilities. Action order is ignored. Unlisted queries
        raise ``KeyError`` unless ``fallback="uniform"``.
        """
        return _TabularPolicyProfile(table, fallback=fallback)

    def save(self, path: str | PathLike[str], *, game: Game | None = None) -> Reference[PolicyProfile]:
        """Save a supported policy to a new directory and return its Reference.

        Manual tables require ``game=load_game(...)`` on first save.
        Arbitrary Python callables cannot be saved.
        """
        from benchmax.execution.artifacts.policies import _PolicySource, write_policy
        from benchmax.execution.jobs.reference import Reference
        from benchmax.execution.jobs.submission import game_entry
        from benchmax.execution.specs import _training_game_entry

        source = self._portable_training
        entry = source.game if source is not None else self._portable_game
        if game is not None:
            requested = game_entry(game)
            if entry is not None and _training_game_entry(requested) != _training_game_entry(entry):
                raise ValueError("game differs from the policy's saved or training game")

            entry = requested

        if entry is None:
            raise ValueError("saving a policy without game provenance requires game=load_game(...)")

        reference = write_policy(self, source or _PolicySource(entry), path)
        _attach_policy_metadata(self, reference=reference, game=entry)
        return Reference.from_path(reference.uri)

    def __call__(
        self, player: Player, observations: np.ndarray, action_space: ActionSpace, legal_action_mask: np.ndarray, /
    ) -> np.ndarray:
        """Return validated dense action distributions for a batch of observations.

        Output column ``j`` corresponds to ``action_space[j]``. The returned
        array must have the same shape as ``legal_action_mask``. Every row must
        form a valid probability distribution over that row's legal actions,
        and every illegal entry must be exactly zero.

        Args:
            player: One Python integer identifying the queried player.
            observations: Floating-point array with shape
                ``(batch_size, *game.observation_spaces[player])``.
            action_space: Complete ordered action space, equal to
                ``game.action_spaces[player]``.
            legal_action_mask: Boolean array with shape
                ``(batch_size, len(action_space))``.

        Returns:
            A floating-point probability array with the same shape as ``legal_action_mask``.

        Raises:
            InvalidPolicyError: If the output violates the probability contract.
                Invalid distributions are rejected without normalization or repair.
        """

        try:
            probabilities = self._batched_policy_profile(player, observations, action_space, legal_action_mask)
        except TypeError as error:
            # An empty traceback tail means the call itself was rejected on
            # arity, rather than the policy running and raising a TypeError.
            traceback = error.__traceback__
            if traceback is not None and traceback.tb_next is None:
                error.add_note(
                    "Note: BenchMAX calls batched policy profiles with four "
                    "positional arguments (player, observations, "
                    "action_space, legal_action_mask). If you passed a scalar "
                    "policy profile, convert it with "
                    "PolicyProfile.from_scalar_callable()."
                )

            raise

        return _validate_batched_distribution(
            probabilities, action_space, legal_action_mask, context=f"policy profile for player {player}"
        )

    def tabulate(
        self, queries: Mapping[Player, Sequence[tuple[Observation, tuple[Action, ...]]]],
        action_spaces: Mapping[Player, ActionSpace]
    ) -> ScalarPolicyProfile:
        """Materialize this profile over a fixed set of information states.

        This is the inverse of :meth:`from_scalar_callable`: it converts a
        batched profile into a scalar one for evaluators that can only be
        queried a single information state at a time. Each player's states are
        sent as one batch, so the profile is invoked exactly once per player
        rather than once per information state.

        Args:
            queries: The information states to materialize for each player,
                as ``(observation, legal_actions)`` pairs without duplicates.
            action_spaces: Complete ordered action space for every player in
                ``queries``, equal to ``game.action_spaces``.

        Returns:
            A :class:`ScalarPolicyProfile` answering from the materialized
            table. It raises :class:`KeyError` for any information state that
            was not supplied in ``queries``.

        Raises:
            InvalidPolicyError: If this profile returns an invalid batch of
                distributions for any player.
        """

        table: dict[tuple[Player, Observation, tuple[Action, ...]], dict[Action, float]] = {}

        for player, player_queries in queries.items():
            player_queries = tuple(player_queries)
            if not player_queries:
                continue

            action_space = action_spaces[player]
            columns = {action: column for column, action in enumerate(action_space)}
            observations = np.asarray([observation for observation, _ in player_queries], dtype=float)
            legal_action_mask = np.zeros((len(player_queries), len(action_space)), dtype=bool)

            for row, (_, legal_actions) in enumerate(player_queries):
                for action in legal_actions:
                    legal_action_mask[row, columns[action]] = True

            probabilities = self(player, observations, action_space, legal_action_mask)

            for row, (observation, legal_actions) in enumerate(player_queries):
                table[(player, observation, legal_actions)] = {
                    action: float(probabilities[row, columns[action]])
                    for action in legal_actions
                }

        def tabulated_policy_profile(
            player: Player, observation: Observation, legal_actions: Sequence[Action], /
        ) -> Mapping[Action, float]:
            key = (player, observation, tuple(legal_actions))

            try:
                return table[key]
            except KeyError:
                raise KeyError(
                    f"player {player} was not materialized at this information "
                    f"state with legal actions {tuple(legal_actions)!r}"
                ) from None

        return tabulated_policy_profile


PolicyProfileLike: TypeAlias = "PolicyProfile | BatchedCallable"


def _attach_policy_metadata(
    profile: PolicyProfile, *, training: TrainingSpec | None = None, reference: _ArtifactMetadata | None = None,
    game: EntrySpec | None = None
) -> None:
    """Remember training settings and saved artifact identity for later serialization."""

    if game is not None:
        profile._portable_game = game

    if training is not None:
        profile._portable_training = training

    if reference is not None:
        profile._portable_reference = reference


def _as_policy_profile(policy_profile: PolicyProfileLike) -> PolicyProfile:
    """Normalize the argument accepted by :meth:`Game.evaluate`."""

    if isinstance(policy_profile, PolicyProfile):
        return policy_profile

    return PolicyProfile(policy_profile)


class _ScalarAdaptedPolicyProfile(PolicyProfile):
    """Policy profile that calls a scalar callable once per batch row.

    Kept as a distinct type so evaluators that cannot afford one Python call
    per information state can reject it with :func:`isinstance`.
    """

    def __init__(self, scalar_policy_profile: ScalarPolicyProfile) -> None:
        if not callable(scalar_policy_profile):
            raise TypeError(
                "a scalar policy profile must be callable; got "
                f"{type(scalar_policy_profile).__name__}"
            )

        self._scalar_policy_profile = scalar_policy_profile
        super().__init__(self._adapt_batch)

    def _adapt_batch(
        self, player: Player, observations: np.ndarray, action_space: ActionSpace, legal_action_mask: np.ndarray, /
    ) -> np.ndarray:
        probabilities = np.zeros_like(legal_action_mask, dtype=float)

        for row, (observation, mask_row) in enumerate(
            zip(observations, legal_action_mask, strict=True)
        ):
            flat_observation = tuple(np.asarray(observation, dtype=float).ravel().tolist())
            legal_actions = tuple(
                action
                for action, is_legal in zip(action_space, mask_row, strict=True)
                if is_legal
            )
            distribution = _validate_scalar_distribution(
                self._scalar_policy_profile(player, flat_observation, legal_actions), legal_actions,
                context=f"scalar policy profile for player {player}, batch row {row}"
            )

            for column, (action, is_legal) in enumerate(
                zip(action_space, mask_row, strict=True)
            ):
                if is_legal:
                    probabilities[row, column] = distribution[action]

        return probabilities


class _TabularPolicyProfile(PolicyProfile):
    """An immutable finite observation/mask lookup with an explicit fallback."""

    def __init__(self, table, *, fallback="error"):
        if fallback not in ("error", "uniform"):
            raise ValueError("fallback must be 'error' or 'uniform'")

        self._table = {}
        self._fallback = fallback
        for (player, observation, legal_actions), distribution in table.items():
            if not isinstance(player, int) or isinstance(player, bool):
                raise InvalidPolicyError("table player IDs must be integers")

            observation = tuple(float(value) for value in observation)
            if not observation or not all(math.isfinite(value) for value in observation):
                raise InvalidPolicyError("table observations must be nonempty and finite")

            legal_actions = tuple(legal_actions)
            if any(not isinstance(action, int) or isinstance(action, bool) for action in legal_actions):
                raise InvalidPolicyError("table action IDs must be integers")

            probabilities = _validate_scalar_distribution(distribution, legal_actions, context="table policy")
            key = (player, observation, tuple(sorted(legal_actions)))
            if key in self._table:
                raise InvalidPolicyError("duplicate table query after canonicalizing legal actions")

            self._table[key] = tuple(probabilities[action] for action in key[2])

        super().__init__(self._query)

    def _query(self, player, observations, action_space, legal_action_mask):
        probabilities = np.zeros(legal_action_mask.shape, dtype=float)
        # Table keys list legal actions in ascending order; sort the columns once per batch.
        order = np.asarray(sorted(range(len(action_space)), key=action_space.__getitem__), dtype=int)
        sorted_mask = np.asarray(legal_action_mask, dtype=bool)[:, order]
        for row, observation in enumerate(observations):
            columns = order[sorted_mask[row]]
            legal = tuple(action_space[column] for column in columns)
            key = (player, tuple(np.asarray(observation, dtype=float).ravel().tolist()), legal)
            values = self._table.get(key)
            if values is None:
                if self._fallback == "error":
                    raise KeyError(f"player {player} was not materialized at this observation and legal-action set")

                if len(columns) == 0:
                    raise InvalidPolicyError("table query has no legal actions")

                values = 1.0 / len(columns)

            probabilities[row, columns] = values

        return probabilities
