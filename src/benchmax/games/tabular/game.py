"""Markov games whose dynamics are given as explicit tensors.

A tabular game states its own model: a finite state set, a transition tensor,
and a reward tensor per player. Algorithms that plan rather than only sample
can read that model directly.

These games are simultaneous-move and infinite-horizon: every player acts at
every decision point, and episodes never end on their own. Their objective is
the discounted return, so :attr:`~benchmax.core.Game.discount` is required
rather than optional.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Self

import numpy as np

from benchmax.core import Game, GameInstance, TimeStep, _validate_joint_actions
from benchmax.policy import PolicyProfile
from benchmax.seeding import derive_seed, numpy_generator
from benchmax.types import Action, Observation, Player

if TYPE_CHECKING:
    from benchmax.evaluation import Evaluator


class TabularMarkovGame(Game):
    """A simultaneous-move Markov game defined by explicit tensors.

    Model arrays are validated and copied at construction, and later edits are the caller's responsibility.

    Algorithms may read :attr:`transitions` and :attr:`rewards` directly.

    Attributes:
        transitions: ``P(s'|s, a_0, ..., a_{n-1})`` with shape
            ``(S, A_0, ..., A_{n-1}, S)``.
        rewards: Immediate reward per player, each with shape
            ``(S, A_0, ..., A_{n-1})``. Rewards are a function of the joint
            action taken in a state, not of the state reached.
        initial_state_distribution: Distribution over starting states, shape
            ``(S,)``.
        observation_table: Row ``s`` is the observation every player receives
            in state ``s``. The state is fully observable, so all players share
            one encoding, independently of their simultaneous choices.
    """

    discount: float

    def __init__(
        self, *, name: str, transitions: np.ndarray, rewards: Mapping[Player, np.ndarray],
        initial_state_distribution: np.ndarray, discount: float, observation_table: np.ndarray | None = None,
        evaluator: Evaluator[Self] | None = None, max_episode_steps: int = 1000, description: str = ""
    ) -> None:
        """Define a game from its model.

        Args:
            name: Stable name identifying the game.
            transitions: Transition tensor of shape ``(S, A_0, ..., A_{n-1}, S)``
                whose last axis sums to one.
            rewards: One reward tensor per player, each of shape
                ``(S, A_0, ..., A_{n-1})``. Its keys define the players.
            initial_state_distribution: Distribution over starting states.
            discount: Discount factor in ``[0, 1)`` defining the objective.
            observation_table: Optional ``(S, d)`` encoding of each state. The
                default is the one-hot encoding, which lets a tabular policy
                recover the state index by ``argmax``.
            evaluator: Method used to compute exploitability. ``None`` leaves
                the game playable and trainable, and training warns before work
                begins that evaluation is not yet available.
            max_episode_steps: Rollout budget for training.
            description: Human-readable account of the game's rules.

        Raises:
            ValueError: If the tensors disagree on shape, the transition rows
                are not distributions, or ``discount`` is outside ``[0, 1)``.
        """

        players = tuple(sorted(rewards))
        transitions = np.asarray(transitions, dtype=float)
        if not players or transitions.ndim != len(players) + 2:
            raise ValueError(
                f"transitions must have one action axis per player; got shape "
                f"{transitions.shape!r} for {len(players)} players"
            )

        num_states = transitions.shape[0]
        action_counts = transitions.shape[1:-1]
        self.transitions = _model_array(
            transitions, (num_states, *action_counts, num_states), "transitions", distribution=True
        )
        self.rewards = {
            player: _model_array(rewards[player], transitions.shape[:-1], f"rewards[{player!r}]") for player in players
        }
        self.initial_state_distribution = _model_array(
            initial_state_distribution, (num_states,), "initial_state_distribution", distribution=True
        )
        observations = np.eye(num_states) if observation_table is None else np.asarray(observation_table, dtype=float)
        if observations.ndim != 2:
            raise ValueError("observation_table must have shape (S, d)")

        self.observation_table = _model_array(observations, (num_states, observations.shape[1]), "observation_table")
        observation_table = self.observation_table
        self._observation_rows = {
            row.tobytes(): index for index, row in enumerate(observation_table)
        }
        if len(self._observation_rows) != num_states:
            raise ValueError(
                "observation_table must encode every state distinctly; "
                "two states share an observation"
            )

        if discount is None:
            raise ValueError("tabular games require a discount in [0, 1)")

        observation_size = observation_table.shape[1]
        super().__init__(
            name=name,
            players=players,
            observation_spaces={
                player: (observation_size,) for player in players
            },
            action_spaces={
                player: tuple(range(action_counts[index]))
                for index, player in enumerate(players)
            },
            state_space=(observation_size,),
            is_fully_observable=True,
            is_simultaneous=True,
            max_episode_steps=max_episode_steps,
            discount=discount,
            description=description,
            evaluator=evaluator,
        )

    @property
    def num_states(self) -> int:
        """Number of states in the game."""
        return len(self.transitions)

    def observation(self, state: int) -> Observation:
        """Return the observation every player receives in ``state``."""
        return tuple(self.observation_table[state].tolist())

    def state_indices(self, observations: np.ndarray) -> np.ndarray:
        """Recover the state index of each observation in a batch.

        This is what lets a policy indexed by state answer the batched queries
        an evaluator makes, without the policy having to know how states are
        encoded.

        Args:
            observations: Array of shape ``(batch_size, d)`` whose rows are
                observations this game produced.

        Returns:
            An integer array of shape ``(batch_size,)`` of state indices.

        Raises:
            KeyError: If any row is not an observation of this game.
        """

        rows = np.ascontiguousarray(np.asarray(observations, dtype=float))

        try:
            return np.array([self._observation_rows[row.tobytes()] for row in rows], dtype=int)
        except KeyError as error:
            raise KeyError(
                f"observation {error.args[0]!r} does not belong to "
                f"{self.name!r}"
            ) from None

    def _policy_tables(self, policy_profile: PolicyProfile) -> dict[Player, np.ndarray]:
        """Materialize each player's validated strategy over every state in one call."""
        observations = np.asarray(self.observation_table, dtype=float)
        policies = {}

        for player in self.players:
            action_space = self.action_spaces[player]
            legal_actions = np.ones((self.num_states, len(action_space)), dtype=bool)
            policies[player] = policy_profile(player, observations, action_space, legal_actions)

        return policies

    def tabular_profile(
        self, policies: Mapping[Player, np.ndarray]
    ) -> PolicyProfile:
        """Adapt one strategy table per player into a policy profile.

        A tabular game knows how it encodes its states, so a strategy indexed by
        state can be evaluated without the caller writing the lookup.

        Args:
            policies: One ``(S, A)`` row-stochastic array per player.

        Returns:
            A profile answering from those tables.

        Raises:
            ValueError: If a player is missing or an array has the wrong shape.
        """

        missing = set(self.players) - set(policies)
        if missing:
            raise ValueError(f"no strategy given for players {sorted(missing)}")

        tables: dict[Player, np.ndarray] = {}

        for player in self.players:
            table = np.asarray(policies[player], dtype=float)
            expected = (self.num_states, len(self.action_spaces[player]))
            if table.shape != expected:
                raise ValueError(
                    f"strategy for player {player} must have shape {expected!r}; "
                    f"got {table.shape!r}"
                )

            tables[player] = table

        entries = {
            (player, tuple(float(value) for value in observation.flat), self.action_spaces[player]):
            dict(zip(self.action_spaces[player], table[state], strict=True))
            for player, table in tables.items() for state, observation in enumerate(self.observation_table)
        }
        return PolicyProfile.from_table(entries)

    def new_env(self) -> GameInstance:
        """Create an independent environment with its own transition RNG."""
        return _TabularMarkovGameInstance(self)

class TabularZeroSumMarkovGame(TabularMarkovGame):
    """A two-player tabular Markov game in which the rewards sum to zero.

    Zero-sum is what makes a single Nash value, and therefore an exploitability
    gap, well defined, so this is the family the minimax solver and the
    discounted Nash-gap evaluator apply to. Player ``0`` maximizes and player
    ``1`` minimizes.

    Attributes:
        payoffs: Player 0's reward tensor, shape ``(S, A_0, A_1)``. Player 1's
            reward is its negation.
    """

    def __init__(
        self, *, name: str, transitions: np.ndarray, payoffs: np.ndarray, initial_state_distribution: np.ndarray,
        discount: float, observation_table: np.ndarray | None = None, evaluator: Evaluator[Self] | None = None,
        max_episode_steps: int = 1000, description: str = ""
    ) -> None:
        """Define a zero-sum game from player 0's payoffs.

        Args:
            name: Stable name identifying the game.
            transitions: Transition tensor of shape ``(S, A_0, A_1, S)``.
            payoffs: Player 0's reward tensor, shape ``(S, A_0, A_1)``.
            initial_state_distribution: Distribution over starting states.
            discount: Discount factor in ``[0, 1)``.
            observation_table: Optional ``(S, d)`` encoding of each state.
            evaluator: Method used to compute exploitability.
            max_episode_steps: Rollout budget for training.
            description: Human-readable account of the game's rules.
        """

        payoffs = np.asarray(payoffs, dtype=float)
        super().__init__(
            name=name, transitions=transitions, rewards={0: payoffs, 1: -payoffs},
            initial_state_distribution=initial_state_distribution, discount=discount,
            observation_table=observation_table, evaluator=evaluator, max_episode_steps=max_episode_steps,
            description=description
        )

        self.payoffs = self.rewards[0]


class _TabularMarkovGameInstance(GameInstance):
    """One episode of a tabular Markov game."""

    game: TabularMarkovGame

    def __init__(self, game: TabularMarkovGame) -> None:
        super().__init__(game)
        self._rng: np.random.Generator | None = None
        self._state: int | None = None

    def _time_step(self, rewards: dict[Player, float]) -> TimeStep:
        assert self._state is not None
        observation = self.game.observation(self._state)
        return TimeStep(
            observations={player: observation for player in self.game.players},
            rewards=rewards,
            players_to_act=self.game.players,
            legal_actions={
                player: self.game.action_spaces[player]
                for player in self.game.players
            },
            terminated=False,
            truncated=False,
            info={},
        )

    def reset(self, seed: int | None = None) -> TimeStep:
        if seed is not None:
            self._rng = numpy_generator(derive_seed(seed, "tabular-environment", 0))

        if self._rng is None:
            raise RuntimeError("the first reset requires an explicit seed")

        self._state = int(
            self._rng.choice(self.game.num_states, p=self.game.initial_state_distribution)
        )
        return self._time_step({player: 0.0 for player in self.game.players})

    def state(self) -> Observation:
        if self._state is None:
            raise RuntimeError("reset must be called before state")

        return self.game.observation(self._state)

    def step(self, actions: Mapping[Player, Action]) -> TimeStep:
        if self._state is None:
            raise RuntimeError("reset must be called before step")

        _validate_joint_actions(actions, self.game.action_spaces)

        joint = (self._state, *(actions[player] for player in self.game.players))
        rewards = {
            player: float(self.game.rewards[player][joint])
            for player in self.game.players
        }
        self._state = int(
            self._rng.choice(self.game.num_states, p=self.game.transitions[joint])
        )
        return self._time_step(rewards)


def _model_array(value: np.ndarray, shape: tuple[int, ...], label: str, *, distribution: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or any(size == 0 for size in shape):
        raise ValueError(f"{label} must have nonempty shape {shape!r}; got {array.shape!r}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")

    if distribution and (np.any(array < 0) or not np.allclose(array.sum(axis=-1), 1.0)):
        raise ValueError(f"{label} must be nonnegative and sum to one along its last axis")

    return array.copy()
