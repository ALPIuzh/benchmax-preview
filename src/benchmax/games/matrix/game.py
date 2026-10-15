"""Finite one-step games represented by explicit payoff matrices."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Self

import numpy as np

from benchmax.core import Game, GameInstance, GameTreeNode, TimeStep
from benchmax.types import Action, ActionSpace, Observation, ObservationSpace, Player, Rewards

if TYPE_CHECKING:
    from benchmax.evaluation import Evaluator


class MatrixGame(Game):
    """A simultaneous one-step game given by one payoff tensor per player.

    Players are the sorted keys of ``rewards``. Axis ``i`` of every payoff
    tensor represents the actions of ``game.players[i]``, regardless of mapping
    insertion order.
    """

    state_space: ObservationSpace
    _rewards: dict[Player, np.ndarray]

    def __init__(
        self, *, name: str, rewards: Mapping[Player, np.ndarray],
        action_spaces: Mapping[Player, ActionSpace] | None = None, evaluator: Evaluator[Self] | None = None,
        max_episode_steps: int = 1, description: str = ""
    ) -> None:
        players = tuple(sorted(rewards))
        if not players:
            raise ValueError("a matrix game needs at least one player")

        tensors = {player: np.array(rewards[player], dtype=float, copy=True) for player in players}
        shape = tensors[players[0]].shape
        if len(shape) != len(players):
            raise ValueError(
                f"payoff tensors need one action axis per player; got shape {shape!r} for {len(players)} players"
            )

        if any(tensor.shape != shape for tensor in tensors.values()):
            raise ValueError("every player's payoff tensor must have the same shape")

        if any(size == 0 for size in shape):
            raise ValueError("every player needs at least one action")

        if any(not np.all(np.isfinite(tensor)) for tensor in tensors.values()):
            raise ValueError("payoff tensors must contain only finite values")

        if action_spaces is None:
            spaces = {player: tuple(range(shape[index])) for index, player in enumerate(players)}
        else:
            if set(action_spaces) != set(players):
                raise ValueError(f"action_spaces must contain exactly players {set(players)!r}")

            spaces = {player: tuple(action_spaces[player]) for player in players}

            for index, player in enumerate(players):
                if len(spaces[player]) != shape[index]:
                    raise ValueError(
                        f"player {player}'s action space has {len(spaces[player])} "
                        f"actions but payoff axis {index} has size {shape[index]}"
                    )

                if len(set(spaces[player])) != len(spaces[player]):
                    raise ValueError(f"player {player}'s actions must be distinct")

        # A matrix game's payoff tensors are its dynamics. Back each array with
        # immutable bytes, rather than only clearing NumPy's WRITEABLE flag on
        # an owning array (which callers could turn on again). The dictionary
        # stays private and the public property returns a detached mapping.
        self._rewards = {player: _immutable_tensor(tensor) for player, tensor in tensors.items()}
        self._action_indices = {
            player: {action: index for index, action in enumerate(spaces[player])} for player in players
        }
        super().__init__(
            name=name, players=players, observation_spaces={player: (1,) for player in players}, action_spaces=spaces,
            is_fully_observable=True, is_simultaneous=True, state_space=(1 + sum(shape),),
            max_episode_steps=max_episode_steps, description=description, evaluator=evaluator
        )

    @property
    def rewards(self) -> Mapping[Player, np.ndarray]:
        """Return immutable payoff tensors indexed by player."""

        return dict(self._rewards)

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore payoff immutability after pickle reconstructs NumPy arrays."""

        self.__dict__.update(state)
        self._rewards = {player: _immutable_tensor(tensor) for player, tensor in self._rewards.items()}

    def new_env(self) -> GameInstance:
        """Create an independent one-step environment."""

        return _MatrixGameInstance(self)

    def new_tree(self) -> GameTreeNode:
        """Return the root simultaneous decision."""

        return _MatrixGameTreeNode(self)

    def _joint_action_indices(self, actions: Mapping[Player, Action]) -> tuple[int, ...]:
        expected = set(self.players)
        if set(actions) != expected:
            raise ValueError(f"actions must contain exactly players {expected!r}")

        indices = []

        for player in self.players:
            try:
                indices.append(self._action_indices[player][actions[player]])
            except KeyError:
                raise ValueError(f"illegal action {actions[player]} for player {player}") from None

        return tuple(indices)


class TwoPlayerZeroSumMatrixGame(MatrixGame):
    """A two-player matrix game defined by player 0's payoff matrix."""

    def __init__(
        self, *, name: str, payoffs: np.ndarray, action_spaces: Mapping[Player, ActionSpace] | None = None,
        evaluator: Evaluator[Self] | None = None, max_episode_steps: int = 1, description: str = ""
    ) -> None:
        payoffs = np.asarray(payoffs, dtype=float)
        if payoffs.ndim != 2:
            raise ValueError(f"a two-player payoff matrix must have two dimensions; got shape {payoffs.shape!r}")

        super().__init__(
            name=name, rewards={0: payoffs, 1: -payoffs}, action_spaces=action_spaces, evaluator=evaluator,
            max_episode_steps=max_episode_steps, description=description
        )

    @property
    def payoffs(self) -> np.ndarray:
        """Return player 0's immutable payoff matrix."""

        return self.rewards[0]


class _MatrixGameInstance(GameInstance):
    """Mutable episode state for one matrix-game play."""

    game: MatrixGame

    def __init__(self, game: MatrixGame) -> None:
        super().__init__(game)
        self._started = False
        self._actions: dict[Player, Action] | None = None

    def reset(self, seed: int | None = None) -> TimeStep:
        del seed
        self._started = True
        self._actions = None
        return TimeStep(
            observations={player: (0.0,) for player in self.game.players},
            rewards={player: 0.0 for player in self.game.players}, players_to_act=self.game.players,
            legal_actions=dict(self.game.action_spaces), terminated=False, truncated=False, info={}
        )

    def state(self) -> Observation:
        if not self._started:
            raise RuntimeError("reset must be called before state")

        if self._actions is None:
            return (1.0, *(0.0 for _ in range(self.game.state_space[0] - 1)))

        encoded = [0.0]

        for player in self.game.players:
            selected = self.game._action_indices[player][self._actions[player]]
            encoded.extend(float(index == selected) for index in range(len(self.game.action_spaces[player])))

        return tuple(encoded)

    def step(self, actions: Mapping[Player, Action]) -> TimeStep:
        if not self._started:
            raise RuntimeError("reset must be called before step")

        if self._actions is not None:
            raise RuntimeError("the episode has terminated")

        indices = self.game._joint_action_indices(actions)
        self._actions = dict(actions)
        return TimeStep(
            observations={},
            rewards={player: float(self.game.rewards[player][indices]) for player in self.game.players},
            players_to_act=(), legal_actions={}, terminated=True, truncated=False, info={}
        )



class _MatrixGameTreeNode(GameTreeNode):
    """Exact tree view of a matrix game's simultaneous decision."""

    def __init__(self, game: MatrixGame, actions: Mapping[Player, Action] | None = None) -> None:
        self._game = game
        self._actions = dict(actions) if actions is not None else None

    def returns(self) -> Rewards | None:
        if self._actions is None:
            return None

        indices = self._game._joint_action_indices(self._actions)
        return {player: float(self._game._rewards[player][indices]) for player in self._game.players}

    def chance_outcomes(self) -> tuple[tuple[Action, float], ...]:
        return ()

    def players_to_act(self) -> tuple[Player, ...]:
        return self._game.players if self._actions is None else ()

    def observation(self, player: Player) -> Observation:
        if player not in self._game.players:
            raise ValueError(f"unknown player {player}")

        return (0.0,)

    def legal_actions(self, player: Player) -> tuple[Action, ...]:
        try:
            return self._game.action_spaces[player]
        except KeyError:
            raise ValueError(f"unknown player {player}") from None

    def chance_child(self, action: Action) -> GameTreeNode:
        del action
        raise RuntimeError("matrix games have no chance nodes")

    def child(self, actions: Mapping[Player, Action]) -> GameTreeNode:
        if self._actions is not None:
            raise RuntimeError("a terminal node has no children")

        self._game._joint_action_indices(actions)
        return _MatrixGameTreeNode(self._game, actions)


def _immutable_tensor(tensor: np.ndarray) -> np.ndarray:
    """Copy one tensor onto an immutable byte buffer."""

    return np.frombuffer(tensor.tobytes(order="C"), dtype=tensor.dtype).reshape(tensor.shape)
