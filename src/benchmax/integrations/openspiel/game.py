"""Generic OpenSpiel-backed game and environment."""

from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Mapping
from itertools import chain
from random import Random
from typing import TYPE_CHECKING, Any, Self

with contextlib.redirect_stdout(io.StringIO()):
    import pyspiel

from benchmax.core import Game, GameInstance, GameTreeNode, TimeStep, _validate_joint_actions
from benchmax.seeding import derive_seed, python_generator
from benchmax.types import Action, Observation, Player, Rewards

if TYPE_CHECKING:
    from benchmax.evaluation import Evaluator


class OpenSpielGame(Game):
    """An OpenSpiel game exposed through the BenchMAX environment interface.

    Chance events are resolved internally and each acting player observes an
    immutable tuple of floats. Information-state tensors are preferred, but fully
    observable games can instead use observation tensors, such as a board
    encoding. Partially observable games must provide information-state tensors
    to retain the player's visible history.

    Attributes:
        spiel_name: OpenSpiel's own name for this game.
        params: The parameters it was loaded with.
        spiel_game: The loaded OpenSpiel game, used by exact evaluators.
        evaluator: The composed evaluator, or ``None`` when no evaluation
            method is available for this game.
    """

    provenance = "openspiel"

    def __init__(
        self, *, spiel_name: str | None = None, params: Mapping[str, Any] | None = None, name: str | None = None,
        evaluator: Evaluator[Self] | None = None, max_episode_steps: int = 1000, description: str = ""
    ) -> None:
        """Load an OpenSpiel game and derive its BenchMAX spaces.

        Args:
            spiel_name: OpenSpiel's own name for the game, such as
                ``"dark_hex"``. Defaults to ``name``, so a configuration whose
                BenchMAX name is already OpenSpiel's needs only one of the two.
            params: Parameters passed to OpenSpiel verbatim. Omitted parameters
                keep OpenSpiel's defaults, so an empty mapping is the game
                exactly as OpenSpiel defines it.
            name: Canonical BenchMAX name. Defaults to ``spiel_name``. At
                least one of the two is required.
            evaluator: Method used to compute exploitability. ``None`` leaves
                the game playable and trainable, and training warns before work
                begins that evaluation is not yet available.
            max_episode_steps: Upper bound on episode length for training.
            description: Human-readable account of the game's rules.
        """

        if spiel_name is None and name is None:
            raise ValueError("an OpenSpiel game needs a spiel_name or a name")

        self.spiel_name = spiel_name if spiel_name is not None else str(name)
        if self.spiel_name not in set(pyspiel.registered_names()):
            raise ValueError(f"OpenSpiel has no game named {self.spiel_name!r}")

        self.params = dict(params or {})
        name = name if name is not None else self.spiel_name
        self.spiel_game = pyspiel.load_game(self.spiel_name, self.params)
        game_type = self.spiel_game.get_type()
        if game_type.dynamics not in (pyspiel.GameType.Dynamics.SEQUENTIAL, pyspiel.GameType.Dynamics.SIMULTANEOUS):
            raise ValueError("the OpenSpiel adapter supports sequential or simultaneous dynamics")

        is_fully_observable = game_type.information in (
            pyspiel.GameType.Information.PERFECT_INFORMATION, pyspiel.GameType.Information.ONE_SHOT,
        )
        self._uses_information_state = game_type.provides_information_state_tensor

        if self._uses_information_state:
            observation_shape = self.spiel_game.information_state_tensor_shape()
        elif is_fully_observable and game_type.provides_observation_tensor:
            observation_shape = self.spiel_game.observation_tensor_shape()
        else:
            raise ValueError(
                f"OpenSpiel game {self.spiel_name!r} requires an information-state tensor "
                "or an observation tensor for a fully observable game"
            )

        players = tuple(range(self.spiel_game.num_players()))
        # The declared tensor shape is not always one-dimensional (phantom_ttt
        # declares (1, 108)), and games whose initial state is a chance node
        # cannot be encoded before it is resolved, so flatten the declared shape
        # rather than measuring an actual observation.
        observation_size = math.prod(observation_shape)
        actions = tuple(range(self.spiel_game.num_distinct_actions()))
        super().__init__(
            name=name, players=players, observation_spaces={player: (observation_size,) for player in players},
            action_spaces={player: actions for player in players}, is_fully_observable=is_fully_observable,
            is_simultaneous=game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS,
            state_space=(observation_size * len(players),), max_episode_steps=max_episode_steps,
            description=description, evaluator=evaluator
        )

    def new_env(self) -> GameInstance:
        """Create an independent environment with its own chance RNG."""
        return _OpenSpielInstance(self)

    def new_tree(self) -> GameTreeNode:
        """Return an exact tree rooted at OpenSpiel's initial state."""

        if not self.spiel_game.get_type().provides_information_state_string:
            raise NotImplementedError(f"{self.name!r} needs OpenSpiel information-state strings for exact traversal")

        return _OpenSpielTreeNode(self, self.spiel_game.new_initial_state())

    def _encode_observation(self, state: pyspiel.State, player: Player) -> Observation:
        """Use the same player input for environments, policies and evaluation."""

        if self._uses_information_state:
            return tuple(state.information_state_tensor(player))

        return tuple(state.observation_tensor(player))

    def _apply_actions(self, state: pyspiel.State, actions: Mapping[Player, Action]) -> None:
        if state.is_simultaneous_node():
            state.apply_actions([actions[player] for player in self.players])
        else:
            state.apply_action(actions[int(state.current_player())])


class _OpenSpielInstance(GameInstance):
    """Mutable state for one independently seeded OpenSpiel environment."""

    game: OpenSpielGame

    def __init__(self, game: OpenSpielGame) -> None:
        super().__init__(game)
        self._state: pyspiel.State | None = None
        self._rng: Random | None = None

    def reset(self, seed: int | None = None) -> TimeStep:
        if seed is not None:
            self._rng = python_generator(derive_seed(seed, "openspiel-environment", 0))

        if self._rng is None:
            raise RuntimeError("the first reset requires an explicit seed")

        self._state = self.game.spiel_game.new_initial_state()
        self._advance_chance_nodes()
        return self._time_step({player: 0.0 for player in self.game.players})

    def state(self) -> Observation:
        if self._state is None:
            raise RuntimeError("reset must be called before state")

        return tuple(chain.from_iterable(
            self.game._encode_observation(self._state, player) for player in self.game.players
        ))

    def spiel_state(self) -> pyspiel.State:
        """Return a copy of the current OpenSpiel state, for solvers that read full positions."""
        if self._state is None:
            raise RuntimeError("reset must be called before spiel_state")

        return self._state.clone()

    def step(self, actions: Mapping[Player, Action]) -> TimeStep:
        if self._state is None:
            raise RuntimeError("reset must be called before step")

        if self._state.is_terminal():
            raise RuntimeError("the episode has terminated")

        _validate_joint_actions(actions, {player: self._legal_actions(player) for player in self._players_to_act()})

        previous_returns = self._state.returns()
        self.game._apply_actions(self._state, actions)
        self._advance_chance_nodes()
        # One exposed transition can include several chance actions. Return
        # their combined reward without counting earlier rewards again.
        values = self._state.returns()
        rewards = {
            player_id: float(values[player_id] - previous_returns[player_id])
            for player_id in self.game.players
        }
        return self._time_step(rewards)

    def _players_to_act(self) -> tuple[Player, ...]:
        if self._state is None or self._state.is_terminal():
            return ()

        if self._state.is_simultaneous_node():
            return self.game.players

        return (int(self._state.current_player()),)

    def _legal_actions(self, player: Player) -> tuple[Action, ...]:
        assert self._state is not None
        return tuple(int(action) for action in self._state.legal_actions(player))

    def _advance_chance_nodes(self) -> None:
        """Sample through any chance nodes, so they are never exposed."""
        assert self._state is not None

        while self._state.is_chance_node():
            outcomes = self._state.chance_outcomes()
            assert self._rng is not None
            action = self._rng.choices(
                [int(action) for action, _ in outcomes], weights=[float(probability) for _, probability in outcomes],
                k=1
            )[0]
            self._state.apply_action(action)

    def _time_step(self, rewards: Rewards) -> TimeStep:
        assert self._state is not None
        players_to_act = self._players_to_act()
        return TimeStep(
            observations={
                player: self.game._encode_observation(self._state, player)
                for player in players_to_act
            },
            rewards=rewards,
            players_to_act=players_to_act,
            legal_actions={
                player: self._legal_actions(player) for player in players_to_act
            },
            terminated=bool(self._state.is_terminal()),
            truncated=False,
            info={},
        )


class _OpenSpielTreeNode(GameTreeNode):
    """Backend-independent exact-tree view of one OpenSpiel state."""

    def __init__(self, game: OpenSpielGame, state: pyspiel.State) -> None:
        self._game = game
        self._state = state

    def returns(self) -> Rewards | None:
        if not self._state.is_terminal():
            return None

        values = self._state.returns()
        return {player: float(values[player]) for player in self._game.players}

    def chance_outcomes(self) -> tuple[tuple[Action, float], ...]:
        if not self._state.is_chance_node():
            return ()

        return tuple(
            (int(action), float(probability))
            for action, probability in self._state.chance_outcomes()
        )

    def players_to_act(self) -> tuple[Player, ...]:
        if self._state.is_terminal() or self._state.is_chance_node():
            return ()

        if self._state.is_simultaneous_node():
            return self._game.players

        return (int(self._state.current_player()),)

    def observation(self, player: Player) -> Observation:
        return self._game._encode_observation(self._state, player)

    def information_state_key(self, player: Player) -> str:
        return self._state.information_state_string(player)

    def legal_actions(self, player: Player) -> tuple[Action, ...]:
        return tuple(int(action) for action in self._state.legal_actions(player))

    def chance_child(self, action: Action) -> GameTreeNode:
        return _OpenSpielTreeNode(self._game, self._state.child(action))

    def child(self, actions: Mapping[Player, Action]) -> GameTreeNode:
        child = self._state.clone()
        self._game._apply_actions(child, actions)
        return _OpenSpielTreeNode(self._game, child)
