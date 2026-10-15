"""A zero-sum gridworld pursuit game as a tabular Markov game.

Two agents move simultaneously on a grid containing one goal cell. Standing on
the goal is worth ``+1`` per step to that agent and ``-1`` to the other, and the
goal is absorbing, so the game has no terminal state and its value is the
discounted sum of that stream.

The dynamics are deterministic. An agent that would leave the grid stays where
it is, and an agent already on the goal cannot move. When the two agents would
land on the same cell, or would exchange cells, neither moves.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from benchmax.games.tabular.game import TabularZeroSumMarkovGame
from benchmax.seeding import numpy_generator, resolve_seed

if TYPE_CHECKING:
    from benchmax.evaluation import Evaluator

#: Action order. Rows increase downwards, so ``up`` decreases the row index.
MOVES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("left", (0, -1)),
    ("right", (0, 1)),
    ("up", (-1, 0)),
    ("down", (1, 0)),
)

#: The settings that decide which gridworld this is, with their defaults. They
#: are the game's ``params`` rather than arguments of their own, so a
#: configuration states them under ``_params_`` and can freeze them by name.
PARAMETERS: Mapping[str, Any] = {
    "rows": 3,
    "columns": 3,
    "goal": None,
    "initial_state": None,
    "discount": 0.9,
    "seed": 0,
}


class Gridworld(TabularZeroSumMarkovGame):
    """Two-player zero-sum pursuit on a grid with one absorbing goal cell.

    States are every ordered pair of distinct cells, enumerated with player 0's
    cell varying most slowly, so a grid of ``rows * columns`` cells has
    ``n * (n - 1)`` states. Cells are numbered in row-major order.

    Attributes:
        params: The settings that produced this gridworld, with a drawn
            layout recorded as explicit cells.
        rows: Grid height.
        columns: Grid width.
        goal_cell: Index of the absorbing goal cell.
        seed: The seed a drawn layout was drawn with, or ``None`` if it was
            drawn from fresh entropy and cannot be reproduced.
        states: Row ``s`` holds the cell indices of player 0 and player 1.
    """

    def __init__(
        self, *, params: Mapping[str, Any] | None = None, evaluator: Evaluator[Self] | None = None,
        name: str = "gridworld", max_episode_steps: int = 1000, description: str = ""
    ) -> None:
        """Build the gridworld's tensors and define the game from them.

        Args:
            params: Which gridworld to build, as the names in
                :data:`PARAMETERS`: ``rows`` and ``columns`` size the grid,
                ``goal`` is the ``(row, column)`` of the absorbing goal cell,
                ``initial_state`` the starting ``(row, column)`` of player 0
                and player 1, which the game starts from deterministically, and
                ``discount`` the discount factor in ``[0, 1)`` defining the
                objective. ``goal`` and ``initial_state`` are drawn when they
                are not given, from ``seed``, which defaults to 0 so an
                unspecified layout is still the same game every time. ``None``
                draws from fresh entropy instead, which produces a game that
                cannot be reproduced or referred to later, and warns.
            evaluator: Method used to compute exploitability.
            name: Stable name identifying the game.
            max_episode_steps: Rollout budget for training.
            description: Human-readable account of the game's rules.

        Raises:
            ValueError: If a parameter is not one this game has, the goal or
                either starting cell is off the grid, or both agents start on
                the same cell.
        """

        given = dict(params or {})
        if unknown := sorted(set(given) - set(PARAMETERS)):
            raise ValueError(
                f"{', '.join(unknown)} is not a gridworld parameter; it takes "
                f"{', '.join(PARAMETERS)}"
            )

        settings = {**PARAMETERS, **given}
        rows: int = settings["rows"]
        columns: int = settings["columns"]
        goal: Sequence[int] | None = settings["goal"]
        initial_state: Sequence[Sequence[int]] | None = settings["initial_state"]
        discount: float = settings["discount"]
        seed: int | None = settings["seed"]

        self.rows = rows
        self.columns = columns
        self.seed = seed

        drawn: dict[str, Any] = {}

        if goal is None or initial_state is None:
            if seed is None:
                warnings.warn(
                    f"{name!r} is drawing its layout from fresh entropy "
                    "because seed=None; the resulting game cannot be "
                    "reproduced or referred to later. Pass an integer seed, or "
                    "a goal and initial_state to choose the layout outright",
                    stacklevel=2,
                )

            cells = rows * columns
            if cells < 3:
                raise ValueError(
                    "a random layout needs at least three cells; got "
                    f"{rows}x{columns}"
                )

            layout_seed = resolve_seed(seed, purpose=f"{name!r} layout")
            draw = numpy_generator(layout_seed).choice(cells, size=3, replace=False)
            drawn = {
                "goal": divmod(int(draw[0]), columns),
                "initial_state": (
                    divmod(int(draw[1]), columns),
                    divmod(int(draw[2]), columns),
                ),
            }
            goal = drawn["goal"] if goal is None else goal
            initial_state = (
                drawn["initial_state"] if initial_state is None else initial_state
            )

        # What this gridworld is, rather than what it was asked for: a drawn
        # layout is recorded here so the game can be named again afterwards,
        # and what the caller stated outright wins over what was drawn for it.
        self.params = {**drawn, **given}

        if not self._on_grid(goal):
            raise ValueError(f"goal {tuple(goal)!r} is off the grid")

        self.goal_cell = self._cell(goal)

        self.states = [
            (first, second)
            for first, second in itertools.product(range(rows * columns), repeat=2)
            if first != second
        ]
        index = {state: position for position, state in enumerate(self.states)}

        if len(initial_state) != 2 or not all(
            self._on_grid(position) for position in initial_state
        ):
            raise ValueError(f"initial_state {tuple(initial_state)!r} is off the grid")

        start = (self._cell(initial_state[0]), self._cell(initial_state[1]))
        if start not in index:
            raise ValueError(
                f"initial_state {tuple(initial_state)!r} places both agents on "
                "one cell"
            )

        transitions, payoffs = self._build_tensors(index)
        initial_state_distribution = np.zeros(len(self.states))
        initial_state_distribution[index[start]] = 1.0

        super().__init__(
            name=name, transitions=transitions, payoffs=payoffs, initial_state_distribution=initial_state_distribution,
            discount=discount, evaluator=evaluator, max_episode_steps=max_episode_steps, description=description
        )

    def state_index_of(
        self, first: Sequence[int], second: Sequence[int]
    ) -> int:
        """Return the state index for two ``(row, column)`` positions.

        Args:
            first: Player 0's cell.
            second: Player 1's cell.

        Returns:
            The index of that joint position in :attr:`states`.

        Raises:
            ValueError: If either position is off the grid or they coincide.
        """

        if not (self._on_grid(first) and self._on_grid(second)):
            raise ValueError(
                f"({tuple(first)!r}, {tuple(second)!r}) is off the grid"
            )

        cells = (self._cell(first), self._cell(second))
        if cells[0] == cells[1]:
            raise ValueError("both agents cannot occupy one cell")

        return self.states.index(cells)

    def _on_grid(self, position: Sequence[int]) -> bool:
        return (
            len(position) == 2
            and 0 <= position[0] < self.rows
            and 0 <= position[1] < self.columns
        )

    def _cell(self, position: Sequence[int]) -> int:
        return int(position[0]) * self.columns + int(position[1])

    def _cell_moves(self) -> np.ndarray:
        """Return ``target[cell, action]``, keeping an agent that hits a wall put."""

        target = np.empty((self.rows * self.columns, len(MOVES)), dtype=int)

        for cell in range(self.rows * self.columns):
            row, column = divmod(cell, self.columns)

            for action, (_, (row_step, column_step)) in enumerate(MOVES):
                next_row, next_column = row + row_step, column + column_step
                inside = 0 <= next_row < self.rows and 0 <= next_column < self.columns
                target[cell, action] = (
                    next_row * self.columns + next_column if inside else cell
                )

        return target

    def _build_tensors(self, index: dict[tuple[int, int], int]) -> tuple[np.ndarray, np.ndarray]:
        """Enumerate the deterministic successor and payoff of every joint move."""

        target = self._cell_moves()
        num_states, num_actions = len(self.states), len(MOVES)
        transitions = np.zeros((num_states, num_actions, num_actions, num_states))
        payoffs = np.zeros((num_states, num_actions, num_actions))

        for state, (first, second) in enumerate(self.states):
            first_targets = (
                [first] * num_actions if first == self.goal_cell else target[first]
            )
            second_targets = (
                [second] * num_actions if second == self.goal_cell else target[second]
            )

            for action_0 in range(num_actions):
                moved_0 = int(first_targets[action_0])

                for action_1 in range(num_actions):
                    moved_1 = int(second_targets[action_1])
                    blocked = moved_0 == moved_1 or (
                        moved_0 == second and moved_1 == first
                    )
                    successor = (first, second) if blocked else (moved_0, moved_1)
                    transitions[state, action_0, action_1, index[successor]] = 1.0

                    if successor[0] == self.goal_cell:
                        payoffs[state, action_0, action_1] = 1.0
                    elif successor[1] == self.goal_cell:
                        payoffs[state, action_0, action_1] = -1.0

        return transitions, payoffs
