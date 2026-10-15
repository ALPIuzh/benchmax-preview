"""Compact exp-a-spiel payoff operator for CFR and CFR+."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmax.core import Game
from benchmax.policy import PolicyProfile
from benchmax.training.cfr import _Level, _Treeplex
from benchmax.types import ActionSpace, Player


class _ExpASpielPayoffOperator:
    def __init__(self, game: Game, traverser: Any) -> None:
        self._traverser = traverser
        self._masks: list[np.ndarray] = []
        treeplexes = []

        for player in game.players:
            actions, counts, masks, parents, depths = traverser.cfr_treeplex(player)
            starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
            levels = []

            for depth in range(1, int(depths.max()) + 1):
                rows = np.flatnonzero(depths == depth)
                level_counts = counts[rows]
                level_starts = np.cumsum(level_counts, dtype=np.int64) - level_counts
                action_indices = (
                    np.repeat(starts[rows], level_counts)
                    + np.arange(int(level_counts.sum()), dtype=np.int64)
                    - np.repeat(level_starts, level_counts)
                )
                levels.append(
                    _Level(
                        parents=None if depth == 1 else parents[rows],
                        actions=actions[action_indices],
                        counts=level_counts,
                        starts=np.concatenate(
                            ([0], np.cumsum(level_counts[:-1], dtype=np.int64))
                        ),
                    )
                )

            treeplexes.append(
                _Treeplex.from_arrays(player, len(counts) * 9, actions, counts, tuple(levels))
            )
            self._masks.append(masks)

        self.treeplexes = tuple(treeplexes)

    def gradient(
        self, strategies: tuple[np.ndarray, ...], realization_plans: tuple[np.ndarray | None, ...], player: int
    ) -> np.ndarray:
        del realization_plans
        return self._traverser.cfr_gradient(strategies[0].reshape(-1, 9), strategies[1].reshape(-1, 9), player)

    def policy(self, averages: tuple[np.ndarray, ...]) -> PolicyProfile:
        strategies = []

        for average, masks in zip(averages, self._masks, strict=True):
            strategy = average.reshape(-1, 9)
            totals = strategy.sum(axis=1, keepdims=True)
            np.divide(strategy, totals, out=strategy, where=totals != 0.0)
            empty = totals[:, 0] == 0.0

            if np.any(empty):
                legal = (masks[empty, None] & (1 << np.arange(9))) != 0
                strategy[empty] = legal / legal.sum(axis=1, keepdims=True)

            strategies.append(strategy)

        return _ExpASpielPolicy(self._traverser, tuple(strategies))


class _ExpASpielPolicy(PolicyProfile):
    def __init__(self, traverser: Any, strategies: tuple[np.ndarray, ...]) -> None:
        self._traverser = traverser
        self._strategies = strategies
        super().__init__(self._query)

    def _query(
        self, player: Player, observations: np.ndarray, action_space: ActionSpace, legal_action_mask: np.ndarray
    ) -> np.ndarray:
        del action_space, legal_action_mask
        return self._traverser.cfr_policy_batch(player, observations, self._strategies[player])

    def _exp_a_spiel_strategies(self, traverser: Any) -> tuple[np.ndarray, ...] | None:
        return self._strategies if traverser is self._traverser else None
