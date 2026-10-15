"""Sequence-form counterfactual regret minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from benchmax.core import Game
from benchmax.policy import PolicyProfile
from benchmax.training._tree import _InformationSet, _SequenceTree
from benchmax.types import Action, Observation, Player


@dataclass(frozen=True, slots=True)
class _Level:
    parents: np.ndarray | None
    actions: np.ndarray
    counts: np.ndarray
    starts: np.ndarray


class _Treeplex:
    def __init__(self, player: Player, information_sets: tuple[_InformationSet, ...], sequence_count: int) -> None:
        self.player = player
        self.information_sets = information_sets
        self.sequence_count = sequence_count
        self.root_sequence: int | None = 0
        self.requires_realization = True
        sequences = [information_set.sequences for information_set in information_sets]
        self.actions = (
            np.concatenate(sequences) if sequences else np.empty(0, dtype=np.int64)
        )
        self.counts = np.asarray([len(information_set.actions) for information_set in information_sets], dtype=np.int64)
        self.starts = np.concatenate(([0], np.cumsum(self.counts[:-1])))
        self.uniform = np.repeat(1.0 / self.counts, self.counts)
        self.levels = tuple(
            self._level(depth)
            for depth in range(
                1,
                max((information_set.depth for information_set in information_sets), default=0)
                + 1,
            )
        )

    @classmethod
    def from_arrays(
        cls, player: Player, sequence_count: int, actions: np.ndarray, counts: np.ndarray, levels: tuple[_Level, ...]
    ) -> _Treeplex:
        treeplex = cls.__new__(cls)
        treeplex.player = player
        treeplex.information_sets = ()
        treeplex.sequence_count = sequence_count
        treeplex.root_sequence = None
        treeplex.requires_realization = False
        treeplex.actions = actions
        treeplex.counts = counts
        treeplex.starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
        treeplex.uniform = np.repeat(1.0 / counts, counts)
        treeplex.levels = levels
        return treeplex

    def realize(self, strategy: np.ndarray, out: np.ndarray) -> None:
        if self.root_sequence is not None:
            out[self.root_sequence] = 1.0

        for level in self.levels:
            parent_reach = (
                1.0
                if level.parents is None
                else np.repeat(out[level.parents], level.counts)
            )
            out[level.actions] = parent_reach * strategy[level.actions]

    def _level(self, depth: int) -> _Level:
        information_sets = tuple(
            information_set
            for information_set in self.information_sets
            if information_set.depth == depth
        )
        counts = np.asarray(
            [len(information_set.actions) for information_set in information_sets]
        )
        actions = np.concatenate(
            [information_set.sequences for information_set in information_sets]
        )
        return _Level(
            parents=(
                None
                if depth == 1
                else np.asarray(
                    [
                        information_set.parent_sequence
                        for information_set in information_sets
                    ]
                )
            ),
            actions=actions,
            counts=counts,
            starts=np.concatenate(([0], np.cumsum(counts[:-1]))),
        )


class _SequenceForm:
    def __init__(self, game: Game) -> None:
        self.players = game.players
        tree = _SequenceTree(game)
        self.treeplexes = tuple(
            _Treeplex(player, tuple(tree.infos[player]), tree.counts[player]) for player in self.players
        )
        self._terminal_sequences = tree.terminal_sequences
        self._terminal_utilities = tree.utilities

    def gradient(
        self, strategies: tuple[np.ndarray, ...], realization_plans: tuple[np.ndarray | None, ...], player: int
    ) -> np.ndarray:
        del strategies
        weights = self._terminal_utilities[:, player].copy()

        for opponent, realization in enumerate(realization_plans):
            if opponent != player:
                assert realization is not None
                weights *= realization[self._terminal_sequences[:, opponent]]

        return np.bincount(
            self._terminal_sequences[:, player], weights=weights, minlength=self.treeplexes[player].sequence_count
        )

    def policy(self, averages: tuple[np.ndarray, ...]) -> PolicyProfile:
        return _policy(self.treeplexes, averages)


class _PayoffOperator(Protocol):
    treeplexes: tuple[_Treeplex, ...]

    def gradient(
        self, strategies: tuple[np.ndarray, ...], realization_plans: tuple[np.ndarray | None, ...], player: int
    ) -> np.ndarray: ...

    def policy(self, averages: tuple[np.ndarray, ...]) -> PolicyProfile: ...


class _RegretMinimizer:
    def __init__(self, treeplex: _Treeplex, plus: bool) -> None:
        self.treeplex = treeplex
        self.plus = plus
        self.regret = np.zeros(treeplex.sequence_count)
        self.strategy = np.zeros(treeplex.sequence_count)
        self._update_strategy()
        self.average = np.zeros(treeplex.sequence_count)
        treeplex.realize(self.strategy, self.average)
        self.realization: np.ndarray | None = self.average.copy() if treeplex.requires_realization else None
        self.average_step = 1

    def _update_strategy(self) -> None:
        if not self.treeplex.actions.size:
            return

        actions = self.treeplex.actions
        strategy = np.maximum(self.regret[actions], 0.0)
        totals = np.add.reduceat(strategy, self.treeplex.starts)
        denominators = np.repeat(totals, self.treeplex.counts)
        np.divide(strategy, denominators, out=strategy, where=denominators != 0.0)
        np.copyto(strategy, self.treeplex.uniform, where=denominators == 0.0)
        self.strategy[actions] = strategy

    def update(self, gradient: np.ndarray) -> None:
        for level in reversed(self.treeplex.levels):
            expected_values = np.add.reduceat(self.strategy[level.actions] * gradient[level.actions], level.starts)
            self.regret[level.actions] += gradient[level.actions] - np.repeat(
                expected_values, level.counts
            )

            if level.parents is not None:
                np.add.at(gradient, level.parents, expected_values)

        if self.plus:
            np.maximum(self.regret, 0.0, out=self.regret)

        self._update_strategy()
        self.average_step += 1

        if self.realization is None:
            self.treeplex.realize(self.strategy, gradient)
            realization = gradient
        else:
            self.treeplex.realize(self.strategy, self.realization)
            realization = self.realization

        self.average += self.average_step**2 * realization


@dataclass(frozen=True, slots=True)
class CFRTrainingFunction:
    """Configured callable for BenchMAX's native CFR implementation."""

    iterations: int = 1000
    plus: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.iterations, int) or isinstance(self.iterations, bool) or self.iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(self.plus, bool):
            raise TypeError("plus must be a bool")

    def __call__(self, game: Game, *, seed: int) -> PolicyProfile:
        del seed
        return train_cfr(game, self.iterations, plus=self.plus)


def _policy(treeplexes: tuple[_Treeplex, ...], averages: tuple[np.ndarray, ...]) -> PolicyProfile:
    table: dict[tuple[Player, Observation, tuple[Action, ...]], dict[Action, float]] = {}

    for treeplex, average in zip(treeplexes, averages, strict=True):
        for information_set in treeplex.information_sets:
            realization = average[information_set.sequences]
            total = realization.sum()
            probabilities = realization / total if total else np.full(len(realization), 1.0 / len(realization))
            key = (treeplex.player, information_set.observation, information_set.actions)
            table[key] = dict(zip(information_set.actions, probabilities, strict=True))

    return PolicyProfile.from_table(table)


def train_cfr(game: Game, iterations: int, *, plus: bool = False) -> PolicyProfile:
    """Train an average tabular strategy with CFR or CFR+."""

    factory = getattr(getattr(game, "evaluator", None), "payoff_operator", None)
    operator: _PayoffOperator = (
        factory(game) if factory is not None else None
    ) or _SequenceForm(game)
    minimizers = tuple(
        _RegretMinimizer(treeplex, plus) for treeplex in operator.treeplexes
    )
    strategies = tuple(minimizer.strategy for minimizer in minimizers)
    realizations = tuple(minimizer.realization for minimizer in minimizers)

    for iteration in range(iterations):
        player = iteration % len(minimizers)
        gradient = operator.gradient(strategies, realizations, player)
        minimizers[player].update(gradient)

    return operator.policy(tuple(minimizer.average for minimizer in minimizers))
