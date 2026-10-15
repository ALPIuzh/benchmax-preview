"""Compile perfect-recall game trees for sequence-form CFR training."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from benchmax.core import Game, GameTreeNode
from benchmax.types import Action, Observation, Player


@dataclass(frozen=True, slots=True)
class _InformationSet:
    observation: Observation
    actions: tuple[Action, ...]
    parent_sequence: int
    sequences: np.ndarray
    depth: int


class _SequenceTree:
    """Compile terminal payoffs and information sets for observation-based policies."""

    def __init__(self, game: Game) -> None:
        self.game = game
        self.infos: dict[Player, list[_InformationSet]] = {player: [] for player in game.players}
        self.counts = {player: 1 for player in game.players}
        indices = {player: {} for player in game.players}
        policy_keys = {player: {} for player in game.players}
        depths = {player: [0] for player in game.players}
        terminal_sequences, utilities = [], []
        # Iterators keep even very deep/wide trees bounded by visited nodes.
        stack = [iter([(game.new_tree(), (0,) * len(game.players), 1.0)])]

        while stack:
            item = next(stack[-1], None)

            if item is None:
                stack.pop()
                continue

            state, sequences, chance_reach = item

            returns = state.returns()

            if returns is not None:
                terminal_sequences.append(sequences)
                utilities.append([chance_reach * returns[p] for p in game.players])
                continue

            outcomes = state.chance_outcomes()

            if outcomes:
                probabilities = np.asarray([probability for _, probability in outcomes])
                if (
                    not np.all(np.isfinite(probabilities))
                    or np.any(probabilities < 0)
                    or not np.isclose(probabilities.sum(), 1.0)
                ):
                    raise ValueError("chance probabilities must be finite, nonnegative and sum to one")

                stack.append(self._chance_children(state, outcomes, sequences, chance_reach))
                continue

            acting = state.players_to_act()
            if not acting or not set(acting) <= set(game.players) or len(set(acting)) != len(acting):
                raise ValueError("a decision node must name distinct game players")

            node_infos = []

            for slot, player in enumerate(game.players):
                if player not in acting:
                    node_infos.append(-1)
                    continue

                legal = state.legal_actions(player)
                actions = tuple(action for action in game.action_spaces[player] if action in legal)
                if not actions or len(actions) != len(legal):
                    raise ValueError(f"invalid legal actions for player {player}")

                observation = tuple(state.observation(player))
                key = state.information_state_key(player)
                # Observation-based policies cannot distinguish information sets
                # sharing observations and legal actions.
                policy_input = (observation, actions)
                if policy_keys[player].setdefault(policy_input, key) != key:
                    raise ValueError(
                        "observations and legal actions must distinguish information sets; "
                        f"player {player!r} has different histories with the same policy input"
                    )

                index = indices[player].get(key)

                if index is None:
                    index = len(self.infos[player])
                    indices[player][key] = index
                    start = self.counts[player]
                    self.infos[player].append(
                        _InformationSet(
                            observation, actions, sequences[slot], np.arange(start, start + len(actions)),
                            depths[player][sequences[slot]] + 1
                        )
                    )
                    self.counts[player] += len(actions)
                    depths[player].extend([self.infos[player][-1].depth] * len(actions))

                info = self.infos[player][index]
                if info.parent_sequence != sequences[slot] or info.actions != actions or info.observation != observation:
                    raise ValueError(f"player {player} has inconsistent information states or imperfect recall")

                node_infos.append(index)

            stack.append(self._decision_children(state, node_infos, sequences, chance_reach))

        n = len(game.players)
        self.terminal_sequences = np.asarray(terminal_sequences, dtype=np.int64).reshape(-1, n)
        self.utilities = np.asarray(utilities, dtype=float).reshape(-1, n)
        if not len(self.utilities) or not np.all(np.isfinite(self.utilities)):
            raise ValueError("the game must have finite terminal utilities")

    @staticmethod
    def _chance_children(state, outcomes, sequences, reach):
        for action, probability in outcomes:
            if probability > 0:
                yield state.chance_child(action), sequences, reach * probability

    def _decision_children(self, state: GameTreeNode, node_infos, sequences, reach):
        acting = [slot for slot, index in enumerate(node_infos) if index >= 0]
        infos = [self.infos[self.game.players[slot]][node_infos[slot]] for slot in acting]

        for choices in itertools.product(*(range(len(info.actions)) for info in infos)):
            child_sequences, actions = list(sequences), {}

            for slot, info, choice in zip(acting, infos, choices, strict=True):
                child_sequences[slot] = int(info.sequences[choice])
                actions[self.game.players[slot]] = info.actions[choice]

            yield state.child(actions), tuple(child_sequences), reach
