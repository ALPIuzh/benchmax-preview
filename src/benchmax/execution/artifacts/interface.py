"""Game compatibility checks shared by policy codecs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchmax.execution.artifacts.policies import ArtifactError
from benchmax.execution.specs import EntrySpec

if TYPE_CHECKING:
    from benchmax.core import Game, Player


@dataclass(frozen=True, slots=True)
class _PolicyInterface:
    """The game-facing shapes and identifiers encoded by a saved policy."""

    players: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]
    state_space: tuple[int, ...] | None

    @classmethod
    def from_game(cls, game: Game, players: Iterable[Player] | None = None) -> _PolicyInterface:
        players = game.players if players is None else players
        entries = tuple((p, tuple(game.observation_spaces[p]), tuple(game.action_spaces[p])) for p in players)
        return cls(entries, tuple(game.state_space) if game.state_space is not None else None)

    @classmethod
    def from_document(cls, value: Any) -> _PolicyInterface:
        if not isinstance(value, Mapping) or set(value) != {"players", "state_space"}:
            raise ArtifactError("policy interface must contain players and state_space")

        entries = value["players"]
        if not isinstance(entries, list) or not entries:
            raise ArtifactError("policy interface players must be a nonempty list")

        players: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []

        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"player", "observation_space", "action_space"}:
                raise ArtifactError("policy interface player entry is malformed")

            player = entry["player"]
            observation_space = _shape(entry["observation_space"], label="policy observation_space")
            action_space = entry["action_space"]
            if (not isinstance(player, int) or isinstance(player, bool)
                    or not isinstance(action_space, list) or not action_space
                    or any(not isinstance(action, int) or isinstance(action, bool) for action in action_space)
                    or len(set(action_space)) != len(action_space)):
                raise ArtifactError("policy interface player entry is malformed")

            players.append((player, observation_space, tuple(action_space)))

        if len({player for player, _, _ in players}) != len(players):
            raise ArtifactError("policy interface contains duplicate players")

        state = value["state_space"]
        state_space = None if state is None else _shape(state, label="policy state_space")
        return cls(tuple(players), state_space)

    def to_document(self) -> dict[str, Any]:
        return {
            "players": [
                {"player": player, "observation_space": list(observation_space), "action_space": list(action_space)}
                for player, observation_space, action_space in self.players
            ],
            "state_space": list(self.state_space) if self.state_space is not None else None,
        }


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in value)):
        raise ArtifactError(f"{label} must be a nonempty list of positive integers")

    return tuple(value)


def _load_policy_game(entry: EntrySpec) -> Game:
    """Load only a saved game's runtime, checking configuration compatibility."""
    from benchmax.registry import load_game

    try:
        entry.validate_resolution("game")
        overrides = entry.overrides
        overrides["evaluator"] = None
        return load_game(entry.name, **overrides)
    except (ImportError, ValueError) as error:
        raise ArtifactError(f"policy game is incompatible: {error}") from error
