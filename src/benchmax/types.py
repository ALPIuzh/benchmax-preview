"""Type aliases shared by the game and policy interfaces."""

from typing import Any, TypeAlias

Player: TypeAlias = int
Action: TypeAlias = int
Observation: TypeAlias = tuple[float, ...]
ObservationSpace: TypeAlias = tuple[int, ...]
ActionSpace: TypeAlias = tuple[Action, ...]

Observations: TypeAlias = dict[Player, Observation]
Rewards: TypeAlias = dict[Player, float]
Info: TypeAlias = dict[str, Any]
