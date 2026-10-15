"""Reusable algorithm configurations and provider construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from benchmax.config import ConfigValue, _EntryOrigin
from benchmax.core import Game
from benchmax.policy import PolicyProfile


@runtime_checkable
class TrainingFunction(Protocol):
    """One configured training implementation."""

    def __call__(self, game: Game, *, seed: int) -> PolicyProfile: ...


@dataclass(frozen=True, slots=True)
class Algorithm:
    """Immutable recipe created by ``load_algorithm()``. Each call builds a fresh learner."""

    _origin: _EntryOrigin
    total_frames: int | None = None
    device: str = "auto"
    verbose: bool = True

    @property
    def name(self) -> str:
        return self._origin.resolved.name

    @property
    def configuration(self) -> dict[str, ConfigValue]:
        """Resolved constructor arguments, including explicit execution controls."""
        return self._origin.resolved.configuration | self._controls

    @property
    def _controls(self) -> dict[str, ConfigValue]:
        return _training_controls(self.total_frames, self.device, self.verbose)

    def __repr__(self) -> str:
        return f"Algorithm(name={self.name!r}, configuration={self.configuration!r})"

    def _build(self) -> TrainingFunction:
        from benchmax.registry import _ALGORITHMS

        settings = self._origin.overrides | self._controls
        if _ALGORITHMS.resolve(self.name, **settings).configuration != self.configuration:
            raise ValueError(f"algorithm {self.name!r} configuration has changed; call load_algorithm() again")

        return load_training_function(self.name, **settings)

    def __call__(self, game: Game, *, seed: int) -> PolicyProfile:
        return self._build()(game, seed=seed)


def load_algorithm(
    name: str, *, total_frames: int | None = None, device: str = "auto", verbose: bool = True,
    **overrides: ConfigValue
) -> Algorithm:
    """Resolve a reusable algorithm without importing its training runtime.

    Bind settings here and pass seeds to ``train()``. Provider-specific validation
    happens when training starts, for both synchronous and submitted runs.
    """
    from benchmax.registry import _ALGORITHMS

    _validate_training_controls(total_frames=total_frames, device=device, verbose=verbose)
    origin = _EntryOrigin(_ALGORITHMS.resolve(name, **overrides), overrides)
    return Algorithm(origin, total_frames, device, verbose)


def load_training_function(
    algorithm: str, *, total_frames: int | None = None, device: str = "auto", verbose: bool = True,
    **overrides: ConfigValue
) -> TrainingFunction:
    """Construct one registered provider callable for internal use."""
    from benchmax.registry import _ALGORITHMS

    _validate_training_controls(total_frames=total_frames, device=device, verbose=verbose)
    settings = overrides | _training_controls(total_frames, device, verbose)
    return _ALGORITHMS.load(algorithm, **settings)


def _training_controls(total_frames: int | None, device: str, verbose: bool) -> dict[str, ConfigValue]:
    settings: dict[str, ConfigValue] = {}
    if total_frames is not None:
        settings["total_frames"] = total_frames

    if device != "auto":
        settings["device"] = device

    if not verbose:
        settings["verbose"] = False

    return settings


def _validate_training_controls(*, total_frames: object, device: object, verbose: object) -> None:
    if total_frames is not None and (
        not isinstance(total_frames, int) or isinstance(total_frames, bool) or total_frames <= 0
    ):
        raise ValueError("total_frames must be a positive integer")

    if not isinstance(device, str) or not device:
        raise TypeError("device must be a non-empty string")

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
