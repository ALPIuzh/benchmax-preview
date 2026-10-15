"""Supplying OpenSpiel's own games as BenchMAX configuration."""

from __future__ import annotations

import contextlib
import io

_GAME_TARGET = "benchmax.integrations.openspiel.OpenSpielGame"


def _registered_names() -> list[str]:
    with contextlib.redirect_stdout(io.StringIO()):
        import pyspiel

    return list(pyspiel.registered_names())


def base_config(entry: str) -> dict[str, object]:
    """Return the configuration for an OpenSpiel game with its own defaults.

    No parameters are set, so the result is the game exactly as OpenSpiel
    defines it. No evaluator is attached, since exploitability is not available
    for an arbitrary game.

    OpenSpiel validates the name when :class:`OpenSpielGame` is constructed on
    the execution target, so resolving a job does not depend on that runtime.
    """

    # Which game this is comes from the entry name and is frozen here, so every
    # OpenSpiel entry inherits it and no name BenchMAX ships can be pointed at
    # another game.
    return {
        "_target_": _GAME_TARGET,
        "spiel_name": entry,
        "_fixed_": ["spiel_name"],
        "evaluator": None,
    }


def available_entries() -> list[str]:
    """Return every game OpenSpiel can build, sorted."""
    return sorted(_registered_names())
