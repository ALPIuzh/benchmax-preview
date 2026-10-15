"""Supplying BenchMARL's own training algorithms as BenchMAX configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarl.algorithms.common import AlgorithmConfig

_TRAINING_TARGET = "benchmax.integrations.benchmarl.training.build_training_function"


def base_config(entry: str) -> dict[str, object]:
    """Return the configuration for a BenchMARL algorithm with its own defaults.

    No parameters are set, so the result is the algorithm exactly as
    BenchMARL defines it, untouched by BenchMAX's tuning.

    The provider validates the name when it builds the training function on the
    execution target, so resolving a job does not depend on that runtime.
    """

    # Which algorithm this is comes from the entry name, and is frozen for the
    # same reason as an OpenSpiel game's: a name BenchMAX ships must keep
    # meaning the algorithm it names.
    return {
        "_target_": _TRAINING_TARGET,
        "provider": "benchmarl",
        "name": entry,
        "_fixed_": ["provider", "name"],
    }


def algorithm_config_class(name: str) -> type[AlgorithmConfig]:
    """Return BenchMARL's configuration class for an algorithm name.

    Raises:
        ValueError: If BenchMARL has no algorithm with that name.
    """

    from benchmarl.algorithms import algorithm_config_registry

    try:
        return algorithm_config_registry[name]
    except KeyError as error:
        available = ", ".join(sorted(algorithm_config_registry))
        raise ValueError(f"unknown benchmarl algorithm {name!r}; available: {available}") from error


def available_entries() -> list[str]:
    """Return every algorithm BenchMARL can build, sorted."""
    from benchmarl.algorithms import algorithm_config_registry

    return sorted(algorithm_config_registry)
