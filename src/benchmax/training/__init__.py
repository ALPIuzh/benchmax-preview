"""Configured training functions for BenchMAX games.

Native algorithms such as CFR remain available from the base installation.
Neural algorithms load the optional BenchMARL and Torch stack only when
selected.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from benchmax.registry import list_available_algorithms as list_available_algorithms
from benchmax.training.algorithm import Algorithm, TrainingFunction, load_algorithm

if TYPE_CHECKING:
    from benchmax.training.aliases import train_ippo as train_ippo
    from benchmax.training.aliases import train_iql as train_iql
    from benchmax.training.aliases import train_mappo as train_mappo
    from benchmax.training.api import train as train

_SOURCES = {
    "train": "benchmax.training.api",
    "train_ippo": "benchmax.training.aliases",
    "train_mappo": "benchmax.training.aliases",
    "train_iql": "benchmax.training.aliases",
}

__all__ = [
    "Algorithm",
    "TrainingFunction",
    "list_available_algorithms",
    "load_algorithm",
    "train",
    "train_ippo",
    "train_iql",
    "train_mappo",
]


def __getattr__(name: str) -> Any:
    """Import a training entry point the first time it is used."""

    try:
        module_name = _SOURCES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"benchmax.training.{name} requires the optional training "
            f"dependencies; install benchmax[training] ({error})"
        ) from error

    return getattr(module, name)
