"""BenchMARL provider: training algorithms and the TorchRL environment adapter.

Importing the integration namespace is safe in BenchMAX's base installation.
Training dependencies are loaded only when a BenchMARL API is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmax.integrations.benchmarl.provider import algorithm_config_class

__all__ = ["algorithm_config_class"]


def __getattr__(name: str) -> Any:
    if name == "algorithm_config_class":
        from benchmax.integrations.benchmarl.provider import (
            algorithm_config_class,
        )

        return algorithm_config_class

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
