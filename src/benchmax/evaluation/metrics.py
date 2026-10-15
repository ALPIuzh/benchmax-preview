"""Shared metric definitions and transformations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmax.types import Player


def deviation_metrics(
    player_improvements: Mapping[Player, float], *, expected_returns: Mapping[Player, float] | None = None
) -> dict[str, Any]:
    """Assemble metrics derived from unilateral-deviation gains."""

    improvements = dict(player_improvements)
    metrics: dict[str, Any] = {
        "player_improvements": improvements,
        "nash_gap": max(improvements.values()),
        "nash_conv": sum(improvements.values()),
    }

    if expected_returns is not None:
        metrics["expected_returns"] = dict(expected_returns)

    return metrics


def zero_sum_deviation_metrics(
    player_improvements: Mapping[Player, float], *, expected_returns: Mapping[Player, float] | None = None
) -> dict[str, Any]:
    """Assemble deviation metrics for a validated two-player zero-sum game."""

    improvements = dict(player_improvements)
    if len(improvements) != 2:
        raise ValueError("zero-sum deviation metrics require exactly two players")

    metrics = deviation_metrics(improvements, expected_returns=expected_returns)
    metrics["exploitability"] = metrics["nash_conv"] / 2.0
    return metrics


__all__ = [
    "deviation_metrics",
    "zero_sum_deviation_metrics",
]
