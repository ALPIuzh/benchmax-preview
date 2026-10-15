"""Evaluation contracts, metrics, and built-in numerical backends."""

from benchmax.evaluation._base import (
    Evaluator,
    IncompatibleEvaluatorError,
    require_discounted,
    require_undiscounted,
)
from benchmax.evaluation.approximate import ApproximateExploitabilityEvaluator
from benchmax.evaluation.matrix import MatrixEvaluator
from benchmax.evaluation.metrics import deviation_metrics, zero_sum_deviation_metrics
from benchmax.evaluation.opponent import Opponent, OpponentEvaluator, OpponentSession
from benchmax.evaluation.tabular import DiscountedNashGapEvaluator

__all__ = [
    "ApproximateExploitabilityEvaluator",
    "DiscountedNashGapEvaluator",
    "Evaluator",
    "IncompatibleEvaluatorError",
    "MatrixEvaluator",
    "Opponent",
    "OpponentEvaluator",
    "OpponentSession",
    "deviation_metrics",
    "require_discounted",
    "require_undiscounted",
    "zero_sum_deviation_metrics",
]
