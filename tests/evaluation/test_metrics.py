import pytest

from benchmax.evaluation import deviation_metrics, zero_sum_deviation_metrics


def test_deviation_metrics_define_the_shared_backend_contract() -> None:
    metrics = deviation_metrics({0: 0.25, 1: 0.75, 2: 0.5}, expected_returns={0: 1.0, 1: 2.0, 2: 3.0})

    assert metrics == {
        "expected_returns": {0: 1.0, 1: 2.0, 2: 3.0},
        "player_improvements": {0: 0.25, 1: 0.75, 2: 0.5},
        "nash_gap": 0.75,
        "nash_conv": 1.5,
    }


def test_zero_sum_assembler_adds_only_the_applicable_metric() -> None:
    metrics = zero_sum_deviation_metrics({0: 0.25, 1: 0.75})

    assert metrics == {
        "player_improvements": {0: 0.25, 1: 0.75},
        "nash_gap": 0.75,
        "nash_conv": 1.0,
        "exploitability": 0.5,
    }


def test_zero_sum_assembler_requires_two_player_evidence() -> None:
    with pytest.raises(ValueError, match="exactly two players"):
        zero_sum_deviation_metrics({0: 0.0})
