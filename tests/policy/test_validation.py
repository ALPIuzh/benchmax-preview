from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import benchmax
from benchmax import InvalidPolicyError
from benchmax.policy import _validate_batched_distribution, _validate_scalar_distribution

_CONTEXT = "test policy for player 7"


def test_invalid_policy_error_is_public_value_error() -> None:
    assert issubclass(InvalidPolicyError, ValueError)
    assert benchmax.InvalidPolicyError is InvalidPolicyError
    assert "InvalidPolicyError" in benchmax.__all__


def test_valid_distribution_is_copied_in_legal_action_order_as_floats() -> None:
    probabilities = {2: Decimal("0.25"), 1: Fraction(3, 4)}

    distribution = _validate_scalar_distribution(probabilities, (1, 2), context=_CONTEXT)

    assert distribution == {1: 0.75, 2: 0.25}
    assert list(distribution) == [1, 2]
    assert all(type(probability) is float for probability in distribution.values())
    assert probabilities == {2: Decimal("0.25"), 1: Fraction(3, 4)}


def test_float32_sized_rounding_error_is_accepted() -> None:
    assert _validate_scalar_distribution({0: 0.5000004, 1: 0.5}, (0, 1), context=_CONTEXT) == {0: 0.5000004, 1: 0.5}


@pytest.mark.parametrize("probabilities", [None, [0.5, 0.5], (0.5, 0.5)])
def test_distribution_must_be_a_mapping(probabilities: object) -> None:
    with pytest.raises(InvalidPolicyError) as error:
        _validate_scalar_distribution(  # type: ignore[arg-type]
            probabilities, (0, 1), context=_CONTEXT
        )

    assert _CONTEXT in str(error.value)
    assert "mapping" in str(error.value)


@pytest.mark.parametrize(
    ("legal_actions", "message"),
    [((), "empty legal-action sequence"), ((2, 1, 2, 1), "duplicate legal actions: [1, 2]")],
)
def test_legal_action_sequence_must_be_nonempty_and_unique(legal_actions: tuple[int, ...], message: str) -> None:
    with pytest.raises(InvalidPolicyError) as error:
        _validate_scalar_distribution({}, legal_actions, context=_CONTEXT)

    assert _CONTEXT in str(error.value)
    assert message in str(error.value)


def test_distribution_requires_exactly_the_legal_actions_in_sorted_errors() -> None:
    with pytest.raises(InvalidPolicyError) as error:
        _validate_scalar_distribution({5: 0.5, 3: 0.5}, (4, 2, 0), context=_CONTEXT)

    assert str(error.value) == (f"{_CONTEXT} omitted legal actions: [0, 2, 4]; returned illegal actions: [3, 5]")


@pytest.mark.parametrize("value", ["1.0", object(), 1 + 0j])
def test_probability_values_must_be_real_numeric_values(value: object) -> None:
    with pytest.raises(InvalidPolicyError) as error:
        _validate_scalar_distribution(
            {0: value},  # type: ignore[dict-item]
            (0,),
            context=_CONTEXT,
        )

    assert _CONTEXT in str(error.value)
    assert "non-numeric probability" in str(error.value)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({0: -0.1, 1: 1.1}, "negative probability"),
        ({0: float("nan"), 1: 1.0}, "non-finite probability"),
        ({0: float("inf"), 1: 0.0}, "non-finite probability"),
        ({0: float("-inf"), 1: float("inf")}, "non-finite probability"),
    ],
)
def test_probability_values_must_be_finite_and_nonnegative(probabilities: dict[int, float], message: str) -> None:
    with pytest.raises(InvalidPolicyError) as error:
        _validate_scalar_distribution(probabilities, (0, 1), context=_CONTEXT)

    assert _CONTEXT in str(error.value)
    assert message in str(error.value)


def test_invalid_total_is_reported_without_repairing_distribution() -> None:
    probabilities = {0: 0.4, 1: 0.5}

    with pytest.raises(InvalidPolicyError) as error:
        _validate_scalar_distribution(probabilities, (0, 1), context=_CONTEXT)

    assert _CONTEXT in str(error.value)
    assert "got total 0.9" in str(error.value)
    assert probabilities == {0: 0.4, 1: 0.5}


def test_probability_tolerance_is_absolute_one_part_per_million() -> None:
    with pytest.raises(InvalidPolicyError):
        _validate_scalar_distribution({0: 1.0000011, 1: 0.0}, (0, 1), context=_CONTEXT)


def test_missing_zero_probability_action_is_rejected() -> None:
    with pytest.raises(InvalidPolicyError, match="omitted legal actions: \\[1\\]"):
        _validate_scalar_distribution({0: 1.0}, (0, 1), context=_CONTEXT)


def test_valid_batched_distribution_is_returned_without_mutating_inputs() -> None:
    probabilities = np.array([[0.25, 0.0, 0.75], [0.0, 1.0, 0.0]])
    mask = np.array([[True, False, True], [False, True, False]])
    original_probabilities = probabilities.copy()
    original_mask = mask.copy()

    result = _validate_batched_distribution(probabilities, (10, 30, 90), mask, context=_CONTEXT)

    np.testing.assert_array_equal(result, probabilities)
    np.testing.assert_array_equal(probabilities, original_probabilities)
    np.testing.assert_array_equal(mask, original_mask)


def test_empty_probability_batch_is_valid() -> None:
    result = _validate_batched_distribution(
        np.empty((0, 3)), (10, 30, 90), np.empty((0, 3), dtype=bool), context=_CONTEXT
    )

    assert result.shape == (0, 3)


@pytest.mark.parametrize("probabilities", [np.array([0.5, 0.5]), np.ones((1, 2, 1)), [["0.5", "0.5"]]])
def test_batched_distribution_must_be_a_two_dimensional_numeric_array(probabilities: object) -> None:
    with pytest.raises(InvalidPolicyError, match="two-dimensional numeric array"):
        _validate_batched_distribution(probabilities, (10, 30), np.ones((1, 2), dtype=bool), context=_CONTEXT)


def test_batched_distribution_shape_must_match_mask_exactly() -> None:
    with pytest.raises(InvalidPolicyError, match=r"returned shape \(2, 2\).+\(2, 3\)"):
        _validate_batched_distribution(
            np.full((2, 2), 0.5), (10, 30, 90), np.ones((2, 3), dtype=bool), context=_CONTEXT
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_batched_probabilities_must_be_finite(value: float) -> None:
    probabilities = np.array([[0.5, value]])

    with pytest.raises(InvalidPolicyError, match=r"non-finite probability in row 0 for action 30"):
        _validate_batched_distribution(probabilities, (10, 30), np.ones((1, 2), dtype=bool), context=_CONTEXT)


def test_batched_probabilities_must_be_nonnegative() -> None:
    with pytest.raises(InvalidPolicyError, match=r"negative probability in row 1 for action 30"):
        _validate_batched_distribution(
            np.array([[1.0, 0.0], [1.1, -0.1]]), (10, 30), np.ones((2, 2), dtype=bool), context=_CONTEXT
        )


def test_illegal_batched_probabilities_must_be_exactly_zero() -> None:
    with pytest.raises(InvalidPolicyError, match=r"illegal action 30 in row 1"):
        _validate_batched_distribution(
            np.array([[1.0, 0.0], [0.9, 1e-15]]), (10, 30), np.array([[True, False], [True, False]]), context=_CONTEXT
        )


def test_each_batched_row_must_sum_to_one_without_repair() -> None:
    probabilities = np.array([[0.5, 0.5], [0.4, 0.5]])
    original = probabilities.copy()

    with pytest.raises(InvalidPolicyError, match=r"row 1 must sum to one; got total 0.9"):
        _validate_batched_distribution(probabilities, (10, 30), np.ones((2, 2), dtype=bool), context=_CONTEXT)

    np.testing.assert_array_equal(probabilities, original)
