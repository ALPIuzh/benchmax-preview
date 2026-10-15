import numpy as np
import pytest

from benchmax import InvalidPolicyError, PolicyProfile
from benchmax.execution.worker import (
    InProcessRuntime,
    PolicyBatch,
    PolicyProfileOracle,
    evaluation_kernel,
    use_evaluation_runtime,
)


@evaluation_kernel
def synthetic_policy_kernel(
    oracle: PolicyProfileOracle, *, rows: int, batch_size: int
) -> tuple[tuple[float, ...], ...]:
    """Build a strategy through the same batched oracle boundary as a kernel."""

    strategy = np.empty((rows, 3), dtype=np.float64)

    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        row_ids = np.arange(start, stop)
        observations = np.column_stack((row_ids, row_ids % 2)).astype(np.float64)
        legal_action_mask = np.ones((stop - start, 3), dtype=bool)
        oracle.evaluate(
            PolicyBatch(
                player=0, observations=observations, action_space=(0, 1, 2), legal_action_mask=legal_action_mask,
                probabilities=strategy[start:stop]
            )
        )

    return tuple(tuple(float(value) for value in row) for row in strategy)


def test_policy_oracle_writes_directly_into_the_kernel_destination() -> None:
    calls: list[tuple[int, np.ndarray, tuple[int, ...], np.ndarray]] = []

    def policy(player, observations, action_space, legal_action_mask):
        calls.append((player, observations.copy(), action_space, legal_action_mask.copy()))
        probabilities = np.zeros_like(legal_action_mask, dtype=np.float64)
        probabilities[:, 1] = 1.0
        return probabilities

    with use_evaluation_runtime(InProcessRuntime()):
        strategy = np.asarray(
            synthetic_policy_kernel(PolicyProfileOracle(PolicyProfile(policy)), rows=10, batch_size=4)
        )

    assert [len(observations) for _, observations, _, _ in calls] == [4, 4, 2]
    np.testing.assert_array_equal(strategy[:, 0], 0.0)
    np.testing.assert_array_equal(strategy[:, 1], 1.0)
    np.testing.assert_array_equal(strategy[:, 2], 0.0)


def test_policy_validation_happens_at_the_oracle_boundary() -> None:
    def invalid_policy(player, observations, action_space, legal_action_mask):
        del player, observations, action_space
        return np.ones_like(legal_action_mask, dtype=np.float64)

    with use_evaluation_runtime(InProcessRuntime()), pytest.raises(InvalidPolicyError, match="probabilities in row 0"):
        synthetic_policy_kernel(PolicyProfileOracle(PolicyProfile(invalid_policy)), rows=3, batch_size=2)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    (
        ({"observations": np.zeros(3)}, ValueError, "batch dimension"),
        ({"legal_action_mask": np.ones((2, 2), dtype=bool)}, ValueError, "mask"),
        ({"legal_action_mask": np.ones((2, 3), dtype=np.float64)}, TypeError, "boolean"),
        ({"probabilities": np.empty((2, 2))}, ValueError, "destination"),
        ({"probabilities": np.empty((2, 3), dtype=np.int64)}, TypeError, "floating-point"),
    ),
)
def test_policy_batches_reject_invalid_buffer_contracts(
    changes: dict[str, np.ndarray], error: type[Exception], message: str
) -> None:
    arguments = {
        "player": 0,
        "observations": np.zeros((2, 4)),
        "action_space": (0, 1, 2),
        "legal_action_mask": np.ones((2, 3), dtype=bool),
        "probabilities": np.empty((2, 3), dtype=np.float64),
    }
    arguments.update(changes)

    with pytest.raises(error, match=message):
        PolicyBatch(**arguments)


def test_policy_batches_require_a_writable_destination() -> None:
    probabilities = np.empty((2, 3), dtype=np.float64)
    probabilities.flags.writeable = False

    with pytest.raises(ValueError, match="writable"):
        PolicyBatch(
            player=0, observations=np.zeros((2, 4)), action_space=(0, 1, 2),
            legal_action_mask=np.ones((2, 3), dtype=bool), probabilities=probabilities
        )


def test_policy_batches_normalize_array_like_values_once() -> None:
    batch = PolicyBatch(
        player=0,
        observations=[[0, 1], [1, 0]],  # type: ignore[arg-type]
        action_space=(0, 1),
        legal_action_mask=[[True, False], [False, True]],  # type: ignore[arg-type]
        probabilities=[[0.0, 0.0], [0.0, 0.0]],  # type: ignore[arg-type]
    )

    assert isinstance(batch.observations, np.ndarray)
    assert isinstance(batch.legal_action_mask, np.ndarray)
    assert isinstance(batch.probabilities, np.ndarray)


def test_policy_batches_reject_non_numeric_observations_consistently() -> None:
    with pytest.raises(TypeError, match="numeric or boolean"):
        PolicyBatch(
            player=0, observations=np.array([["private"]]), action_space=(0,),
            legal_action_mask=np.ones((1, 1), dtype=bool), probabilities=np.empty((1, 1), dtype=np.float64)
        )
