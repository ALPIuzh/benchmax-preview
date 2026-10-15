"""Small importable kernels used to exercise worker backends."""

from __future__ import annotations

import itertools
import os
import time

import numpy as np

from benchmax.execution.worker.kernel import evaluation_kernel
from benchmax.execution.worker.oracle import PolicyBatch, PolicyOracle
from benchmax.execution.worker.prepared import prepared_resource

_preparation_ids = itertools.count(1)


class _PreparedProbe:
    def __init__(self, label: str) -> None:
        self.label = label
        self.identifier = next(_preparation_ids)


@prepared_resource
def _prepare_probe(label: str) -> _PreparedProbe:
    return _PreparedProbe(label)


@evaluation_kernel
def process_identity() -> int:
    return os.getpid()


@evaluation_kernel
def prepared_probe(label: str) -> tuple[int, int]:
    probe = _prepare_probe(label)
    return os.getpid(), probe.identifier


@evaluation_kernel
def synthetic_policy_evaluation(
    policy: PolicyOracle, *, rows: int, observation_columns: int, actions: int, batch_size: int
) -> dict[str, int | float]:
    strategy = np.empty((rows, actions), dtype=np.float64)
    batches = 0

    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        row_ids = np.arange(start, stop, dtype=np.float64)
        observations = np.empty((stop - start, observation_columns), dtype=np.float64)

        for column in range(observation_columns):
            observations[:, column] = row_ids + column / 1000

        legal_action_mask = np.ones((stop - start, actions), dtype=bool)
        policy.evaluate(
            PolicyBatch(
                player=0, observations=observations, action_space=tuple(range(actions)),
                legal_action_mask=legal_action_mask, probabilities=strategy[start:stop]
            )
        )
        batches += 1

    return {
        "pid": os.getpid(),
        "batches": batches,
        "strategy_sum": float(strategy.sum()),
        "weighted_sum": float(strategy @ np.arange(1, actions + 1, dtype=np.float64) @ np.ones(rows, dtype=np.float64)),
    }


@evaluation_kernel
def raise_in_worker(message: str) -> None:
    raise ValueError(message)


@evaluation_kernel
def crash_worker(exit_code: int) -> None:
    os._exit(exit_code)


@evaluation_kernel
def sleep_in_worker(seconds: float) -> int:
    time.sleep(seconds)
    return os.getpid()


@evaluation_kernel
def return_array() -> np.ndarray:
    return np.zeros(1)
