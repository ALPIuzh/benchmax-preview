"""Worker primitives for transparent, isolated evaluation kernels.

The in-process backend defines reference semantics. Ordinary decorated calls
use the lazy persistent local worker.
"""

from typing import TYPE_CHECKING, Any

from benchmax.execution.worker.errors import EvaluationWorkerError
from benchmax.execution.worker.kernel import (
    EvaluationRuntime,
    InProcessRuntime,
    KernelValue,
    close_default_evaluation_runtime,
    evaluation_kernel,
    get_evaluation_runtime,
    use_evaluation_runtime,
)
from benchmax.execution.worker.oracle import PolicyBatch, PolicyOracle, PolicyProfileOracle
from benchmax.execution.worker.prepared import prepared_resource

if TYPE_CHECKING:
    from benchmax.execution.worker.local import (
        EvaluationCancelled,
        LocalInvocationProfile,
        LocalWorkerRuntime,
    )

_LOCAL_EXPORTS = {
    "EvaluationCancelled",
    "LocalInvocationProfile",
    "LocalWorkerRuntime",
}


def __getattr__(name: str) -> Any:
    if name not in _LOCAL_EXPORTS:
        raise AttributeError(name)

    from benchmax.execution.worker import local

    value = getattr(local, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LOCAL_EXPORTS)

__all__ = [
    "EvaluationCancelled",
    "EvaluationRuntime",
    "EvaluationWorkerError",
    "InProcessRuntime",
    "KernelValue",
    "LocalInvocationProfile",
    "LocalWorkerRuntime",
    "PolicyBatch",
    "PolicyOracle",
    "PolicyProfileOracle",
    "close_default_evaluation_runtime",
    "evaluation_kernel",
    "get_evaluation_runtime",
    "prepared_resource",
    "use_evaluation_runtime",
]
