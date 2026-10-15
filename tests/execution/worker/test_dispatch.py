import inspect
import subprocess
import sys
from typing import Any

import pytest

from benchmax.execution.worker import (
    InProcessRuntime,
    LocalWorkerRuntime,
    close_default_evaluation_runtime,
    evaluation_kernel,
    get_evaluation_runtime,
    prepared_resource,
    use_evaluation_runtime,
)
from benchmax.execution.worker import kernel as worker_kernel
from benchmax.execution.worker.kernel import KernelDefinition, get_kernel_definition, resolve_kernel
from tests.execution.worker._kernels import process_identity


@evaluation_kernel
def synthetic_kernel(value: int, *, scale: int = 2) -> int:
    """Multiply a synthetic input."""

    return value * scale


class RecordingRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[KernelDefinition, tuple[Any, ...], dict[str, Any]]] = []

    def invoke(self, kernel: KernelDefinition, args: tuple[Any, ...], kwargs: dict[str, Any], /) -> Any:
        self.calls.append((kernel, args, kwargs))
        return kernel.function(*args, **kwargs)


def test_a_kernel_is_an_ordinary_transparent_call_in_process() -> None:
    runtime = InProcessRuntime()

    with use_evaluation_runtime(runtime):
        assert synthetic_kernel(3) == 6
        assert synthetic_kernel(3, scale=4) == 12


def test_the_default_runtime_is_lazy_local_and_transparent() -> None:
    close_default_evaluation_runtime()

    try:
        runtime = get_evaluation_runtime()
        assert isinstance(runtime, LocalWorkerRuntime)
        assert runtime.worker_pid is None
        assert process_identity() == runtime.worker_pid
    finally:
        close_default_evaluation_runtime()


def test_importing_the_public_kernel_surface_does_not_import_local_ipc() -> None:
    program = """
import sys
import benchmax.execution.worker
assert "benchmax.execution.worker.local" not in sys.modules
assert "torch" not in sys.modules
assert "benchmarl" not in sys.modules
assert "benchmax.integrations.exp_a_spiel._exp_a_spiel" not in sys.modules
"""

    subprocess.run([sys.executable, "-c", program], check=True)


def test_the_decorator_preserves_function_introspection() -> None:
    assert synthetic_kernel.__name__ == "synthetic_kernel"
    assert synthetic_kernel.__doc__ == "Multiply a synthetic input."
    assert str(inspect.signature(synthetic_kernel)) == "(value: int, *, scale: int = 2) -> int"
    assert synthetic_kernel.__wrapped__(3, scale=5) == 15


def test_runtime_selection_is_context_local_and_restored() -> None:
    outer = RecordingRuntime()
    inner = RecordingRuntime()
    default = get_evaluation_runtime()

    with use_evaluation_runtime(outer):
        assert synthetic_kernel(2) == 4
        assert get_evaluation_runtime() is outer

        with use_evaluation_runtime(inner):
            assert synthetic_kernel(3, scale=3) == 9
            assert get_evaluation_runtime() is inner

        assert get_evaluation_runtime() is outer
        assert synthetic_kernel(4) == 8

    assert get_evaluation_runtime() is default
    assert [(args, kwargs) for _, args, kwargs in outer.calls] == [((2,), {}), ((4,), {})]
    assert [(args, kwargs) for _, args, kwargs in inner.calls] == [((3,), {"scale": 3})]


def test_the_in_process_runtime_preserves_exception_identity() -> None:
    expected = RuntimeError("policy failed")
    definition = KernelDefinition("tests:failure", lambda: _raise(expected))

    with pytest.raises(RuntimeError) as raised:
        InProcessRuntime().invoke(definition, (), {})

    assert raised.value is expected


def _raise(error: Exception) -> None:
    raise error


def test_the_reference_runtime_enforces_portable_arguments_and_results() -> None:
    runtime = InProcessRuntime()
    identity = KernelDefinition("tests:identity", lambda value: value)
    arbitrary_result = KernelDefinition("tests:result", object)

    class UnrelatedEvaluator:
        def evaluate(self, value: object) -> object:
            return value

    with pytest.raises(TypeError, match="kernel arguments.*explicit data plane"):
        runtime.invoke(identity, (object(),), {})

    with pytest.raises(TypeError, match="kernel arguments.*explicit data plane"):
        runtime.invoke(identity, (UnrelatedEvaluator(),), {})

    with pytest.raises(TypeError, match="kernel result.*explicit data plane"):
        runtime.invoke(arbitrary_result, (), {})


def test_the_default_runtime_lock_is_reinitialized_after_fork() -> None:
    inherited = worker_kernel._DEFAULT_RUNTIME_LOCK
    inherited.acquire()

    try:
        worker_kernel._reset_default_runtime_lock_after_fork()
        replacement = worker_kernel._DEFAULT_RUNTIME_LOCK

        assert replacement is not inherited
        assert replacement.acquire(blocking=False)
        replacement.release()
    finally:
        inherited.release()


def test_kernel_definitions_have_stable_import_identities() -> None:
    definition = get_kernel_definition(synthetic_kernel)

    assert definition.identifier == (f"{synthetic_kernel.__module__}:synthetic_kernel")
    assert definition.function is synthetic_kernel.__wrapped__
    assert resolve_kernel(definition.identifier) is definition


async def asynchronous_kernel():
    pass


def generator_kernel():
    yield None


async def asynchronous_generator_kernel():
    yield None


def test_kernels_reject_nested_functions() -> None:
    def nested():
        pass

    with pytest.raises(TypeError, match="module-level named"):
        evaluation_kernel(nested)


@pytest.mark.parametrize("decorator", [evaluation_kernel, prepared_resource])
@pytest.mark.parametrize("candidate", [asynchronous_kernel, generator_kernel, asynchronous_generator_kernel])
def test_worker_decorators_require_synchronous_functions(decorator, candidate) -> None:
    with pytest.raises(TypeError, match="synchronous"):
        decorator(candidate)


def test_kernels_require_functions_not_classes_or_methods() -> None:
    class Candidate:
        def method(self):
            pass

    with pytest.raises(TypeError, match="module-level named"):
        evaluation_kernel(Candidate.method)

    with pytest.raises(TypeError, match="only decorate a function"):
        evaluation_kernel(Candidate)


def test_invalid_runtime_and_kernel_objects_are_rejected() -> None:
    with (
        pytest.raises(TypeError, match=r"provide invoke\(\)"),
        use_evaluation_runtime(object()),  # type: ignore[arg-type]
    ):
        pass

    with pytest.raises(TypeError, match="not an evaluation kernel"):
        get_kernel_definition(lambda: None)


@pytest.mark.parametrize("identifier", ("", "missing-colon", ":symbol", "module:", "module:Class.method"))
def test_malformed_kernel_identifiers_are_rejected(identifier: str) -> None:
    with pytest.raises(ValueError, match="kernel identifier"):
        resolve_kernel(identifier)
