"""Transparent evaluation-kernel dispatch and the reference worker backend."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import os
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ParamSpec, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

from benchmax.execution.values import (
    PolicyEndpoint,
    PortableValue,
    marshal_value,
)
from benchmax.execution.worker.oracle import PolicyOracle
from benchmax.execution.worker.prepared import (
    PreparedResourceStore,
    _require_importable_function,
    use_prepared_resource_store,
)

P = ParamSpec("P")
KernelValue: TypeAlias = PortableValue
R = TypeVar("R", bound=KernelValue)

_KERNEL_ATTRIBUTE = "__benchmax_kernel__"


@dataclass(frozen=True, slots=True)
class KernelDefinition:
    """Stable identity and undecorated implementation of one kernel.

    The function is used only by an in-process runtime. Isolated runtimes use
    ``identifier`` to import the same decorated symbol in their own process and
    recover its definition with :func:`get_kernel_definition`.
    """

    identifier: str
    function: Callable[..., Any]


@runtime_checkable
class EvaluationRuntime(Protocol):
    """Backend capable of executing a decorated evaluation kernel."""

    def invoke(self, kernel: KernelDefinition, args: tuple[Any, ...], kwargs: dict[str, Any], /) -> Any:
        """Execute one invocation and return its result."""


class InProcessRuntime:
    """Semantic reference backend that calls the original function directly."""

    def __init__(self) -> None:
        self._prepared_resources = PreparedResourceStore()

    def invoke(self, kernel: KernelDefinition, args: tuple[Any, ...], kwargs: dict[str, Any], /) -> Any:
        """Execute ``kernel`` without serialization or exception translation."""

        _validate_kernel_arguments(args, kwargs)

        with use_prepared_resource_store(self._prepared_resources):
            result = kernel.function(*args, **kwargs)

        _validate_kernel_result(result)
        return result

    def clear_prepared_resources(self) -> None:
        """Release every prepared value owned by this runtime."""

        self._prepared_resources.clear()

    @property
    def num_prepared_resources(self) -> int:
        """Return the number of prepared values retained by this runtime."""

        return len(self._prepared_resources)


_DEFAULT_RUNTIME: EvaluationRuntime | None = None
_DEFAULT_RUNTIME_LOCK = threading.Lock()
_RUNTIME: contextvars.ContextVar[EvaluationRuntime | None] = contextvars.ContextVar(
    "benchmax_evaluation_runtime", default=None
)


def _reset_default_runtime_lock_after_fork() -> None:
    """Replace a possibly locked parent mutex in a freshly forked child."""

    global _DEFAULT_RUNTIME_LOCK
    _DEFAULT_RUNTIME_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_default_runtime_lock_after_fork)


def get_evaluation_runtime() -> EvaluationRuntime:
    """Return the context override or the lazy default local runtime."""

    runtime = _RUNTIME.get()
    if runtime is not None:
        return runtime

    return _get_default_evaluation_runtime()


def _get_default_evaluation_runtime() -> EvaluationRuntime:
    global _DEFAULT_RUNTIME

    with _DEFAULT_RUNTIME_LOCK:
        if _DEFAULT_RUNTIME is None:
            # Deferred to avoid importing process machinery in library modules
            # that only define kernels, and to keep the worker's import graph
            # free of a construction-time cycle.
            from benchmax.execution.worker.local import LocalWorkerRuntime

            _DEFAULT_RUNTIME = LocalWorkerRuntime()

        return _DEFAULT_RUNTIME


def close_default_evaluation_runtime() -> None:
    """Close and forget the process-wide default runtime, if it was used.

    Ordinary applications can rely on interpreter cleanup. Tests, notebooks,
    and applications with explicit lifecycle boundaries can call this function
    to release prepared memory immediately. A later kernel call creates a new
    clean worker.
    """

    global _DEFAULT_RUNTIME

    with _DEFAULT_RUNTIME_LOCK:
        runtime = _DEFAULT_RUNTIME
        _DEFAULT_RUNTIME = None

    if runtime is not None:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()


@contextlib.contextmanager
def use_evaluation_runtime(runtime: EvaluationRuntime) -> Iterator[None]:
    """Select ``runtime`` within the dynamic scope of this context manager.

    Context-local selection keeps tests and concurrent tasks independent. It
    is an advanced override, since ordinary library code calls decorated kernels
    without choosing a backend itself.
    """

    if not isinstance(runtime, EvaluationRuntime):
        raise TypeError("an evaluation runtime must provide invoke()")

    token = _RUNTIME.set(runtime)

    try:
        yield
    finally:
        _RUNTIME.reset(token)


def evaluation_kernel(function: Callable[P, R]) -> Callable[P, R]:
    """Mark a module-level function as an evaluation kernel.

    The returned wrapper has the same call signature and transparently sends
    each invocation through the active runtime.

    Portable arguments are finite scalar/container values plus policy oracles,
    and results are finite scalar/container :data:`KernelValue` values. NumPy
    arrays and arbitrary Python objects require an explicit runtime-owned data
    plane or durable artifact. The in-process reference backend enforces the
    same boundary as isolated workers.

    Raises:
        TypeError: If applied to something other than a synchronous,
            module-level named function.
    """

    _require_importable_function(function, decorator="evaluation_kernel", noun="an evaluation kernel")

    definition = KernelDefinition(identifier=f"{function.__module__}:{function.__qualname__}", function=function)

    @functools.wraps(function)
    def dispatched(*args: P.args, **kwargs: P.kwargs) -> R:
        runtime = get_evaluation_runtime()
        return cast(R, runtime.invoke(definition, args, kwargs))

    setattr(dispatched, _KERNEL_ATTRIBUTE, definition)
    return dispatched


def get_kernel_definition(function: object) -> KernelDefinition:
    """Return the definition attached by :func:`evaluation_kernel`.

    Raises:
        TypeError: If ``function`` is not a decorated evaluation kernel.
    """

    definition = getattr(function, _KERNEL_ATTRIBUTE, None)
    if not isinstance(definition, KernelDefinition):
        raise TypeError("object is not an evaluation kernel")

    return definition


def resolve_kernel(identifier: str) -> KernelDefinition:
    """Import and validate the decorated kernel named by ``identifier``.

    This is the import path an isolated worker will use. Resolving the public
    decorated symbol rather than an arbitrary pickled callable makes code
    identity explicit and reproducible.

    Raises:
        ValueError: If the identifier is malformed or resolves to a different
            kernel identity.
        ImportError: If its module cannot be imported.
        AttributeError: If its symbol does not exist.
        TypeError: If its symbol is not an evaluation kernel.
    """

    module_name, separator, symbol_name = identifier.partition(":")
    if not separator or not module_name or not symbol_name or "." in symbol_name:
        raise ValueError("a kernel identifier must have the form 'importable.module:function'")

    module = importlib.import_module(module_name)
    definition = get_kernel_definition(getattr(module, symbol_name))
    if definition.identifier != identifier:
        raise ValueError(f"kernel {identifier!r} resolved as {definition.identifier!r}")

    return definition


def _validate_kernel_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    def policy_reference(value: object) -> PolicyEndpoint | None:
        return PolicyEndpoint(0) if isinstance(value, PolicyOracle) else None

    marshal_value(args, location="kernel arguments", policy_reference=policy_reference)
    marshal_value(kwargs, location="kernel arguments", policy_reference=policy_reference)


def _validate_kernel_result(result: object) -> None:
    marshal_value(result, location="kernel result")
