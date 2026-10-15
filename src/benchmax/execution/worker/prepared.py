"""Worker-owned construction and reuse of expensive prepared resources."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class PreparedResourceDefinition:
    """Stable identity and constructor for one prepared-resource kind."""

    identifier: str
    function: Callable[..., Any]
    signature: inspect.Signature


class PreparedResourceStore:
    """Own prepared values for the lifetime chosen by one runtime."""

    def __init__(self) -> None:
        self._values: dict[tuple[Any, ...], Any] = {}
        self._lock = threading.RLock()

    def get(self, definition: PreparedResourceDefinition, args: tuple[Any, ...], kwargs: dict[str, Any], /) -> Any:
        """Return one value, constructing it once for an equal invocation."""

        bound = definition.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = []

        for name, value in bound.arguments.items():
            parameter = definition.signature.parameters[name]
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                value = tuple(sorted(value.items()))

            arguments.append((name, value))

        key = (definition.identifier, tuple(arguments))

        try:
            hash(key)
        except TypeError as error:
            raise TypeError(f"arguments to prepared resource {definition.identifier!r} must be hashable") from error

        with self._lock:
            try:
                return self._values[key]
            except KeyError:
                value = definition.function(*args, **kwargs)
                self._values[key] = value
                return value

    def clear(self) -> None:
        """Release every value, closing resources that expose ``close()``."""

        with self._lock:
            resources = {id(value): value for value in self._values.values()}
            self._values.clear()

        callbacks = contextlib.ExitStack()

        for resource in resources.values():
            close = getattr(resource, "close", None)
            if callable(close):
                callbacks.callback(close)

        callbacks.close()

    def __len__(self) -> int:
        return len(self._values)


_STORE: contextvars.ContextVar[PreparedResourceStore | None] = contextvars.ContextVar(
    "benchmax_prepared_resource_store", default=None
)


@contextlib.contextmanager
def use_prepared_resource_store(store: PreparedResourceStore) -> Iterator[None]:
    """Make ``store`` available while a runtime invokes one kernel."""

    token = _STORE.set(store)

    try:
        yield
    finally:
        _STORE.reset(token)


def _require_importable_function(function: object, *, decorator: str, noun: str) -> None:
    """Reject callables a worker process cannot re-import by module and name."""

    if not inspect.isfunction(function):
        raise TypeError(f"{decorator} can only decorate a function")

    if function.__qualname__ != function.__name__ or function.__name__ == "<lambda>":
        raise TypeError(f"{noun} must be a module-level named function")

    if function.__module__ == "__main__":
        raise TypeError(f"{noun} must live in an importable module")

    if any(check(function) for check in (
        inspect.iscoroutinefunction, inspect.isgeneratorfunction, inspect.isasyncgenfunction,
    )):
        raise TypeError(f"{noun} must be a synchronous function")


def prepared_resource(function: Callable[P, R]) -> Callable[P, R]:
    """Construct one runtime-owned value for each distinct argument set.

    The active runtime owns the store and decides when to retain or clear its prepared values.
    """

    _require_importable_function(function, decorator="prepared_resource", noun="a prepared resource")

    definition = PreparedResourceDefinition(
        identifier=f"{function.__module__}:{function.__qualname__}", function=function,
        signature=inspect.signature(function)
    )

    @functools.wraps(function)
    def get_or_create(*args: P.args, **kwargs: P.kwargs) -> R:
        store = _STORE.get()
        if store is None:
            raise RuntimeError("a prepared resource can only be used while an evaluation kernel is running")

        return cast(R, store.get(definition, args, kwargs))

    return get_or_create
