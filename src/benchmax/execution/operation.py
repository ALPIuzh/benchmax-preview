"""Add job submission controls to training and evaluation functions."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import Parameter, Signature, isasyncgenfunction, iscoroutinefunction, isgeneratorfunction, signature
from typing import TypeVar

from benchmax.execution.jobs.job import Job
from benchmax.execution.targets.base import Target

R = TypeVar("R")


def executable(*, job: Callable[..., Job[R]]) -> Callable[[Callable[..., R]], Callable[..., R | Job[R]]]:
    """Add keyword-only ``submit`` and ``target`` options to a synchronous function.

    ``job`` receives domain arguments with defaults applied and returns an
    unstarted Job. Reserved names fail at decoration time. Methods retain normal
    binding, and public overloads can specify synchronous/deferred return types.
    """

    def decorate(function: Callable[..., R]) -> Callable[..., R | Job[R]]:
        if iscoroutinefunction(function) or isgeneratorfunction(function) or isasyncgenfunction(function):
            raise TypeError("executable requires a synchronous, non-generator function")

        original = signature(function)
        conflicts = {"submit", "target"}.intersection(original.parameters)
        if conflicts:
            raise TypeError(f"{function.__qualname__} uses reserved execution parameters: {', '.join(sorted(conflicts))}")

        @wraps(function)
        def execute(*args, submit: bool = False, target: str | Target | None = None, **kwargs):
            if not isinstance(submit, bool):
                raise TypeError("submit must be a bool")

            if target is not None and not (isinstance(target, Target) or isinstance(target, str) and bool(target)):
                raise TypeError("target must be a Target, a non-empty string, or None")

            if target is not None and not submit:
                raise ValueError(
                    "target selects where a submitted job runs; pass submit=True "
                    f"or omit target to {function.__name__} synchronously"
                )

            if not submit:
                return function(*args, **kwargs)

            arguments = original.bind(*args, **kwargs)
            arguments.apply_defaults()
            deferred = job(*arguments.args, **arguments.kwargs)
            if not isinstance(deferred, Job):
                raise TypeError("the job adapter must return a Job")

            return deferred.run(target) if target is not None else deferred

        parameters = list(original.parameters.values())
        index = next((i for i, p in enumerate(parameters) if p.kind == Parameter.VAR_KEYWORD), len(parameters))
        parameters[index:index] = [
            Parameter("submit", Parameter.KEYWORD_ONLY, default=False, annotation=bool),
            Parameter("target", Parameter.KEYWORD_ONLY, default=None, annotation="str | Target | None"),
        ]
        execute.__annotations__ = {**function.__annotations__, "submit": bool, "target": "str | Target | None"}
        result = original.return_annotation
        if result is not Signature.empty:
            result = f"{result} | Job[{result}]" if isinstance(result, str) else result | Job[result]
            execute.__annotations__["return"] = result

        execute.__signature__ = original.replace(parameters=parameters, return_annotation=result)
        return execute

    return decorate
