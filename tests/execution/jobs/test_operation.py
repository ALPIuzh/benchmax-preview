"""The execution decorator owns routing without changing domain arguments."""

from inspect import Parameter, signature, unwrap
from unittest.mock import Mock

import pytest

from benchmax import Game
from benchmax.execution import Job, LocalTarget, executable
from benchmax.training import train


@pytest.mark.parametrize("function", [
    lambda submit: None, lambda *, target: None, lambda submit, /: None,
    lambda *target: None, lambda **submit: None,
])
def test_reserved_names_are_rejected_when_decorating(function) -> None:
    with pytest.raises(TypeError, match="reserved execution parameters"):
        executable(job=Mock())(function)


def test_decorating_twice_is_rejected() -> None:
    function = executable(job=Mock())(lambda value: value)

    with pytest.raises(TypeError, match="reserved execution parameters"):
        executable(job=Mock())(function)


def test_direct_calls_keep_domain_arguments_and_skip_job_preparation() -> None:
    factory = Mock()

    @executable(job=factory)
    def collect(value: int, **options) -> tuple:
        """A domain operation."""
        return value, options

    assert collect(3, submit=False, target=None, episodes=10) == (3, {"episodes": 10})
    factory.assert_not_called()
    assert collect.__name__ == "collect"
    assert collect.__doc__ == "A domain operation."
    assert set(signature(unwrap(collect)).parameters) == {"value", "options"}


@pytest.mark.parametrize("target", [None, "research", LocalTarget(runs="/tmp/benchmax-operation-test")])
def test_submitted_calls_prepare_a_job_and_run_it_only_with_a_target(target) -> None:
    deferred = Mock(spec=Job)
    deferred.run.return_value = deferred
    factory = Mock(return_value=deferred)

    @executable(job=factory)
    def collect(value, /, count=5, *, seed=7, **options):
        raise AssertionError("scientific work must not run during submission")

    assert collect(3, submit=True, target=target, format="arrays") is deferred
    factory.assert_called_once_with(3, 5, seed=7, format="arrays")
    if target is None:
        deferred.run.assert_not_called()
    else:
        deferred.run.assert_called_once_with(target)


def test_an_adapter_must_return_a_job() -> None:
    collect = executable(job=lambda value: value)(lambda value: value)

    with pytest.raises(TypeError, match="job adapter must return a Job"):
        collect(3, submit=True)


@pytest.mark.parametrize("options,error", [
    ({"submit": 1}, TypeError), ({"submit": None}, TypeError),
    ({"submit": True, "target": ""}, TypeError), ({"submit": True, "target": object()}, TypeError),
    ({"target": "local"}, ValueError),
])
def test_invalid_execution_options_fail_before_any_work(options, error) -> None:
    factory = Mock()

    @executable(job=factory)
    def collect(value):
        raise AssertionError("invalid execution options must fail before scientific work")

    with pytest.raises(error):
        collect(3, **options)

    factory.assert_not_called()


@pytest.mark.parametrize("args,kwargs", [
    ((), {}), ((3, 4), {}), ((3,), {"value": 4}), ((3,), {"unknown": 4}),
])
def test_deferred_calls_validate_the_scientific_signature(args, kwargs) -> None:
    factory = Mock()
    collect = executable(job=factory)(lambda value: value)

    with pytest.raises(TypeError):
        collect(*args, submit=True, **kwargs)

    factory.assert_not_called()


def test_methods_bind_the_instance_in_both_modes() -> None:
    deferred = Mock(spec=Job)
    factory = Mock(return_value=deferred)

    class Collector:
        @executable(job=factory)
        def collect(self, value=3):
            return self, value

    collector = Collector()
    assert collector.collect() == (collector, 3)
    assert collector.collect(value=5, submit=True) is deferred
    factory.assert_called_once_with(collector, 5)
    assert "self" not in signature(collector.collect).parameters


@pytest.mark.parametrize("function", [train, Game.evaluate])
def test_public_signatures_advertise_injected_execution_controls(function) -> None:
    public = signature(function)
    assert public.parameters["submit"].kind == Parameter.KEYWORD_ONLY
    assert public.parameters["submit"].default is False
    assert public.parameters["target"].kind == Parameter.KEYWORD_ONLY
    assert public.parameters["target"].default is None
    assert "Job" in str(public.return_annotation)
    assert "submit" not in signature(unwrap(function)).parameters
    assert "target" not in signature(unwrap(function)).parameters


def test_execution_rejects_functions_whose_bodies_run_after_return() -> None:
    async def coroutine():
        return 7

    async def async_generator():
        yield 7

    def generator():
        yield 7

    for function in (coroutine, async_generator, generator):
        with pytest.raises(TypeError, match="synchronous, non-generator"):
            executable(job=Mock())(function)
