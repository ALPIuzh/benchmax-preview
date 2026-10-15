from dataclasses import dataclass

import pytest

from benchmax.execution.worker import InProcessRuntime, evaluation_kernel, prepared_resource, use_evaluation_runtime

_constructions: list[str] = []


@dataclass
class ClosableResource:
    name: str
    closed: bool = False

    def close(self) -> None:
        self.closed = True


_built_resources: list[ClosableResource] = []


@prepared_resource
def build_resource(name: str) -> ClosableResource:
    _constructions.append(name)
    resource = ClosableResource(name)
    _built_resources.append(resource)
    return resource


@prepared_resource
def failing_resource(name: str) -> None:
    _constructions.append(name)
    raise RuntimeError(name)


@prepared_resource
def defaulted_resource(name: str, copies: int = 1) -> ClosableResource:
    _constructions.append(f"{name}:{copies}")
    return ClosableResource(name)


@evaluation_kernel
def resource_identity(name: str) -> int:
    return id(build_resource(name))


@evaluation_kernel
def fail_during_preparation(name: str) -> None:
    failing_resource(name)


@evaluation_kernel
def defaulted_resource_identity(style: str) -> int:
    if style == "positional":
        resource = defaulted_resource("tree", 1)
    elif style == "keyword":
        resource = defaulted_resource(name="tree", copies=1)
    else:
        resource = defaulted_resource("tree")

    return id(resource)


def test_prepared_resources_are_reused_within_one_runtime() -> None:
    runtime = InProcessRuntime()

    with use_evaluation_runtime(runtime):
        first_identity = resource_identity("tree")
        second_identity = resource_identity(name="tree")
        other_identity = resource_identity("other")

    assert first_identity == second_identity
    assert other_identity != first_identity
    assert _constructions == ["tree", "other"]
    assert runtime.num_prepared_resources == 2


def test_prepared_resources_are_scoped_to_the_runtime() -> None:
    first_runtime = InProcessRuntime()
    second_runtime = InProcessRuntime()

    with use_evaluation_runtime(first_runtime):
        first = resource_identity("tree")

    with use_evaluation_runtime(second_runtime):
        second = resource_identity("tree")

    assert first != second
    assert _constructions == ["tree", "tree"]


def test_equivalent_call_spellings_share_one_prepared_resource() -> None:
    runtime = InProcessRuntime()

    with use_evaluation_runtime(runtime):
        identities = {defaulted_resource_identity(style) for style in ("positional", "keyword", "default")}

    assert len(identities) == 1
    assert _constructions == ["tree:1"]


def test_clear_releases_resources_and_allows_reconstruction() -> None:
    runtime = InProcessRuntime()

    with use_evaluation_runtime(runtime):
        first_identity = resource_identity("tree")

    first = _built_resources[0]
    runtime.clear_prepared_resources()

    assert first.closed is True
    assert runtime.num_prepared_resources == 0

    with use_evaluation_runtime(runtime):
        second_identity = resource_identity("tree")

    assert second_identity != first_identity
    assert _constructions == ["tree", "tree"]


def test_failed_construction_is_not_cached() -> None:
    runtime = InProcessRuntime()

    with use_evaluation_runtime(runtime):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="broken"):
                fail_during_preparation("broken")

    assert _constructions == ["broken", "broken"]
    assert runtime.num_prepared_resources == 0


def test_prepared_resources_require_hashable_arguments() -> None:
    runtime = InProcessRuntime()

    with use_evaluation_runtime(runtime), pytest.raises(TypeError, match="must be hashable"):
        resource_identity(["not", "hashable"])  # type: ignore[arg-type]


def test_prepared_resources_can_only_be_used_inside_a_kernel() -> None:
    with pytest.raises(RuntimeError, match="kernel is running"):
        build_resource("outside")


@pytest.fixture(autouse=True)
def clear_recorded_resources():
    _constructions.clear()
    _built_resources.clear()
    yield
    _constructions.clear()
    _built_resources.clear()
