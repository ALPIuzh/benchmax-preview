from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from benchmax import InvalidPolicyError, PolicyProfile
from benchmax.execution.worker import (
    EvaluationCancelled,
    EvaluationWorkerError,
    LocalWorkerRuntime,
    PolicyProfileOracle,
    use_evaluation_runtime,
)
from benchmax.execution.worker import local as worker_local
from benchmax.execution.worker.local import EvaluationWorkerCrashed, EvaluationWorkerExecutionError
from tests.execution.worker._kernels import (
    crash_worker,
    prepared_probe,
    process_identity,
    raise_in_worker,
    return_array,
    sleep_in_worker,
    synthetic_policy_evaluation,
)


def _uniform_policy(calls: list[np.ndarray] | None = None) -> PolicyProfileOracle:
    def policy(player, observations, action_space, legal_action_mask):
        del player, action_space

        if calls is not None:
            calls.append(observations.copy())

        probabilities = legal_action_mask.astype(np.float64)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    return PolicyProfileOracle(PolicyProfile(policy))


def test_policy_batches_and_results_match_the_in_process_reference() -> None:
    local_calls: list[np.ndarray] = []
    reference_calls: list[np.ndarray] = []
    arguments = {"rows": 19, "observation_columns": 5, "actions": 3, "batch_size": 7}
    reference = synthetic_policy_evaluation(_uniform_policy(reference_calls), **arguments)

    with LocalWorkerRuntime() as runtime, use_evaluation_runtime(runtime):
        isolated = synthetic_policy_evaluation(_uniform_policy(local_calls), **arguments)
        profile = runtime.last_profile

    assert isolated == {**reference, "pid": isolated["pid"]}
    assert isolated["pid"] != reference["pid"]
    assert profile is not None
    assert profile.policy_batches == 3
    assert profile.observation_bytes == 19 * 5 * np.dtype(np.float64).itemsize
    assert profile.legal_action_mask_bytes == 19 * 3
    assert profile.probability_bytes == 19 * 3 * np.dtype(np.float64).itemsize
    assert profile.wall_seconds >= profile.worker_kernel_seconds >= 0.0
    assert profile.worker_policy_roundtrip_seconds >= profile.parent_policy_seconds
    assert profile.worker_input_copy_seconds >= 0.0
    assert profile.worker_output_copy_seconds >= 0.0
    assert len(local_calls) == len(reference_calls) == 3

    for local, expected in zip(local_calls, reference_calls, strict=True):
        np.testing.assert_array_equal(local, expected)


def test_prepared_resources_are_reused_and_explicitly_clearable() -> None:
    with LocalWorkerRuntime() as runtime, use_evaluation_runtime(runtime):
        first_pid, first_resource = prepared_probe("tree")
        second_pid, second_resource = prepared_probe("tree")
        _, other_resource = prepared_probe("other")

        assert first_pid == second_pid == runtime.worker_pid
        assert first_resource == second_resource
        assert other_resource != first_resource

        runtime.clear_prepared_resources()
        rebuilt_pid, rebuilt_resource = prepared_probe("tree")

        assert rebuilt_pid == first_pid
        assert rebuilt_resource not in {first_resource, other_resource}


def test_prepared_resource_cleanup_discards_a_broken_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = LocalWorkerRuntime()
    runtime.start()
    state = runtime._state
    assert state is not None

    def fail_to_send(value: object) -> None:
        del value
        raise BrokenPipeError("closed")

    monkeypatch.setattr(state.channel, "send", fail_to_send)

    with pytest.raises(EvaluationWorkerCrashed, match="control channel"):
        runtime.clear_prepared_resources()

    assert runtime.worker_pid is None


def test_worker_exceptions_preserve_remote_type_and_traceback() -> None:
    with LocalWorkerRuntime() as runtime, use_evaluation_runtime(runtime):
        pid = process_identity()

        with pytest.raises(EvaluationWorkerExecutionError) as raised:
            raise_in_worker("broken kernel")

        assert raised.value.remote_type == "builtins.ValueError"
        assert "broken kernel" in str(raised.value)
        assert "raise_in_worker" in raised.value.remote_traceback
        assert process_identity() == pid


def test_policy_exceptions_are_reraised_in_the_parent_and_worker_survives() -> None:
    expected = InvalidPolicyError("bad policy")

    def fail(player, observations, action_space, legal_action_mask):
        del player, observations, action_space, legal_action_mask
        raise expected

    with LocalWorkerRuntime() as runtime, use_evaluation_runtime(runtime):
        pid = process_identity()

        with pytest.raises(InvalidPolicyError) as raised:
            synthetic_policy_evaluation(
                PolicyProfileOracle(PolicyProfile(fail)), rows=2, observation_columns=2, actions=2, batch_size=2
            )

        assert raised.value is expected
        assert process_identity() == pid


def test_native_style_worker_crashes_are_detected_without_retry() -> None:
    with LocalWorkerRuntime() as runtime, use_evaluation_runtime(runtime):
        first_pid = process_identity()

        with pytest.raises(EvaluationWorkerCrashed) as raised:
            crash_worker(73)

        assert raised.value.exit_code == 73
        second_pid = process_identity()
        assert second_pid != first_pid


def test_large_array_results_cannot_leak_into_the_control_plane() -> None:
    with LocalWorkerRuntime() as runtime, use_evaluation_runtime(runtime):
        pid = process_identity()

        with pytest.raises(EvaluationWorkerExecutionError, match="data plane"):
            return_array()

        assert process_identity() == pid


def test_batches_larger_than_the_arena_fail_without_killing_the_worker() -> None:
    with LocalWorkerRuntime(arena_bytes=1024) as runtime, use_evaluation_runtime(runtime):
        pid = process_identity()

        with pytest.raises(EvaluationWorkerExecutionError, match="arena has 1024"):
            synthetic_policy_evaluation(_uniform_policy(), rows=10, observation_columns=100, actions=2, batch_size=10)

        assert process_identity() == pid


def test_close_waits_for_the_worker_to_exit() -> None:
    runtime = LocalWorkerRuntime()

    with use_evaluation_runtime(runtime):
        pid = process_identity()

    runtime.close()

    assert runtime.worker_pid is None

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("failure", ["arena", "mmap", "log", "launch", "handshake"])
def test_failed_startup_releases_resources_and_can_be_retried(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    sockets, resources, processes = [], [], []
    socketpair = worker_local.socket.socketpair
    temporary_file = worker_local.tempfile.TemporaryFile
    mmap = worker_local.mmap.mmap
    popen = worker_local.subprocess.Popen

    def track_sockets():
        pair = socketpair()
        sockets.extend(pair)
        return pair

    def acquire(stage, factory, tracked, *args, **kwargs):
        if failure == stage:
            raise RuntimeError("startup failed")

        value = factory(*args, **kwargs)
        tracked.append(value)
        return value

    def create_file(**kwargs):
        stage = "arena" if kwargs["prefix"] == "benchmax-arena-" else "log"
        return acquire(stage, temporary_file, resources, **kwargs)

    def fail_handshake(*args, **kwargs):
        raise RuntimeError("startup failed")

    with LocalWorkerRuntime(arena_bytes=1024) as runtime:
        with monkeypatch.context() as patch:
            patch.setattr(worker_local.socket, "socketpair", track_sockets)
            patch.setattr(worker_local.tempfile, "TemporaryFile", create_file)
            patch.setattr(worker_local.mmap, "mmap", lambda *a, **kw: acquire("mmap", mmap, resources, *a, **kw))
            patch.setattr(worker_local.subprocess, "Popen", lambda *a, **kw: acquire("launch", popen, processes, *a, **kw))

            if failure == "handshake":
                patch.setattr(runtime, "_receive", fail_handshake)

            with pytest.raises(RuntimeError, match="startup failed"):
                runtime.start()

        assert all(sock.fileno() == -1 for sock in sockets)
        assert all(resource.closed for resource in resources)
        assert all(process.poll() is not None for process in processes)
        assert runtime.worker_pid is None

        runtime.start()
        assert runtime.worker_pid is not None


def test_an_active_native_call_can_be_cancelled_from_another_thread() -> None:
    runtime = LocalWorkerRuntime()
    failures: list[BaseException] = []

    def run() -> None:
        try:
            with use_evaluation_runtime(runtime):
                sleep_in_worker(30.0)
        except EvaluationWorkerError as error:
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5.0

    while not runtime.cancel():
        if time.monotonic() >= deadline:
            pytest.fail("worker invocation did not become cancellable")

        time.sleep(0.01)

    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], EvaluationCancelled)

    with use_evaluation_runtime(runtime):
        assert process_identity() != os.getpid()

    runtime.close()


def test_calls_from_multiple_threads_are_serialized_onto_one_worker() -> None:
    runtime = LocalWorkerRuntime()
    worker_pids: list[int] = []

    def run() -> None:
        with use_evaluation_runtime(runtime):
            worker_pids.append(process_identity())

    threads = [threading.Thread(target=run) for _ in range(8)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(worker_pids) == 8
    assert len(set(worker_pids)) == 1
    runtime.close()


def test_worker_exits_when_its_parent_disappears() -> None:
    program = """
import os
from benchmax.execution.worker import LocalWorkerRuntime, use_evaluation_runtime
from tests.execution.worker._kernels import process_identity

runtime = LocalWorkerRuntime()
with use_evaluation_runtime(runtime):
    print(process_identity(), flush=True)
os._exit(0)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", program], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert parent.stdout is not None
    worker_pid = int(parent.stdout.readline())
    _, stderr = parent.communicate(timeout=5)
    assert parent.returncode == 0, stderr

    deadline = time.monotonic() + 5.0

    while _process_exists(worker_pid):
        if time.monotonic() >= deadline:
            pytest.fail(f"orphaned worker {worker_pid} did not exit")

        time.sleep(0.01)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_a_forked_process_detaches_and_starts_its_own_worker() -> None:
    runtime = LocalWorkerRuntime()

    with use_evaluation_runtime(runtime):
        parent_worker_pid = process_identity()

    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        try:
            os.close(read_fd)

            with use_evaluation_runtime(runtime):
                child_worker_pid = process_identity()

            os.write(write_fd, str(child_worker_pid).encode())
            runtime.close()
            os._exit(0)
        except BaseException:  # noqa: BLE001 - fork child must not re-enter pytest
            os._exit(1)

    os.close(write_fd)
    child_worker_pid = int(os.read(read_fd, 64))
    os.close(read_fd)
    waited_pid, status = os.waitpid(child_pid, 0)

    assert waited_pid == child_pid
    assert os.waitstatus_to_exitcode(status) == 0
    assert child_worker_pid != parent_worker_pid

    with use_evaluation_runtime(runtime):
        assert process_identity() == parent_worker_pid

    runtime.close()


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False

    return True
