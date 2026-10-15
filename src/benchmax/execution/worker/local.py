"""Persistent isolated worker with a bounded shared data plane."""

from __future__ import annotations

import mmap
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Self

from benchmax.execution.values import PolicyEndpoint, marshal_value, restore_value
from benchmax.execution.worker import protocol
from benchmax.execution.worker.data_plane import DataPlaneError
from benchmax.execution.worker.errors import EvaluationWorkerError
from benchmax.execution.worker.kernel import KernelDefinition
from benchmax.execution.worker.oracle import PolicyBatch, PolicyOracle

_DEFAULT_ARENA_BYTES = 64 * 1024 * 1024
_START_TIMEOUT_SECONDS = 10.0
_SHUTDOWN_TIMEOUT_SECONDS = 3.0
_LOG_TAIL_BYTES = 64 * 1024


class EvaluationWorkerCrashed(EvaluationWorkerError):
    """Raised when the worker exits without returning a structured result."""

    def __init__(self, message: str, *, exit_code: int | None, log_tail: str) -> None:
        details = message

        if exit_code is not None:
            details += f" (exit code {exit_code})"

        if log_tail:
            details += f"\nworker output:\n{log_tail}"

        super().__init__(details)
        self.exit_code = exit_code
        self.log_tail = log_tail


class EvaluationWorkerExecutionError(EvaluationWorkerError):
    """Raised for a Python exception reported by the worker."""

    def __init__(self, remote_type: str, message: str, remote_traceback: str) -> None:
        text = f"worker raised {remote_type}: {message}"

        if remote_traceback:
            text += f"\nremote traceback:\n{remote_traceback}"

        super().__init__(text)
        self.remote_type = remote_type
        self.remote_traceback = remote_traceback


class EvaluationWorkerProtocolError(EvaluationWorkerError):
    """Raised when the worker violates the control or data-plane protocol."""


class EvaluationCancelled(EvaluationWorkerError):
    """Raised when an explicit cancellation terminates the active worker."""


class _PolicyCallFailed(Exception):
    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


@dataclass(slots=True)
class _WorkerState:
    process: subprocess.Popen[bytes]
    channel: protocol.ControlChannel
    arena: mmap.mmap
    log_file: Any
    resources: ExitStack


@dataclass(frozen=True, slots=True)
class LocalInvocationProfile:
    """End-to-end data-plane measurements for one successful invocation."""

    wall_seconds: float
    policy_batches: int
    observation_bytes: int
    legal_action_mask_bytes: int
    probability_bytes: int
    worker_kernel_seconds: float
    worker_input_copy_seconds: float
    worker_policy_roundtrip_seconds: float
    parent_policy_seconds: float
    worker_output_copy_seconds: float


@dataclass(slots=True)
class _ProfileAccumulator:
    started: float
    policy_batches: int = 0
    observation_bytes: int = 0
    legal_action_mask_bytes: int = 0
    probability_bytes: int = 0
    parent_policy_seconds: float = 0.0


class LocalWorkerRuntime:
    """Execute kernels in one reusable, clean Python subprocess.

    The child is launched with ``python -m`` rather than multiprocessing, so it
    never imports the user's main module. One fixed-size memory-mapped arena
    carries policy batches, and the control channel structurally rejects arrays.
    Calls are serialized because prepared numerical kernels are not assumed to
    be reentrant.
    """

    def __init__(self, *, arena_bytes: int = _DEFAULT_ARENA_BYTES, python_executable: str | None = None) -> None:
        if not isinstance(arena_bytes, int) or isinstance(arena_bytes, bool):
            raise TypeError("arena_bytes must be an integer")

        if arena_bytes <= 0:
            raise ValueError("arena_bytes must be positive")

        self._arena_bytes = arena_bytes
        self._python_executable = python_executable or sys.executable
        self._state: _WorkerState | None = None
        self._finalizer: weakref.finalize | None = None
        self._lock = threading.RLock()
        self._activity_lock = threading.Lock()
        self._active_worker_pid: int | None = None
        self._cancelled_worker_pids: set[int] = set()
        self._next_invocation_id = 0
        self._last_profile: LocalInvocationProfile | None = None
        _LIVE_RUNTIMES.add(self)

    def invoke(self, kernel: KernelDefinition, args: tuple[Any, ...], kwargs: dict[str, Any], /) -> Any:
        """Execute one kernel, serving its policy queries in this process."""

        with self._lock:
            state = self._ensure_started()
            self._next_invocation_id += 1
            invocation_id = self._next_invocation_id
            endpoints: dict[int, PolicyOracle] = {}
            endpoint_ids: dict[int, int] = {}

            def policy_reference(value: object) -> PolicyEndpoint | None:
                if not isinstance(value, PolicyOracle):
                    return None

                identity = id(value)

                try:
                    endpoint_id = endpoint_ids[identity]
                except KeyError:
                    endpoint_id = len(endpoints)
                    endpoint_ids[identity] = endpoint_id
                    endpoints[endpoint_id] = value

                return PolicyEndpoint(endpoint_id)

            marshalled_args = marshal_value(args, location="kernel arguments", policy_reference=policy_reference)
            marshalled_kwargs = marshal_value(kwargs, location="kernel arguments", policy_reference=policy_reference)
            pending_policy_error: Exception | None = None
            profile = _ProfileAccumulator(started=time.perf_counter())
            self._begin_invocation(state.process.pid)

            try:
                state.channel.send(
                    protocol.Invoke(
                        invocation_id=invocation_id, kernel=kernel.identifier, args=marshalled_args,
                        kwargs=marshalled_kwargs
                    )
                )

                while True:
                    message = self._receive(state)

                    if self._consume_cancellation(state.process.pid):
                        self._discard_state(state)
                        raise EvaluationCancelled("evaluation was cancelled")

                    match message:
                        case protocol.PolicyQuery():
                            _require_invocation(message.invocation_id, invocation_id)

                            if pending_policy_error is not None:
                                raise EvaluationWorkerProtocolError(
                                    "worker requested another policy batch after a policy failure"
                                )

                            try:
                                self._serve_policy_query(state, message, endpoints, profile)
                            except _PolicyCallFailed as failure:
                                pending_policy_error = failure.error
                                state.channel.send(
                                    protocol.PolicyError(
                                        invocation_id=invocation_id, batch_id=message.batch_id,
                                        error_type=_qualified_type(failure.error), message=str(failure.error)
                                    )
                                )
                        case protocol.Result():
                            _require_invocation(message.invocation_id, invocation_id)

                            if pending_policy_error is not None:
                                raise EvaluationWorkerProtocolError("worker returned a result after a policy failure")

                            self._last_profile = _complete_profile(message, profile)
                            return restore_value(message.value)
                        case protocol.Error():
                            _require_invocation(message.invocation_id, invocation_id)

                            if pending_policy_error is not None:
                                raise pending_policy_error

                            raise EvaluationWorkerExecutionError(message.error_type, message.message, message.traceback)
                        case _:
                            raise EvaluationWorkerProtocolError(f"unexpected worker message {type(message).__name__}")
            except EvaluationWorkerCrashed as error:
                cancelled = self._consume_cancellation(state.process.pid)
                self._discard_state(state)

                if cancelled:
                    raise EvaluationCancelled("evaluation was cancelled") from error

                raise
            except (protocol.ControlChannelClosed, OSError) as error:
                crashed = _crashed(state, "lost the worker control channel")
                cancelled = self._consume_cancellation(state.process.pid)
                self._discard_state(state)

                if cancelled:
                    raise EvaluationCancelled("evaluation was cancelled") from error

                raise crashed from error
            except (EvaluationWorkerProtocolError, DataPlaneError, KeyboardInterrupt, SystemExit):
                self._discard_state(state)
                raise
            finally:
                self._end_invocation(state.process.pid)

    def cancel(self) -> bool:
        """Terminate the active invocation and return whether one was found.

        Cancellation is safe to call from another thread while ``invoke()`` is
        blocked in native code. It necessarily discards prepared state in that
        worker, so the next invocation starts a clean process.
        """

        with self._activity_lock:
            state = self._state
            if state is None or self._active_worker_pid != state.process.pid or state.process.poll() is not None:
                return False

            self._cancelled_worker_pids.add(state.process.pid)
            state.process.terminate()
            return True

    def clear_prepared_resources(self) -> None:
        """Release every prepared resource retained by the current worker."""

        with self._lock:
            if self._state is None:
                return

            state = self._state

            try:
                state.channel.send(protocol.ClearPrepared())
                message = self._receive(state)
                if not isinstance(message, protocol.PreparedCleared):
                    raise EvaluationWorkerProtocolError("worker did not acknowledge prepared-resource cleanup")
            except (EvaluationWorkerCrashed, EvaluationWorkerProtocolError):
                self._discard_state(state)
                raise
            except OSError as error:
                crashed = _crashed(state, "lost the worker control channel")
                self._discard_state(state)
                raise crashed from error

    @property
    def worker_pid(self) -> int | None:
        """Return the live worker PID for diagnostics, without starting it."""

        state = self._state
        if state is None or state.process.poll() is not None:
            return None

        return state.process.pid

    @property
    def last_profile(self) -> LocalInvocationProfile | None:
        """Return data-plane telemetry from the last successful invocation."""

        return self._last_profile

    def start(self) -> None:
        """Start and handshake with the worker without invoking a kernel."""

        with self._lock:
            self._ensure_started()

    def close(self) -> None:
        """Shut down the worker and release its arena and temporary files."""

        with self._lock:
            if self._state is not None:
                self._discard_state(self._state, graceful=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self.close()

    def _after_fork_in_child(self) -> None:
        """Detach descriptors inherited from the parent without signalling it."""

        state = self._state
        self._state = None
        self._detach_finalizer()

        if state is not None:
            state.resources.close()

        self._lock = threading.RLock()
        self._activity_lock = threading.Lock()
        self._active_worker_pid = None
        self._cancelled_worker_pids = set()
        self._next_invocation_id = 0
        self._last_profile = None

    def _ensure_started(self) -> _WorkerState:
        if self._state is not None:
            if self._state.process.poll() is None:
                return self._state

            self._discard_state(self._state)

        with ExitStack() as resources:
            parent_socket, child_socket = socket.socketpair()
            resources.enter_context(parent_socket)

            with child_socket:
                arena_file = resources.enter_context(tempfile.TemporaryFile(prefix="benchmax-arena-"))
                arena_file.truncate(self._arena_bytes)
                arena = resources.enter_context(mmap.mmap(arena_file.fileno(), self._arena_bytes, access=mmap.ACCESS_WRITE))
                log_file = resources.enter_context(tempfile.TemporaryFile(prefix="benchmax-worker-log-"))
                command = [
                    self._python_executable, "-m", "benchmax.execution.worker.process",
                    "--control-fd", str(child_socket.fileno()), "--arena-fd", str(arena_file.fileno()),
                    "--arena-bytes", str(self._arena_bytes)
                ]
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, close_fds=True,
                    pass_fds=(child_socket.fileno(), arena_file.fileno()),
                    # A managed local attempt owns the group used to cancel all its computation.
                    start_new_session=os.environ.get("BENCHMAX_LOCAL_PROCESS_GROUP") != str(os.getpgrp())
                )

            # The runtime owns the descriptors after launch; fork children only close their inherited copies.
            state = _WorkerState(
                process=process, channel=protocol.ControlChannel(parent_socket), arena=arena, log_file=log_file,
                resources=resources.pop_all()
            )

        self._state = state
        self._finalizer = weakref.finalize(self, _close_state, state, False)

        try:
            hello = self._receive(state, timeout=_START_TIMEOUT_SECONDS)
            if (
                not isinstance(hello, protocol.Hello)
                or hello.protocol_version != protocol.PROTOCOL_VERSION
                or hello.arena_bytes != self._arena_bytes
                or hello.pid != process.pid
            ):
                raise EvaluationWorkerProtocolError("invalid worker handshake")
        except BaseException:
            self._discard_state(state)
            raise

        return state

    def _serve_policy_query(
        self, state: _WorkerState, message: protocol.PolicyQuery, endpoints: dict[int, PolicyOracle],
        profile: _ProfileAccumulator
    ) -> None:
        try:
            endpoint = endpoints[message.endpoint_id]
        except KeyError as error:
            raise EvaluationWorkerProtocolError(
                f"worker requested unknown policy endpoint {message.endpoint_id}"
            ) from error

        packed = message.arrays
        observations = packed.observations.view(state.arena, writable=False)
        mask = packed.legal_action_mask.view(state.arena, writable=False)
        probabilities = packed.probabilities.view(state.arena)
        profile.policy_batches += 1
        profile.observation_bytes += packed.observations.nbytes
        profile.legal_action_mask_bytes += packed.legal_action_mask.nbytes
        profile.probability_bytes += packed.probabilities.nbytes
        policy_started = time.perf_counter()

        try:
            endpoint.evaluate(
                PolicyBatch(
                    player=message.player, observations=observations, action_space=message.action_space,
                    legal_action_mask=mask, probabilities=probabilities
                )
            )
        except Exception as error:
            raise _PolicyCallFailed(error) from error
        finally:
            profile.parent_policy_seconds += time.perf_counter() - policy_started

        state.channel.send(protocol.PolicyResult(invocation_id=message.invocation_id, batch_id=message.batch_id))

    def _receive(self, state: _WorkerState, *, timeout: float | None = None) -> protocol.ControlMessage:
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            wait = 0.1

            if deadline is not None:
                wait = min(wait, max(0.0, deadline - time.monotonic()))

            readable, _, _ = select.select([state.channel.connection], [], [], wait)

            if readable:
                try:
                    value = state.channel.receive(deadline=deadline)
                except TimeoutError as error:
                    raise EvaluationWorkerCrashed(
                        "worker did not complete its startup handshake", exit_code=None, log_tail=_log_tail(state)
                    ) from error
                except protocol.ControlChannelClosed as error:
                    raise _crashed(state, "worker closed its control channel") from error
                except protocol.ControlProtocolError as error:
                    raise EvaluationWorkerProtocolError(str(error)) from error

                return value

            exit_code = state.process.poll()
            if exit_code is not None:
                raise _crashed(state, "worker exited unexpectedly")

            if deadline is not None and time.monotonic() >= deadline:
                raise EvaluationWorkerCrashed(
                    "worker did not complete its startup handshake", exit_code=None, log_tail=_log_tail(state)
                )

    def _discard_state(self, state: _WorkerState, *, graceful: bool = False) -> None:
        if self._state is state:
            self._state = None

        self._detach_finalizer()
        _close_state(state, graceful=graceful)

    def _detach_finalizer(self) -> None:
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None

    def _begin_invocation(self, worker_pid: int) -> None:
        with self._activity_lock:
            self._active_worker_pid = worker_pid

    def _end_invocation(self, worker_pid: int) -> None:
        with self._activity_lock:
            if self._active_worker_pid == worker_pid:
                self._active_worker_pid = None

    def _consume_cancellation(self, worker_pid: int) -> bool:
        with self._activity_lock:
            if worker_pid not in self._cancelled_worker_pids:
                return False

            self._cancelled_worker_pids.remove(worker_pid)
            return True


def _require_invocation(actual: int, expected: int) -> None:
    if actual != expected:
        raise EvaluationWorkerProtocolError(f"worker message belongs to invocation {actual}, expected {expected}")


def _qualified_type(error: BaseException) -> str:
    return f"{type(error).__module__}.{type(error).__qualname__}"


def _complete_profile(message: protocol.Result, profile: _ProfileAccumulator) -> LocalInvocationProfile:
    telemetry = message.telemetry
    if (
        telemetry.policy_batches != profile.policy_batches
        or telemetry.observation_bytes != profile.observation_bytes
        or telemetry.legal_action_mask_bytes != profile.legal_action_mask_bytes
        or telemetry.probability_bytes != profile.probability_bytes
    ):
        raise EvaluationWorkerProtocolError("worker and parent disagree about policy-batch data volumes")

    return LocalInvocationProfile(
        wall_seconds=time.perf_counter() - profile.started, policy_batches=profile.policy_batches,
        observation_bytes=profile.observation_bytes, legal_action_mask_bytes=profile.legal_action_mask_bytes,
        probability_bytes=profile.probability_bytes, worker_kernel_seconds=telemetry.kernel_seconds,
        worker_input_copy_seconds=telemetry.input_copy_seconds,
        worker_policy_roundtrip_seconds=telemetry.policy_roundtrip_seconds,
        parent_policy_seconds=profile.parent_policy_seconds, worker_output_copy_seconds=telemetry.output_copy_seconds
    )


def _log_tail(state: _WorkerState) -> str:
    try:
        size = os.fstat(state.log_file.fileno()).st_size
        start = max(0, size - _LOG_TAIL_BYTES)
        return os.pread(state.log_file.fileno(), size - start, start).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _crashed(state: _WorkerState, message: str) -> EvaluationWorkerCrashed:
    exit_code = state.process.poll()
    if exit_code is None:
        try:
            exit_code = state.process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass

    return EvaluationWorkerCrashed(message, exit_code=exit_code, log_tail=_log_tail(state))


def _close_state(state: _WorkerState, graceful: bool) -> None:
    with state.resources:
        if graceful and state.process.poll() is None:
            try:
                state.channel.send(protocol.Shutdown())
                state.process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass

        if state.process.poll() is None:
            state.process.terminate()

            try:
                state.process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                state.process.kill()
                state.process.wait()


_LIVE_RUNTIMES: weakref.WeakSet[LocalWorkerRuntime] = weakref.WeakSet()


def _detach_live_runtimes_after_fork() -> None:
    for runtime in list(_LIVE_RUNTIMES):
        runtime._after_fork_in_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_detach_live_runtimes_after_fork)
