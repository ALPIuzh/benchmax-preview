"""Dedicated subprocess entry point for :class:`LocalWorkerRuntime`."""

from __future__ import annotations

import argparse
import mmap
import os
import socket
import time
import traceback
from typing import cast

import numpy as np

from benchmax.execution.values import marshal_value, restore_value
from benchmax.execution.worker import protocol
from benchmax.execution.worker.data_plane import pack_policy_batch
from benchmax.execution.worker.kernel import InProcessRuntime, resolve_kernel
from benchmax.execution.worker.oracle import PolicyBatch, PolicyOracle


class _RemotePolicyError(RuntimeError):
    pass


class _WorkerPolicyOracle(PolicyOracle):
    def __init__(
        self, channel: protocol.ControlChannel, arena: mmap.mmap, invocation_id: int, endpoint_id: int
    ) -> None:
        self._channel = channel
        self._arena = arena
        self._invocation_id = invocation_id
        self._endpoint_id = endpoint_id
        self._next_batch_id = 0
        self.policy_batches = 0
        self.observation_bytes = 0
        self.legal_action_mask_bytes = 0
        self.probability_bytes = 0
        self.input_copy_seconds = 0.0
        self.policy_roundtrip_seconds = 0.0
        self.output_copy_seconds = 0.0

    def evaluate(self, batch: PolicyBatch, /) -> None:
        self._next_batch_id += 1
        batch_id = self._next_batch_id
        copy_started = time.perf_counter()
        packed = pack_policy_batch(self._arena, batch)
        self.input_copy_seconds += time.perf_counter() - copy_started
        self.policy_batches += 1
        self.observation_bytes += packed.observations.nbytes
        self.legal_action_mask_bytes += packed.legal_action_mask.nbytes
        self.probability_bytes += packed.probabilities.nbytes
        roundtrip_started = time.perf_counter()

        try:
            self._channel.send(
                protocol.PolicyQuery(
                    invocation_id=self._invocation_id, endpoint_id=self._endpoint_id, batch_id=batch_id,
                    player=batch.player, action_space=batch.action_space, arrays=packed
                )
            )
            response = self._channel.receive()
        finally:
            self.policy_roundtrip_seconds += time.perf_counter() - roundtrip_started

        match response:
            case protocol.PolicyResult() | protocol.PolicyError():
                pass
            case _:
                raise protocol.ControlProtocolError(f"parent sent unexpected {type(response).__name__} response")

        if response.invocation_id != self._invocation_id:
            raise protocol.ControlProtocolError("parent response has the wrong invocation ID")

        if response.batch_id != batch_id:
            raise protocol.ControlProtocolError("parent response has the wrong policy batch ID")

        match response:
            case protocol.PolicyError(message=error_message):
                raise _RemotePolicyError(error_message)
            case protocol.PolicyResult():
                pass

        copy_started = time.perf_counter()
        np.copyto(batch.probabilities, packed.probabilities.view(self._arena, writable=False), casting="no")
        self.output_copy_seconds += time.perf_counter() - copy_started


def _serve(channel: protocol.ControlChannel, arena: mmap.mmap) -> None:
    runtime = InProcessRuntime()
    channel.send(
        protocol.Hello(protocol_version=protocol.PROTOCOL_VERSION, pid=os.getpid(), arena_bytes=len(arena))
    )

    while True:
        message = channel.receive()

        match message:
            case protocol.Shutdown():
                runtime.clear_prepared_resources()
                return
            case protocol.ClearPrepared():
                runtime.clear_prepared_resources()
                channel.send(protocol.PreparedCleared())
                continue
            case protocol.Invoke():
                pass
            case _:
                raise protocol.ControlProtocolError(f"unexpected parent message {type(message).__name__}")

        invocation_id = message.invocation_id

        try:
            args, kwargs, policy_oracles = _restore_invocation_arguments(message, channel, arena)
            kernel_started = time.perf_counter()
            result = runtime.invoke(resolve_kernel(message.kernel), args, kwargs)
            kernel_seconds = time.perf_counter() - kernel_started
            channel.send(
                protocol.Result(
                    invocation_id=invocation_id, value=marshal_value(result, location="kernel result"),
                    telemetry=_combined_telemetry(policy_oracles, kernel_seconds=kernel_seconds)
                )
            )
        except Exception as error:  # noqa: BLE001 - report arbitrary kernel errors
            channel.send(
                protocol.Error(
                    invocation_id=invocation_id, error_type=(f"{type(error).__module__}.{type(error).__qualname__}"),
                    message=str(error), traceback=traceback.format_exc()
                )
            )


def _restore_invocation_arguments(
    message: protocol.Invoke, channel: protocol.ControlChannel, arena: mmap.mmap
) -> tuple[tuple[object, ...], dict[str, object], dict[int, _WorkerPolicyOracle]]:
    policy_oracles: dict[int, _WorkerPolicyOracle] = {}

    def policy_oracle(endpoint_id: int) -> _WorkerPolicyOracle:
        try:
            return policy_oracles[endpoint_id]
        except KeyError:
            oracle = _WorkerPolicyOracle(channel, arena, message.invocation_id, endpoint_id)
            policy_oracles[endpoint_id] = oracle
            return oracle

    args = restore_value(message.args, policy_oracle)
    kwargs = restore_value(message.kwargs, policy_oracle)
    if not isinstance(args, tuple) or not isinstance(kwargs, dict) or not all(isinstance(key, str) for key in kwargs):
        raise TypeError("invocation arguments have invalid containers")

    return args, cast(dict[str, object], kwargs), policy_oracles


def _combined_telemetry(
    policy_oracles: dict[int, _WorkerPolicyOracle], *, kernel_seconds: float
) -> protocol.WorkerTelemetry:
    return protocol.WorkerTelemetry(
        policy_batches=sum(oracle.policy_batches for oracle in policy_oracles.values()),
        observation_bytes=sum(oracle.observation_bytes for oracle in policy_oracles.values()),
        legal_action_mask_bytes=sum(oracle.legal_action_mask_bytes for oracle in policy_oracles.values()),
        probability_bytes=sum(oracle.probability_bytes for oracle in policy_oracles.values()),
        input_copy_seconds=sum(oracle.input_copy_seconds for oracle in policy_oracles.values()),
        policy_roundtrip_seconds=sum(oracle.policy_roundtrip_seconds for oracle in policy_oracles.values()),
        output_copy_seconds=sum(oracle.output_copy_seconds for oracle in policy_oracles.values()),
        kernel_seconds=kernel_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--arena-fd", type=int, required=True)
    parser.add_argument("--arena-bytes", type=int, required=True)
    args = parser.parse_args()

    connection = socket.socket(fileno=args.control_fd)
    arena = mmap.mmap(args.arena_fd, args.arena_bytes, access=mmap.ACCESS_WRITE)
    os.close(args.arena_fd)
    channel = protocol.ControlChannel(connection)

    try:
        _serve(channel, arena)
    finally:
        channel.close()
        arena.close()


if __name__ == "__main__":
    main()
