import socket
import struct
import threading
import time

import msgspec
import numpy as np
import pytest

from benchmax.execution.values import PolicyEndpoint, WireList, WireMap, WireTuple, marshal_value, restore_value
from benchmax.execution.worker.data_plane import ArrayDescriptor, PackedPolicyBatch
from benchmax.execution.worker.protocol import (
    ControlChannel,
    ControlChannelClosed,
    ControlProtocolError,
    Error,
    Hello,
    Invoke,
    PolicyQuery,
    Result,
    Shutdown,
    WorkerTelemetry,
    decode_message,
    encode_message,
)


def test_control_values_round_trip_without_losing_container_or_key_types() -> None:
    policy = object()
    value = {"message": "invoke", "args": (1, 2.5, True, None, ["x"]), 7: {0: (1, 2), "policy": policy}}
    wire_value = marshal_value(
        value, location="test value", policy_reference=lambda item: PolicyEndpoint(3) if item is policy else None
    )

    assert restore_value(wire_value, lambda endpoint_id: ("policy", endpoint_id)) == {
        "message": "invoke",
        "args": (1, 2.5, True, None, ["x"]),
        7: {0: (1, 2), "policy": ("policy", 3)},
    }


@pytest.mark.parametrize("value", (np.zeros((1, 1)), {"nested": [np.zeros(1)]}, b"opaque bytes", object()))
def test_arbitrary_objects_and_arrays_cannot_enter_the_control_plane(value) -> None:
    with pytest.raises(TypeError, match="data plane"):
        marshal_value(value, location="test value")


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_non_finite_control_floats_are_rejected(value: float) -> None:
    with pytest.raises(TypeError, match="finite"):
        marshal_value(value, location="test value")


def test_wire_maps_reject_duplicate_and_unhashable_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        WireMap(((1, "first"), (True, "same Python key")))

    with pytest.raises(ValueError, match="hashable"):
        WireMap(((WireList(()), "value"),))


def test_oversized_control_payloads_are_rejected() -> None:
    with pytest.raises(ControlProtocolError, match="maximum"):
        encode_message(Error(1, "Failure", "x" * (4 * 1024 * 1024), ""))


@pytest.mark.parametrize(
    "message",
    (
        Invoke(
            invocation_id=1, kernel="package.module:kernel", args=WireTuple((1, PolicyEndpoint(2))),
            kwargs=WireMap((("name", "value"),))
        ),
        PolicyQuery(
            invocation_id=1, endpoint_id=2, batch_id=3, player=0, action_space=(2, 3),
            arrays=PackedPolicyBatch(
                ArrayDescriptor(0, (2, 8), "<f8", 128), ArrayDescriptor(128, (2, 2), "|b1", 4),
                ArrayDescriptor(192, (2, 2), "<f8", 32)
            )
        ),
        Result(
            invocation_id=1,
            value=marshal_value({"exploitability": 0.25}, location="test result"),
            telemetry=WorkerTelemetry(
                policy_batches=2, observation_bytes=128, legal_action_mask_bytes=32, probability_bytes=64,
                input_copy_seconds=0.01, policy_roundtrip_seconds=0.02, output_copy_seconds=0.01, kernel_seconds=0.5
            ),
        ),
    ),
)
def test_nested_control_payloads_round_trip_as_typed_values(message: object) -> None:
    assert decode_message(encode_message(message)) == message


@pytest.mark.parametrize(
    "value",
    (
        {"type": "unknown"},
        {
            "type": "policy_query", "invocation_id": 1, "endpoint_id": 2, "batch_id": 3,
            "player": 0, "action_space": [0, 1], "arrays": {},
        },
        {"type": "shutdown", "extra": True},
        {"type": "hello", "protocol_version": 1, "pid": 17},
        {"type": "hello", "protocol_version": 1, "pid": 0, "arena_bytes": 4096},
        {"type": "invoke", "invocation_id": 1, "kernel": "x", "args": [], "kwargs": {}},
        {
            "type": "result",
            "invocation_id": 1,
            "value": 0.25,
            "telemetry": {
                "policy_batches": 0,
                "observation_bytes": 0,
                "legal_action_mask_bytes": 0,
                "probability_bytes": 0,
                "input_copy_seconds": 0.0,
                "policy_roundtrip_seconds": 0.0,
                "output_copy_seconds": 0.0,
                "kernel_seconds": -1.0,
            },
        },
    ),
)
def test_malformed_control_messages_are_rejected(value: object) -> None:
    with pytest.raises(ControlProtocolError):
        decode_message(msgspec.json.encode(value))


@pytest.mark.parametrize("payload", (b"not JSON", b"[]", b"{}", b"null"))
def test_non_message_payloads_are_rejected(payload: bytes) -> None:
    with pytest.raises(ControlProtocolError):
        decode_message(payload)


def test_connected_channels_exchange_multiple_frames() -> None:
    first_socket, second_socket = socket.socketpair()
    first = ControlChannel(first_socket)
    second = ControlChannel(second_socket)

    try:
        first.send(Hello(protocol_version=1, pid=17, arena_bytes=4096))
        first.send(Shutdown())

        assert second.receive() == Hello(protocol_version=1, pid=17, arena_bytes=4096)
        assert second.receive() == Shutdown()
    finally:
        first.close()
        second.close()


def test_peer_closure_and_truncated_frames_are_distinguished() -> None:
    first_socket, second_socket = socket.socketpair()
    receiver = ControlChannel(second_socket)
    first_socket.sendall(struct.pack("!I", 10) + b"short")
    first_socket.close()

    with pytest.raises(ControlChannelClosed, match="5 of 10"):
        receiver.receive()

    receiver.close()


def test_fragmented_frame_arrives_before_deadline() -> None:
    first, second = socket.socketpair()
    payload = encode_message(Shutdown())
    frame = struct.pack("!I", len(payload)) + payload

    def send():
        for part in (frame[:1], frame[1:5], frame[5:]):
            first.sendall(part)
            time.sleep(0.01)

    with first, second:
        sender = threading.Thread(target=send)
        sender.start()
        try:
            assert ControlChannel(second).receive(deadline=time.monotonic() + 1) == Shutdown()
        finally:
            sender.join(timeout=1)


@pytest.mark.parametrize("fragment", [b"\x00", struct.pack("!I", 10) + b"short"], ids=["header", "payload"])
def test_worker_startup_deadline_covers_partial_frames(fragment, tmp_path) -> None:
    from types import SimpleNamespace

    from benchmax.execution.worker.local import EvaluationWorkerCrashed, LocalWorkerRuntime

    first, second = socket.socketpair()
    with first, second, (tmp_path / "worker.log").open("w+b") as log:
        first.sendall(fragment)
        state = SimpleNamespace(channel=ControlChannel(second), log_file=log)
        started = time.monotonic()
        release = threading.Timer(1, first.shutdown, args=(socket.SHUT_WR,))
        release.start()
        try:
            with pytest.raises(EvaluationWorkerCrashed, match="startup handshake"):
                LocalWorkerRuntime()._receive(state, timeout=0.02)
        finally:
            release.cancel()

        assert time.monotonic() - started < 1


def test_declared_oversized_frames_are_rejected_before_reading_the_payload() -> None:
    first_socket, second_socket = socket.socketpair()
    receiver = ControlChannel(second_socket)
    first_socket.sendall(struct.pack("!I", 4 * 1024 * 1024 + 1))

    try:
        with pytest.raises(ControlProtocolError, match="declares"):
            receiver.receive()
    finally:
        first_socket.close()
        receiver.close()
