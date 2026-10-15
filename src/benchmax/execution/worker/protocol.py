"""Typed, bounded control protocol for local evaluation workers."""

from __future__ import annotations

import select
import socket
import struct
import threading
import time
from typing import Annotated, TypeAlias

import msgspec

from benchmax.execution.values import WireMap, WireTuple, WireValue
from benchmax.execution.worker.data_plane import PackedPolicyBatch

PROTOCOL_VERSION = 1

_FRAME_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 4 * 1024 * 1024

_NonnegativeInt: TypeAlias = Annotated[int, msgspec.Meta(ge=0)]
_PositiveInt: TypeAlias = Annotated[int, msgspec.Meta(gt=0)]
_NonnegativeFloat: TypeAlias = Annotated[float, msgspec.Meta(ge=0.0)]


class ControlProtocolError(RuntimeError):
    """Raised when a control frame or message violates the protocol."""


class ControlChannelClosed(EOFError):
    """Raised when the peer closes before one complete control frame arrives."""


class _Message(msgspec.Struct, frozen=True, forbid_unknown_fields=True, tag_field="type"):
    pass


class Hello(_Message, tag="hello"):
    protocol_version: _NonnegativeInt
    pid: _PositiveInt
    arena_bytes: _PositiveInt


class Invoke(_Message, tag="invoke"):
    invocation_id: _NonnegativeInt
    kernel: str
    args: WireTuple
    kwargs: WireMap


class PolicyQuery(_Message, tag="policy_query"):
    invocation_id: _NonnegativeInt
    endpoint_id: _NonnegativeInt
    batch_id: _NonnegativeInt
    player: int
    action_space: tuple[int, ...]
    arrays: PackedPolicyBatch


class PolicyResult(_Message, tag="policy_result"):
    invocation_id: _NonnegativeInt
    batch_id: _NonnegativeInt


class PolicyError(_Message, tag="policy_error"):
    invocation_id: _NonnegativeInt
    batch_id: _NonnegativeInt
    error_type: str
    message: str


class WorkerTelemetry(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    policy_batches: _NonnegativeInt
    observation_bytes: _NonnegativeInt
    legal_action_mask_bytes: _NonnegativeInt
    probability_bytes: _NonnegativeInt
    input_copy_seconds: _NonnegativeFloat
    policy_roundtrip_seconds: _NonnegativeFloat
    output_copy_seconds: _NonnegativeFloat
    kernel_seconds: _NonnegativeFloat


class Result(_Message, tag="result"):
    invocation_id: _NonnegativeInt
    value: WireValue
    telemetry: WorkerTelemetry


class Error(_Message, tag="error"):
    invocation_id: _NonnegativeInt
    error_type: str
    message: str
    traceback: str


class ClearPrepared(_Message, tag="clear_prepared"):
    pass


class PreparedCleared(_Message, tag="prepared_cleared"):
    pass


class Shutdown(_Message, tag="shutdown"):
    pass


ControlMessage: TypeAlias = Hello | Invoke | PolicyQuery | PolicyResult | PolicyError | Result | Error | ClearPrepared | PreparedCleared | Shutdown  # fmt: skip

# Control payloads are intentionally small. JSON retains Python's full integer
# range, while numerical arrays travel through the bounded shared-memory arena.
_ENCODER = msgspec.json.Encoder()
_DECODER = msgspec.json.Decoder(ControlMessage)


def encode_message(message: ControlMessage) -> bytes:
    """Encode one typed message and enforce the control-frame bound."""

    if not isinstance(message, _Message):
        raise ControlProtocolError(f"unsupported control message {type(message).__name__}")

    try:
        payload = _ENCODER.encode(message)
    except (msgspec.MsgspecError, TypeError) as error:
        raise ControlProtocolError("control message cannot be encoded") from error

    if len(payload) > _MAX_FRAME_BYTES:
        raise ControlProtocolError(f"control payload is {len(payload)} bytes; maximum is {_MAX_FRAME_BYTES} bytes")

    return payload


def decode_message(payload: bytes) -> ControlMessage:
    """Decode and validate one typed message."""

    if len(payload) > _MAX_FRAME_BYTES:
        raise ControlProtocolError(f"control payload is {len(payload)} bytes; maximum is {_MAX_FRAME_BYTES} bytes")

    try:
        return _DECODER.decode(payload)
    except msgspec.DecodeError as error:
        raise ControlProtocolError(str(error)) from error


class ControlChannel:
    """Send and receive length-prefixed typed messages over a socket."""

    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection
        self._send_lock = threading.Lock()

    @property
    def connection(self) -> socket.socket:
        """Return the socket for readiness and process-death monitoring."""

        return self._connection

    def send(self, message: ControlMessage, /) -> None:
        """Send one complete bounded message."""

        payload = encode_message(message)
        frame = _FRAME_HEADER.pack(len(payload)) + payload

        with self._send_lock:
            self._connection.sendall(frame)

    def receive(self, *, deadline: float | None = None) -> ControlMessage:
        """Receive a complete frame before an optional monotonic deadline."""

        header = self._read_exact(_FRAME_HEADER.size, deadline)
        (length,) = _FRAME_HEADER.unpack(header)
        if length > _MAX_FRAME_BYTES:
            raise ControlProtocolError(f"control frame declares {length} bytes; maximum is {_MAX_FRAME_BYTES} bytes")

        return decode_message(self._read_exact(length, deadline))

    def close(self) -> None:
        """Close this endpoint. Repeated calls are harmless."""

        self._connection.close()

    def _read_exact(self, length: int, deadline: float | None) -> bytes:
        chunks = bytearray(length)
        view = memoryview(chunks)
        received = 0

        while received < length:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self._connection], [], [], remaining)[0]:
                    raise TimeoutError("control frame did not arrive before its deadline")

            count = self._connection.recv_into(view[received:])
            if count == 0:
                raise ControlChannelClosed(f"control channel closed after {received} of {length} bytes")

            received += count

        return bytes(chunks)
