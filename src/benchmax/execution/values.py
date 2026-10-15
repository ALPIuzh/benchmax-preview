"""Portable values shared by durable jobs and isolated worker runtimes."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Annotated, Any, TypeAlias, overload

import msgspec

_MAX_DEPTH = 64

ControlScalar: TypeAlias = None | bool | int | float | str
PortableMapKey: TypeAlias = "ControlScalar | tuple[PortableMapKey, ...]"
PortableValue: TypeAlias = (
    "ControlScalar | list[PortableValue] | tuple[PortableValue, ...] | dict[PortableMapKey, PortableValue]"
)
PolicyRefHook: TypeAlias = Callable[[object], "PolicyEndpoint | None"] | None
PolicyEndpointResolver: TypeAlias = Callable[[int], object]
_EndpointId: TypeAlias = Annotated[int, msgspec.Meta(ge=0)]
_FiniteFloat: TypeAlias = Annotated[float, msgspec.Meta(ge=-sys.float_info.max, le=sys.float_info.max)]

# Containers are explicit tagged structs so decoding preserves list/tuple and
# map-key semantics without ever importing or reconstructing arbitrary objects.


class PolicyEndpoint(msgspec.Struct, tag="policy_reference", frozen=True, forbid_unknown_fields=True):
    """Wire placeholder for a policy retained by the parent process."""

    endpoint_id: _EndpointId


class WireList(msgspec.Struct, tag="list", frozen=True, forbid_unknown_fields=True):
    items: tuple[WireValue, ...]


class WireTuple(msgspec.Struct, tag="tuple", frozen=True, forbid_unknown_fields=True):
    items: tuple[WireValue, ...]


class WireMap(msgspec.Struct, tag="map", frozen=True, forbid_unknown_fields=True):
    # Pairs preserve scalar and tuple keys; JSON object keys would coerce them.
    items: tuple[tuple[WireValue, WireValue], ...]

    def __post_init__(self) -> None:
        seen: set[object] = set()

        for key, _ in self.items:
            if not _wire_key_is_hashable(key):
                raise ValueError("wire map key is not hashable")

            if key in seen:
                raise ValueError("wire map contains a duplicate key")

            seen.add(key)


WireValue: TypeAlias = None | bool | int | _FiniteFloat | str | PolicyEndpoint | WireList | WireTuple | WireMap

_DOCUMENT_ENCODER = msgspec.json.Encoder()
_DOCUMENT_DECODER = msgspec.json.Decoder(WireValue)


def _wire_key_is_hashable(value: WireValue) -> bool:
    match value:
        case WireList() | WireMap():
            return False
        case WireTuple():
            return all(_wire_key_is_hashable(item) for item in value.items)
        case _:
            return True


@overload
def marshal_value(value: tuple[Any, ...], *, location: str, policy_reference: PolicyRefHook = None) -> WireTuple: ...


@overload
def marshal_value(value: dict[Any, Any], *, location: str, policy_reference: PolicyRefHook = None) -> WireMap: ...


@overload
def marshal_value(value: object, *, location: str, policy_reference: PolicyRefHook = None) -> WireValue: ...


def marshal_value(value: object, *, location: str, policy_reference: PolicyRefHook = None) -> WireValue:
    """Validate and copy a value into the portable wire model."""

    return _marshal(value, (location, policy_reference), 0)


def _marshal(value: object, context: tuple[str, PolicyRefHook], depth: int) -> WireValue:
    if depth > _MAX_DEPTH:
        location, _ = context
        raise TypeError(f"{location} is nested too deeply")

    location, policy_reference = context

    match value:
        case None | bool() | int() | str():
            return value
        case float():
            if not math.isfinite(value):
                raise TypeError(f"{location} contains a non-finite float")

            return value
        case PolicyEndpoint() | WireList() | WireTuple() | WireMap():
            raise TypeError(f"{location} contains an internal wire value")

    if policy_reference is not None:
        reference = policy_reference(value)

        if reference is not None:
            if reference.endpoint_id < 0:
                raise ValueError("policy endpoint IDs must be nonnegative")

            return reference

    nested = depth + 1

    match value:
        case list():
            return WireList(tuple(_marshal(item, context, nested) for item in value))
        case tuple():
            return WireTuple(tuple(_marshal(item, context, nested) for item in value))
        case dict():
            return WireMap(
                tuple((_marshal(key, context, nested), _marshal(item, context, nested)) for key, item in value.items())
            )
        case _:
            qualifier = " or policy oracle" if policy_reference is not None else ""
            raise TypeError(
                f"{location} contains {type(value).__name__}; expected a small "
                f"control value{qualifier}. Large arrays and arbitrary objects "
                "require an explicit data plane or artifact"
            )


def restore_value(value: WireValue, policy_reference: PolicyEndpointResolver | None = None) -> PortableValue | object:
    """Restore ordinary containers and resolve policy references when allowed."""

    return _restore(value, policy_reference, 0)


def _restore(value: WireValue, policy_reference: PolicyEndpointResolver | None, depth: int) -> PortableValue | object:
    if depth > _MAX_DEPTH:
        raise TypeError("wire value is nested too deeply")

    nested = depth + 1

    match value:
        case PolicyEndpoint(endpoint_id):
            if policy_reference is None:
                raise TypeError("wire value contains an unexpected policy reference")

            return policy_reference(endpoint_id)
        case WireList(items):
            return [_restore(item, policy_reference, nested) for item in items]
        case WireTuple(items):
            return tuple(_restore(item, policy_reference, nested) for item in items)
        case WireMap(items):
            return {
                _restore(key, policy_reference, nested): _restore(item, policy_reference, nested) for key, item in items
            }
        case _:
            return value


def value_to_document(value: object, *, location: str) -> object:
    """Encode one natural value as an unambiguous JSON-compatible document."""

    encoded = _DOCUMENT_ENCODER.encode(marshal_value(value, location=location))
    return msgspec.json.decode(encoded)


def value_from_document(document: object) -> PortableValue:
    """Validate and restore one value produced by :func:`value_to_document`."""

    try:
        wire_value = _DOCUMENT_DECODER.decode(_DOCUMENT_ENCODER.encode(document))
    except msgspec.MsgspecError as error:
        raise TypeError(str(error)) from error

    value = restore_value(wire_value)
    assert not isinstance(value, PolicyEndpoint)
    return value
