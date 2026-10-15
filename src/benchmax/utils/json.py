"""Canonical JSON identities and atomic document writing."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from benchmax.utils.files import _atomic_writer

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def canonical_json(value: Mapping[str, Any]) -> str:
    """Encode a mapping as sorted, compact JSON, rejecting non-finite values."""

    if not isinstance(value, Mapping):
        raise TypeError("a JSON object must be a mapping")

    return json.dumps(dict(value), allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class _Document(Protocol):
    """A value accepted by the shared document writer."""

    def to_document(self) -> dict[str, Any]: ...


def write_document(
    document: _Document | Mapping[str, Any], path: str | os.PathLike[str], *, replace: bool = True
) -> Path:
    """Atomically publish a document as stable, readable JSON.

    ``replace=False`` uses an atomic hard-link publication in the destination
    directory, so a concurrently created output is never overwritten.
    """

    destination = Path(path)
    value = dict(document) if isinstance(document, Mapping) else document.to_document()
    text = json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)

    with _atomic_writer(destination, replace=replace) as stream:
        stream.write((text + "\n").encode("utf-8"))

    return destination
