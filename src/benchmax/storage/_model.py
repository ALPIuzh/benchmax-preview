"""Transport-neutral objects and catalog records."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import msgspec

from benchmax.utils.hashing import _digest_bytes, _digest_file, _is_sha256
from benchmax.utils.json import JsonObject, canonical_json

BlobSource: TypeAlias = bytes | str | os.PathLike[str]

_NAME = re.compile(r"[a-z][a-z0-9_-]*")


class StorageError(RuntimeError):
    """A store is unavailable, malformed, or internally inconsistent."""


class IntegrityError(StorageError):
    """Stored bytes do not match their content identity."""


def canonical_document(document: Mapping[str, object]) -> bytes:
    """Encode one finite JSON object deterministically."""

    try:
        return canonical_json(document).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("a stored document must contain only finite JSON values") from error


def _validate_digest(digest: object) -> str:
    if not _is_sha256(digest):
        raise ValueError("a content digest must be a lowercase SHA-256 hex digest")
    return digest


def _validate_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must match {_NAME.pattern!r}")
    return value


def _validated_dimensions(dimensions: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    values = dict(dimensions or {})
    for name, value in values.items():
        _validate_name(name, label="catalog dimension")
        if not isinstance(value, str) or not value:
            raise ValueError(f"catalog dimension {name!r} must have a non-empty string value")
    return tuple(sorted(values.items()))


@dataclass(frozen=True, slots=True)
class ObjectReference:
    """Location-independent identity of one canonical JSON document."""

    digest: str

    def __post_init__(self) -> None:
        _validate_digest(self.digest)

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> ObjectReference:
        return cls(_digest_bytes(canonical_document(document)))

    @property
    def path(self) -> str:
        return f"objects/{self.digest[:2]}/{self.digest}.json"


@dataclass(frozen=True, slots=True)
class BlobReference:
    """Location-independent identity and length of one binary object."""

    digest: str
    size: int

    def __post_init__(self) -> None:
        _validate_digest(self.digest)
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("blob size must be a non-negative integer")

    @classmethod
    def from_source(cls, source: BlobSource) -> BlobReference:
        if isinstance(source, bytes):
            return cls(_digest_bytes(source), len(source))
        digest, size = _digest_file(Path(source))
        return cls(digest, size)

    @property
    def path(self) -> str:
        return f"blobs/{self.digest[:2]}/{self.digest}"

    def to_document(self) -> JsonObject:
        return {"digest": self.digest, "size": self.size}


class _CatalogRow(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    kind: str
    object: str
    dimensions: dict[str, str]


class _CatalogDocument(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: Literal[1]
    entries: list[_CatalogRow]


@dataclass(frozen=True, slots=True, init=False)
class CatalogEntry:
    """One searchable projection of an immutable stored document."""

    kind: str
    object: ObjectReference
    _dimensions: tuple[tuple[str, str], ...] = field(repr=False)

    def __init__(self, kind: str, reference: ObjectReference, dimensions: Mapping[str, str] | None = None) -> None:
        object.__setattr__(self, "kind", _validate_name(kind, label="catalog kind"))
        if not isinstance(reference, ObjectReference):
            raise TypeError("catalog object must be an ObjectReference")
        object.__setattr__(self, "object", reference)
        object.__setattr__(self, "_dimensions", _validated_dimensions(dimensions))

    @property
    def dimensions(self) -> dict[str, str]:
        return dict(self._dimensions)

    def to_document(self) -> JsonObject:
        return {"kind": self.kind, "object": self.object.digest, "dimensions": self.dimensions}

    @classmethod
    def from_document(cls, document: object) -> CatalogEntry:
        try:
            row = msgspec.convert(document, type=_CatalogRow, strict=True)
            return cls(row.kind, ObjectReference(row.object), row.dimensions)
        except (TypeError, ValueError) as error:
            raise StorageError(f"invalid catalog entry: {error}") from error


@dataclass(frozen=True, slots=True, init=False)
class CatalogQuery:
    """Exact indexed dimensions used to shortlist stored objects."""

    kind: str | None
    _dimensions: tuple[tuple[str, str], ...] = field(repr=False)

    def __init__(self, *, kind: str | None = None, dimensions: Mapping[str, str] | None = None) -> None:
        if kind is not None:
            _validate_name(kind, label="catalog kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "_dimensions", _validated_dimensions(dimensions))

    @property
    def dimensions(self) -> dict[str, str]:
        return dict(self._dimensions)

    def matches(self, entry: CatalogEntry) -> bool:
        if self.kind is not None and entry.kind != self.kind:
            return False
        actual = entry.dimensions
        return all(actual.get(name) == value for name, value in self._dimensions)


class PolicyNotFoundError(StorageError):
    """No policy satisfies the requested selection."""

class AmbiguousPolicyError(StorageError):
    """Several policies match and the caller must narrow the selection."""


def _read_catalog(value: bytes | None) -> tuple[CatalogEntry, ...]:
    if value is None:
        return ()
    try:
        document = msgspec.json.decode(value, type=_CatalogDocument)
        entries = tuple(CatalogEntry(row.kind, ObjectReference(row.object), row.dimensions) for row in document.entries)
    except (msgspec.DecodeError, TypeError, ValueError) as error:
        raise StorageError(f"invalid catalog document: {error}") from error

    identities = [(entry.kind, entry.object.digest) for entry in entries]
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise StorageError("catalog entries must be unique and sorted by kind and object")
    return entries


def _merge_entries(entries: Iterable[CatalogEntry]) -> tuple[CatalogEntry, ...]:
    unique = {}
    for entry in entries:
        existing = unique.setdefault(entry.object, entry)
        if existing != entry:
            raise StorageError("the same stored document cannot have conflicting catalog metadata")
    return tuple(sorted(unique.values(), key=lambda entry: (entry.kind, entry.object.digest)))


def _prepare_publication(
    document: Mapping[str, object], *, dimensions: Mapping[str, str] | None,
    blobs: Mapping[BlobReference, BlobSource] | None,
) -> tuple[bytes, CatalogEntry, tuple[tuple[BlobReference, BlobSource], ...]]:
    encoded = canonical_document(document)
    entry = CatalogEntry(document.get("kind"), ObjectReference(_digest_bytes(encoded)), dimensions)
    prepared = []
    for reference, source in (blobs or {}).items():
        if not isinstance(reference, BlobReference):
            raise TypeError("publication blob keys must be BlobReferences")
        actual = BlobReference.from_source(source)
        if actual != reference:
            raise IntegrityError(f"blob source does not match {reference.digest}")
        prepared.append((reference, source))
    return encoded, entry, tuple(sorted(prepared, key=lambda item: item[0].digest))


def _decode_document(value: bytes, reference: ObjectReference) -> JsonObject:
    if _digest_bytes(value) != reference.digest:
        raise IntegrityError(f"stored document does not match {reference.digest}")
    try:
        document = json.loads(value)
    except json.JSONDecodeError as error:  # pragma: no cover - digest mismatch catches ordinary corruption
        raise StorageError(f"invalid stored document JSON: {error}") from error
    if not isinstance(document, dict):
        raise StorageError("stored document must be an object")
    return document
