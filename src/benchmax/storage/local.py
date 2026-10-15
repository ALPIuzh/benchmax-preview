"""Atomic local implementation of the BenchMAX object store."""

from __future__ import annotations  # noqa: I001

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from benchmax.storage._files import _atomic_blob, _atomic_bytes, _verify_blob
from benchmax.storage._model import (
    BlobReference, BlobSource, CatalogEntry, ObjectReference,
    StorageError, _decode_document, _merge_entries, _prepare_publication, _read_catalog,
)
from benchmax.storage.store import Store
from benchmax.utils.files import _exclusive_file_lock
from benchmax.utils.json import JsonObject, write_document

if TYPE_CHECKING:
    from benchmax.storage.requirements import PolicyRequirements


class LocalStore(Store):
    """An immutable content store and atomically replaced local catalog."""

    def __init__(self, root: str | os.PathLike[str], *, requirements: PolicyRequirements | None = None) -> None:
        super().__init__(requirements=requirements)
        self.root = Path(root).expanduser().resolve()

    def _catalog(self) -> tuple[CatalogEntry, ...]:
        try:
            value = (self.root / "catalog.json").read_bytes()
        except FileNotFoundError:
            value = None
        return _read_catalog(value)

    def read_document(self, reference: ObjectReference) -> JsonObject:
        try:
            value = (self.root / reference.path).read_bytes()
        except OSError as error:
            raise StorageError(f"cannot read stored document {reference.digest}: {error}") from error
        return _decode_document(value, reference)

    def _verified_blob_path(self, reference: BlobReference) -> Path:
        return _verify_blob(self.root / reference.path, reference)

    def publish(
        self, document: Mapping[str, object], *, dimensions: Mapping[str, str] | None = None,
        blobs: Mapping[BlobReference, BlobSource] | None = None,
    ) -> CatalogEntry:
        encoded, entry, prepared_blobs = _prepare_publication(document, dimensions=dimensions, blobs=blobs)
        self.root.mkdir(parents=True, exist_ok=True)
        with _exclusive_file_lock(self.root / ".publish.lock"):
            entries = self._catalog()
            updated = _merge_entries((*entries, entry))
            already_published = updated == entries
            if already_published:
                self.read_document(entry.object)

            for reference, source in prepared_blobs:
                target = self.root / reference.path
                if already_published or target.exists():
                    self._verified_blob_path(reference)
                else:
                    _atomic_blob(target, reference, source)

            if not already_published:
                _atomic_bytes(self.root / entry.object.path, encoded)
                write_document({"schema_version": 1, "entries": [item.to_document() for item in updated]},
                               self.root / "catalog.json")
        return entry
