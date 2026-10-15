"""Content-addressed policy storage in a Hugging Face bucket."""

from __future__ import annotations  # noqa: I001

import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmax.storage._files import _verify_blob
from benchmax.storage._model import (
    BlobReference, BlobSource, CatalogEntry, IntegrityError, ObjectReference,
    StorageError, _decode_document, _merge_entries, _prepare_publication, canonical_document,
)
from benchmax.storage.store import Store
from benchmax.utils.files import cache_directory
from benchmax.utils.hashing import _digest_bytes, _digest_file
from benchmax.utils.json import JsonObject

if TYPE_CHECKING:
    from benchmax.storage.requirements import PolicyRequirements

_ENTRY = re.compile(r"entries/([0-9a-f]{64})/([0-9a-f]{64})[.]json")


def _validate_bucket_location(bucket_id: str, prefix: str) -> None:
    if not isinstance(bucket_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", bucket_id) is None:
        raise ValueError("a Hugging Face bucket ID must have the form 'owner/name'")

    if not isinstance(prefix, str) or any(part in {"", ".", ".."} for part in prefix.split("/")) and prefix:
        raise ValueError("prefix must be empty or a relative path without empty, '.' or '..' components")

    if "\\" in prefix:
        raise ValueError("prefix must use forward slashes")


class StorageAuthenticationError(StorageError):
    """The Hub requires authentication or the token lacks bucket access."""


class HuggingFaceStore(Store):
    """A bucket with hash-verified objects and independent publication records.

    The bucket must already exist. token=None uses the saved login or HF_TOKEN,
    True requires a token, and False selects anonymous access. Construction is offline.
    Buckets have no Git revisions. A snapshot freezes discovery in memory, while
    object hashes verify reads. A snapshot does not retain deleted remote files.
    """

    def __init__(
        self, bucket_id: str, *, prefix: str = "", token: str | bool | None = None,
        cache_dir: str | os.PathLike[str] | None = None, requirements: PolicyRequirements | None = None,
    ) -> None:
        super().__init__(requirements=requirements)
        _validate_bucket_location(bucket_id, prefix)

        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("token must be a string, bool, or None")

        if token == "":
            raise ValueError("token must not be empty; use None for saved credentials or False for anonymous access")

        self.bucket_id = bucket_id
        self.prefix = prefix
        self.token = token
        cache = Path(cache_dir) if cache_dir is not None else cache_directory() / "buckets"
        self.cache_dir = cache.expanduser().resolve()
        self._snapshot: tuple[CatalogEntry, ...] | None = None

    def _remote(self, path: str) -> str:
        return f"{self.prefix}/{path}" if self.prefix else path

    def _api(self) -> Any:
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def _check_authentication_error(self, error: Exception, *, writing: bool = False) -> None:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in {401, 403} or type(error).__name__ == "LocalTokenNotFoundError":
            permission = "write" if writing else "read"
            raise StorageAuthenticationError(
                f"Hugging Face access denied for {self.bucket_id}; run 'hf auth login', set HF_TOKEN, or pass token=... "
                f"with {permission} access to this bucket"
            ) from None

    def _download(self, reference: ObjectReference | BlobReference) -> Path:
        return self._download_files({reference.path: reference})[reference.path]

    def _download_files(self, references: Mapping[str, ObjectReference | BlobReference]) -> dict[str, Path]:
        # Paths are constructed from validated content identities, never arbitrary remote names.
        root = self.cache_dir / self.bucket_id / self.prefix
        destinations = {path: root / path for path in references}
        pending = {}
        for path, reference in references.items():
            destination = destinations[path]
            if destination.is_file():
                digest, size = _digest_file(destination)
                if digest == reference.digest and (not isinstance(reference, BlobReference) or size == reference.size):
                    continue
            pending[path] = reference

        if not pending:
            return destinations

        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix=".download-") as directory:
            temporary = {path: Path(directory) / str(index) for index, path in enumerate(pending)}
            try:
                self._api().download_bucket_files(
                    self.bucket_id, files=[(self._remote(path), local) for path, local in temporary.items()],
                    raise_on_missing_files=True,
                )
            except Exception as error:
                self._check_authentication_error(error)
                raise StorageError(f"cannot download content from {self.bucket_id}") from error

            # Verify the whole response before exposing any new cache entries.
            for path, reference in pending.items():
                digest, size = _digest_file(temporary[path])
                if digest != reference.digest or (isinstance(reference, BlobReference) and size != reference.size):
                    raise IntegrityError(f"stored content does not match {reference.digest}")

            for path, local in temporary.items():
                destinations[path].parent.mkdir(parents=True, exist_ok=True)
                os.replace(local, destinations[path])

        return destinations

    def snapshot(self) -> HuggingFaceStore:
        """Freeze the observed discovery entries in memory."""

        snapshot = copy(self)
        snapshot._snapshot = self._catalog()
        return snapshot

    def _catalog(self, reference: ObjectReference | None = None) -> tuple[CatalogEntry, ...]:
        if self._snapshot is not None:
            return tuple(entry for entry in self._snapshot if reference is None or entry.object == reference)

        prefix = "entries/" if reference is None else f"entries/{reference.digest}/"
        try:
            files = list(self._api().list_bucket_tree(self.bucket_id, prefix=self._remote(prefix), recursive=True))
        except Exception as error:
            self._check_authentication_error(error)
            raise StorageError(f"cannot list publications in {self.bucket_id}") from error

        publications = []
        for item in files:
            if item.type != "file":
                continue

            path = item.path.removeprefix(self.prefix + "/") if self.prefix else item.path
            match = _ENTRY.fullmatch(path)
            if match is None:
                raise StorageError(f"invalid publication path: {item.path!r}")

            object_digest, entry_digest = match.groups()
            publications.append((path, object_digest, ObjectReference(entry_digest)))

        downloaded = self._download_files({path: marker for path, _, marker in publications})
        entries = []
        for path, object_digest, marker in publications:
            entry = CatalogEntry.from_document(_decode_document(downloaded[path].read_bytes(), marker))
            if entry.object.digest != object_digest:
                raise IntegrityError("publication path and object identity disagree")

            entries.append(entry)

        # Concurrent, conflicting metadata cannot silently replace an earlier entry.
        return _merge_entries(entries)

    def read_document(self, reference: ObjectReference) -> JsonObject:
        path = self._download(reference)
        return _decode_document(path.read_bytes(), reference)

    def _verified_blob_path(self, reference: BlobReference) -> Path:
        return self._download(reference)

    def publish(
        self, document: Mapping[str, object], *, dimensions: Mapping[str, str] | None = None,
        blobs: Mapping[BlobReference, BlobSource] | None = None,
    ) -> CatalogEntry:
        if self._snapshot is not None:
            raise StorageError("a HuggingFaceStore snapshot is read-only")

        encoded, entry, prepared_blobs = _prepare_publication(document, dimensions=dimensions, blobs=blobs)
        entries = self._catalog(entry.object)
        if _merge_entries((*entries, entry)) == entries:
            self.read_document(entry.object)
            self._download_files({reference.path: reference for reference, _ in prepared_blobs})
            return entry

        # Bucket batches are not transactional. Freeze/verify input files, upload
        # dependencies first, then publish one independent discovery marker.
        with tempfile.TemporaryDirectory(prefix="benchmax-bucket-upload-") as directory:
            uploads: list[tuple[BlobSource, str]] = [(encoded, self._remote(entry.object.path))]
            for reference, source in prepared_blobs:
                frozen = Path(directory) / reference.digest
                if isinstance(source, bytes):
                    frozen.write_bytes(source)
                else:
                    shutil.copyfile(source, frozen)
                _verify_blob(frozen, reference)
                uploads.append((frozen, self._remote(reference.path)))

            marker = canonical_document(entry.to_document())
            marker_digest = _digest_bytes(marker)
            marker_path = self._remote(f"entries/{entry.object.digest}/{marker_digest}.json")
            try:
                api = self._api()
                api.batch_bucket_files(self.bucket_id, add=uploads)
                api.batch_bucket_files(self.bucket_id, add=[(marker, marker_path)])
            except Exception as error:
                self._check_authentication_error(error, writing=True)
                raise StorageError(f"cannot publish to {self.bucket_id}; retry the operation") from error

        return entry
