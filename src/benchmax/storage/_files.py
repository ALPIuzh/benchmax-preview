"""Verified file operations shared by local and bucket storage."""

from __future__ import annotations

import shutil
from pathlib import Path

from benchmax.storage._model import BlobReference, BlobSource, IntegrityError, StorageError
from benchmax.utils.files import _atomic_writer
from benchmax.utils.hashing import _digest_file


def _verify_blob(path: Path, reference: BlobReference) -> Path:
    try:
        actual = _digest_file(path)
    except OSError as error:
        raise StorageError(f"cannot read stored blob {reference.digest}: {error}") from error
    if actual != (reference.digest, reference.size):
        raise IntegrityError(f"stored blob does not match {reference.digest}")
    return path


def _atomic_bytes(path: Path, value: bytes) -> None:
    with _atomic_writer(path) as stream:
        stream.write(value)


def _atomic_blob(path: Path, reference: BlobReference, source: BlobSource, *, replace_existing: bool = True) -> None:
    with _atomic_writer(path, replace=replace_existing) as stream:
        if isinstance(source, bytes):
            stream.write(source)
        else:
            with Path(source).open("rb") as input_stream:
                shutil.copyfileobj(input_stream, stream, length=1024 * 1024)
        stream.flush()
        _verify_blob(Path(stream.name), reference)
