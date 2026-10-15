"""Cache location, durable atomic publication and blocking file locks."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


def cache_directory() -> Path:
    """Resolve BenchMAX's cache root without creating it.

    An unset or empty XDG_CACHE_HOME uses ~/.cache. Subsystems append their
    own directories, and explicit caller-supplied locations bypass this default.
    """
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root.expanduser().resolve() / "benchmax"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _atomic_writer(destination: Path, *, replace: bool = True) -> Iterator[BinaryIO]:
    """Flush and close a writable temporary stream before atomically publishing it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, destination)
        else:
            # A destination created by another caller must never be overwritten.
            os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold a blocking exclusive lock. The caller owns the parent directory."""

    with path.open("a+b") as stream:
        if os.name == "nt":  # pragma: no cover - Windows only
            import msvcrt

            stream.seek(0)
            if stream.read(1) == b"":
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":  # pragma: no cover - Windows only
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
