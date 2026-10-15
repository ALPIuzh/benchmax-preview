"""SHA-256 content identities shared by artifacts, workloads, and storage."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TypeGuard


def _is_sha256(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> str:
    return _digest_file(path)[0]
