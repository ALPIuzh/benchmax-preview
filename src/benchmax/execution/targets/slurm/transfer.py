"""Bounded transfers of the canonical policy manifest and tensor archive."""

from __future__ import annotations

import base64
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from benchmax.execution.artifacts.policies import ArtifactError, resolve_policy_artifact, validate_policy_artifact
from benchmax.execution.specs import _ArtifactMetadata
from benchmax.utils.hashing import _digest_bytes


def cached_policy(root: Path, digest: str, manifest: str, *, limit: int,
                  bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect or atomically populate a content-addressed target policy cache."""
    _ArtifactMetadata("cache", digest)
    if (digest is None or not isinstance(manifest, str) or manifest in {"", ".", ".."}
            or "/" in manifest or "\\" in manifest):
        raise ArtifactError("invalid cached policy identity")

    destination = root / digest / _digest_bytes(manifest.encode())[:16]
    reference = _ArtifactMetadata(str(destination / manifest), digest)
    if bundle is not None:
        import fcntl

        root.mkdir(parents=True, exist_ok=True)
        with (root / f"{digest}.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not destination.exists():
                with tempfile.TemporaryDirectory(prefix=".upload-", dir=root) as temporary:
                    staged = Path(temporary) / "policy"
                    unpack_policy(bundle, staged, digest=digest, limit=limit)
                    if bundle["manifest"] != manifest:
                        raise ArtifactError("uploaded manifest name mismatch")

                    destination.parent.mkdir(exist_ok=True)
                    os.rename(staged, destination)

    present = destination.exists()
    if present:
        path, document, _ = validate_policy_artifact(reference, max_tensor_bytes=limit)
        if path.stat().st_size + (path.parent / document["tensor_file"]).stat().st_size > limit:
            raise ArtifactError("policy exceeds transfer_limit_mb")

    return {"policy": reference.to_document(), "cached": present}


def pack_policy(reference: _ArtifactMetadata, *, limit: int) -> dict[str, Any]:
    manifest_path, manifest, _ = resolve_policy_artifact(reference)
    paths = [manifest_path, manifest_path.parent / manifest["tensor_file"]]
    if len({path.name for path in paths}) != len(paths):
        raise ArtifactError("policy artifact contains duplicate file names")

    total = 0
    files = {}

    for path in paths:
        path = Path(path)
        if not path.is_file() or path.is_symlink() or path.resolve().parent != manifest_path.parent:
            raise ArtifactError("policy files must be regular files beside the manifest")

        size = path.stat().st_size
        total += size

        if total > limit:
            raise ArtifactError("policy exceeds transfer_limit_mb; reuse it on the target or transfer it explicitly")

        with path.open("rb") as stream:
            data = stream.read(min(size, limit) + 1)

        if len(data) != size:
            raise ArtifactError("policy changed during transfer")

        files[path.name] = base64.b64encode(data).decode("ascii")

    validate_policy_artifact(reference, max_tensor_bytes=limit)
    return {"manifest": manifest_path.name, "digest": reference.digest, "files": files}


def unpack_policy(bundle: dict[str, Any], destination: Path, *, digest: str, limit: int) -> _ArtifactMetadata:
    manifest = bundle.get("manifest")
    files = bundle.get("files")
    if bundle.get("digest") != digest or not isinstance(files, dict) or manifest not in files:
        raise ArtifactError("downloaded policy does not match the requested artifact")

    decoded = {}
    total = 0

    for name, value in files.items():
        if not isinstance(name, str) or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
            raise ArtifactError("invalid policy file name")

        if not isinstance(value, str) or len(value) > (limit + 2) // 3 * 4:
            raise ArtifactError("downloaded policy exceeds transfer limit")

        data = base64.b64decode(value, validate=True)
        total += len(data)

        if total > limit:
            raise ArtifactError("downloaded policy exceeds transfer limit")

        decoded[name] = data

    if _digest_bytes(decoded[manifest]) != digest:
        raise ArtifactError("downloaded manifest digest mismatch")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
        staged = Path(temporary)
        for name, data in decoded.items():
            (staged / name).write_bytes(data)

        _, document, _ = validate_policy_artifact(_ArtifactMetadata(str(staged / manifest), digest), max_tensor_bytes=limit)
        if set(files) != {manifest, document["tensor_file"]}:
            raise ArtifactError("downloaded policy contains files outside its manifest")

        destination.mkdir(exist_ok=False)
        try:
            for name in decoded:
                os.replace(staged / name, destination / name)
        except OSError:
            shutil.rmtree(destination)
            raise

    return _ArtifactMetadata(str(destination / manifest), digest)
