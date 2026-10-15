"""Validated NPZ storage shared by policy codecs."""

from __future__ import annotations

import math
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmax.execution.artifacts.policies import ArtifactError
from benchmax.utils.hashing import _sha256_file

_MAX_NPY_HEADER_BYTES = 128 * 1024
_ZIP_FIXED_OVERHEAD_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class _TensorEntry:
    array: str
    player: int
    state_name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    nbytes: int

    @property
    def member(self) -> str:
        return f"{self.array}.npy"


class _HeaderReader:
    """Bound reads while NumPy parses an untrusted NPY header."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0 or self.bytes_read + size > _MAX_NPY_HEADER_BYTES:
            raise ArtifactError("policy tensor NPY header exceeds the safety limit")

        value = self._stream.read(size)
        self.bytes_read += len(value)
        return value


def _parse_tensor_index(entries: list[Any], *, max_tensor_bytes: int) -> list[_TensorEntry]:
    if not isinstance(max_tensor_bytes, int) or isinstance(max_tensor_bytes, bool) or max_tensor_bytes <= 0:
        raise ArtifactError("max_tensor_bytes must be a positive integer")

    parsed: list[_TensorEntry] = []
    arrays: set[str] = set()
    states: set[tuple[int, str]] = set()
    total_bytes = 0

    for entry in entries:
        if not isinstance(entry, dict):
            raise ArtifactError("policy tensor index entry must be an object")

        array_name = entry.get("array")
        player = entry.get("player")
        state_name = entry.get("state_name")
        shape = entry.get("shape")
        dtype_name = entry.get("dtype")
        if (not isinstance(array_name, str) or not array_name.isidentifier()
                or not isinstance(player, int) or isinstance(player, bool)
                or not isinstance(state_name, str) or not state_name
                or not isinstance(shape, list) or len(shape) > 32
                or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)
                or not isinstance(dtype_name, str)):
            raise ArtifactError("policy tensor index entry is malformed")

        try:
            dtype = np.dtype(dtype_name)
        except TypeError as error:
            raise ArtifactError("policy tensor index has an invalid dtype") from error

        if dtype.str != dtype_name or dtype.hasobject or dtype.kind not in "biufc":
            raise ArtifactError("policy tensor index has an unsupported dtype")

        if array_name in arrays:
            raise ArtifactError(f"duplicate policy tensor array {array_name!r}")

        state_key = (player, state_name)
        if state_key in states:
            raise ArtifactError(f"duplicate policy state entry {state_name!r}")

        nbytes = math.prod(shape) * dtype.itemsize
        total_bytes += nbytes

        if total_bytes > max_tensor_bytes:
            raise ArtifactError(
                f"policy tensors require {total_bytes} bytes, exceeding the {max_tensor_bytes}-byte load limit"
            )

        arrays.add(array_name)
        states.add(state_key)
        parsed.append(_TensorEntry(array_name, player, state_name, tuple(shape), dtype, nbytes))

    return parsed


def _preflight_tensor_archive(tensor_path: Path, entries: list[_TensorEntry]) -> None:
    expected_payload = sum(entry.nbytes for entry in entries)
    maximum_archive_bytes = expected_payload + len(entries) * (_MAX_NPY_HEADER_BYTES + 1024) + _ZIP_FIXED_OVERHEAD_BYTES
    if tensor_path.stat().st_size > maximum_archive_bytes:
        raise ArtifactError("policy tensor archive exceeds its indexed size bound")

    with zipfile.ZipFile(tensor_path) as archive:
        infos = archive.infolist()
        members = [info.filename for info in infos]
        expected = [entry.member for entry in entries]
        if len(members) != len(set(members)) or set(members) != set(expected):
            raise ArtifactError("policy tensor archive members do not match its index")

        by_name = {info.filename: info for info in infos}

        for entry in entries:
            info = by_name[entry.member]
            if info.flag_bits & 1:
                raise ArtifactError("encrypted policy tensor archives are unsupported")

            if not entry.nbytes <= info.file_size <= entry.nbytes + _MAX_NPY_HEADER_BYTES:
                raise ArtifactError(f"policy tensor {entry.state_name!r} has an invalid archive size")

            with archive.open(info) as stream:
                reader = _HeaderReader(stream)
                version = np.lib.format.read_magic(reader)

                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(reader)
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(reader)
                else:
                    raise ArtifactError(f"policy tensor archive uses unsupported NPY {version!r}")

                if (tuple(shape) != entry.shape or np.dtype(dtype) != entry.dtype or fortran_order
                        or reader.bytes_read + entry.nbytes != info.file_size):
                    raise ArtifactError(f"policy tensor {entry.state_name!r} metadata does not match the archive")


def _inspect_tensor_archive(
    manifest_path: Path, manifest: Mapping[str, Any], *, max_tensor_bytes: int
) -> tuple[Path, list[_TensorEntry]]:
    tensor_name = manifest.get("tensor_file")
    expected_digest = manifest.get("tensor_sha256")
    entries = manifest.get("tensors")

    if not isinstance(tensor_name, str) or not isinstance(expected_digest, str):
        raise ArtifactError("policy manifest has no valid tensor archive reference")

    if not isinstance(entries, list):
        raise ArtifactError("policy manifest tensor index must be a list")

    tensor_path = (manifest_path.parent / tensor_name).resolve()
    if tensor_path.parent != manifest_path.parent:
        raise ArtifactError("policy tensor archive must remain beside its manifest")

    try:
        actual_digest = _sha256_file(tensor_path)
    except OSError as error:
        raise ArtifactError(f"cannot read policy tensor archive: {error}") from error

    if actual_digest != expected_digest:
        raise ArtifactError(f"policy tensor digest does not match {tensor_path}")

    try:
        parsed_entries = _parse_tensor_index(entries, max_tensor_bytes=max_tensor_bytes)
        _preflight_tensor_archive(tensor_path, parsed_entries)
    except ArtifactError:
        raise
    except (OSError, EOFError, MemoryError, TypeError, ValueError, KeyError,
            zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ArtifactError(f"cannot load policy tensor archive: {error}") from error

    return tensor_path, parsed_entries


def _read_tensor_array(archive: zipfile.ZipFile, entry: _TensorEntry) -> np.ndarray:
    with archive.open(entry.member) as stream:
        array = np.load(stream, allow_pickle=False)

    if array.shape != entry.shape or array.dtype != entry.dtype or array.dtype.hasobject:
        raise ArtifactError(f"policy tensor {entry.state_name!r} metadata does not match the archive")

    return array


def _write_array_policy(arrays, source, destination, interface, format_name, *, annotations=None, **metadata):
    """Publish data arrays and a manifest, removing incomplete owned output."""
    import shutil

    from benchmax.execution.artifacts.policies import (  # isort: skip
        POLICY_ANNOTATION_KEYS, POLICY_ARTIFACT_SCHEMA_VERSION, validate_policy_artifact)
    from benchmax.execution.specs import TrainingSpec, _ArtifactMetadata
    from benchmax.utils.files import _atomic_writer
    from benchmax.utils.json import write_document

    directory = Path(destination).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    try:
        named, index = {}, []
        for (player, name), array in arrays.items():
            key = f"tensor_{len(named)}"
            named[key] = array
            index.append({"array": key, "player": player, "state_name": name,
                          "shape": list(array.shape), "dtype": array.dtype.str})

        tensor_path = directory / "tensors.npz"
        with _atomic_writer(tensor_path) as stream:
            np.savez(stream, **named)

        manifest = {"schema_version": POLICY_ARTIFACT_SCHEMA_VERSION, "kind": "policy-artifact",
                    "format": format_name, "policy_interface": interface.to_document(),
                    "training": source.to_document() if isinstance(source, TrainingSpec) else None,
                    "game": source.game.to_document(), "tensor_file": tensor_path.name,
                    "tensor_sha256": _sha256_file(tensor_path), "tensors": index, "state_values": [], **metadata}
        manifest.update({key: value for key, value in (annotations or {}).items() if key in POLICY_ANNOTATION_KEYS})
        path = write_document(manifest, directory / "manifest.json")
        reference = _ArtifactMetadata.from_path(path)
        validate_policy_artifact(reference)
        return reference
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
