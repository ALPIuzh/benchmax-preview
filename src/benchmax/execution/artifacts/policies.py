"""The canonical safe policy-artifact envelope."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from benchmax.execution.specs import EntrySpec, TrainingSpec, _ArtifactMetadata, workload_from_document
from benchmax.policy import PolicyProfile, _attach_policy_metadata
from benchmax.utils.hashing import _digest_bytes, _is_sha256

_MAX_POLICY_DOCUMENT_BYTES = 16 * 1024**2
_ARTIFACTS_ENVIRONMENT = "BENCHMAX_ARTIFACTS"
POLICY_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_MAX_POLICY_TENSOR_BYTES = 16 * 1024**3
POLICY_ANNOTATION_KEYS = frozenset({"producer", "evaluation", "progress"})


class ArtifactError(ValueError):
    """A policy artifact is incompatible, malformed, or fails verification."""


@dataclass(frozen=True)
class _PolicySource:
    """Game provenance for a policy created without a registered trainer."""

    game: EntrySpec


class PolicyCodec(Protocol):
    """Installed data-only codec. Importing and matching must not load optional runtimes."""

    POLICY_FORMAT: str
    REQUIRES_TRAINING: bool

    def supports(self, profile: PolicyProfile) -> bool: ...

    def write_policy(
        self, profile: PolicyProfile, source: TrainingSpec | _PolicySource, destination: str | Path
    ) -> _ArtifactMetadata: ...

    def validate_policy_document(self, path: Path, manifest: Mapping[str, Any], *, max_tensor_bytes: int) -> None: ...

    def load_policy_document(
        self, path: Path, manifest: Mapping[str, Any], source: TrainingSpec | _PolicySource, *,
        expected_game: EntrySpec, max_tensor_bytes: int
    ) -> PolicyProfile: ...


def _policy_codecs() -> dict[str, EntryPoint]:
    entries = {}
    for entry in entry_points(group="benchmax.policy_codecs"):
        if entry.name in entries:
            raise ArtifactError(f"multiple policy codecs are registered for {entry.name!r}")

        entries[entry.name] = entry

    return entries


def _load_codec(entry: EntryPoint) -> PolicyCodec:
    codec = entry.load()
    if (getattr(codec, "POLICY_FORMAT", None) != entry.name
            or not isinstance(getattr(codec, "REQUIRES_TRAINING", None), bool)
            or any(not callable(getattr(codec, name, None)) for name in
                   ("supports", "write_policy", "validate_policy_document", "load_policy_document"))):
        raise ArtifactError(f"invalid policy codec registered for {entry.name!r}")

    return codec


def _policy_codec(format_name: object) -> PolicyCodec:
    entries = _policy_codecs()
    if not isinstance(format_name, str) or format_name not in entries:
        raise ArtifactError(f"unsupported policy artifact format {format_name!r}")

    return _load_codec(entries[format_name])


def _check_source(codec: PolicyCodec, source: TrainingSpec | _PolicySource) -> None:
    if codec.REQUIRES_TRAINING and not isinstance(source, TrainingSpec):
        raise ArtifactError(f"policy format {codec.POLICY_FORMAT!r} requires training provenance")


def artifact_directory(output: Path | None = None) -> Path:
    """Allocate a caller-owned directory for one policy artifact."""

    if output is not None:
        return output.parent / f"{output.stem}.policy-{uuid.uuid4().hex}"

    configured = os.environ.get(_ARTIFACTS_ENVIRONMENT)
    root = Path(configured).expanduser() if configured else Path.home() / ".local" / "share" / "benchmax" / "artifacts"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"policy-{uuid.uuid4().hex}"


def _read_document(reference: _ArtifactMetadata, *, relative_to: Path | None) -> tuple[Path, dict[str, Any]]:
    path = Path(reference.uri).expanduser()
    if not path.is_absolute():
        path = (relative_to or Path.cwd()) / path

    path = path.resolve()

    try:
        with path.open("rb") as stream:
            value = stream.read(_MAX_POLICY_DOCUMENT_BYTES + 1)
    except OSError as error:
        raise ArtifactError(f"cannot read policy document {path}: {error}") from error

    if len(value) > _MAX_POLICY_DOCUMENT_BYTES:
        raise ArtifactError(f"policy document {path} exceeds the {_MAX_POLICY_DOCUMENT_BYTES}-byte safety limit")

    if reference.digest is not None and _digest_bytes(value) != reference.digest:
        raise ArtifactError(f"policy document digest does not match {path}")

    try:
        document = json.loads(value)
    except json.JSONDecodeError as error:
        raise ArtifactError(f"cannot read policy document {path}: {error}") from error

    if not isinstance(document, dict):
        raise ArtifactError(f"policy document {path} must be a JSON object")

    return path, document


def resolve_policy_artifact(
    reference: _ArtifactMetadata | str, *, relative_to: Path | None = None, remaining_redirects: int = 4,
    seen: frozenset[Path] = frozenset()
) -> tuple[Path, dict[str, Any], TrainingSpec | _PolicySource]:
    """Verify redirects and return one canonical policy manifest."""

    if isinstance(reference, str):
        reference = _ArtifactMetadata(reference)

    path, document = _read_document(reference, relative_to=relative_to)
    if path in seen:
        raise ArtifactError(f"policy document reference cycle reaches {path}")

    if document.get("kind") == "training-result":
        if remaining_redirects == 0:
            raise ArtifactError("policy document reference depth exceeds four")

        nested = document.get("policy")
        if not isinstance(nested, Mapping):
            raise ArtifactError(f"training result {path} has no policy reference")

        try:
            policy = _ArtifactMetadata.from_document(nested)
        except (TypeError, ValueError) as error:
            raise ArtifactError(f"training result {path} has an invalid policy") from error

        return resolve_policy_artifact(
            policy, relative_to=path.parent, remaining_redirects=remaining_redirects - 1, seen=seen | {path}
        )

    return path, document, validate_policy_manifest(document)


def validate_policy_manifest(manifest: Mapping[str, Any]) -> TrainingSpec | _PolicySource:
    """Check the canonical envelope and file references without reading tensor data."""

    if (type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != POLICY_ARTIFACT_SCHEMA_VERSION
            or manifest.get("kind") != "policy-artifact"):
        raise ArtifactError("unsupported policy artifact")

    codec = _policy_codec(manifest.get("format"))
    try:
        if manifest.get("training") is None:
            source = _PolicySource(EntrySpec.from_document(manifest.get("game"), label="policy game"))
        else:
            source = workload_from_document(manifest["training"])
            if not isinstance(source, TrainingSpec):
                raise ArtifactError("policy artifact does not contain a training workload")

            if "game" in manifest and EntrySpec.from_document(manifest["game"], label="policy game") != source.game:
                raise ArtifactError("policy game differs from its training workload")
    except ValueError as error:
        raise ArtifactError(f"policy artifact has an invalid training workload or game: {error}") from error

    _check_source(codec, source)

    tensor_name = manifest.get("tensor_file")
    tensor_digest = manifest.get("tensor_sha256")

    if not isinstance(tensor_name, str) or not tensor_name or "\\" in tensor_name or "\x00" in tensor_name:
        raise ArtifactError("policy manifest has no valid tensor archive reference")

    tensor_file = PurePosixPath(tensor_name)
    if (tensor_file.is_absolute() or len(tensor_file.parts) != 1 or tensor_file.as_posix() != tensor_name
            or tensor_name in {".", "..", "manifest.json"}):
        raise ArtifactError("policy tensor archive must remain beside its manifest")

    if not _is_sha256(tensor_digest):
        raise ArtifactError("policy manifest has no valid tensor digest")

    return source


def validate_policy_artifact(
    reference: _ArtifactMetadata | str, *, max_tensor_bytes: int = DEFAULT_MAX_POLICY_TENSOR_BYTES,
) -> tuple[Path, dict[str, Any], TrainingSpec | _PolicySource]:
    """Verify an artifact without importing Torch, loading arrays, or constructing a policy."""

    path, manifest, training = resolve_policy_artifact(reference)
    _policy_codec(manifest["format"]).validate_policy_document(path, manifest, max_tensor_bytes=max_tensor_bytes)
    return path, manifest, training


def write_policy(
    profile: PolicyProfile, training: TrainingSpec | _PolicySource, destination: str | Path
) -> _ArtifactMetadata:
    """Publish a supported representation without serializing Python code."""

    codecs = (_load_codec(entry) for _, entry in sorted(_policy_codecs().items()))
    matches = [codec for codec in codecs if codec.supports(profile)]

    if not matches:
        raise ArtifactError(
            "policy has no registered safe representation; use PolicyProfile.from_table() for finite policies"
        )

    if len(matches) != 1:
        formats = [codec.POLICY_FORMAT for codec in matches]
        raise ArtifactError(f"multiple policy codecs support this profile: {formats}")

    codec = matches[0]
    _check_source(codec, training)
    return codec.write_policy(profile, training, destination)


def load_policy(
    reference: _ArtifactMetadata | str, *, expected_game: EntrySpec,
    max_tensor_bytes: int = DEFAULT_MAX_POLICY_TENSOR_BYTES, retain_source: bool = True,
) -> PolicyProfile:
    """Verify and load a policy. Temporary sources must opt out of retaining their path."""

    path, manifest, source = resolve_policy_artifact(reference)
    profile = _policy_codec(manifest["format"]).load_policy_document(
        path, manifest, source, expected_game=expected_game, max_tensor_bytes=max_tensor_bytes
    )
    _attach_policy_metadata(profile, training=source if isinstance(source, TrainingSpec) else None,
                            reference=_ArtifactMetadata.from_path(path) if retain_source else None, game=source.game)
    profile._artifact_annotations = {key: manifest[key] for key in POLICY_ANNOTATION_KEYS if key in manifest}
    return profile


def _copy_policy_files(reference: _ArtifactMetadata, destination: Path) -> _ArtifactMetadata:
    """Copy a verified canonical artifact into a new directory without loading a model."""
    manifest_path, manifest, _ = validate_policy_artifact(reference)
    paths = (manifest_path, manifest_path.parent / manifest["tensor_file"])

    for path in paths:
        if not path.is_file() or path.is_symlink() or path.resolve().parent != manifest_path.parent:
            raise ArtifactError("policy files must be regular files beside the manifest")

    destination.mkdir(parents=True, exist_ok=False)

    for path in paths:
        shutil.copyfile(path, destination / path.name)

    copied = _ArtifactMetadata.from_path(destination / manifest_path.name)
    if copied.digest != reference.digest:
        raise ArtifactError("copied policy manifest differs from the completed result")

    validate_policy_artifact(copied)
    return copied
