"""Stored policy records and their searchable metadata."""

from __future__ import annotations  # noqa: I001

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import msgspec

from benchmax.execution.artifacts.policies import ArtifactError, resolve_policy_artifact, validate_policy_manifest
from benchmax.execution.specs import EntrySpec, TrainingSpec, _ArtifactMetadata, workload_from_document
from benchmax.release import LibraryRelease
from benchmax.storage._model import (
    BlobReference, BlobSource, ObjectReference, StorageError, canonical_document,
)
from benchmax.utils.json import JsonObject

if TYPE_CHECKING:
    from benchmax.core import Game


def _game_document(game: EntrySpec) -> JsonObject:
    # Evaluators do not affect the policy's game or training interface. The
    # portability layer deliberately saves training games with evaluator=None.
    document = game.to_document()
    document["overrides"].pop("evaluator", None)
    if document["resolved"] is not None:
        document["resolved"].pop("evaluator", None)
    return document


def _configuration_document(training: TrainingSpec) -> JsonObject:
    document = training.to_document()
    del document["seed"]
    document["game"] = _game_document(training.game)
    return document


def _identity(document: Mapping[str, object]) -> str:
    return ObjectReference.from_document(document).digest


class _PolicyRecord(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """A canonical policy manifest, its tensor archive, and producing release."""

    schema_version: Literal[1]
    kind: Literal["policy"]
    manifest: dict[str, Any]
    tensor: BlobReference

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, BlobReference):
            raise TypeError("policy tensor must be a BlobReference")
        copied = json.loads(canonical_document(self.manifest))
        assert isinstance(copied, dict)
        try:
            validate_policy_manifest(copied)
        except ArtifactError as error:
            raise StorageError(str(error)) from error
        tensor_digest = copied["tensor_sha256"]
        if self.tensor.digest != tensor_digest:
            raise StorageError("policy tensor reference does not match its manifest digest")
        msgspec.structs.force_setattr(self, "manifest", copied)
        _ = self.release  # Validate the producer at the decoding boundary.
        if "evaluation" in copied and not isinstance(copied["evaluation"], dict):
            raise StorageError("policy evaluation metadata must be an object")

    @property
    def release(self) -> LibraryRelease:
        producer = self.manifest.get("producer")
        if not isinstance(producer, dict) or set(producer) - {"release", "provenance"}:
            raise StorageError("policy producer must contain release and optional provenance")
        try:
            return LibraryRelease.from_document(producer.get("release"))
        except (TypeError, ValueError) as error:
            raise StorageError(f"invalid policy release: {error}") from error

    @property
    def training(self) -> TrainingSpec:
        value = workload_from_document(self.manifest["training"])
        assert isinstance(value, TrainingSpec)
        return value

    def to_document(self) -> JsonObject:
        return {
            "schema_version": 1,
            "kind": "policy",
            "manifest": self.manifest,
            "tensor": self.tensor.to_document(),
        }

    @classmethod
    def from_document(cls, document: object) -> _PolicyRecord:
        try:
            if not isinstance(document, Mapping):
                raise TypeError("stored policy must be an object")

            tensor = document.get("tensor")
            if not isinstance(tensor, Mapping) or set(tensor) != {"digest", "size"}:
                raise ValueError("policy tensor must contain digest and size")

            return msgspec.convert(document, type=cls, strict=True)
        except (TypeError, ValueError) as error:
            raise StorageError(f"invalid stored policy: {error}") from error

    @classmethod
    def from_artifact(
        cls, reference: _ArtifactMetadata | str, release: LibraryRelease | None = None, *,
        provenance: Mapping[str, object] | None = None, evaluation: Mapping[str, object] | None = None,
    ) -> tuple[_PolicyRecord, dict[BlobReference, BlobSource]]:
        """Read the canonical artifact without importing the training implementation."""

        manifest_path, manifest, _ = resolve_policy_artifact(reference)
        producer = manifest.get("producer", {})
        if not isinstance(producer, Mapping):
            raise StorageError("policy producer metadata must be an object")
        producer = dict(producer)
        if release is not None:
            if not isinstance(release, LibraryRelease):
                raise TypeError("release must be a LibraryRelease")
            if "release" in producer and producer["release"] != release.to_document():
                raise StorageError("cannot replace the recorded producing release")
            producer["release"] = release.to_document()
        if "release" not in producer:
            raise StorageError("policy has no producing release; supply release explicitly for legacy artifacts")
        if provenance is not None:
            if "provenance" in producer and producer["provenance"] != provenance:
                raise StorageError("cannot replace the recorded producing provenance")
            producer["provenance"] = dict(provenance)
        manifest["producer"] = producer
        if evaluation is not None:
            manifest["evaluation"] = dict(evaluation)
        tensor_digest = manifest["tensor_sha256"]
        tensor_path = (manifest_path.parent / manifest["tensor_file"]).resolve()
        if tensor_path.parent != manifest_path.parent:
            raise StorageError("policy tensor file must remain beside its manifest")
        try:
            tensor = BlobReference.from_source(tensor_path)
        except OSError as error:
            raise StorageError(f"cannot read policy tensor file {tensor_path}: {error}") from error
        if tensor.digest != tensor_digest:
            raise StorageError("policy tensor file does not match its manifest digest")
        return cls(schema_version=1, kind="policy", manifest=manifest, tensor=tensor), {tensor: tensor_path}


def _policy_dimensions(record: _PolicyRecord) -> dict[str, str]:
    training = record.training
    return {
        "game": _identity(_game_document(training.game)),
        "algorithm": _identity(training.algorithm.to_document()),
        "configuration": _identity(_configuration_document(training)),
        "workload": training.digest,
        "seed": str(training.seed),
        "seed_scheme": training.seed_scheme,
        "release": record.release.digest,
    }


def _game_entry(game: Game | EntrySpec | str) -> EntrySpec:
    if isinstance(game, EntrySpec):
        return game
    if isinstance(game, str):
        return EntrySpec.create("game", game)
    from benchmax.core import Game
    from benchmax.execution.jobs.submission import game_entry

    if not isinstance(game, Game):
        raise TypeError("game must be a Game, EntrySpec, or registered game name")
    return game_entry(game)
