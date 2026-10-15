"""Shared policy operations and the storage backend contract."""

from __future__ import annotations  # noqa: I001

import contextlib
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from benchmax.config import ConfigValue
from benchmax.execution.artifacts.policies import (
    POLICY_ANNOTATION_KEYS, ArtifactError, validate_policy_artifact, write_policy,
)
from benchmax.execution.jobs.reference import Reference
from benchmax.execution.specs import EntrySpec, TrainingSpec, _ArtifactMetadata
from benchmax.policy import PolicyProfile
from benchmax.release import LibraryRelease
from benchmax.storage._files import _atomic_blob
from benchmax.storage._model import (
    AmbiguousPolicyError, BlobReference, BlobSource, CatalogEntry, CatalogQuery, ObjectReference,
    PolicyNotFoundError, StorageError,
)
from benchmax.storage._policy_record import (
    _game_document, _game_entry, _policy_dimensions, _PolicyRecord,
)
from benchmax.storage.requirements import DEFAULT_REQUIREMENTS, PolicyRequirements, PolicyValidation, metadata_errors
from benchmax.utils.json import JsonObject, write_document

if TYPE_CHECKING:
    from benchmax.core import Game


class Store(ABC):
    """Common object and policy operations for every storage backend."""

    def __init__(self, *, requirements: PolicyRequirements | None = None) -> None:
        if requirements is not None and not isinstance(requirements, PolicyRequirements):
            raise TypeError("requirements must be a PolicyRequirements")
        self.requirements = DEFAULT_REQUIREMENTS if requirements is None else requirements

    @abstractmethod
    def _catalog(self) -> tuple[CatalogEntry, ...]: ...

    @abstractmethod
    def _verified_blob_path(self, reference: BlobReference) -> Path: ...

    def find(self, query: CatalogQuery | None = None) -> tuple[CatalogEntry, ...]:
        query = query or CatalogQuery()
        return tuple(entry for entry in self._catalog() if query.matches(entry))

    @abstractmethod
    def read_document(self, reference: ObjectReference) -> JsonObject: ...

    def materialize_blob(self, reference: BlobReference, destination: str | os.PathLike[str]) -> Path:
        destination_path = Path(destination).expanduser().resolve()
        if destination_path.exists():
            raise FileExistsError(destination_path)
        _atomic_blob(destination_path, reference, self._verified_blob_path(reference), replace_existing=False)
        return destination_path

    @abstractmethod
    def publish(
        self, document: Mapping[str, object], *, dimensions: Mapping[str, str] | None = None,
        blobs: Mapping[BlobReference, BlobSource] | None = None,
    ) -> CatalogEntry: ...

    def publish_policy(
        self, policy: PolicyProfile | Reference[PolicyProfile] | str | os.PathLike[str], *,
        release: LibraryRelease | None = None,
        provenance: Mapping[str, object] | None = None, evaluation: Mapping[str, object] | None = None,
    ) -> Reference[PolicyProfile]:
        """Validate against the store's requirements before making a policy visible."""

        from benchmax.storage.reference import _policy_reference

        with _policy_source(policy) as artifact:
            record, sources = _PolicyRecord.from_artifact(
                artifact, release, provenance=provenance, evaluation=evaluation,
            )
            self._validation(record, artifact).raise_for_errors()
            published = self.publish(record.to_document(), dimensions=_policy_dimensions(record), blobs=sources)
            return _policy_reference(self, published.object, record.training)

    def _save_reference(
        self, reference: ObjectReference | Reference[PolicyProfile], destination: str | os.PathLike[str],
    ) -> Reference[PolicyProfile]:
        """Save into an exclusively created directory, cleaning it up if validation fails."""

        record = self._reference_record(reference)
        destination_path = Path(destination).expanduser().resolve()
        destination_path.mkdir(parents=True, exist_ok=False)
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(shutil.rmtree, destination_path)
            manifest = self._materialize_record(record, destination_path)
            self._validation(record, manifest).raise_for_errors()
            saved = Reference.from_path(manifest)
            cleanup.pop_all()
            return saved

    def validate_policy(
        self, policy: PolicyProfile | ObjectReference | Reference[PolicyProfile] | str | os.PathLike[str], *,
        release: LibraryRelease | None = None, provenance: Mapping[str, object] | None = None,
        evaluation: Mapping[str, object] | None = None,
    ) -> PolicyValidation:
        """Verify archive and metadata without reconstructing a policy or executing stored configuration."""

        try:
            stored_reference = isinstance(policy, Reference) and policy.kind == "stored-policy"
            if isinstance(policy, ObjectReference) or stored_reference:
                source = policy._storage() if stored_reference else self
                if provenance is not None or evaluation is not None:
                    raise ValueError("stored metadata cannot be replaced during validation")
                with source._stored_artifact(policy) as (record, manifest_path):
                    report = self._validation(record, manifest_path)
                    if release is not None and release != record.release:
                        return PolicyValidation(report.requirements, report.errors + ("producing release differs",))
                    return report
            with _policy_source(policy) as artifact:
                record, _ = _PolicyRecord.from_artifact(artifact, release, provenance=provenance, evaluation=evaluation)
                return self._validation(record, artifact)
        except (OSError, ValueError, TypeError, StorageError) as error:
            return PolicyValidation(self.requirements.name, (str(error),))

    def find_policies(
        self, game: Game | EntrySpec | str | None = None, *, algorithm: str | EntrySpec | None = None,
        seed: int | None = None, release: LibraryRelease | None = None,
        game_config: Mapping[str, ConfigValue] | Literal["default", "any"] = "default",
        training_config: Mapping[str, ConfigValue] | Literal["default", "any"] = "default",
    ) -> tuple[Reference[PolicyProfile], ...]:
        """Return references to stored policies matching the filters.

        Configuration mappings resolve overrides against the same defaults as game creation and training.
        "any" ignores that configuration while retaining the supplied game or algorithm name.
        A configured game is preserved by default. Omitted names leave that dimension unrestricted.
        Discovery includes every seed unless specified, without downloading or validating tensors.
        """

        from benchmax.storage.reference import _policy_reference

        for name, config in (("game_config", game_config), ("training_config", training_config)):
            if not isinstance(config, Mapping) and config not in ("default", "any"):
                raise ValueError(f"{name} must be 'default', 'any', or an overrides mapping")

        game_entry = _game_entry(game) if game is not None and not isinstance(game, str) else None
        game_name = game_entry.name if game_entry is not None else game
        expected_game = None
        if game_config != "any" and game is not None:
            if isinstance(game_config, Mapping):
                if not isinstance(game, str):
                    raise TypeError("game_config overrides require a game name; a configured game already has settings")
                expected_game = EntrySpec.create("game", game, game_config)
            else:
                expected_game = game_entry if game_entry is not None else _game_entry(game)
        elif isinstance(game_config, Mapping):
            raise ValueError("game_config overrides require a game name")

        if algorithm is not None and not isinstance(algorithm, (str, EntrySpec)):
            raise TypeError("algorithm must be a recorded name or EntrySpec")

        if algorithm == "":
            raise ValueError("algorithm must be non-empty")

        if seed is not None:
            from benchmax.seeding import validate_seed

            validate_seed(seed)

        if release is not None and not isinstance(release, LibraryRelease):
            raise TypeError("release must be a LibraryRelease")

        release = release or self.requirements.expected_release
        expected_training = None
        if training_config != "any" and algorithm is not None:
            overrides = dict(training_config) if isinstance(training_config, Mapping) else {}
            if "seed" in overrides:
                raise ValueError("use the seed filter separately from training_config")

            expected_training = TrainingSpec.create(
                game=EntrySpec("selection"), algorithm=algorithm, seed=0,
                device=overrides.pop("device", "auto"), total_frames=overrides.pop("total_frames", None),
                algorithm_overrides=overrides,
            )
        elif isinstance(training_config, Mapping):
            raise ValueError("training_config overrides require an algorithm")

        game_configuration = _entry_configuration(_game_document(expected_game)) if expected_game is not None else None
        training_configuration = _training_configuration(expected_training) if expected_training is not None else None
        dimensions = {}
        if seed is not None:
            dimensions["seed"] = str(seed)
        if release is not None:
            dimensions["release"] = release.digest
        query = CatalogQuery(kind="policy", dimensions=dimensions)
        matches = []
        for entry in self.find(query):
            record = _PolicyRecord.from_document(self.read_document(entry.object))
            actual = CatalogEntry("policy", entry.object, _policy_dimensions(record))
            if entry != actual:
                raise StorageError("policy catalog metadata does not match its authoritative record")
            recorded = record.training
            if game_configuration is not None:
                if _entry_configuration(_game_document(recorded.game)) != game_configuration:
                    continue
            elif game_name is not None and recorded.game.name != game_name:
                continue

            if training_configuration is not None:
                if _training_configuration(recorded) != training_configuration:
                    continue
            elif algorithm is not None:
                name = algorithm.name if isinstance(algorithm, EntrySpec) else algorithm
                if recorded.algorithm.name != name:
                    continue

            matches.append(_policy_reference(self, entry.object, recorded))
        return tuple(matches)

    def load_policy(
        self, game: Game | EntrySpec | str, *, algorithm: str | EntrySpec | None = None, seed: int | None = None,
        release: LibraryRelease | None = None,
        game_config: Mapping[str, ConfigValue] | Literal["default", "any"] = "default",
        training_config: Mapping[str, ConfigValue] | Literal["default", "any"] = "default",
    ) -> PolicyProfile:
        """Load the single policy matching the filters."""

        matches = self.find_policies(game, algorithm=algorithm, seed=seed, release=release,
                                     game_config=game_config, training_config=training_config)
        if not matches:
            raise PolicyNotFoundError("no policy matches the requested game, algorithm, configuration, seed, and release")
        if len(matches) != 1:
            raise AmbiguousPolicyError(f"{len(matches)} policies match; specify seed, configuration, or release")
        return matches[0].result()

    def _load_reference(self, reference: Reference[PolicyProfile], *, expected_game: EntrySpec) -> PolicyProfile:
        from benchmax.execution.artifacts.policies import load_policy as load_artifact

        with self._stored_artifact(reference) as (record, manifest_path):
            self._validation(record, manifest_path).raise_for_errors()
            return load_artifact(str(manifest_path), expected_game=expected_game,
                                 max_tensor_bytes=self.requirements.max_tensor_bytes, retain_source=False)

    def _validation(self, record: _PolicyRecord, artifact: _ArtifactMetadata | str | Path) -> PolicyValidation:
        errors = metadata_errors(record, self.requirements)
        try:
            reference = os.fspath(artifact) if isinstance(artifact, os.PathLike) else artifact
            path, manifest, _ = validate_policy_artifact(reference, max_tensor_bytes=self.requirements.max_tensor_bytes)
            # Producer/evaluation annotations may have been supplied for a legacy artifact.
            content = {key: value for key, value in manifest.items() if key not in POLICY_ANNOTATION_KEYS}
            recorded = {key: value for key, value in record.manifest.items() if key not in POLICY_ANNOTATION_KEYS}
            if content != recorded:
                raise StorageError("policy manifest changed during validation")
            if (path.parent / manifest["tensor_file"]).stat().st_size != record.tensor.size:
                raise StorageError("policy tensor size does not match its content reference")
        except (OSError, ValueError, TypeError, StorageError) as error:
            errors.append(str(error))
        return PolicyValidation(self.requirements.name, tuple(errors))

    def _materialize_record(self, record: _PolicyRecord, root: Path) -> Path:
        self.materialize_blob(record.tensor, root / record.manifest["tensor_file"])
        return write_document(record.manifest, root / "manifest.json")

    def _reference_record(self, reference: ObjectReference | Reference[PolicyProfile]) -> _PolicyRecord:
        identity = reference.object if isinstance(reference, Reference) else reference
        record = _PolicyRecord.from_document(self.read_document(identity))
        if isinstance(reference, Reference) and record.training != reference.training:
            raise StorageError("stored policy reference training does not match its artifact")

        return record

    @contextlib.contextmanager
    def _stored_artifact(
        self, reference: ObjectReference | Reference[PolicyProfile],
    ) -> Iterator[tuple[_PolicyRecord, Path]]:
        record = self._reference_record(reference)
        with tempfile.TemporaryDirectory(prefix="benchmax-policy-read-") as directory:
            yield record, self._materialize_record(record, Path(directory))


@contextlib.contextmanager
def _policy_source(policy: PolicyProfile | Reference[PolicyProfile] | str | os.PathLike[str]):
    with contextlib.ExitStack() as resources:
        if isinstance(policy, Reference):
            if policy.output != "policy":
                raise TypeError("storage requires a policy output")

            if policy.kind == "stored-policy":
                _, manifest = resources.enter_context(policy._storage()._stored_artifact(policy))
                yield str(manifest)
                return

            if policy.kind != "file":
                directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="benchmax-storage-input-"))
                policy = policy.save(path=Path(directory) / "policy")

            yield policy._artifact_metadata()
        elif isinstance(policy, PolicyProfile):
            training = policy._portable_training
            if not isinstance(training, TrainingSpec):
                raise ArtifactError("policy has no portable training metadata; provide a saved policy artifact")

            directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="benchmax-storage-input-"))
            yield write_policy(policy, training, Path(directory) / "policy")
        elif isinstance(policy, (str, os.PathLike)):
            yield os.fspath(policy)
        else:
            raise TypeError("policy must be a PolicyProfile, policy Reference, or artifact path")


def _training_configuration(training: TrainingSpec) -> dict[str, object]:
    document = training.to_document()
    for key in ("game", "seed"):
        del document[key]
    document["algorithm"] = _entry_configuration(document["algorithm"])
    return document


def _entry_configuration(document: dict[str, object]) -> dict[str, object]:
    # Explicitly repeating a default override does not change the effective configuration.
    field = "resolved" if document["resolved"] is not None else "overrides"
    return {"name": document["name"], field: document[field]}
