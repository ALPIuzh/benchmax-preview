"""Versioned descriptions of work BenchMAX can schedule."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

from benchmax.config import ConfigValue
from benchmax.seeding import SEED_SCHEME, derive_seed, resolve_seed, validate_seed, validate_seed_scheme
from benchmax.utils.hashing import _digest_bytes, _is_sha256, _sha256_file
from benchmax.utils.json import canonical_json

_SCHEMA_VERSION = 1
_REFERENCE_SOURCE_FIELDS = {
    "job-output": {"job_id", "target", "workload_digest"},
    "file": {"uri", "digest"},
    "stored-policy": {"store", "digest", "training"},
}


class SpecError(ValueError):
    """A workload document is malformed or has the wrong kind."""


def _structural_json(value: Mapping[str, ConfigValue], *, label: str) -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError) as error:
        raise SpecError(f"{label} must contain only finite JSON values") from error


def _reject_loader_controls(value: Any, *, path: str = "overrides") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.startswith("_"):
                raise SpecError(
                    f"{path} contains reserved configuration key {key!r}; "
                    "workloads may select registered entries but may not supply "
                    "Hydra loader controls"
                )

            _reject_loader_controls(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_loader_controls(item, path=f"{path}[{index}]")


def _required_fields(document: Mapping[str, Any], *, kind: str, fields: set[str]) -> None:
    unknown = sorted(set(document) - fields)
    if unknown:
        raise SpecError(f"unknown {kind} fields: {', '.join(unknown)}")

    missing = sorted(fields - set(document))
    if missing:
        raise SpecError(f"missing {kind} fields: {', '.join(missing)}")


def _canonical(document: Mapping[str, Any]) -> bytes:
    return _structural_json(document, label="a workload").encode("utf-8")


def _resolve_entry(
    group: Literal["game", "algorithm"], name: str, overrides: Mapping[str, ConfigValue]
) -> dict[str, ConfigValue]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)

        if group == "game":
            from benchmax.registry import _GAMES

            return _GAMES.resolve(name, **overrides).configuration

        from benchmax.registry import _ALGORITHMS

        return _ALGORITHMS.resolve(name, **overrides).configuration


@dataclass(frozen=True, slots=True, init=False)
class EntrySpec:
    """A registered configuration entry and explicit caller overrides."""

    name: str
    _overrides_json: str = field(repr=False)
    _resolved_json: str | None = field(repr=False)

    def __init__(
        self, name: str, overrides: Mapping[str, ConfigValue] | None = None,
        resolved: Mapping[str, ConfigValue] | None = None
    ) -> None:
        if not isinstance(name, str) or not name:
            raise SpecError("entry name must be a non-empty string")

        object.__setattr__(self, "name", name)
        settings = _structural_json(overrides if overrides is not None else {}, label=f"overrides for {name!r}")
        _reject_loader_controls(json.loads(settings))
        object.__setattr__(self, "_overrides_json", settings)
        snapshot = None if resolved is None else _structural_json(resolved, label=f"resolved configuration for {name!r}")
        object.__setattr__(self, "_resolved_json", snapshot)

    @property
    def overrides(self) -> dict[str, ConfigValue]:
        """Return a detached copy of caller overrides."""

        value = json.loads(self._overrides_json)
        assert isinstance(value, dict)
        return value

    @property
    def resolved(self) -> dict[str, ConfigValue] | None:
        """Return a detached copy of the resolved compatibility snapshot."""

        if self._resolved_json is None:
            return None

        value = json.loads(self._resolved_json)
        assert isinstance(value, dict)
        return value

    @classmethod
    def create(
        cls, group: Literal["game", "algorithm"], name: str, overrides: Mapping[str, ConfigValue] | None = None
    ) -> EntrySpec:
        settings = dict(overrides or {})

        try:
            resolved = _resolve_entry(group, name, settings)
        except ImportError:
            # Audit mode must remain able to describe a missing optional
            # provider. An allocated attempt rejects unresolved entries.
            resolved = None

        return cls(name, settings, resolved)

    def to_document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "overrides": self.overrides,
            "resolved": self.resolved,
        }

    @classmethod
    def from_document(cls, document: Any, *, label: str) -> EntrySpec:
        if not isinstance(document, Mapping):
            raise SpecError(f"{label} must be an object")

        _required_fields(document, kind=label, fields={"name", "overrides", "resolved"})
        overrides = document["overrides"]
        if not isinstance(overrides, Mapping):
            raise SpecError(f"{label}.overrides must be an object")

        resolved = document["resolved"]
        if resolved is not None and not isinstance(resolved, Mapping):
            raise SpecError(f"{label}.resolved must be an object or null")

        return cls(
            name=document["name"], overrides=dict(overrides), resolved=dict(resolved) if resolved is not None else None
        )

    def validate_resolution(self, group: Literal["game", "algorithm"]) -> None:
        """Reject unresolved entries and local configuration drift."""

        if self.resolved is None:
            raise SpecError(
                f"{group} {self.name!r} was not fully resolved when the workload "
                "was created; install its provider and recreate or audit the spec"
            )

        try:
            current = _resolve_entry(group, self.name, self.overrides)
        except ImportError as error:
            raise SpecError(str(error)) from error

        if current != self.resolved:
            raise SpecError(
                f"resolved {group} {self.name!r} differs from the workload; "
                "the configuration or provider environment has drifted"
            )


def _training_game_entry(game: EntrySpec) -> EntrySpec:
    """Remove evaluation composition from a training game description."""

    overrides = game.overrides
    overrides["evaluator"] = None
    resolved = game.resolved
    if resolved is not None:
        resolved["evaluator"] = None

    return EntrySpec(game.name, overrides, resolved)


@dataclass(frozen=True, slots=True)
class _ArtifactMetadata:
    """Internal location/checksum metadata stored in workloads and training results."""

    uri: str
    digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise SpecError("policy URI must be a non-empty string")

        if self.digest is not None and not _is_sha256(self.digest):
            raise SpecError("policy digest must be a lowercase SHA-256 hex digest")

    def to_document(self) -> dict[str, Any]:
        return {"uri": self.uri, "digest": self.digest}

    def content_identity(self) -> dict[str, str]:
        """Return the location-independent identity used by a workload."""

        if self.digest is None:
            raise SpecError("a policy without a digest has no content identity")

        return {"digest": self.digest}

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> _ArtifactMetadata:
        """Pin one local policy or training-result document by content."""

        source = Path(path).expanduser().resolve()

        try:
            digest = _sha256_file(source)
        except OSError as error:
            raise SpecError(
                f"cannot pin policy document {source}: {error}; a scheduled "
                "evaluation requires an existing content-addressed artifact"
            ) from error

        return cls(str(source), digest)

    @classmethod
    def from_document(cls, document: Any) -> _ArtifactMetadata:
        if not isinstance(document, Mapping):
            raise SpecError("policy must be an object")

        _required_fields(document, kind="policy", fields={"uri", "digest"})
        return cls(uri=document["uri"], digest=document["digest"])


@dataclass(frozen=True, init=False)
class _ReferenceSpec:
    """Immutable reference data stored in portable workloads, without resolution or I/O."""

    _document_json: str = field(repr=False)

    def __init__(self, document: Mapping[str, Any]) -> None:
        if not isinstance(document, Mapping):
            raise SpecError("reference must be an object")

        kind = document.get("kind")
        if not isinstance(kind, str) or kind not in _REFERENCE_SOURCE_FIELDS:
            supported = ", ".join(repr(value) for value in _REFERENCE_SOURCE_FIELDS)
            raise SpecError(f"reference kind must be one of: {supported}")

        fields = {"schema_version", "kind", "output"} | _REFERENCE_SOURCE_FIELDS[kind]
        _required_fields(document, kind="reference", fields=fields)
        _check_header(document, expected_kind=kind)
        output = document["output"]
        if not isinstance(output, str) or output not in {"policy", "metrics"}:
            raise SpecError("reference output must be 'policy' or 'metrics'")

        if kind == "job-output":
            job_id, target, digest = document["job_id"], document["target"], document["workload_digest"]
            if not isinstance(job_id, str) or not job_id or any(c in job_id for c in "\x00\n\r"):
                raise SpecError("job reference requires a non-empty job ID")

            if not _is_sha256(digest):
                raise SpecError("job reference requires a SHA-256 workload digest")

            if (not isinstance(target, Mapping) or not isinstance(target.get("kind"), str)
                    or target["kind"] not in {"local", "slurm", "adapter"}):
                raise SpecError("job reference requires a target configuration")
        elif kind == "stored-policy":
            from benchmax.storage.reference import _validate_store_document

            if output != "policy":
                raise SpecError("stored references require a policy output")

            if not _is_sha256(document["digest"]):
                raise SpecError("stored reference requires a lowercase SHA-256 digest")

            TrainingSpec.from_document(document["training"])
            _validate_store_document(document["store"])
        else:
            metadata = _ArtifactMetadata(document["uri"], document["digest"])
            if metadata.digest is None or not Path(metadata.uri).is_absolute() or "\x00" in metadata.uri:
                raise SpecError("file reference requires an absolute local path and a SHA-256 digest")

        object.__setattr__(self, "_document_json", _canonical(document).decode("utf-8"))

    @property
    def kind(self) -> Literal["job-output", "file", "stored-policy"]:
        return self.to_document()["kind"]

    @property
    def output(self) -> Literal["policy", "metrics"]:
        return self.to_document()["output"]

    def _source(self, kind: str) -> dict[str, Any]:
        if self.kind != kind:
            raise SpecError(f"{self.kind} reference has no {kind} source")

        return self.to_document()

    @property
    def job_id(self) -> str:
        return self._source("job-output")["job_id"]

    @property
    def workload_digest(self) -> str:
        return self._source("job-output")["workload_digest"]

    @property
    def target(self) -> dict[str, Any]:
        """Return a detached, inspectable target configuration."""
        return self._source("job-output")["target"]

    @property
    def path(self) -> Path:
        """The local source file, available only for file references."""
        return Path(self._source("file")["uri"])

    def _artifact_metadata(self) -> _ArtifactMetadata:
        source = self._source("file")
        return _ArtifactMetadata(source["uri"], source["digest"])

    def to_document(self) -> dict[str, Any]:
        return json.loads(self._document_json)

    def content_identity(self) -> dict[str, Any]:
        if self.kind == "file":
            return self._artifact_metadata().content_identity()

        if self.kind == "stored-policy":
            return {"stored_policy": self.to_document()["digest"]}

        return self.to_document()

    @classmethod
    def _from_file(cls, metadata: _ArtifactMetadata, output: Literal["policy", "metrics"]) -> Self:
        return cls({"schema_version": _SCHEMA_VERSION, "kind": "file", "output": output, **metadata.to_document()})

    @classmethod
    def from_document(cls, document: Any) -> Self:
        return cls(document)


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    """One reproducible neural self-play training workload."""

    game: EntrySpec
    algorithm: EntrySpec = field(default_factory=lambda: EntrySpec("mappo"))
    seed: int = 0
    seed_scheme: str = SEED_SCHEME
    device: str = "auto"
    total_frames: int | None = None

    kind: Literal["training"] = field(default="training", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.game, EntrySpec):
            raise SpecError("training game must be an EntrySpec")

        object.__setattr__(self, "game", _training_game_entry(self.game))

        if not isinstance(self.algorithm, EntrySpec):
            raise SpecError("training algorithm must be an EntrySpec")

        if "total_frames" in self.algorithm.overrides:
            raise SpecError("total_frames is a training workload field, not an algorithm override")

        try:
            validate_seed(self.seed, label="training seed")
            validate_seed_scheme(self.seed_scheme)
        except (TypeError, ValueError) as error:
            raise SpecError(str(error)) from error

        if not isinstance(self.device, str) or not self.device:
            raise SpecError("training device must be a non-empty string")

        if self.total_frames is not None and (
            not isinstance(self.total_frames, int) or isinstance(self.total_frames, bool) or self.total_frames <= 0
        ):
            raise SpecError("total_frames must be a positive integer or null")

    @classmethod
    def create(
        cls, *, game: str | EntrySpec, algorithm: str | EntrySpec = "mappo", seed: int | None = None,
        device: str = "auto", total_frames: int | None = None, game_overrides: Mapping[str, ConfigValue] | None = None,
        algorithm_overrides: Mapping[str, ConfigValue] | None = None
    ) -> TrainingSpec:
        if isinstance(game, str):
            game = EntrySpec.create("game", game, game_overrides)
        elif game_overrides:
            raise SpecError("game_overrides cannot accompany an EntrySpec")

        if isinstance(algorithm, str):
            algorithm = EntrySpec.create("algorithm", algorithm, algorithm_overrides)
        elif algorithm_overrides:
            raise SpecError("algorithm_overrides cannot accompany an EntrySpec")

        return cls(
            game=game, algorithm=algorithm, seed=resolve_seed(seed, purpose="training workload"), device=device,
            total_frames=total_frames
        )

    @property
    def digest(self) -> str:
        return _digest_bytes(_canonical(self.to_document()))

    @property
    def output_kind(self) -> Literal["policy"]:
        return "policy"

    @property
    def python_hash_seed(self) -> int:
        """PYTHONHASHSEED an attempt must start with, set before Python launches."""
        return derive_seed(self.seed, "python-hash", scheme=self.seed_scheme)

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": self.kind,
            "game": self.game.to_document(),
            "algorithm": self.algorithm.to_document(),
            "seed": self.seed,
            "seed_scheme": self.seed_scheme,
            "device": self.device,
            "total_frames": self.total_frames,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> TrainingSpec:
        fields = {
            "schema_version",
            "kind",
            "game",
            "algorithm",
            "seed",
            "seed_scheme",
            "device",
            "total_frames",
        }
        _required_fields(document, kind="training", fields=fields)
        _check_header(document, expected_kind="training")
        return cls(
            game=EntrySpec.from_document(document["game"], label="game"),
            algorithm=EntrySpec.from_document(document["algorithm"], label="algorithm"), seed=document["seed"],
            seed_scheme=document["seed_scheme"], device=document["device"], total_frames=document["total_frames"]
        )


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """One evaluation of an artifact-backed policy on a registered game."""

    game: EntrySpec
    policy: _ArtifactMetadata | _ReferenceSpec

    kind: Literal["evaluation"] = field(default="evaluation", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.game, EntrySpec):
            raise SpecError("evaluation game must be an EntrySpec")

        if not isinstance(self.policy, (_ArtifactMetadata, _ReferenceSpec)):
            raise SpecError("evaluation requires artifact metadata or a policy job-output reference")

        if isinstance(self.policy, _ReferenceSpec):
            if self.policy.output != "policy":
                raise SpecError("evaluation requires a policy output")

            if self.policy.kind == "file":
                object.__setattr__(self, "policy", self.policy._artifact_metadata())
            else:
                object.__setattr__(self, "policy", _ReferenceSpec(self.policy.to_document()))
        elif self.policy.digest is None:
            raise SpecError("evaluation policy references must include a SHA-256 digest")

    @classmethod
    def create(
        cls, *, game: str | EntrySpec, policy: str | _ArtifactMetadata | _ReferenceSpec,
        game_overrides: Mapping[str, ConfigValue] | None = None
    ) -> EvaluationSpec:
        if isinstance(game, str):
            game = EntrySpec.create("game", game, game_overrides)
        elif game_overrides:
            raise SpecError("game_overrides cannot accompany an EntrySpec")

        if isinstance(policy, str):
            policy = _ArtifactMetadata.from_path(policy)

        return cls(game=game, policy=policy)

    @property
    def digest(self) -> str:
        document = self.to_document()
        document["policy"] = self.policy.content_identity()
        return _digest_bytes(_canonical(document))

    @property
    def output_kind(self) -> Literal["metrics"]:
        return "metrics"

    @property
    def python_hash_seed(self) -> int:
        return 0

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": self.kind,
            "game": self.game.to_document(),
            "policy": self.policy.to_document(),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> EvaluationSpec:
        fields = {"schema_version", "kind", "game", "policy"}
        _required_fields(document, kind="evaluation", fields=fields)
        _check_header(document, expected_kind="evaluation")
        return cls(
            game=EntrySpec.from_document(document["game"], label="game"),
            policy=(_ReferenceSpec.from_document(document["policy"])
                    if isinstance(document["policy"], Mapping) and "kind" in document["policy"]
                    else _ArtifactMetadata.from_document(document["policy"])),
        )


WorkloadSpec: TypeAlias = TrainingSpec | EvaluationSpec


def _check_header(document: Mapping[str, Any], *, expected_kind: str) -> None:
    version = document.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise SpecError(f"unsupported schema_version {version!r}; expected {_SCHEMA_VERSION}")

    if document.get("kind") != expected_kind:
        raise SpecError(f"document kind is {document.get('kind')!r}; expected {expected_kind!r}")


def workload_from_document(document: Any) -> WorkloadSpec:
    """Validate and unwrap one workload or audited-workload document."""

    if not isinstance(document, Mapping):
        raise SpecError("workload document must be a JSON object")

    kind = document.get("kind")

    if kind == "audit":
        fields = {"schema_version", "kind", "workload", "evidence"}
        _required_fields(document, kind="audit", fields=fields)
        _check_header(document, expected_kind="audit")

        if not isinstance(document["evidence"], Mapping):
            raise SpecError("audit evidence must be an object")

        return workload_from_document(document["workload"])

    if kind == "training":
        return TrainingSpec.from_document(document)

    if kind == "evaluation":
        return EvaluationSpec.from_document(document)

    raise SpecError(f"unknown workload kind {kind!r}")


def validate_workload_resolution(workload: WorkloadSpec) -> None:
    """Check that this installed registry still means what the spec recorded."""

    workload.game.validate_resolution("game")

    if isinstance(workload, TrainingSpec):
        workload.algorithm.validate_resolution("algorithm")


def load_workload(
    path: str | os.PathLike[str], *, expected_kind: Literal["training", "evaluation"] | None = None
) -> WorkloadSpec:
    """Load a workload, transparently accepting a prior audit envelope."""

    source = Path(path)

    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise SpecError(f"cannot read workload {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise SpecError(f"invalid JSON in workload {source}: {error}") from error

    workload = workload_from_document(document)
    if expected_kind is not None and workload.kind != expected_kind:
        raise SpecError(f"{source} contains an {workload.kind} workload; expected {expected_kind}")

    return workload
