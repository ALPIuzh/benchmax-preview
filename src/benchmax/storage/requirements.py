"""Explicit, reusable requirements for publishing and loading policies."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from packaging.utils import canonicalize_name

from benchmax.execution.artifacts.policies import DEFAULT_MAX_POLICY_TENSOR_BYTES
from benchmax.release import LibraryRelease
from benchmax.storage._model import StorageError

if TYPE_CHECKING:
    from benchmax.storage._policy_record import _PolicyRecord


@dataclass(frozen=True, slots=True)
class PolicyRequirements:
    """Metadata requirements. Integrity and safe archive checks are always mandatory.

    Profiles are local configuration, never executable instructions from a store.
    """

    name: str = "basic-v1"
    require_resolved_config: bool = False
    require_explicit_device: bool = False
    require_frame_budget: bool = False
    require_provenance: bool = False
    required_metrics: tuple[str, ...] = ()
    expected_release: LibraryRelease | None = None
    max_tensor_bytes: int = DEFAULT_MAX_POLICY_TENSOR_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("requirements name must be a non-empty string")
        for field in ("require_resolved_config", "require_explicit_device",
                      "require_frame_budget", "require_provenance"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be a bool")
        if (not isinstance(self.required_metrics, tuple)
                or any(not isinstance(name, str) or not name for name in self.required_metrics)
                or len(set(self.required_metrics)) != len(self.required_metrics)):
            raise ValueError("required_metrics must be a tuple of distinct non-empty names")
        if self.expected_release is not None and not isinstance(self.expected_release, LibraryRelease):
            raise TypeError("expected_release must be a LibraryRelease")
        if (not isinstance(self.max_tensor_bytes, int) or isinstance(self.max_tensor_bytes, bool)
                or self.max_tensor_bytes < 1):
            raise ValueError("max_tensor_bytes must be a positive integer")


DEFAULT_REQUIREMENTS = PolicyRequirements()
STRICT_REQUIREMENTS = PolicyRequirements(
    name="strict-v1", require_resolved_config=True, require_provenance=True,
)


class PolicyValidationError(StorageError):
    """A policy does not satisfy its store's validation requirements."""


@dataclass(frozen=True, slots=True)
class PolicyValidation:
    """All detected metadata problems plus any artifact verification failure."""

    requirements: str
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise PolicyValidationError(f"policy fails {self.requirements}: " + "; ".join(self.errors))


def _provenance_errors(provenance: object, release: LibraryRelease) -> list[str]:
    if not isinstance(provenance, Mapping):
        return ["provenance must contain source_revision, python_version, and packages"]
    errors = []
    revision = provenance.get("source_revision")
    if "source_revision" not in provenance or (revision is not None and
            (not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None)):
        errors.append("provenance.source_revision must be a full lowercase source commit or null when unknown")
    python = provenance.get("python_version")
    if not isinstance(python, str) or re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)", python) is None:
        errors.append("provenance.python_version must be an exact Python version")
    packages = provenance.get("packages")
    if (not isinstance(packages, Mapping) or not packages
            or any(not isinstance(name, str) or not name or not isinstance(version, str)
                   or re.fullmatch(r"[0-9][a-zA-Z0-9.!+_-]*", version) is None for name, version in packages.items())):
        errors.append("provenance.packages must map distribution names to exact versions")
        return errors
    normalized = {canonicalize_name(name): version for name, version in packages.items()}
    if len(normalized) != len(packages):
        errors.append("provenance.packages contains duplicate normalized distribution names")
    if canonicalize_name(release.name) not in normalized:
        errors.append("provenance.packages is missing: " + release.name)
    if normalized.get(canonicalize_name(release.name)) != release.version:
        errors.append("provenance library version does not match the producing release")
    return errors


def metadata_errors(record: _PolicyRecord, requirements: PolicyRequirements) -> list[str]:
    training = record.training
    errors = []
    if requirements.require_resolved_config:
        for label, entry in (("game", training.game), ("algorithm", training.algorithm)):
            if not entry.resolved:
                errors.append(f"training.{label} must include its resolved configuration")
    if requirements.require_explicit_device and training.device == "auto":
        errors.append("training.device must be explicitly recorded rather than 'auto'")
    if requirements.require_frame_budget and training.total_frames is None:
        errors.append("training.total_frames must be explicitly recorded")
    if requirements.expected_release is not None and requirements.expected_release != record.release:
        errors.append("producing library name or version differs from the expected release")
    provenance = record.manifest["producer"].get("provenance")
    if requirements.require_provenance or provenance is not None:
        errors.extend(_provenance_errors(provenance, record.release))
    evaluation = record.manifest.get("evaluation")
    if evaluation is not None or requirements.required_metrics:
        if not isinstance(evaluation, Mapping):
            errors.append("evaluation must contain workload_digest and metrics")
        else:
            if evaluation.get("workload_digest") != training.digest:
                errors.append("evaluation.workload_digest must match the producing training workload")
            if evaluation.get("tensor_sha256") != record.tensor.digest:
                errors.append("evaluation.tensor_sha256 must match the policy tensor archive")
            metrics = evaluation.get("metrics")
            if (not isinstance(metrics, Mapping)
                    or any(not isinstance(name, str) or not name or isinstance(value, bool)
                           or not isinstance(value, (int, float)) or not math.isfinite(value)
                           for name, value in metrics.items())):
                errors.append("evaluation.metrics must map names to finite numbers")
            else:
                errors.extend(f"evaluation metric {name!r} is missing" for name in requirements.required_metrics
                              if name not in metrics)
    return errors
