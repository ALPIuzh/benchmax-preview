"""The BenchMAX reference collection and portable storage references."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import copy
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from benchmax.release import LibraryRelease
from benchmax.storage._model import ObjectReference, StorageError
from benchmax.storage.huggingface import HuggingFaceStore, _validate_bucket_location
from benchmax.storage.requirements import DEFAULT_REQUIREMENTS, STRICT_REQUIREMENTS, PolicyRequirements

if TYPE_CHECKING:
    from benchmax.execution.jobs.reference import Reference
    from benchmax.execution.specs import TrainingSpec
    from benchmax.policy import PolicyProfile
    from benchmax.storage.store import Store


def reference_store(
    *, token: str | bool | None = None, cache_dir: str | os.PathLike[str] | None = None,
) -> HuggingFaceStore:
    """Build the configured reference store without network access.

    BENCHMAX_REFERENCE_BUCKET overrides the shipped configuration. Validation uses exactly the same requirements
    and policy methods available to user-owned stores.
    """

    with files("benchmax.conf").joinpath("storage/reference.yaml").open("r", encoding="utf-8") as stream:
        config = OmegaConf.to_container(OmegaConf.load(stream), resolve=True)
    if not isinstance(config, dict) or set(config) != {"bucket_id", "requirements"}:
        raise StorageError("invalid reference store configuration")
    profiles = {profile.name: profile for profile in (DEFAULT_REQUIREMENTS, STRICT_REQUIREMENTS)}
    name = config["requirements"]
    if not isinstance(name, str) or name not in profiles:
        raise StorageError(f"unknown reference policy requirements {name!r}")
    selected_bucket = os.environ.get("BENCHMAX_REFERENCE_BUCKET", config["bucket_id"])
    if selected_bucket is None:
        raise StorageError("reference bucket is not configured; set BENCHMAX_REFERENCE_BUCKET")
    return HuggingFaceStore(selected_bucket,
                           token=token, cache_dir=cache_dir, requirements=profiles[name])

def _store_document(store: Store) -> dict[str, Any]:
    """Portable store location and validation settings, excluding credentials and caches."""
    from benchmax.storage.local import LocalStore

    requirements = asdict(store.requirements)
    if store.requirements.expected_release is not None:
        requirements["expected_release"] = store.requirements.expected_release.to_document()

    if isinstance(store, LocalStore):
        return {"kind": "local", "root": str(store.root), "requirements": requirements}

    if isinstance(store, HuggingFaceStore):
        return {"kind": "huggingface", "bucket_id": store.bucket_id, "prefix": store.prefix,
                "requirements": requirements}
    raise TypeError("this store does not support portable references")


def _validate_store_document(document: object) -> PolicyRequirements:
    if not isinstance(document, Mapping):
        raise StorageError("stored reference requires a store configuration")

    fields = {"kind", "requirements"}
    kind = document.get("kind")
    fields |= {"root"} if kind == "local" else {"bucket_id", "prefix"}
    if not isinstance(kind, str) or kind not in {"local", "huggingface"} or set(document) != fields:
        raise StorageError("unsupported store configuration")

    try:
        if not isinstance(document["requirements"], Mapping):
            raise TypeError("requirements must be an object")

        settings = dict(document["requirements"])
        metrics = settings.get("required_metrics", ())
        if not isinstance(metrics, (list, tuple)):
            raise TypeError("required_metrics must be a list of names")

        settings["required_metrics"] = tuple(metrics)
        if settings.get("expected_release") is not None:
            settings["expected_release"] = LibraryRelease.from_document(settings["expected_release"])
        requirements = PolicyRequirements(**settings)
    except (TypeError, ValueError) as error:
        raise StorageError(f"invalid stored reference requirements: {error}") from error

    if kind == "local":
        if not isinstance(document["root"], str) or not Path(document["root"]).is_absolute():
            raise StorageError("local store references require an absolute path")
    else:
        _validate_bucket_location(document["bucket_id"], document["prefix"])

    return requirements


def _store_from_document(document: Mapping[str, Any]) -> Store:
    from benchmax.storage.local import LocalStore

    requirements = _validate_store_document(document)
    if document["kind"] == "local":
        return LocalStore(document["root"], requirements=requirements)
    return HuggingFaceStore(document["bucket_id"], prefix=document["prefix"], requirements=requirements)


def _policy_reference(store: Store, identity: ObjectReference, training: TrainingSpec) -> Reference[PolicyProfile]:
    from benchmax.execution.jobs.reference import Reference

    reference = Reference({"schema_version": 1, "kind": "stored-policy", "output": "policy",
                           "store": _store_document(store), "digest": identity.digest,
                           "training": training.to_document()})
    # Snapshot runtime settings without serializing credentials or constructing another backend.
    object.__setattr__(reference, "_store", copy(store))
    return reference
