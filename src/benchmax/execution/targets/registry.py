"""Process-local target registration and optional user-owned JSON profiles."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from benchmax.execution.targets.base import AdapterTarget, LocalTarget, Target
from benchmax.execution.targets.service import JobError
from benchmax.utils.files import _exclusive_file_lock
from benchmax.utils.json import write_document

_REGISTERED: dict[str, Target] = {}
_LOCK = threading.RLock()
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")


def _profile_path() -> Path:
    configured = os.environ.get("BENCHMAX_TARGETS")
    if configured:
        return Path(configured).expanduser()

    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "benchmax" / "targets.json"


def _read_profiles(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise JobError(f"cannot read target profiles {path}: {error}") from error

    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise JobError(f"invalid target profiles in {path}")

    profiles = document.get("targets")
    if not isinstance(profiles, dict) or any(not isinstance(k, str) or not _NAME.fullmatch(k) for k in profiles):
        raise JobError(f"invalid target names in {path}")

    if "local" in profiles:
        raise JobError("the built-in target name 'local' is reserved")

    return profiles


def target_from_document(document: dict[str, Any]) -> Target:
    from benchmax.execution.targets.slurm.target import SlurmTarget

    if not isinstance(document, dict):
        raise JobError("target profile must be an object")

    fields = dict(document)
    kind = fields.pop("kind", None)
    constructors = {"local": LocalTarget, "adapter": AdapterTarget, "slurm": SlurmTarget}

    try:
        constructor = constructors[kind]
        return constructor(**fields)
    except (KeyError, TypeError, ValueError) as error:
        raise JobError(f"invalid {kind!r} target profile: {error}") from error


def dependency_target(document: dict[str, Any], consumer: Target) -> Target:
    """Require a shared job store before a target resolves a producer's files."""
    producer = target_from_document(document)
    if not consumer._shares_job_store(producer):
        raise JobError("job-output dependencies require the same target endpoint and run store; transfer explicitly")
    return producer


def target_label(document: dict[str, Any]) -> str:
    """Describe captured target settings without constructing or contacting a target."""
    kind = document["kind"]
    if kind == "local":
        return "Local"

    if kind == "slurm":
        host, user, port = document.get("host"), document.get("user"), document.get("port", 22)
        address = f"{user}@{host}" if user else str(host)

        if port != 22:
            address += f":{port}"

        return f"Slurm · {address}" if host else "Slurm · direct"

    return f"Adapter · {document.get('uri', '(not configured)')}"


def register_target(name: str, target: Target, *, persist: bool = False, replace: bool = False) -> Target:
    """Register a target under a name. Saved profiles contain configuration only.

    ``persist=True`` writes to ``$BENCHMAX_TARGETS`` or
    ``$XDG_CONFIG_HOME/benchmax/targets.json`` (default ``~/.config``).
    Existing names require ``replace=True``.
    """

    if not isinstance(name, str) or not _NAME.fullmatch(name) or name == "local":
        raise ValueError("target names use letters, digits, '.', '_' or '-' and start with a letter; local is reserved")

    if not isinstance(target, Target):
        raise TypeError("target must be a Target")

    with _LOCK:
        path = _profile_path()
        profiles = _read_profiles(path)
        if not replace and (name in _REGISTERED or name in profiles):
            raise ValueError(f"target {name!r} already exists; use replace=True")

        if persist:
            configuration = target.to_document()
            path.parent.mkdir(parents=True, exist_ok=True)

            with _exclusive_file_lock(path.with_suffix(path.suffix + ".lock")):
                profiles = _read_profiles(path)
                if not replace and name in profiles:
                    raise ValueError(f"target {name!r} already exists; use replace=True")

                profiles[name] = configuration
                write_document({"schema_version": 1, "targets": profiles}, path)

        _REGISTERED[name] = target

    return target


def load_target(name: str | Target) -> Target:
    """Resolve a target object, registered name, saved profile, or adapter URI."""

    if isinstance(name, Target):
        return name

    if not isinstance(name, str) or not name:
        raise JobError("target must be a Target or a non-empty name or URI")

    if name == "local":
        return LocalTarget()

    with _LOCK:
        if name in _REGISTERED:
            return _REGISTERED[name]

        profiles = _read_profiles(_profile_path())
        if name in profiles:
            return target_from_document(profiles[name])

    if ":" in name:
        return AdapterTarget(name)

    raise JobError(
        f"unknown target {name!r}; configure a LocalTarget or SlurmTarget and save it with "
        f"target.register({name!r}, persist=True). Profiles: {_profile_path()}"
    )


def list_available_targets() -> list[str]:
    """List target names without opening connections or importing adapters."""

    with _LOCK:
        return sorted({"local", *_REGISTERED, *_read_profiles(_profile_path())})
