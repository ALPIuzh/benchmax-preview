"""A minimal valid data-only policy artifact; no training dependencies."""

import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmax.execution.artifacts.policies import POLICY_ARTIFACT_SCHEMA_VERSION
from benchmax.execution.specs import TrainingSpec
from benchmax.integrations.benchmarl.artifact import POLICY_FORMAT


def write_policy_artifact(directory: Path, training: TrainingSpec) -> Path:
    directory.mkdir()
    array = np.array([1.0, 2.0], dtype=np.float32)
    np.savez(directory / "tensors.npz", tensor_0=array)
    tensor = (directory / "tensors.npz").read_bytes()
    manifest = {
        "schema_version": POLICY_ARTIFACT_SCHEMA_VERSION,
        "kind": "policy-artifact",
        "format": POLICY_FORMAT,
        "training": training.to_document(),
        "tensor_file": "tensors.npz",
        "tensor_sha256": hashlib.sha256(tensor).hexdigest(),
        "tensors": [{"array": "tensor_0", "player": 0, "state_name": "weight", "shape": [2], "dtype": "<f4"}],
        "state_values": [],
        "policy_interface": {
            "players": [{"player": 0, "observation_space": [2], "action_space": [0, 1]}],
            "state_space": None,
        },
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def forbid_neural_imports(monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    """Fail the test if Torch, TorchRL, or BenchMARL is imported while patched."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchrl", "benchmarl"}:
            pytest.fail(f"{reason} must not import {name}")

        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
