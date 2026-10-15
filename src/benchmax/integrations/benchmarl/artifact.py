"""Safe BenchMARL policy artifacts for cross-process evaluation."""

from __future__ import annotations

import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmax.execution.artifacts.arrays import (  # isort: skip
    _inspect_tensor_archive, _read_tensor_array, _TensorEntry, _write_array_policy)
from benchmax.execution.artifacts.interface import _PolicyInterface

from benchmax.execution.artifacts.policies import (  # isort: skip
    DEFAULT_MAX_POLICY_TENSOR_BYTES, POLICY_ARTIFACT_SCHEMA_VERSION, ArtifactError)
from benchmax.execution.specs import EntrySpec, SpecError, TrainingSpec, _ArtifactMetadata
from benchmax.policy import PolicyProfile

POLICY_FORMAT = "benchmax.neural-policy.npz"
REQUIRES_TRAINING = True


def supports(profile: PolicyProfile) -> bool:
    # A real adapter already has its runtime loaded; other profiles need no neural imports.
    if not hasattr(profile._batched_policy_profile, "_state_dicts"):
        return False

    from benchmax.integrations.benchmarl.env import _BenchMARLPolicyProfile

    return isinstance(profile._batched_policy_profile, _BenchMARLPolicyProfile)


def write_policy(profile: PolicyProfile, training: TrainingSpec, destination: str | Path) -> _ArtifactMetadata:
    """Extract neural state and publish it through the shared array writer."""
    if not supports(profile):
        raise ArtifactError("policy has no supported neural representation")

    import torch

    implementation = profile._batched_policy_profile
    state_dicts = implementation._state_dicts()
    arrays, structural = {}, []
    for player in sorted(state_dicts):
        for name, value in state_dicts[player].items():
            if isinstance(value, torch.Size) or value is None:
                structural.append({"player": player, "state_name": name,
                                   "type": "none" if value is None else "torch.Size",
                                   "value": None if value is None else list(value)})
                continue

            try:
                arrays[player, name] = value.detach().cpu().contiguous().numpy()
            except (AttributeError, TypeError) as error:
                raise ArtifactError(f"policy tensor {name!r} has no portable NumPy dtype") from error

            if arrays[player, name].dtype.hasobject:
                raise ArtifactError(f"policy tensor {name!r} has object dtype")

    interface = _PolicyInterface.from_document(implementation._interface.to_document())
    return _write_array_policy(arrays, training, destination, interface, POLICY_FORMAT,
                               annotations=profile._artifact_annotations, state_values=structural)


def _restore_structural_states(entries: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(entries, list):
        raise ArtifactError("policy manifest structural state index must be a list")

    states: dict[int, dict[str, Any]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            raise ArtifactError("policy structural state entry must be an object")

        player = entry.get("player")
        state_name = entry.get("state_name")
        value_type = entry.get("type")
        value = entry.get("value")

        if (not isinstance(player, int) or isinstance(player, bool)
                or not isinstance(state_name, str) or not state_name):
            raise ArtifactError("policy structural state entry is malformed")

        if (value_type == "torch.Size" and isinstance(value, list)
                and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value)):
            restored: Any = tuple(value)
        elif value_type == "none" and value is None:
            restored = None
        else:
            raise ArtifactError(f"unsupported structural state type {value_type!r}")

        player_state = states.setdefault(player, {})
        if state_name in player_state:
            raise ArtifactError(f"duplicate policy state entry {state_name!r}")

        player_state[state_name] = restored

    return states


def _inspect_neural_archive(
    path: Path, manifest: Mapping[str, Any], *, max_tensor_bytes: int
) -> tuple[Path, list[_TensorEntry], dict[int, dict[str, Any]]]:
    tensor_path, entries = _inspect_tensor_archive(path, manifest, max_tensor_bytes=max_tensor_bytes)
    states = _restore_structural_states(manifest.get("state_values"))
    for entry in entries:
        if entry.state_name in states.get(entry.player, {}):
            raise ArtifactError(f"duplicate policy state entry {entry.state_name!r}")

    return tensor_path, entries, states


def validate_policy_document(
    manifest_path: Path, manifest: Mapping[str, Any], *, max_tensor_bytes: int = DEFAULT_MAX_POLICY_TENSOR_BYTES,
) -> None:
    """Verify the neural artifact's archive and interface without loading tensor arrays."""

    interface = _PolicyInterface.from_document(manifest.get("policy_interface"))
    _, entries, states = _inspect_neural_archive(manifest_path, manifest, max_tensor_bytes=max_tensor_bytes)
    players = set(states) | {entry.player for entry in entries}
    if players != {player for player, _, _ in interface.players}:
        raise ArtifactError("policy tensor players do not match its interface")


def _load_tensor_states(
    manifest_path: Path, manifest: Mapping[str, Any], *, max_tensor_bytes: int,
) -> dict[int, dict[str, Any]]:
    tensor_path, entries, states = _inspect_neural_archive(manifest_path, manifest, max_tensor_bytes=max_tensor_bytes)
    import torch

    for state in states.values():
        for name, value in state.items():
            if isinstance(value, tuple):
                state[name] = torch.Size(value)

    try:
        with zipfile.ZipFile(tensor_path) as archive:
            for entry in entries:
                array = _read_tensor_array(archive, entry)
                states.setdefault(entry.player, {})[entry.state_name] = torch.from_numpy(array)
    except ArtifactError:
        raise
    except (OSError, EOFError, MemoryError, TypeError, ValueError, KeyError,
            zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ArtifactError(f"cannot load policy tensor archive: {error}") from error

    return states


def load_policy_document(
    manifest_path: Path, manifest: Mapping[str, Any], training: TrainingSpec, *, expected_game: EntrySpec,
    max_tensor_bytes: int = DEFAULT_MAX_POLICY_TENSOR_BYTES
) -> PolicyProfile:
    """Load one already-verified common policy-artifact envelope."""

    if (manifest.get("schema_version") != POLICY_ARTIFACT_SCHEMA_VERSION
            or manifest.get("kind") != "policy-artifact" or manifest.get("format") != POLICY_FORMAT):
        raise ArtifactError(f"unsupported policy artifact in {manifest_path}")

    try:
        training.algorithm.validate_resolution("algorithm")
    except SpecError as error:
        raise ArtifactError(f"policy workload is incompatible: {error}") from error

    from benchmax.integrations.benchmarl.env import _BenchMARLPolicyProfile, _group_name
    from benchmax.integrations.benchmarl.training import _create_experiment, load_benchmarl_training_function
    from benchmax.registry import load_game

    training_function = load_benchmarl_training_function(training)
    game_overrides = expected_game.overrides
    game_overrides["evaluator"] = None
    game = load_game(expected_game.name, **game_overrides)
    stored_interface = _PolicyInterface.from_document(manifest.get("policy_interface"))
    if stored_interface != _PolicyInterface.from_game(game):
        raise ArtifactError(f"policy interface is incompatible with game {expected_game.name!r}")

    states = _load_tensor_states(manifest_path, manifest, max_tensor_bytes=max_tensor_bytes)

    with tempfile.TemporaryDirectory(prefix="benchmax-policy-load-") as output_dir:
        experiment = _create_experiment(
            game, training_function, seed=training.seed, device="cpu", output_dir=output_dir
        )

        try:
            if set(states) != set(game.players):
                raise ArtifactError("policy artifact players do not match the registered game")

            for player in game.players:
                experiment.group_policies[_group_name(player)].load_state_dict(states[player], strict=True)

            profile = PolicyProfile(_BenchMARLPolicyProfile(experiment, game))
        finally:
            experiment.close()

    return profile
