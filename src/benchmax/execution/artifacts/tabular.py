"""Finite observation/action tables stored as validated, non-pickle arrays."""

from __future__ import annotations

import math
import zipfile

import numpy as np

from benchmax.execution.artifacts.arrays import (  # isort: skip
    _inspect_tensor_archive, _read_tensor_array, _write_array_policy)
from benchmax.execution.artifacts.interface import _load_policy_game, _PolicyInterface
from benchmax.execution.artifacts.policies import DEFAULT_MAX_POLICY_TENSOR_BYTES, ArtifactError
from benchmax.execution.specs import _ArtifactMetadata, _training_game_entry
from benchmax.policy import PolicyProfile, _TabularPolicyProfile, _validate_batched_distribution

POLICY_FORMAT = "benchmax.tabular-policy.npz"
REQUIRES_TRAINING = False


def supports(profile: PolicyProfile) -> bool:
    return isinstance(profile, _TabularPolicyProfile)


def write_policy(profile: _TabularPolicyProfile, source, destination) -> _ArtifactMetadata:
    game = _load_policy_game(source.game)
    interface = _PolicyInterface.from_game(game)
    arrays = {}
    unknown = {key[0] for key in profile._table} - set(game.players)
    if unknown:
        raise ArtifactError(f"policy table contains unknown players: {sorted(unknown)}")

    table = sorted(profile._table.items())
    for player, shape, actions in interface.players:
        rows = [(key, values) for key, values in table if key[0] == player]
        observations = np.zeros((len(rows), math.prod(shape)), dtype="<f8")
        masks = np.zeros((len(rows), len(actions)), dtype=bool)
        probabilities = np.zeros((len(rows), len(actions)), dtype="<f8")
        columns = {action: index for index, action in enumerate(actions)}
        for row, ((_, obs, legal), values) in enumerate(rows):
            if len(obs) != observations.shape[1] or any(action not in columns for action in legal):
                raise ArtifactError(f"policy table is incompatible with the interface of player {player}")

            observations[row] = obs
            selected = [columns[action] for action in legal]
            masks[row, selected] = True
            probabilities[row, selected] = values

        for name, array in (("observations", observations), ("legal_masks", masks), ("probabilities", probabilities)):
            arrays[player, name] = array

    return _write_array_policy(arrays, source, destination, interface, POLICY_FORMAT,
                               annotations=profile._artifact_annotations, fallback=profile._fallback)


def _inspect(path, manifest, max_tensor_bytes):
    interface = _PolicyInterface.from_document(manifest.get("policy_interface"))
    if manifest.get("fallback") not in ("error", "uniform"):
        raise ArtifactError("invalid tabular policy fallback")

    tensor_path, entries = _inspect_tensor_archive(path, manifest, max_tensor_bytes=max_tensor_bytes)
    if manifest.get("state_values") != []:
        raise ArtifactError("tabular policies may not contain structural neural state")

    indexed = {(entry.player, entry.state_name): entry for entry in entries}
    expected = {(player, name) for player, _, _ in interface.players
                for name in ("observations", "legal_masks", "probabilities")}
    if set(indexed) != expected:
        raise ArtifactError("tabular policy arrays do not match its interface")

    for player, shape, actions in interface.players:
        observations = indexed[player, "observations"]
        masks = indexed[player, "legal_masks"]
        probabilities = indexed[player, "probabilities"]
        if (len(observations.shape) != 2 or observations.shape[1] != math.prod(shape)
                or masks.shape != (observations.shape[0], len(actions)) or probabilities.shape != masks.shape
                or observations.dtype != np.dtype("<f8") or probabilities.dtype != np.dtype("<f8")
                or masks.dtype != np.dtype(bool)):
            raise ArtifactError("tabular policy array shapes or dtypes are incompatible")

    return tensor_path, entries, interface


def validate_policy_document(path, manifest, *, max_tensor_bytes=DEFAULT_MAX_POLICY_TENSOR_BYTES):
    """Verify file bounds, checksums and array headers without loading arrays."""
    _inspect(path, manifest, max_tensor_bytes)


def load_policy_document(path, manifest, source, *, expected_game, max_tensor_bytes=DEFAULT_MAX_POLICY_TENSOR_BYTES):
    tensor_path, entries, interface = _inspect(path, manifest, max_tensor_bytes)
    if _training_game_entry(source.game) != _training_game_entry(expected_game):
        raise ArtifactError(f"tabular policy is incompatible with game {expected_game.name!r}")

    game = _load_policy_game(expected_game)
    if interface != _PolicyInterface.from_game(game):
        raise ArtifactError(f"policy interface is incompatible with game {expected_game.name!r}")

    arrays = {}
    try:
        with zipfile.ZipFile(tensor_path) as archive:
            for entry in entries:
                arrays[entry.player, entry.state_name] = _read_tensor_array(archive, entry)

        table = {}
        for player, _, actions in interface.players:
            observations = arrays[player, "observations"]
            masks = arrays[player, "legal_masks"]
            probabilities = arrays[player, "probabilities"]
            if not np.all(np.isfinite(observations)):
                raise ArtifactError("table observations must be finite")

            _validate_batched_distribution(probabilities, actions, masks, context="saved table policy")
            for obs, mask, row in zip(observations, masks, probabilities, strict=True):
                distribution = {action: float(value) for action, allowed, value in zip(actions, mask, row, strict=True)
                                if allowed}
                key = (player, tuple(obs), tuple(distribution))
                if key in table:
                    raise ArtifactError("saved policy contains duplicate table queries")

                table[key] = distribution

        profile = PolicyProfile.from_table(table, fallback=manifest["fallback"])
    except ArtifactError:
        raise
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ArtifactError(f"invalid tabular policy data: {error}") from error

    return profile
