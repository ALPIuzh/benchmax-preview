"""Indexed policies tied to a known native information-state layout."""

from __future__ import annotations

import hashlib
import zipfile

import numpy as np

from benchmax.execution.artifacts.arrays import (  # isort: skip
    _inspect_tensor_archive, _read_tensor_array, _write_array_policy)
from benchmax.execution.artifacts.interface import _load_policy_game, _PolicyInterface
from benchmax.execution.artifacts.policies import DEFAULT_MAX_POLICY_TENSOR_BYTES, ArtifactError
from benchmax.execution.specs import _training_game_entry
from benchmax.integrations.exp_a_spiel.evaluation import ExpASpielEvaluator
from benchmax.integrations.exp_a_spiel.payoff import _ExpASpielPolicy
from benchmax.policy import PolicyProfile, _validate_batched_distribution
from benchmax.utils.hashing import _is_sha256

POLICY_FORMAT = "benchmax.exp-a-spiel-indexed-policy.npz"
REQUIRES_TRAINING = False

_TRAVERSERS = ("PtttEvaluator", "AbruptPtttEvaluator", "DhEvaluator", "AbruptDhEvaluator")


def supports(profile: PolicyProfile) -> bool:
    return isinstance(profile, _ExpASpielPolicy)


def _layout(traverser, player):
    # Include ordering and parent links, not only the number of information sets.
    layout = traverser.cfr_treeplex(player)
    digest = hashlib.sha256()
    for array in layout:
        values = np.ascontiguousarray(array)
        digest.update(str((values.shape, values.dtype.str)).encode())
        digest.update(memoryview(values).cast("B"))

    masks = (layout[2][:, None] & (1 << np.arange(9))) != 0
    return masks, digest.hexdigest()


def write_policy(profile, source, destination):
    name = type(profile._traverser).__name__
    if name not in _TRAVERSERS:
        raise ArtifactError("unsupported indexed policy traverser")

    game = _load_policy_game(source.game)
    if not ExpASpielEvaluator(name)._supports(game):
        raise ArtifactError("indexed policy traverser is incompatible with the policy game")

    arrays, layouts = {}, {}
    for player in game.players:
        masks, digest = _layout(profile._traverser, player)
        strategy = np.ascontiguousarray(profile._strategies[player], dtype="<f8")
        _validate_batched_distribution(strategy, game.action_spaces[player], masks, context="indexed policy")
        arrays[player, "probabilities"] = strategy
        layouts[str(player)] = digest

    return _write_array_policy(arrays, source, destination, _PolicyInterface.from_game(game),
                               POLICY_FORMAT, annotations=profile._artifact_annotations, traverser=name, layouts=layouts)


def _inspect(path, manifest, max_tensor_bytes):
    if manifest.get("traverser") not in _TRAVERSERS:
        raise ArtifactError("unsupported indexed policy traverser")

    interface = _PolicyInterface.from_document(manifest.get("policy_interface"))
    if tuple(player for player, _, _ in interface.players) != (0, 1):
        raise ArtifactError("indexed policy requires players 0 and 1")

    tensor, entries = _inspect_tensor_archive(path, manifest, max_tensor_bytes=max_tensor_bytes)
    expected = {(0, "probabilities"), (1, "probabilities")}
    if manifest.get("state_values") != [] or {(entry.player, entry.state_name) for entry in entries} != expected:
        raise ArtifactError("invalid indexed policy array index")

    layouts = manifest.get("layouts")
    if not isinstance(layouts, dict) or set(layouts) != {"0", "1"}:
        raise ArtifactError("invalid indexed policy layout index")

    for entry in entries:
        if (len(entry.shape) != 2 or entry.shape[1] != 9 or entry.dtype != np.dtype("<f8")
                or interface.players[entry.player][2] != tuple(range(9))):
            raise ArtifactError("invalid indexed policy strategy shape or dtype")

        digest = layouts[str(entry.player)]
        if not _is_sha256(digest):
            raise ArtifactError("invalid indexed policy layout digest")

    return tensor, entries, interface


def validate_policy_document(path, manifest, *, max_tensor_bytes=DEFAULT_MAX_POLICY_TENSOR_BYTES):
    _inspect(path, manifest, max_tensor_bytes)


def load_policy_document(path, manifest, source, *, expected_game, max_tensor_bytes=DEFAULT_MAX_POLICY_TENSOR_BYTES):
    tensor, entries, interface = _inspect(path, manifest, max_tensor_bytes)
    if _training_game_entry(source.game) != _training_game_entry(expected_game):
        raise ArtifactError("indexed policy is incompatible with game")

    game = _load_policy_game(expected_game)
    evaluator = ExpASpielEvaluator(manifest["traverser"])
    if interface != _PolicyInterface.from_game(game) or not evaluator._supports(game):
        raise ArtifactError("indexed policy interface is incompatible with game")

    traverser = evaluator._get_traverser()
    strategies = {}
    try:
        with zipfile.ZipFile(tensor) as archive:
            for entry in entries:
                masks, digest = _layout(traverser, entry.player)
                if digest != manifest["layouts"][str(entry.player)] or masks.shape != entry.shape:
                    raise ArtifactError("indexed policy information-state layout has changed")

                values = _read_tensor_array(archive, entry)

                _validate_batched_distribution(values, game.action_spaces[entry.player], masks,
                                               context="saved indexed policy")
                strategies[entry.player] = values
    except ArtifactError:
        raise
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactError(f"invalid indexed policy data: {error}") from error

    return _ExpASpielPolicy(traverser, tuple(strategies[p] for p in game.players))
