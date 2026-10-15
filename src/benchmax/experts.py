"""An expert is a pretrained :doc:`policy profile </reference/policy>` selected as a strong reference for a supported game.
You can use it as an evaluation baseline or to collect demonstrations for imitation learning.

Experts are selected by their recorded exact exploitability. They are the best-found policies, not guaranteed
equilibria. Each library version pins a specific checkpoint, so ``load_expert(game)`` loads the selected policy without
retraining or choosing from a changing ranking.
"""

from __future__ import annotations

import os
from importlib.resources import files

from omegaconf import OmegaConf

from benchmax.core import Game
from benchmax.policy import PolicyProfile
from benchmax.storage import (
    STRICT_REQUIREMENTS,
    HuggingFaceStore,
    ObjectReference,
    PolicyNotFoundError,
    StorageError,
)
from benchmax.storage._policy_record import _game_document, _game_entry
from benchmax.storage.reference import _policy_reference
from benchmax.storage.store import _entry_configuration


def _expert_references() -> tuple[str, dict[str, str]]:
    with files("benchmax.conf").joinpath("expert/policies.yaml").open("r", encoding="utf-8") as stream:
        config = OmegaConf.to_container(OmegaConf.load(stream), resolve=True)
    if (not isinstance(config, dict) or set(config) != {"bucket_id", "policies"}
            or not isinstance(config["policies"], dict)):
        raise StorageError("invalid expert policy configuration")
    return config["bucket_id"], config["policies"]


def load_expert(
    game: str | Game, *, token: str | bool | None = None, cache_dir: str | os.PathLike[str] | None = None,
) -> PolicyProfile:
    """Download and load this library version's selected expert, without training.

    Accept a canonical game name or a game created by ``load_game()``. The
    checkpoint must match its game configuration, but evaluator choices are ignored.
    Experts are selected by recorded exact exploitability and pinned in the
    shipped configuration, rather than ranked again against the live bucket.
    They are best-found policies, not guaranteed equilibria.

    ``token=None`` uses the saved Hugging Face login or ``HF_TOKEN``. The current
    collection is private and requires read access. Verified downloads use the
    normal storage cache (or ``cache_dir``), including offline reuse after the
    first successful load. Neural experts require ``benchmax[training]``.

    Raises:
        PolicyNotFoundError: No expert is registered for the game name.
        ValueError: The requested game settings differ from the pinned policy.
        StorageError: Download, authentication, or artifact validation fails.
    """
    name = game if isinstance(game, str) else _game_entry(game).name
    bucket_id, references = _expert_references()
    if name not in references:
        available = ", ".join(sorted(references))
        raise PolicyNotFoundError(f"no expert is registered for {name!r}; available games: {available}")

    requested = _game_entry(game)
    identity = ObjectReference(references[name])
    store = HuggingFaceStore(bucket_id, token=token, cache_dir=cache_dir, requirements=STRICT_REQUIREMENTS)
    record = store._reference_record(identity)
    if (_entry_configuration(_game_document(requested))
            != _entry_configuration(_game_document(record.training.game))):
        raise ValueError(f"the pinned expert for {name!r} does not match the requested game configuration")

    reference = _policy_reference(store, identity, record.training)
    return store._load_reference(reference, expected_game=requested)
