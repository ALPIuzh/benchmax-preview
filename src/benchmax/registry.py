"""Discovery and loading of the games, algorithms, and evaluators BenchMAX ships.

The groups are resolved by the same machinery, whose rules are
documented in ``conf/README.md``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from benchmax.config import ConfigGroup, ConfigValue, _EntryOrigin, _resolve, entry_definition, is_provider_entry
from benchmax.core import Game
from benchmax.providers import PROVIDERS, require_provider

if TYPE_CHECKING:
    from benchmax.evaluation import Evaluator
    from benchmax.training.algorithm import TrainingFunction

_GAMES: ConfigGroup[Game] = ConfigGroup("game", Game, references={"evaluator": "evaluator"})
_EVALUATORS: ConfigGroup[Evaluator] = ConfigGroup("evaluator", "benchmax.evaluation.Evaluator")

#: Algorithm entries build configured training callables. Their provider
#: dependencies are imported only when an entry is instantiated.
_ALGORITHMS: ConfigGroup[TrainingFunction] = ConfigGroup("algorithm", "benchmax.training.algorithm.TrainingFunction")


def _provider_of(name: str, config: DictConfig) -> str:
    """Which library implements this entry."""
    declared = config.get("provider")
    if isinstance(declared, str):
        return declared

    if is_provider_entry(name):
        return name.partition(".")[0]

    target = str(config.get("_target_", ""))

    for provider in PROVIDERS:
        if f"integrations.{provider}." in target:
            return provider

    return "benchmax"


def _inspect_entry(group: str, name: str) -> dict[str, Any]:
    """Describe an entry and its availability without building it, locally or on a target."""
    report = {"name": name, "group": group, "config": None, "definition": {}, "provider": "benchmax",
              "available": False, "error": "", "python": sys.executable, "workspace": str(Path.cwd()),
              "scope": "local execution environment; GPU/runtime capacity is not tested"}
    try:
        config = _resolve(group, name)
    except ValueError as error:
        return {**report, "error": str(error)}
    except ModuleNotFoundError as error:
        return {**report, "config": {}, "error": str(error)}

    provider = _provider_of(name, config)
    report.update(config=OmegaConf.to_container(config, resolve=True), provider=provider,
                  definition=OmegaConf.to_container(entry_definition(group, name), resolve=True))
    try:
        if provider in PROVIDERS:
            source = require_provider(provider)
            if source.implementation and group in source.supplies:
                entries = importlib.import_module(source.implementation).available_entries()
                entry = config[source.entry_key]
                if entry not in entries:
                    return {**report, "error": f"unknown {provider} {group} {entry!r}"}

        module, _, attribute = str(config["_target_"]).rpartition(".")
        if not callable(getattr(importlib.import_module(module), attribute)):
            raise TypeError(f"{group} implementation {config['_target_']!r} is not callable")
    except ModuleNotFoundError as error:
        extra = PROVIDERS[provider].extra if provider in PROVIDERS else None
        install = f"benchmax[{extra}]" if extra else "benchmax"
        return {**report, "error": f"{error}. Install {install} in the inspected Python environment."}

    return {**report, "available": True}


def list_available_games() -> list[str]:
    """Return the names of loadable games in sorted order.

    Includes bare names, provider entries that have a configuration, and the
    games implemented in BenchMAX. Provider entries without a configuration are
    loadable but not listed, since a provider may supply hundreds.
    """

    return _GAMES.names()


def list_available_algorithms() -> list[str]:
    """Return the names of bundled training algorithms in sorted order.

    Listing does not require the training dependencies to be installed.
    """

    return _ALGORITHMS.names()


def load_game(name: str, **overrides: ConfigValue) -> Game:
    """Create a game from its configuration.

    Args:
        name: A bare name such as ``"kuhn_poker"``, a provider entry such as
            ``"openspiel.leduc_poker"``, or the name of a game BenchMAX
            configures itself.
        overrides: Configuration keys to replace, such as
            ``params={"board_size": 4}`` or ``evaluator="nash_conv"``.

    Returns:
        The configured game.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If no game has that name.
        FixedParameterError: If an override changes a parameter this game's
            name depends on.
        ImportError: If the named provider is not installed.
    """

    game, resolved = _GAMES.load_resolved(name, **overrides)
    # A detached job must reconstruct the exact registered entry rather than
    # guess from Game.name, which custom games may share. This private origin
    # is deliberately structural and contains no execution machinery.
    game._benchmax_origin = _EntryOrigin(resolved, overrides)
    # Portable evaluation must reproduce the configured evaluator exactly.
    # Keep only an identity marker; directly composed evaluators remain fully
    # supported for ordinary synchronous evaluation.
    game._benchmax_configured_evaluator = game.evaluator
    return game
