"""Loading the configuration groups BenchMAX ships.

The rules these functions implement are documented in ``conf/README.md``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any, Generic, TypeAlias, TypeVar

from omegaconf import DictConfig, OmegaConf

from benchmax.providers import PROVIDERS
from benchmax.utils.json import canonical_json

ConfigValue: TypeAlias = (
    "str | int | float | bool | None | list[ConfigValue] | dict[str, ConfigValue]"
)

_CONFIG_PACKAGE = "benchmax.conf"

#: Keys wrapped in underscores belong to the loader and are consumed here.
#: Every other key is an argument of the thing being built, so what a
#: configuration may say is bounded by that constructor rather than by a list
#: kept in this module.
#:
#: Replaces this key with the named entry of another group, so a game can say
#: ``evaluator: {_use_: exp_a_spiel, traverser: DhEvaluator}`` instead of
#: repeating an import path. Sibling keys override the referenced entry.
_USE_KEY = "_use_"

#: Starts this configuration from another entry of the same group, which must
#: exist.
_EXTENDS_KEY = "_extends_"

#: Starts this configuration from a provider's own defaults, whether or not a
#: configuration exists for them.
_SOURCE_KEY = "_source_"

#: Becomes the target's ``params`` argument, which every game and algorithm
#: takes: the settings that decide which game or algorithm this is. They stay in
#: a mapping of their own rather than becoming keyword arguments, so a provider
#: that names parameters per game -- OpenSpiel has 228 names across its 122
#: games -- can never collide with an argument BenchMAX declares.
_PARAMS_KEY = "_params_"

#: Arguments of the built object that overrides may not change, and the same
#: for names under ``_params_``, so a name such as ``dark_hex_3x3`` cannot be
#: made to mean something else -- neither by changing a parameter it depends on
#: nor by swapping the game underneath it. Both accumulate as configurations
#: layer: an entry adds to what it inherits rather than replacing it, so
#: freezing something new cannot quietly unfreeze the rest.
_FIXED_KEY = "_fixed_"
_FIXED_PARAMS_KEY = "_fixed_params_"

#: The target's name for it, and what a caller writes. The underscores mark a
#: key as the loader's rather than the target's, which a keyword argument has no
#: need to do.
_PARAMS_ARGUMENT = "params"

#: Bare names for raw provider entries.
_ALIASES_ENTRY = "_aliases"

T = TypeVar("T")


@dataclass(frozen=True, slots=True, init=False)
class ResolvedEntry:
    """Canonical, non-executable snapshot of one configured entry.

    ``configuration`` is retained for identity and compatibility checks. Code
    always instantiates through its local, allow-listed :class:`ConfigGroup`
    and never trusts an import target supplied by a workload document.
    """

    group: str
    name: str
    _configuration_json: str = field(repr=False)

    def __init__(self, group: str, name: str, configuration: Mapping[str, ConfigValue]) -> None:
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_configuration_json", canonical_json(configuration))

    @property
    def configuration(self) -> dict[str, ConfigValue]:
        """Return a detached copy of the configuration used to build this entry."""
        return json.loads(self._configuration_json)


@dataclass(frozen=True, slots=True, init=False)
class _EntryOrigin:
    """Exact registered configuration behind one instantiated object."""

    resolved: ResolvedEntry
    _overrides_json: str = field(repr=False)

    def __init__(self, resolved: ResolvedEntry, overrides: Mapping[str, ConfigValue]) -> None:
        object.__setattr__(self, "resolved", resolved)
        object.__setattr__(self, "_overrides_json", canonical_json(overrides))

    @property
    def overrides(self) -> dict[str, ConfigValue]:
        """Return a detached copy of the original caller's overrides."""
        return json.loads(self._overrides_json)


class FixedParameterError(ValueError):
    """Raised when an override would change a parameter a name depends on."""


def _group_directory(group: str) -> Traversable:
    return files(_CONFIG_PACKAGE).joinpath(group)


def _read_entry(group: str, name: str) -> DictConfig | None:
    """Read the file backing one entry, or return ``None`` if there is none.

    A dotted name addresses a provider subdirectory, so ``openspiel.dark_hex``
    is ``game/openspiel/dark_hex.yaml``.
    """

    *directories, leaf = name.split(".")
    entry = _group_directory(group)

    for part in directories:
        entry = entry.joinpath(part)

    entry = entry.joinpath(f"{leaf}.yaml")
    return _read(entry, f"{group}/{name}") if entry.is_file() else None


def _read(entry: Traversable, label: str) -> DictConfig:
    """Return a fresh, mutable copy of one packaged configuration file."""
    with entry.open(encoding="utf-8") as stream:
        config = OmegaConf.load(stream)

    if not isinstance(config, DictConfig):
        raise TypeError(f"configuration {label!r} must be a mapping")

    return config


def _aliases(group: str) -> DictConfig:
    """Return this group's bare-name definitions, if it has any."""
    entry = _group_directory(group).joinpath(f"{_ALIASES_ENTRY}.yaml")
    if not entry.is_file():
        return OmegaConf.create({})

    return _read(entry, f"{group}/{_ALIASES_ENTRY}")


def entry_definition(group: str, name: str) -> DictConfig:
    """Read an entry without resolving references or importing its provider.

    Aliases take precedence over files. An entry without a bundled definition
    returns an empty mapping, including provider entries using their defaults.
    """

    aliases = _aliases(group)
    if name in aliases:
        return OmegaConf.create(aliases[name])

    definition = _read_entry(group, name)
    return definition if definition is not None else OmegaConf.create({})


def list_names(group: str) -> list[str]:
    """Return the selectable names in a group, sorted.

    Includes bare names from ``_aliases.yaml`` and raw provider entries that
    have a configuration, and omits underscore-prefixed files, which hold
    shared defaults rather than loadable entries.

    Listing imports nothing the configurations refer to, so a group can be
    discovered without the providers that build it being installed.
    """

    directory = _group_directory(group)
    names = {
        entry.name.removesuffix(".yaml")
        for entry in directory.iterdir()
        if entry.name.endswith(".yaml") and not entry.name.startswith("_")
    }

    for provider in directory.iterdir():
        if provider.is_dir() and not provider.name.startswith("_"):
            names |= {
                f"{provider.name}.{entry.name.removesuffix('.yaml')}"
                for entry in provider.iterdir()
                if entry.name.endswith(".yaml")
            }

    names |= set(_aliases(group).keys())
    return sorted(names)


def _provider_defaults(group: str, provider_name: str, entry: str) -> DictConfig:
    """Return the structural configuration for one provider entry."""

    try:
        provider = PROVIDERS[provider_name]
    except KeyError as error:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {provider_name!r}; known providers: {known}") from error

    if provider.implementation is None or group not in provider.supplies:
        raise ValueError(f"the {provider_name!r} provider does not supply {group} entries")

    module = importlib.import_module(provider.implementation)
    return OmegaConf.create(module.base_config(entry))


def _resolve(group: str, name: str, _seen: frozenset[str] = frozenset()) -> DictConfig:
    """Assemble one entry's configuration, following its references."""

    if name in _seen:
        raise ValueError(f"{group} configuration {name!r} refers to itself")

    seen = _seen | {name}

    aliases = _aliases(group)

    if name in aliases:
        config = OmegaConf.create(aliases[name])
    elif is_provider_entry(name):
        # A provider entry is the provider's own defaults. Any configuration
        # file for it adds BenchMAX's concerns on top rather than replacing it.
        provider_name, _, remainder = name.partition(".")
        base = _provider_defaults(group, provider_name, remainder)
        contribution = _read_entry(group, name)
        if contribution is None:
            return base

        if contribution.get(_PARAMS_KEY):
            raise ValueError(
                f"{group} configuration {name!r} is a provider entry and may "
                f"not set {_PARAMS_KEY!r}; anything that changes what the "
                "provider supplies needs its own name"
            )

        return _layer(base, contribution)
    else:
        config = _read_entry(group, name)
        if config is None:
            available = ", ".join(list_names(group))
            raise ValueError(
                f"unknown {group} {name!r}; available {group}s: {available}"
            )

    base_name = config.pop(_EXTENDS_KEY, None)
    source_name = config.pop(_SOURCE_KEY, None)

    # Layers apply outermost first: the provider's own defaults, then any
    # BenchMAX configuration this entry extends, then the entry itself. An
    # entry may use both, which is how a tuned algorithm starts from its
    # library's defaults and then picks up settings shared across algorithms.
    layers: list[DictConfig] = []

    if source_name is not None:
        if not is_provider_entry(str(source_name)):
            raise ValueError(
                f"{group} configuration {name!r} has {_SOURCE_KEY} "
                f"{source_name!r}, which does not name a provider entry"
            )

        provider_name, _, remainder = str(source_name).partition(".")
        layers.append(_provider_defaults(group, provider_name, remainder))

    if base_name is not None:
        layers.append(_resolve(group, str(base_name), seen))

    if not layers:
        return config

    return _layer(*layers, config)


def _layer(*configurations: DictConfig) -> DictConfig:
    """Merge configuration layers, accumulating the frozen names."""
    merged = OmegaConf.merge(*configurations)
    assert isinstance(merged, DictConfig)

    for key in (_FIXED_KEY, _FIXED_PARAMS_KEY):
        frozen = {
            str(frozen_name)
            for configuration in configurations
            for frozen_name in configuration.get(key) or ()
        }
        if frozen:
            merged[key] = sorted(frozen)

    return merged


def is_provider_entry(name: str) -> bool:
    """Whether this name addresses a provider's own entry, such as ``openspiel.kuhn_poker``."""
    provider_name, separator, _ = name.partition(".")
    return bool(separator) and provider_name in PROVIDERS


def _as_configuration(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return keyword overrides as the configuration keys they stand for."""
    keys = dict(overrides)
    if _PARAMS_ARGUMENT in keys:
        keys[_PARAMS_KEY] = keys.pop(_PARAMS_ARGUMENT)

    return keys


def _check_fixed(name: str, config: DictConfig, overrides: Mapping[str, Any]) -> None:
    """Refuse overrides of anything the entry's name depends on.

    Only callers are refused. Derived configurations may change frozen values,
    as described in ``conf/README.md``.
    """

    clashing = set(config.get(_FIXED_KEY) or ()) & set(overrides)
    parameters = overrides.get(_PARAMS_KEY)
    if isinstance(parameters, Mapping):
        clashing |= set(config.get(_FIXED_PARAMS_KEY) or ()) & set(parameters)

    if clashing:
        raise FixedParameterError(
            f"{', '.join(sorted(clashing))} cannot be changed for {name!r}; "
            "it is part of what that name means, so choosing it needs a "
            "configuration of its own"
        )


def _arguments(config: DictConfig, label: str) -> DictConfig:
    """Reduce one configuration to the arguments its target is called with.

    The loader's own keys are consumed here, and ``_params_`` becomes the
    ``params`` argument. ``conf/README.md`` explains why it is kept separate.
    """

    config.pop(_FIXED_KEY, None)
    config.pop(_FIXED_PARAMS_KEY, None)
    parameters = config.pop(_PARAMS_KEY, None)
    if parameters is None:
        return config

    if not isinstance(parameters, DictConfig):
        raise TypeError(f"{_PARAMS_KEY} of {label!r} must be a mapping")

    config[_PARAMS_ARGUMENT] = parameters
    return config


class ConfigGroup(Generic[T]):
    """One directory of configurations that build objects of one type.

    Every group is listed, resolved, and instantiated the same way.

    Attributes:
        group: Name of the configuration directory, such as ``"game"``.
        references: Configuration keys whose value names an entry of another
            group, mapped to that group's name.
    """

    def __init__(self, group: str, builds: type[T] | str, references: Mapping[str, str] | None = None) -> None:
        """Declare a configuration group.

        Args:
            group: Name of the configuration directory.
            builds: The type every configuration in this group instantiates,
                either directly or as an import path. Naming it as a path
                defers the import, so a group whose type comes from an optional
                dependency can still be declared and listed without it.
            references: Keys that name entries of another group.
        """

        self.group = group
        self._builds = builds
        self.references = dict(references or {})

    @property
    def builds(self) -> type[T]:
        """The type this group builds, imported on first use."""

        if isinstance(self._builds, str):
            from hydra.utils import get_class

            self._builds = get_class(self._builds)

        return self._builds

    def names(self) -> list[str]:
        """Return the selectable names in this group, sorted."""
        return list_names(self.group)

    def _resolve_references(self, config: DictConfig) -> DictConfig:
        """Replace group references with the configurations they name."""

        for key, group in self.references.items():
            value = config.get(key)

            if isinstance(value, str):
                referenced = _resolve(group, value)
                label = value
            elif isinstance(value, DictConfig) and (_USE_KEY in value or "name" in value):
                # Public named references carry settings in portable workloads;
                # _use_ remains the configuration-file shorthand.
                selector = _USE_KEY if _USE_KEY in value else "name"
                label = str(value[selector])
                overrides = {
                    name: item for name, item in value.items() if name != selector
                }
                referenced = _layer(_resolve(group, label), OmegaConf.create(overrides))
            else:
                continue

            config[key] = _arguments(referenced, label)

        return config

    def load(self, name: str, **overrides: ConfigValue) -> T:
        """Build one configured object.

        Args:
            name: Name of an entry in this group.
            overrides: Arguments of the built object to replace, ``params``
                among them. Nested keys merge into the configuration rather
                than replacing it, so overriding one parameter leaves the
                others in place.

        Returns:
            The instantiated object.

        Raises:
            TypeError: If ``name`` is not a string, or the configuration builds
                something other than :attr:`builds`.
            ValueError: If this group has no entry with that name.
            FixedParameterError: If an override changes a parameter the name
                depends on.
        """

        prepared = self._prepare_named(name, overrides)
        return self._instantiate(prepared, label=name)

    def load_resolved(self, name: str, **overrides: ConfigValue) -> tuple[T, ResolvedEntry]:
        """Build an entry and return the exact structural snapshot used."""

        prepared = self._prepare_named(name, overrides)
        return self._instantiate(prepared, label=name), self._snapshot(prepared, name=name)

    def resolve(self, name: str, **overrides: ConfigValue) -> ResolvedEntry:
        """Resolve an entry to structural arguments without instantiating it.

        The snapshot follows references and applies exactly the same override
        and fixed-parameter rules as :meth:`load`. It is suitable for workload
        identity and environment-drift checks, but is not an executable config.
        """

        prepared = self._prepare_named(name, overrides)
        return self._snapshot(prepared, name=name)

    def _snapshot(self, prepared: DictConfig, *, name: str) -> ResolvedEntry:
        value = OmegaConf.to_container(prepared, resolve=True, enum_to_str=True)
        if not isinstance(value, dict):  # pragma: no cover - configs are mappings
            raise TypeError(f"resolved {self.group} {name!r} is not a mapping")

        return ResolvedEntry(self.group, name, value)

    def _prepare_named(self, name: str, overrides: Mapping[str, ConfigValue]) -> DictConfig:
        if not isinstance(name, str):
            raise TypeError(f"{self.group} name must be a string")

        return self._prepare(_resolve(self.group, name), label=name, **overrides)

    def build(self, config: DictConfig, *, label: str = "<inline>", **overrides: ConfigValue) -> T:
        """Build an inline configuration, applying overrides and fixed-parameter checks."""
        config = self._prepare(config, label=label, **overrides)
        return self._instantiate(config, label=label)

    def _instantiate(self, config: DictConfig, *, label: str) -> T:
        from hydra.errors import InstantiationException
        from hydra.utils import instantiate

        try:
            built = instantiate(config, _convert_="all")
        except InstantiationException as error:
            # Hydra adds useful configuration context to unexpected failures,
            # but ordinary public input and installation errors are clearer in
            # their original form.
            if isinstance(error.__cause__, (TypeError, ValueError, ImportError)):
                raise error.__cause__

            raise

        if not isinstance(built, self.builds):
            raise TypeError(
                f"{self.group} configuration {label!r} built "
                f"{type(built).__name__}; expected {self.builds.__name__}"
            )

        return built

    def _prepare(self, config: DictConfig, *, label: str, **overrides: ConfigValue) -> DictConfig:
        """Apply caller overrides and references before build or snapshot."""

        config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        assert isinstance(config, DictConfig)
        replacements = _as_configuration(overrides)
        _check_fixed(label, config, replacements)

        if replacements:
            for key in self.references:
                value = replacements.get(key)
                if isinstance(value, Mapping) and (_USE_KEY in value or "name" in value):
                    # Selecting a backend starts from that backend's settings;
                    # constructor arguments from the old one must not leak in.
                    config.pop(key, None)
            merged = OmegaConf.merge(config, OmegaConf.create(replacements))
            assert isinstance(merged, DictConfig)
            config = merged

        config = self._resolve_references(_arguments(config, label))
        return config
