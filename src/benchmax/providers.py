"""Providers connect external libraries to BenchMAX, supplying games, training algorithms, or evaluators through a common
interface. Provider names identify where entries come from, such as ``openspiel.kuhn_poker``, and let BenchMAX report
missing dependencies clearly.

Adding a library
----------------

Adding a provider currently means adding an integration to the BenchMAX source. For a library supplying games:

1. **Write the adapter** under ``src/benchmax/integrations/mylib/``. Wrap the library's game and environment with the
   :doc:`Core interfaces </reference/core>`. For training algorithms, adapt the library to the :doc:`training interface
   </reference/training>` instead.
2. **Expose its entries** in ``provider.py``: ``available_entries()`` lists supported names, and ``base_config(entry)``
   returns the constructor configuration for one name. Keep configuration resolution independent of the optional
   runtime by importing the library only when listing entries or constructing the adapter.
3. **Declare the provider** in ``PROVIDERS`` in ``src/benchmax/providers.py``, including its importable module,
   configuration group, and implementation module.
4. **Declare the dependency** in ``pyproject.toml``, using an optional extra if appropriate. Set the provider's
   ``extra`` to that extra's name so missing-dependency messages explain how to install it.
5. **Check a small example** with ``load_game("mylib.example")``: create an environment, reset it, and step through
   legal actions. A game adapter alone has no evaluator, so attach a compatible one before evaluating policies.

For example, the entry in ``PROVIDERS`` would be:

.. code-block:: python

    "mylib": Provider(
        name="mylib", supplies=("game",), module="mylib", extra="mylib",
        implementation="benchmax.integrations.mylib.provider", entry_key="library_name"
    ),

Its ``base_config(entry)`` would return a configuration matching your adapter's constructor:

.. code-block:: python

    return {
        "_target_": "benchmax.integrations.mylib.MyGame",
        "library_name": entry,
        "_fixed_": ["library_name"],
        "evaluator": None,
    }

Use the existing ``integrations/openspiel/provider.py`` for a game example or ``integrations/benchmarl/provider.py`` for
algorithms. Optional presets under ``src/benchmax/conf/game/mylib/`` can add parameters and evaluators, but you do
not need one YAML file per library game.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class Provider:
    """One external library BenchMAX can draw benchmark content from.

    Attributes:
        name: Namespace used to address this provider's content.
        supplies: Configuration groups this provider contributes entries to,
            named exactly as the groups are: ``"game"``, ``"algorithm"``.
        module: Module whose presence indicates the provider is installed.
        extra: Installation extra that provides it, when it is optional.
        implementation: Module that turns one of this provider's own entry
            names into BenchMAX configuration. It exposes ``base_config``
            for one entry and ``available_entries`` for everything the
            provider can build. Providers without one supply nothing
            addressable as ``provider.name``.
        entry_key: Configuration key naming the entry supplied by that implementation.
    """

    name: str
    supplies: tuple[str, ...]
    module: str
    extra: str | None = None
    implementation: str | None = None
    entry_key: str = "name"

    @property
    def available(self) -> bool:
        """Whether this provider is installed.

        Presence is probed without importing the module, so checking a provider
        never runs its initialisation. That matters for ``exp_a_spiel``, whose
        native binding must not be loaded before torch (see #9).
        """

        try:
            return find_spec(self.module) is not None
        except (ImportError, ValueError):
            return False


PROVIDERS: dict[str, Provider] = {
    "openspiel": Provider(
        name="openspiel", supplies=("game",), module="pyspiel",
        implementation="benchmax.integrations.openspiel.provider", entry_key="spiel_name"
    ),
    "exp_a_spiel": Provider(
        name="exp_a_spiel", supplies=("evaluator",), module="benchmax.integrations.exp_a_spiel._exp_a_spiel"
    ),
    "benchmarl": Provider(
        name="benchmarl", supplies=("algorithm",), module="benchmarl", extra="training",
        implementation="benchmax.integrations.benchmarl.provider"
    ),
}


def list_providers() -> dict[str, bool]:
    """Return each known provider and whether it is installed."""
    return {name: provider.available for name, provider in PROVIDERS.items()}


def require_provider(name: str) -> Provider:
    """Return an installed provider, or explain how to install it.

    Raises:
        ValueError: If no provider has that name.
        ModuleNotFoundError: If the provider is known but not installed.
    """

    try:
        provider = PROVIDERS[name]
    except KeyError as error:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"unknown provider {name!r}; known providers: {known}"
        ) from error

    if not provider.available:
        install = f"benchmax[{provider.extra}]" if provider.extra else "benchmax"
        raise ModuleNotFoundError(
            f"the {name!r} provider is not installed; install {install}"
        )

    return provider
