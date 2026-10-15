"""Inspection commands for what a BenchMAX installation can do.

Everything here reads configuration rather than building anything, so it works
on a base installation and never loads a provider it does not need.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from typing import Annotated, Any

from cyclopts import Parameter
from omegaconf import DictConfig

from benchmax.config import (
    _EXTENDS_KEY,
    _FIXED_KEY,
    _FIXED_PARAMS_KEY,
    _PARAMS_KEY,
    _SOURCE_KEY,
    _USE_KEY,
    _resolve,
    entry_definition,
    is_provider_entry,
    list_names,
)
from benchmax.providers import PROVIDERS
from benchmax.registry import _inspect_entry

from ._render import heading, table

_GROUPS = ("game", "algorithm", "evaluator")


_WIDTH = 76


def _evaluator_of(config: DictConfig) -> str:
    """Which evaluator an entry uses, as configuration spells it."""

    if "evaluator" not in config:
        return "built in"

    evaluator: Any = config.get("evaluator")
    if evaluator is None:
        return "none"

    if isinstance(evaluator, str):
        return evaluator

    if isinstance(evaluator, DictConfig) and _USE_KEY in evaluator:
        return str(evaluator[_USE_KEY])

    return "custom"


def _origin(name: str, definition: DictConfig, provider: str) -> str:
    """Describe where an entry comes from."""

    if not definition:
        if is_provider_entry(name):
            head, _, entry = name.partition(".")
            return f"{entry}, straight from {head}, with its own defaults"

        return f"from {provider}"

    if extends := definition.get(_EXTENDS_KEY):
        if not definition.get(_PARAMS_KEY):
            return f"based on {extends}"

        return f"{extends}, with some parameters changed"

    if source := definition.get(_SOURCE_KEY):
        return f"{source}, with some parameters changed"

    return "implemented in benchmax"


def providers() -> int:
    """List provider libraries and whether this installation can use them."""

    rows = []

    for name, provider in PROVIDERS.items():
        if provider.available:
            status = "installed"
        elif provider.extra:
            status = f"run: pip install benchmax[{provider.extra}]"
        else:
            status = "not installed"

        rows.append([name, ", ".join(provider.supplies), status])

    print("\n".join(table(["provider", "supplies", "status"], rows, indent="  ")))
    return 0


def targets() -> int:
    """List named targets and installed adapter schemes."""

    from benchmax.execution.targets.registry import list_available_targets
    from benchmax.execution.targets.service import available_target_adapters

    rows = [[name] for name in list_available_targets()]
    print("\n".join(table(["target"], rows, indent="  ")))
    print("Adapters: " + ", ".join(available_target_adapters()))
    return 0


def _from_provider(group: str, name: str) -> str:
    """Where this name comes from, and how it relates to it."""
    definition = entry_definition(group, name)
    if alias := definition.get(_EXTENDS_KEY):
        return f"{alias} (alias)"

    if source := definition.get(_SOURCE_KEY):
        return str(source)

    return ""


def _entries(group: str, show_all: bool) -> int:
    """List configured entries, then the other entries providers can supply."""

    print("\n".join(_paragraphs(
        f"Additional {group}s come from providers. Install the ones you need; "
        f"benchmax providers shows which are available."
    )))
    print()
    print(heading(f"These {group}s are configured and tested."))
    print()

    configured = list_names(group)
    print("\n".join(table(
        [group, "from provider", "evaluator"],
        [
            [name, _from_provider(group, name), _evaluator_of(_resolve(group, name)) if group == "game" else ""]
            for name in configured if not is_provider_entry(name)
        ],
        indent="  ",
    )))
    print()

    for name, provider in PROVIDERS.items():
        if group not in provider.supplies or not provider.available:
            continue

        if provider.implementation is None:
            continue

        own = [entry.partition(".")[2] for entry in configured if entry.startswith(f"{name}.")]

        if own:
            print()
            print(f"  These are configured from external {group} providers.")
            print()
            print(f"  {name}.{{name}}")
            print("\n".join(table(
                [group, "evaluator"],
                [
                    [entry, _evaluator_of(_resolve(group, f"{name}.{entry}")) if group == "game" else ""]
                    for entry in own
                ],
            )))
            print()

        module = importlib.import_module(provider.implementation)
        rest = [e for e in module.available_entries() if e not in own]
        print(heading(f"These {group}s are not configured but loadable."))
        print()

        if not show_all:
            print(
                f"  {name} supplies {len(rest)} of them. Use --all to list "
                f"them,\n  or benchmax show {name}.<name>."
            )
            continue

        print(f"  {name}.{{name}}")
        print(textwrap.fill(
            ", ".join(rest), width=_WIDTH, initial_indent="    ", subsequent_indent="    ", break_on_hyphens=False
        ))

    return 0


def _paragraphs(text: str, indent: str = "  ") -> list[str]:
    lines: list[str] = []

    for paragraph in text.strip().split("\n\n"):
        collapsed = " ".join(paragraph.split())
        lines += textwrap.wrap(collapsed, width=_WIDTH, initial_indent=indent, subsequent_indent=indent) or [""]
        lines.append("")

    return lines[:-1] if lines else lines


def show(
    name: Annotated[str, Parameter(help="Registered entry name.")], /, *,
    target: Annotated[str | None, Parameter(help="Inspect an algorithm in a saved execution target.")] = None
) -> int:
    """Describe one game, algorithm, or evaluator."""

    if target is not None:
        from benchmax.execution.targets.registry import load_target

        destination = load_target(target)
        try:
            report = destination.service().inspect_algorithm(name)
        except Exception as error:  # noqa: BLE001 -- report probe failures at the CLI boundary
            print(f"  {name} — inspection failed on {target}: {error}", file=sys.stderr)
            return 2

        print(f"  Target: {target}")
        print("\n".join(table(["", ""], [
            ["Python", report["python"]], ["workspace", report["workspace"]], ["scope", report["scope"]],
        ], indent="  ")[1:]))
        print()
        return _show_entry(report)

    for group in _GROUPS:
        report = _inspect_entry(group, name)
        if report["config"] is not None:
            return _show_entry(report)

    print(f"  nothing named {name!r} in {', '.join(_GROUPS)}")
    return 1


def _show_entry(report: dict[str, Any]) -> int:
    name, group, provider = report["name"], report["group"], report["provider"]
    if not report["available"]:
        print(f"  {name}\n  {group} — not available here\n")
        print("\n".join(_paragraphs(report["error"])))
        return 1

    config = DictConfig(report["config"])
    definition = DictConfig(report["definition"])
    print(f"  {name}")
    print(f"  {group} — {_origin(name, definition, provider)}")
    print()

    if not definition:
        print("\n".join(_paragraphs(
            "This is the provider's own entry, with the provider's own "
            "defaults. BenchMAX has not configured it: nothing here has "
            "been tuned or selected, and no evaluator has been chosen."
        )))
        print()

    if description := config.get("description"):
        print("\n".join(_paragraphs(str(description))))
        print()

    facts = [["availability", "available"]]

    if group == "game":
        facts.append(["evaluator", _evaluator_of(config)])

        if parameters := config.get(_PARAMS_KEY):
            facts.append(["parameters", ", ".join(f"{k}={v}" for k, v in parameters.items())])

        # The two halves are frozen separately, and saying which is which
        # is the difference between "this game is 3x3" and "this name is
        # this game".
        if fixed := config.get(_FIXED_PARAMS_KEY):
            facts.append(["fixed parameters", ", ".join(str(key) for key in fixed)])

        if fixed := config.get(_FIXED_KEY):
            facts.append(["fixed", ", ".join(str(key) for key in fixed)])

    if provider := config.get("provider"):
        facts.append(["provider", str(provider)])

    print("\n".join(table(["", ""], facts, indent="  ")[1:]))

    return 0


def games(
    *, all: Annotated[bool, Parameter(negative=(), help="Also list provider games not configured by BenchMAX.")] = False
) -> int:
    """List games available to this installation."""

    return _entries("game", all)


def algorithms(
    *, all: Annotated[
        bool, Parameter(negative=(), help="Also list provider algorithms not configured by BenchMAX.")
    ] = False
) -> int:
    """List training algorithms available to this installation."""

    return _entries("algorithm", all)
