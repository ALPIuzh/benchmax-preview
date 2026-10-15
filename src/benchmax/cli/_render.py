"""Small output helpers shared by CLI commands."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence


def table(headers: Sequence[str] | None, rows: Iterable[Sequence[str]], indent: str = "    ") -> list[str]:
    """Render aligned columns, dropping columns nothing fills."""

    body = [list(row) for row in rows]
    if not body:
        return [f"{indent}none"]

    keep = [column for column in range(len(body[0])) if any(row[column] for row in body)]
    body = [[row[column] for column in keep] for row in body]
    shown = [headers[column] for column in keep] if headers else None
    widths = [
        max(len(shown[column]) if shown else 0, *(len(row[column]) for row in body))
        for column in range(len(body[0]))
    ]
    lines = []

    if shown:
        lines.append(indent + "  ".join(heading.upper().ljust(width) for heading, width in zip(shown, widths)))

    lines += [indent + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in body]
    return [line.rstrip() for line in lines]


def heading(text: str, indent: str = "  ") -> str:
    """Render a section sentence with a rule under it."""

    return f"{indent}{text}\n{indent}{'-' * len(text)}"


def render_value(value: object, *, as_json: bool) -> None:
    """Print a scalar directly and structured data as canonical JSON."""

    if as_json or isinstance(value, (dict, list, tuple)):
        print(json.dumps(value, sort_keys=True))
    else:
        print(value)
