"""Explicit installation of optional external opponents."""

from __future__ import annotations

import sys
from pathlib import Path

from cyclopts import App

app = App(name="setup", help="Install optional external opponents before evaluation.", help_format="plaintext")


@app.command
def pons(*, rows: int = 6, columns: int = 7, directory: Path | None = None, timeout: float = 300) -> int:
    """Download and compile the pinned Connect Four engine for one board size."""
    from benchmax.integrations.pons.install import install

    try:
        executable = install(rows=rows, columns=columns, directory=directory, timeout=timeout)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"benchmax: {error}", file=sys.stderr)
        return 1
    print(executable)
    return 0


@app.command
def stockfish(*, directory: Path | None = None, timeout: float = 300) -> int:
    """Download the pinned Stockfish binary for a supported platform."""
    from benchmax.integrations.stockfish.install import install

    try:
        executable = install(directory=directory, timeout=timeout)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"benchmax: {error}", file=sys.stderr)
        return 1
    print(executable)
    return 0
