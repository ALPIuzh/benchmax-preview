"""Root application and dispatch for the ``benchmax`` command."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from cyclopts import App, CycloptsError

from benchmax.cli.attempts import attempt
from benchmax.cli.inspect import algorithms, games, providers, show, targets
from benchmax.cli.jobs import job
from benchmax.cli.setup import app as setup_app
from benchmax.cli.storage import app as store_app
from benchmax.cli.workloads import evaluate, train
from benchmax.execution.targets.service import JobError

app = App(name="benchmax", help="Inspect and run BenchMAX workloads.", help_format="plaintext")
for command in (providers, targets, games, algorithms, show, train, evaluate, job):
    app.command(command)
app.command(attempt, name="_attempt", show=False)
app.command(store_app)
app.command(setup_app)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` from signatures and dispatch one BenchMAX command."""

    try:
        result = app(argv, result_action="return_value")
    except CycloptsError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error
    except JobError as error:
        print(f"benchmax: {error}", file=sys.stderr)
        return 1

    return int(result or 0)
