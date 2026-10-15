"""Run assertions in fresh interpreters without hiding subprocess diagnostics."""

import subprocess
import sys
import textwrap


def assert_python_succeeds(program: str, *, timeout: float | None = 30) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)], capture_output=True, text=True, timeout=timeout, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
