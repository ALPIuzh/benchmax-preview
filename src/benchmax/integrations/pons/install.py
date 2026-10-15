"""Explicit installation of a pinned Pascal Pons Connect Four executable.

The AGPL engine stays external to BenchMAX. Evaluation never downloads or builds it.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from benchmax.utils.files import cache_directory

COMMIT = "d6ba50d8aaf2308c769d9bf2abd42d90f34baf41"
SOURCE_URL = f"https://github.com/PascalPons/connect4/archive/{COMMIT}.tar.gz"
BOOK_URL = "https://github.com/PascalPons/connect4/releases/download/book/7x6.book"
BOOK_SHA256 = "f346cd449626fb81da93be0958e017ee854e5f74d85b6d38062357f5403aec53"


def validate_board(rows: int, columns: int) -> None:
    """Restrict builds to four-in-a-row boards supported by the 64-bit adapter."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 4 for v in (rows, columns)):
        raise ValueError("Pons requires integer rows and columns >= 4")

    if columns > 9 or columns * (rows + 1) > 64:
        raise ValueError("Pons requires at most 9 columns and 64 cells including a spare row")


def default_directory(rows: int, columns: int) -> Path:
    host = f"{platform.system().lower()}-{platform.machine().lower()}"
    return cache_directory() / "pons" / COMMIT[:12] / host / f"{columns}x{rows}"


def install(*, rows: int = 6, columns: int = 7, directory: Path | None = None, timeout: float = 300) -> Path:
    """Download and build the engine explicitly, and return its executable path.

    ``directory`` contains both the executable and the standard board's book.
    Source is pinned by commit, and the separately released book is checksum checked.
    The timeout applies to network operations and the compiler invocation.
    """
    validate_board(rows, columns)
    if not 0 < timeout < float("inf"):
        raise ValueError("timeout must be finite and positive")

    directory = (default_directory(rows, columns) if directory is None else Path(directory)).expanduser().resolve()
    executable = directory / "c4solver"
    if is_complete_installation(directory, rows, columns):
        return executable

    if directory.exists():
        raise FileExistsError(
            f"incomplete or mismatched installation at {directory}; choose an empty installation directory"
        )

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as build:
        try:
            with (urllib.request.urlopen(SOURCE_URL, timeout=timeout) as response,
                  tarfile.open(fileobj=response, mode="r|gz") as archive):
                archive.extractall(build, filter="data")
            source = Path(build) / f"connect4-{COMMIT}"
            header = source / "Position.hpp"
            contents = header.read_text()
            if contents.count("WIDTH = 7;") != 1 or contents.count("HEIGHT = 6;") != 1:
                raise RuntimeError("Pons source board constants differ from the pinned installation recipe")
            header.write_text(
                contents.replace("WIDTH = 7;", f"WIDTH = {columns};").replace("HEIGHT = 6;", f"HEIGHT = {rows};")
            )
            subprocess.run(
                ["make", "c4solver"], cwd=source, check=True, capture_output=True, text=True, timeout=timeout
            )
            staged = Path(build) / "installed"
            staged.mkdir()
            shutil.move(source / "c4solver", staged / "c4solver")
            if (rows, columns) == (6, 7):
                _download_book(staged / "7x6.book", timeout)
            shutil.copyfile(source / "LICENSE", staged / "LICENSE")
            (staged / "board.txt").write_text(f"{rows} {columns}\n")
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"compiling Pons failed: {error.stderr}") from error
        except (OSError, subprocess.TimeoutExpired, tarfile.TarError) as error:
            raise RuntimeError(
                f"Pons setup failed; internet access, make and a C++ compiler are required: {error}"
            ) from error

        try:
            staged.rename(directory)
        except OSError:
            if not is_complete_installation(directory, rows, columns):
                raise
    return executable


def built_for(directory: Path, rows: int, columns: int) -> bool | None:
    """Whether the build in ``directory`` targets this board, or None if it records no board."""
    board = directory / "board.txt"
    return board.read_text().strip() == f"{rows} {columns}" if board.is_file() else None


def is_complete_installation(directory: Path, rows: int, columns: int) -> bool:
    return (
        (directory / "c4solver").is_file() and built_for(directory, rows, columns) is True
        and ((rows, columns) != (6, 7) or (directory / "7x6.book").is_file())
    )


def _download_book(path: Path, timeout: float) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(BOOK_URL, timeout=timeout) as response, path.open("wb") as book:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            book.write(chunk)
    if digest.hexdigest() != BOOK_SHA256:
        raise OSError("Pons opening book checksum mismatch")
