"""Explicit installation of checksummed Stockfish 17.1 release binaries."""

from __future__ import annotations

import hashlib
import math
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from benchmax.utils.files import cache_directory

VERSION = "17.1"
RELEASE_URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1"
# Baseline x86 binaries avoid requiring AVX2/BMI2 on the installation machine.
ASSETS = {
    ("Darwin", "arm64"): (
        "stockfish-macos-m1-apple-silicon", "4e23165eb8f353c221ff7ab6716f0a160c3993dadf90d0c0ad982a7ade4091c9"
    ),
    ("Darwin", "x86_64"): (
        "stockfish-macos-x86-64", "067f100a31d3d6f0e45826e6495513c3e0518044e006caa3477993689049d658"
    ),
    ("Linux", "x86_64"): (
        "stockfish-ubuntu-x86-64", "4dafdd04f71e70755a327b5be258937b281e60ba87bc0a5801399908240d4a73"
    ),
}


def default_executable() -> Path:
    host = f"{platform.system().lower()}-{platform.machine().lower()}"
    return cache_directory() / "stockfish" / VERSION / host / "stockfish"


def install(*, directory: Path | None = None, timeout: float = 300) -> Path:
    """Install a pinned official binary. Unsupported hosts can supply their own."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")

    key = (platform.system(), platform.machine())
    if key not in ASSETS:
        raise RuntimeError(f"No bundled Stockfish recipe for {key}; install Stockfish and set opponent.executable")

    name, checksum = ASSETS[key]
    target = Path(directory).expanduser().resolve() / "stockfish" if directory is not None else default_executable()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".install-") as temp:
        archive = Path(temp) / "stockfish.tar"
        url = f"{RELEASE_URL}/{name}.tar"
        with urllib.request.urlopen(url, timeout=timeout) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)

        with archive.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != checksum:
            raise RuntimeError("Stockfish archive checksum mismatch")

        with tarfile.open(archive) as source:
            members = [member for member in source.getmembers() if member.isfile() and Path(member.name).name == name]
            if len(members) != 1:
                raise RuntimeError("Stockfish archive does not contain the expected executable")

            with source.extractfile(members[0]) as binary:
                staged = Path(temp) / "stockfish"
                with staged.open("wb") as output:
                    shutil.copyfileobj(binary, output)

        staged.chmod(0o755)
        staged.replace(target)
    return target
