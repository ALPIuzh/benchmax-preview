"""Explicit seeds make experiments reproducible: they let you repeat the same random choices and compare methods under
controlled conditions. BenchMAX derives separate random streams from a root seed and stable names, so their values do
not depend on the order in which streams are requested. Matching seeds alone does not guarantee identical results across
library versions or nondeterministic hardware operations.

Fix the seed
------------

.. code-block:: python

    from benchmax import load_game
    from benchmax.training import train

    game = load_game("kuhn_poker")
    policy = train(game, "cfr", iterations=1_000, seed=7)
    env = game.new_env()
    time_step = env.reset(seed=7)

For your own sampling, use an isolated generator instead of changing global random state:

.. code-block:: python

    from benchmax.seeding import derive_seed, numpy_generator

    rng = numpy_generator(derive_seed(7, "demonstrations"))

Catch missing seeds
-------------------

When an optional seed is omitted, BenchMAX generates one without warning. Use ``forbid_entropy()`` during development to
turn these omissions into errors:

.. code-block:: python

    from benchmax.seeding import forbid_entropy

    with forbid_entropy():
        train(game, "cfr", iterations=1_000)  # Raises EntropyDisabledError: pass seed=7.

This only checks randomness managed by BenchMAX, not independent random generators in your code or third-party
libraries.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random as stdlib_random
import secrets
import threading
from collections.abc import Iterator

import numpy as np

SEED_LIMIT = 2**31
SEED_SCHEME = "benchmax-sha256-v1"


class EntropyDisabledError(RuntimeError):
    """Fresh entropy was requested while strict mode was active."""


_entropy_lock = threading.RLock()
_entropy_forbidden = 0


def validate_seed(seed: object, *, label: str = "seed") -> int:
    """Return a valid portable seed, rejecting booleans and oversized values."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"{label} must be an integer")

    if not 0 <= seed < SEED_LIMIT:
        raise ValueError(f"{label} must lie in [0, 2**31)")

    return seed


def validate_seed_scheme(scheme: object) -> str:
    """Reject seed recipes that this version of BenchMAX cannot replay."""

    if scheme != SEED_SCHEME:
        raise ValueError(f"unsupported seed scheme {scheme!r}; expected {SEED_SCHEME!r}")

    return scheme


def resolve_seed(seed: int | None, *, purpose: str) -> int:
    """Resolve an optional public seed to one concrete, recordable value."""

    if seed is not None:
        return validate_seed(seed)

    with _entropy_lock:
        if _entropy_forbidden:
            raise EntropyDisabledError(
                f"{purpose} requested fresh entropy because no seed was supplied; pass an explicit seed"
            )

        return secrets.randbelow(SEED_LIMIT)


@contextlib.contextmanager
def forbid_entropy() -> Iterator[None]:
    """Make omitted BenchMAX seeds fail anywhere in this process.

    This is primarily a test aid. It covers all BenchMAX-owned
    randomness because :func:`resolve_seed` is its only fresh-entropy source.
    It cannot police arbitrary native code in third-party libraries.
    """

    global _entropy_forbidden

    with _entropy_lock:
        _entropy_forbidden += 1

    try:
        yield
    finally:
        with _entropy_lock:
            _entropy_forbidden -= 1


def derive_seed(seed: int, *path: str | int, scheme: str = SEED_SCHEME) -> int:
    """Derive a seed from a stable semantic path, independently of call order."""

    seed = validate_seed(seed)
    validate_seed_scheme(scheme)

    if not path:
        raise ValueError("a derived seed needs a non-empty path")

    encoded_path: list[tuple[str, str | int]] = []

    for part in path:
        if isinstance(part, str):
            if not part:
                raise ValueError("seed path strings must be non-empty")

            encoded_path.append(("string", part))
        elif isinstance(part, int) and not isinstance(part, bool):
            encoded_path.append(("integer", part))
        else:
            raise TypeError("seed path parts must be strings or integers")

    # The tagged JSON representation makes strings and integers unambiguous.
    # This exact recipe is the persisted v1 contract and must not be edited.
    payload = json.dumps([seed, encoded_path], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(scheme.encode("ascii") + b"\0" + payload).digest()
    return int.from_bytes(digest[:8], "big") % SEED_LIMIT


def numpy_generator(seed: int) -> np.random.Generator:
    """Create a NumPy generator with an explicitly pinned bit generator."""

    return np.random.Generator(np.random.PCG64(validate_seed(seed)))


def python_generator(seed: int) -> stdlib_random.Random:
    """Create an isolated standard-library generator without fresh entropy."""

    return stdlib_random.Random(validate_seed(seed))


def seed_process(seed: int) -> None:
    """Seed Python and NumPy's legacy process-global generators."""

    seed = validate_seed(seed)
    stdlib_random.seed(seed, version=2)
    np.random.seed(seed)
