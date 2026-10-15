"""Describe stores and browse policy metadata without downloading weights."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from typing import Annotated

from cyclopts import App, Parameter

from benchmax.storage import HuggingFaceStore, LocalStore, StorageError, reference_store

from ._render import render_value, table
from .jobs import JsonFlag

app = App(name="store", help="Describe a store or browse its policies.", help_format="plaintext")
StoreLocation = Annotated[str | None, Parameter(
    help="Store location: local:PATH or huggingface:OWNER/BUCKET[/PREFIX]; default: reference store."
)]


def _store(use: str | None) -> LocalStore | HuggingFaceStore:
    if use is None:
        return reference_store()

    scheme, separator, location = use.partition(":")
    if not separator or not location or location.startswith("//"):
        raise ValueError("--use expects local:PATH or huggingface:OWNER/BUCKET[/PREFIX]")

    if scheme == "local":
        return LocalStore(location)

    if scheme == "huggingface":
        parts = location.split("/", 2)
        if len(parts) >= 2 and parts[0] and parts[1]:
            return HuggingFaceStore("/".join(parts[:2]), prefix=parts[2] if len(parts) == 3 else "")

    raise ValueError("--use expects local:PATH or huggingface:OWNER/BUCKET[/PREFIX]")


@app.command
def info(*, use: StoreLocation = None, as_json: JsonFlag = False) -> int:
    """Describe the selected store without contacting it or scanning its catalog."""

    try:
        store = _store(use)
    except (StorageError, ValueError) as error:
        print(f"benchmax: {error}", file=sys.stderr)
        return 1

    document = {"backend": "local" if isinstance(store, LocalStore) else "huggingface",
                "location": str(store.root) if isinstance(store, LocalStore) else store.bucket_id,
                "prefix": "" if isinstance(store, LocalStore) else store.prefix,
                "requirements": asdict(store.requirements)}
    if as_json:
        render_value(document, as_json=True)
    else:
        if use is None:
            print("Reference store for BenchMAX\n")

        rows = [[key, str(document[key]) or "(root)"] for key in ("backend", "location", "prefix")]
        rows += [[key, json.dumps(value)] for key, value in document["requirements"].items()]
        print("\n".join(table(["setting", "value"], rows, indent="")))

    return 0


@app.command
def show(*, use: StoreLocation = None, game: str | None = None, algorithm: str | None = None,
         seed: int | None = None, as_json: JsonFlag = False) -> int:
    """List policies across all configurations, optionally filtering by game, algorithm, and seed."""

    try:
        store = _store(use)
        references = store.find_policies(game, algorithm=algorithm, seed=seed,
                                         game_config="any", training_config="any")
        rows = []
        for reference in references:
            training = reference.training.to_document()
            configuration = {key: value for key, value in training.items() if key != "seed"}
            encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(encoded).hexdigest()
            rows.append({"id": reference.object.digest, "configuration": digest, "training": training})
    except (StorageError, ValueError, OSError) as error:
        print(f"benchmax: {error}", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: (row["training"]["game"]["name"], row["training"]["algorithm"]["name"],
                              row["training"]["seed"], row["id"]))
    if as_json:
        render_value(rows, as_json=True)
    else:
        body = [[row["id"][:12], row["training"]["game"]["name"], row["training"]["algorithm"]["name"],
                 str(row["training"]["seed"]), row["configuration"][:12]] for row in rows]
        print("\n".join(table(["policy", "game", "algorithm", "seed", "config"], body, indent="")))

    return 0

