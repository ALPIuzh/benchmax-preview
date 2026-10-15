"""Remove the parts of a notebook that record who ran it rather than what it does.

Notebooks carry state from the machine that executed them, and Jupyter and
nbconvert write it back on every run, so it cannot be kept out by discipline.
The result is that two people who change nothing still produce a diff.

Outputs are deliberately kept. They are what makes a notebook readable on its
own, and every notebook this repository ships stores them.

Run over the paths given, or with no arguments over every notebook in the tree.
Exits non-zero when it changed something, which is what pre-commit expects.
With --check, report notebooks needing normalization without rewriting them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Kept as the interpreter's name alone. Jupyter records the full patch version
#: it happened to run under, which differs between contributors and says nothing
#: about the notebook.
LANGUAGE_INFO = {"name": "python"}

#: The kernel a notebook asks for, pinned so that whichever kernel a
#: contributor's editor happens to select does not become a diff.
KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}


def normalize(notebook: dict[str, Any]) -> dict[str, Any]:
    """Return the notebook with machine-specific state replaced by fixed values.

    Execution counts are renumbered from one in cell order, so a partial re-run
    that leaves them scattered does not show up as a change to every cell. Cell
    identifiers are left alone: nbformat requires them, and Jupyter keeps them
    stable unless cells are actually added or removed.

    Args:
        notebook: The parsed notebook.

    Returns:
        The same object, modified in place.
    """

    notebook.get("metadata", {}).update(
        language_info=dict(LANGUAGE_INFO), kernelspec=dict(KERNELSPEC)
    )

    counter = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            counter += 1
            cell["execution_count"] = counter
            for output in cell.get("outputs", []):
                if "execution_count" in output:
                    output["execution_count"] = counter
        else:
            cell["execution_count"] = None
    return notebook


def normalize_file(path: Path, *, check: bool = False) -> bool:
    """Normalize one notebook, or only check whether normalization is needed.

    Returns:
        Whether the file needs normalization (and was rewritten unless checking).
    """

    original = path.read_text(encoding="utf-8")
    notebook = json.loads(original)
    normalized = json.dumps(normalize(notebook), indent=1, ensure_ascii=False)
    normalized = normalized.rstrip("\n") + "\n"
    if normalized == original:
        return False
    if not check:
        path.write_text(normalized, encoding="utf-8")
    return True


def main(arguments: list[str]) -> int:
    """Normalize the notebooks named, or every notebook in the tree."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report changes needed without rewriting files")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(arguments)
    paths = args.paths or sorted(
        path
        for path in Path().rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts and ".venv" not in path.parts
    )
    changed = [path for path in paths if normalize_file(path, check=args.check)]
    for path in changed:
        print(f"{'needs normalization' if args.check else 'normalized'} {path}")
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
