import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests import REPOSITORY

sys.path.insert(0, str(REPOSITORY / ".github" / "scripts"))

from normalize_notebooks import normalize, normalize_file


def notebook(**metadata):
    return {
        "cells": [
            {"cell_type": "markdown", "id": "keep-me", "source": ["text"]},
            {
                "cell_type": "code",
                "id": "also-keep",
                "source": ["1 + 1"],
                "execution_count": 7,
                "outputs": [{"output_type": "execute_result", "execution_count": 7}],
            },
            {"cell_type": "code", "id": "unrun", "source": ["2 + 2"], "execution_count": None, "outputs": []},
        ],
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_normalization_preserves_content_and_removes_machine_metadata() -> None:
    result = normalize(
        notebook(
            language_info={"name": "python", "version": "3.11.8"}, kernelspec={"display_name": "personal (3.12.11)"}
        )
    )
    assert result["metadata"] == {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }
    assert [cell["id"] for cell in result["cells"]] == ["keep-me", "also-keep", "unrun"]
    assert [cell["source"] for cell in result["cells"]] == [["text"], ["1 + 1"], ["2 + 2"]]
    assert result["cells"][1]["execution_count"] == 1
    assert result["cells"][1]["outputs"] == [{"output_type": "execute_result", "execution_count": 1}]
    assert result["cells"][2]["execution_count"] is None
    assert result["cells"][2]["outputs"] == []


def test_normalizing_twice_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "example.ipynb"
    path.write_text(json.dumps(notebook(language_info={"name": "python", "version": "3.11.8"})), encoding="utf-8")
    assert normalize_file(path) is True
    assert normalize_file(path) is False


NOTEBOOKS = subprocess.check_output(["git", "ls-files", "-z", "--", "*.ipynb"], cwd=REPOSITORY).decode().split("\0")


@pytest.mark.parametrize("name", [name for name in NOTEBOOKS if name])
def test_committed_notebooks_are_normalized(name: str) -> None:
    original = (REPOSITORY / name).read_text(encoding="utf-8")
    expected = json.dumps(normalize(json.loads(original)), indent=1, ensure_ascii=False).rstrip("\n") + "\n"
    assert original == expected, f"Run python .github/scripts/normalize_notebooks.py {name}"
