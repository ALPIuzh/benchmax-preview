"""Obfuscate saved notebook outputs; never execute cells or change local settings."""

from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = {
    "host": "demo-cluster", "user": "demo_user", "account": "example_project", "partition": "standard",
    "workspace": "/shared/example_project/benchmax", "runs": "/scratch/demo_user/benchmax/runs",
    "artifacts": "/shared/example_project/benchmax/artifacts",
    "python": "/shared/example_project/benchmax/.venv/bin/python", "control_path": "/tmp/benchmax-demo-%C",
}


def obfuscate(notebook: dict, configuration: dict) -> None:
    replacements = {}
    for field, replacement in PUBLIC.items():
        value = configuration["target"].get(field)
        if isinstance(value, str) and value and value != replacement:
            replacements[value] = replacement

    replacements.update(configuration.get("replacements", {}))
    if any(not key or not isinstance(value, str) for key, value in replacements.items()):
        raise ValueError("replacement mappings require nonempty keys and string values")

    # HTML output escapes values independently of the surrounding notebook JSON.
    replacements.update({escape(key): escape(value) for key, value in list(replacements.items())})
    pattern = re.compile("|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True)))
    identities: dict[str, str] = {}
    slurm_ids: dict[str, str] = {}
    timestamps: dict[str, str] = {}

    def text(value: str) -> str:
        # Preserve diagnostic package paths while hiding local checkout locations.
        value = re.sub(r"/(?:Users|home)/[^\s<>\"\']+?/\.venv/", "/workspace/benchmax/.venv/", value)
        value = re.sub(r"/(?:Users|home)/[^/\s<>\"\']+", "/home/demo_user", value)
        if replacements:
            value = pattern.sub(lambda match: replacements[match[0]], value)

        def job(match: re.Match) -> str:
            original = match[0]
            if original not in identities:
                identities[original] = f"{match[1]}-{len(identities) + 1:016x}"
            return identities[original]

        def slurm(match: re.Match) -> str:
            original = match[2]
            if original not in slurm_ids:
                slurm_ids[original] = str(100001 + len(slurm_ids))
            return match[1] + slurm_ids[original]

        def timestamp(match: re.Match) -> str:
            original = match[0]
            if original not in timestamps:
                timestamps[original] = f"2000-01-01T00:{len(timestamps) // 60:02d}:{len(timestamps) % 60:02d}+00:00"
            return timestamps[original]

        value = re.sub(r"\b(training|evaluation)-[0-9a-f]{16}\b", job, value)
        value = re.sub(r"(Slurm job:\s*)(\d+)\b", slurm, value)
        return re.sub(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", timestamp, value)

    def walk(value):
        if isinstance(value, str):
            return text(value)
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            for mime in ("text/plain", "text/html"):
                if mime in output.get("data", {}):
                    output["data"][mime] = walk(output["data"][mime])
            for key in ("text", "traceback", "evalue"):
                if key in output:
                    output[key] = walk(output[key])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path, default=sorted((ROOT / "notebooks").glob("*.ipynb")))
    args = parser.parse_args()
    try:
        configuration = json.loads((ROOT / ".local/slurm-demo.json").read_text())
    except FileNotFoundError:
        if any(path.name == "slurm_demo.ipynb" for path in args.paths):
            parser.exit(1, "Missing .local/slurm-demo.json; see Setup in notebooks/slurm_demo.ipynb.\n")
        configuration = {"target": {}}

    changed = False
    for path in args.paths:
        original = path.read_text()
        notebook = json.loads(original)
        obfuscate(notebook, configuration)
        rendered = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
        if rendered != original:
            changed = True
            if not args.check:
                path.write_text(rendered)
            print(f"{'Needs obfuscation' if args.check else 'Obfuscated; review and stage'}: {path}")
    return int(changed)


if __name__ == "__main__":
    raise SystemExit(main())
