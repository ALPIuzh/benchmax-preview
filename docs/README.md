# Working on the documentation

The site uses Sphinx, MyST-NB (Markdown and notebooks), and Furo. The generated HTML stays in `docs/_build/`.

Sub-folders correspond to sections of the main TOC.
Module pages in `reference/` use `automodule` while public members and docstrings come directly from the code.
All README links to documented Python modules resolve to their API pages on the website.

## Build and preview

From the repository root, install the locked docs environment without compiling BenchMAX's native extension:

```sh
uv sync --locked --python 3.12 --group docs --group dev --no-default-groups --no-install-project
uv run --no-sync python .github/scripts/normalize_notebooks.py --check notebooks/*.ipynb
uv run --no-sync ruff check --no-fix notebooks/*.ipynb
uv run --no-sync sphinx-build -b html -W --keep-going docs docs/_build/html
uv run --no-sync python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open http://127.0.0.1:8000. The configuration imports the Python source from `src/`, so documenting public interfaces does not require the native backend or optional neural-training dependencies.

The docs-only sync changes the selected environment. In an existing development environment, use `uv sync --group docs` (and retain any desired extras) instead.

## Preview GitHub Pages with Docker

From the repository root:

```sh
docker build -f docs/Dockerfile -t benchmax-pages:local .
docker run --rm -d --name benchmax-pages-preview -p 127.0.0.1:8080:80 benchmax-pages:local
```

Open http://localhost:8080/benchmax-preview/. The container builds with Python 3.12 and the locked dependencies,
checks notebook normalization and Ruff without modifying files, then runs the strict Sphinx build and documentation checks before serving
the generated site with Nginx. The `/benchmax-preview/` prefix matches the future GitHub Pages project URL.

Stop the preview with `docker stop benchmax-pages-preview`. Rebuild the image after editing the documentation.

## Check links

Examples live directly in Markdown code blocks. After building the site, check local links and anchors:

```sh
uv run --no-sync python docs/check.py
```

The documentation workflow runs the base checks and uploads HTML as a build artifact on pull requests. Successful builds on `main` also deploy to GitHub Pages through the `github-pages` environment. Manual runs on `main` can republish the site.

In repository Settings → Pages, select **GitHub Actions** as the source. GitHub Pages must be enabled before deployment, and private repositories require a plan that supports Pages. The site URL is reported by the deployment job.

## Notebook quality gate

All notebook sources live in root `notebooks/`, with saved outputs in their `.ipynb` files. The Sphinx configuration stages generated copies in ignored `docs/notebooks/` so MyST-NB can resolve local links. Edit only the originals. Each rendered notebook automatically includes a **View notebook on GitHub** link to its original source, and its title and introduction come from the notebook itself. Links from notebooks to `../docs/` are adjusted in the generated copies. `nb_execution_mode = "off"` keeps website rendering static. Notebooks must be ready before the build because CI and Docker render saved outputs without executing or modifying notebooks.

Before rendering, CI and Docker check all root notebooks with the same normalization rules and Ruff configuration as pre-commit:

```sh
uv run --no-sync python .github/scripts/normalize_notebooks.py --check notebooks/*.ipynb
uv run --no-sync ruff check --no-fix notebooks/*.ipynb
```

These commands fail with the affected filenames without rewriting files. To fix issues locally:

```sh
uv run --no-sync python .github/scripts/normalize_notebooks.py notebooks/*.ipynb
uv run --no-sync ruff check --fix notebooks/*.ipynb
```

The normalizer preserves outputs and cell IDs, standardizes kernel/language metadata, and renumbers execution counts. `docs/check.py` checks generated HTML links and anchors.

## Editorial and visual direction

- Lead with a runnable task, then explain the output and assumptions.
- Keep each small example self-contained, or explicitly say when it continues a previous block.
- Use notebooks for longer experiments and exploration, but keep docs useful without opening one.
- Define metric conventions, payoff units, supported cases, and computational limitations.
- Draw API signatures from the source. Include existing guides instead of maintaining duplicate copies.
- Preserve Furo's navigation, search, light/dark controls, and mobile layout.
- Use ALPI blue, restrained typography, and a custom homepage, and avoid decorative clutter.
- Keep branding in `_templates/` and `_static/benchmax.css`.
- Use `$...$` for inline math and `equation*` for displayed equations, with each equation body on one source line.

## Repository links

Set `repository_url` and `repository_branch` in `docs/conf.py` when the repository or source branch changes. The notebook source links are generated from these settings and the notebook filename. Markdown pages can use `{{ repository_url }}` in source links and commands, which Sphinx expands during the build. The footer uses the same value. Links between documentation pages and notebooks stay relative.
