"""Sphinx configuration for the BenchMAX documentation."""

import json
import os
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Stage generated copies instead of a symlink: Sphinx resolves symlinks outside its
# source tree, breaking relative links between notebooks. Originals stay in notebooks/.
notebook_sources = ROOT / "notebooks"
notebook_docs = ROOT / "docs" / "notebooks"
notebook_docs.mkdir(exist_ok=True)
for stale in notebook_docs.glob("*.ipynb"):
    if not (notebook_sources / stale.name).exists():
        stale.unlink()
for notebook in notebook_sources.glob("*.ipynb"):
    content = json.loads(notebook.read_text())
    for cell in content["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = [line.replace("](../docs/", "](../") for line in cell["source"]]
    (notebook_docs / notebook.name).write_text(json.dumps(content, indent=1, ensure_ascii=False) + "\n")

project = "BenchMAX"
author = "ALPI Lab"
copyright = "2026, ALPI Lab, University of Zurich"
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax", "sphinx_copybutton"]
myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence"]
myst_heading_anchors = 3
nb_execution_mode = "off"
nb_output_stderr = "warn"
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autoclass_content = "both"
# Decorated execution APIs expose synthetic signatures; source defaults cannot be matched to them.
autodoc_preserve_defaults = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_title = "BenchMAX"
repository_url = "https://github.com/ALPIuzh/benchmax-preview"
repository_branch = "main"
html_context = {"repository_url": repository_url, "repository_branch": repository_branch}
html_static_path = ["_static"]
html_css_files = ["benchmax.css"]
html_js_files = ["api-members.js"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "README.md"]
html_show_sourcelink = False
html_sidebars = {"**": [
    "sidebar/brand.html", "sidebar/search.html", "sidebar/scroll-start.html", "sidebar/navigation.html",
    "sidebar/ethical-ads.html", "sidebar/scroll-end.html", "sidebar/variant-selector.html", "sidebar/lab.html",
]}
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#244e83",
        "color-brand-content": "#244e83",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f6f7f9",
        "color-foreground-primary": "#20252e",
        "color-foreground-secondary": "#525a67",
        "font-stack": '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        "font-stack--monospace": '"SFMono-Regular", Consolas, "Liberation Mono", monospace',
    },
    "dark_css_variables": {
        "color-brand-primary": "#a4bfe3",
        "color-brand-content": "#a4bfe3",
        "color-background-primary": "#141820",
        "color-background-secondary": "#1b202b",
        "color-foreground-primary": "#edf0f7",
        "color-foreground-secondary": "#b1bacb",
    },
    "footer_icons": [],
}
pygments_style = "friendly"
pygments_dark_style = "native"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Copy executable inputs, not printed notebook results.
copybutton_selector = ".highlight pre:not(.cell_output pre)"


def module_readme(module):
    return ROOT / "src" / module.replace(".", "/") / "README.md"


def resolve_repository_links(app, docname, source):
    """Reuse module READMEs and resolve their links for the documentation site."""
    if app.env.doc2path(docname).endswith(".md"):
        source[0] = source[0].replace("{{ repository_url }}", app.config.html_context["repository_url"])
        page = Path(app.env.doc2path(docname))

        def readme(match):
            path = module_readme(match[1])
            app.env.note_dependency(str(path))
            content = path.read_text().split("\n", 1)[1].strip()

            def link(match):
                target = urlsplit(match[1])
                if target.scheme or target.netloc or not target.path:
                    return match[0]

                resolved = (path.parent / target.path).resolve()
                destinations = {
                    ROOT / "README.md": ROOT / "docs/getting-started/installation.md",
                    ROOT / "src/benchmax/cli/README.md": ROOT / "docs/getting-started/cli.md",
                }
                for document in (ROOT / "docs/reference").rglob("*.md"):
                    text = document.read_text()
                    for module in re.findall(r"\{\{ readme: ([\w.]+) \}\}", text):
                        destinations[module_readme(module)] = document
                    for module in re.findall(r"^\.\. automodule:: ([\w.]+)", text, re.MULTILINE):
                        module_path = ROOT / "src" / module.replace(".", "/")
                        module_path = module_path / "__init__.py" if module_path.is_dir() else module_path.with_suffix(".py")
                        destinations[module_path] = document
                destination = destinations.get(resolved)
                if resolved.is_relative_to(ROOT / "docs"):
                    destination = resolved
                if resolved.is_relative_to(ROOT / "notebooks"):
                    destination = ROOT / "docs/notebooks" / resolved.name

                if destination:
                    href = Path(os.path.relpath(destination, page.parent)).as_posix()
                    if target.fragment and resolved != ROOT / "README.md":
                        fragment = target.fragment
                        if resolved == ROOT / "src/benchmax/cli/README.md" and fragment == "policy-stores":
                            fragment = "browse-policy-stores"
                        href += "#" + fragment
                else:
                    context = app.config.html_context
                    href = (f"{context['repository_url']}/blob/{context['repository_branch']}/"
                            f"{resolved.relative_to(ROOT).as_posix()}")
                    if target.fragment:
                        href += "#" + target.fragment
                return "](" + href + ")"

            return re.sub(r"\]\(([^)]+)\)", link, content)

        source[0] = re.sub(r"\{\{ readme: ([\w.]+) \}\}", readme, source[0])


def add_api_heading(app, what, name, obj, options, lines):
    """Separate module guides from their generated member documentation."""
    guides = {"benchmax.policy", "benchmax.providers", "benchmax.experts", "benchmax.seeding",
              "benchmax.execution.jobs.reference"}
    if what == "module" and name in guides:
        lines.extend(["", "Python API", "----------", ""])


def prepare_syntax_styles(app, pagename, templatename, context, doctree):
    # Furo normally writes this after HTML generation, too late for Sphinx's asset hash.
    from furo import get_pygments_stylesheet

    stylesheet = Path(app.outdir) / "_static" / "pygments.css"
    content = get_pygments_stylesheet()
    if not stylesheet.exists() or stylesheet.read_text() != content:
        stylesheet.parent.mkdir(parents=True, exist_ok=True)
        stylesheet.write_text(content)


def setup(app):
    app.connect("source-read", resolve_repository_links)
    app.connect("html-page-context", prepare_syntax_styles)
    app.connect("autodoc-process-docstring", add_api_heading)
