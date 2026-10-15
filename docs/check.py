"""Check generated documentation links and anchors."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])


def main():
    build = ROOT / "docs/_build/html"
    pages = {}
    for path in build.rglob("*.html"):
        page = Links()
        page.feed(path.read_text())
        pages[path.resolve()] = page
    if not pages:
        raise SystemExit("Build the HTML documentation before checking links.")
    failures = []
    count = 0
    for source, page in pages.items():
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target = (source.parent / unquote(url.path)).resolve() if url.path else source
            if target.is_dir():
                target /= "index.html"
            count += 1
            if not target.exists():
                failures.append(f"{source.relative_to(build)}: missing {href}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                failures.append(f"{source.relative_to(build)}: missing anchor {href}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(pages)} HTML pages; {count} local links and anchors checked.")


if __name__ == "__main__":
    main()
