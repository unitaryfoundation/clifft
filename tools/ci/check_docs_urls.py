"""Check rendered documentation URLs before publishing a MkDocs build."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree


class PageLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canonicals: list[str] = []
        self.links: list[str] = []
        self.resources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"img", "script"} and attributes.get("src"):
            self.resources.append(str(attributes["src"]))
        href = attributes.get("href")
        if not href:
            return
        if tag == "link" and "canonical" in (attributes.get("rel") or "").split():
            self.canonicals.append(href)
        elif tag == "link" and attributes.get("rel") in {"stylesheet", "icon"}:
            self.resources.append(href)
        if tag == "a":
            self.links.append(href)


def check_site(site_dir: Path, site_url: str) -> None:
    site_dir = site_dir.resolve()
    site_url = site_url.rstrip("/") + "/"
    expected = urlsplit(site_url)
    if expected.scheme not in {"http", "https"} or not expected.netloc:
        raise ValueError("site_url must be an absolute HTTP or HTTPS URL")

    def check_page_url(url: str, source: Path) -> None:
        if not url.startswith(site_url):
            raise ValueError(f"{source}: URL is outside {site_url}: {url}")
        relative = unquote(urlsplit(url).path[len(expected.path) :])
        target = (site_dir / relative).resolve()
        if not target.is_relative_to(site_dir):
            raise ValueError(f"{source}: URL escapes the site directory: {url}")
        if urlsplit(url).path.endswith("/"):
            target /= "index.html"
        if not target.is_file():
            raise ValueError(f"{source}: URL has no built page: {url}")

    canonical_count = 0
    playground_count = 0
    for page in sorted(site_dir.rglob("*.html")):
        relative = page.relative_to(site_dir).as_posix()
        # The separately built application has no documentation canonical URL.
        if relative.startswith("playground/"):
            continue
        page_url = urljoin(site_url, relative.removesuffix("index.html"))
        parsed = PageLinks()
        parsed.feed(page.read_text())
        for canonical in parsed.canonicals:
            check_page_url(urljoin(page_url, canonical), page)
            canonical_count += 1
        for resource in parsed.resources:
            resolved_resource = urljoin(page_url, resource)
            if resolved_resource.startswith(site_url):
                check_page_url(resolved_resource, page)
        for href in parsed.links:
            resolved = urlsplit(urljoin(page_url, href))
            if resolved.netloc != expected.netloc or not resolved.path.endswith("/playground/"):
                continue
            if (resolved.scheme, resolved.path) != (
                expected.scheme,
                expected.path + "playground/",
            ):
                raise ValueError(f"{page}: Playground URL uses the wrong site prefix: {href}")
            playground_count += 1

    sitemap = site_dir / "sitemap.xml"
    locations = ElementTree.parse(sitemap).findall(
        ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
    )
    if not canonical_count or not playground_count or not locations:
        raise ValueError("build must contain canonical URLs, Playground links, and sitemap entries")
    for location in locations:
        check_page_url(location.text or "", sitemap)
    print(
        f"Verified {canonical_count} canonical URLs, {playground_count} Playground links, "
        f"and {len(locations)} sitemap entries for {site_url}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_dir", type=Path)
    parser.add_argument(
        "site_url", help="expected published URL including the version or preview path"
    )
    args = parser.parse_args()
    check_site(args.site_dir, args.site_url)
