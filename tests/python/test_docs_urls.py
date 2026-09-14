"""Reject broken publication URLs even when MkDocs builds successfully."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_docs_urls", Path(__file__).parents[2] / "tools/ci/check_docs_urls.py"
)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def write_site(site: Path, base: str, broken: str | None = None) -> None:
    canonical = base + ("stable/" if broken == "canonical" else "")
    playground = base + ("stable/" if broken == "playground" else "") + "playground/"
    location = base + ("stable/" if broken == "sitemap" else "")
    (site / "index.html").write_text(
        f'<link rel="canonical" href="{canonical}"><a href="{playground}">Playground</a>'
    )
    (site / "sitemap.xml").write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{location}</loc></url></urlset>"
    )


@pytest.mark.parametrize("broken", ["canonical", "sitemap", "playground"])
def test_duplicate_publication_prefix_is_rejected(tmp_path, broken):
    base = "https://example.com/clifft/stable/"
    write_site(tmp_path, base, broken)
    with pytest.raises(ValueError, match="no built page|wrong site prefix"):
        checker.check_site(tmp_path, base)


@pytest.mark.parametrize("prefix", ["", "0.10.1/", "stable/", "dev/", "pr-preview/pr-123/"])
def test_valid_publication_prefixes_and_nested_pages(tmp_path, prefix):
    base = "https://example.com/clifft/" + prefix
    write_site(tmp_path, base)
    nested = tmp_path / "guide" / "example"
    nested.mkdir(parents=True)
    (nested / "index.html").write_text(
        f'<link rel="canonical" href="{base}guide/example/">'
        '<a href="../../playground/?url=https%3A%2F%2Fexample.com%2Fcircuit.stim">Try</a>'
    )
    redirect = tmp_path / "old"
    redirect.mkdir()
    (redirect / "index.html").write_text('<link rel="canonical" href="../guide/example/">')
    checker.check_site(tmp_path, base)


def test_empty_site_is_not_accepted(tmp_path):
    (tmp_path / "sitemap.xml").write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>'
    )
    with pytest.raises(ValueError, match="must contain"):
        checker.check_site(tmp_path, "https://example.com/clifft/")


def test_duplicate_asset_prefix_on_error_page_is_rejected(tmp_path):
    base = "https://example.com/clifft/stable/"
    write_site(tmp_path, base)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/site.css").write_text("body {}")
    (tmp_path / "404.html").write_text(
        '<link rel="stylesheet" href="/clifft/stable/stable/assets/site.css">'
    )
    with pytest.raises(ValueError, match="no built page"):
        checker.check_site(tmp_path, base)
