# UCC Documentation Site Plan

## Goal

Ship a documentation website at **https://unitaryfoundation.github.io/ucc-next/** using
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/). The site must:

1. Serve as the primary user-facing documentation for the `ucc` Python package.
2. Render LaTeX math (KaTeX) for the theoretical foundations.
3. Embed the existing Compiler Explorer as a live, interactive page.
4. Deploy automatically from `main` via GitHub Actions to the existing `gh-pages` branch.
5. Coexist with the PR preview system already using `gh-pages/pr-preview/`.

---

## Current State

- **GitHub Pages** is active, source = `gh-pages` branch, root `/`.
- `gh-pages` currently only contains `pr-preview/` directories (explorer PR previews).
- No docs site exists yet.
- The explorer builds to `explorer/dist/` as a standalone Vite SPA.
- Python API surface is small (~15 public symbols in `ucc/__init__.py`).
- Rich design docs exist in `design/` (overview, architecture, data structures, gates, etc.).
- `DEVELOPMENT.md` has comprehensive build/test/coverage instructions.

---

## Architecture Decision: MkDocs root + Explorer iframe

MkDocs Material will own the `gh-pages` root. The explorer will be:
- Built as a Vite SPA and copied into the MkDocs output at `explorer/` subpath.
- Embedded in a docs page via a full-viewport `<iframe>` (or linked directly).
- This avoids any bundler conflict between Vite and MkDocs.

The `pr-preview/` directory continues to be managed by `rossjrw/pr-preview-action`
and is unaffected (it writes directly to `gh-pages` branch, not to the MkDocs build).

---

## Phase 1 — Scaffolding & CI (Initial Scope)

### 1.1 Add MkDocs Material as a dev dependency

- Add to `pyproject.toml` `[dependency-groups]` dev group:
  - `mkdocs-material>=9.6`
  - `mkdocs-minify-plugin>=0.8` (HTML minification)
  - `pymdown-extensions>=10.7` (for math, tasklists, etc.)
- `uv sync` to update lockfile.

### 1.2 Create `mkdocs.yml` at repo root

```yaml
site_name: UCC — Unitary Compiler Collection
site_url: https://unitaryfoundation.github.io/ucc-next/
repo_url: https://github.com/unitaryfoundation/ucc-next
repo_name: unitaryfoundation/ucc-next

theme:
  name: material
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: deep purple
      accent: amber
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: deep purple
      accent: amber
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.instant
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - content.code.copy
    - search.highlight
    - toc.follow

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details
  - attr_list
  - md_in_html
  - toc:
      permalink: true

extra_javascript:
  - https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js
  - https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js
  - javascripts/katex.js

extra_css:
  - https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css

plugins:
  - search
  - minify:
      minify_html: true

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
  - User Guide:
    - Compiling Circuits: guide/compiling.md
    - Simulation: guide/simulation.md
    - Supported Gates: guide/gates.md
  - Theory:
    - Overview: theory/overview.md
    - Architecture: theory/architecture.md
  - Explorer: explorer.md
  - Development:
    - Contributing: development/contributing.md
    - Building from Source: development/building.md
```

### 1.3 Create `docs/` directory with initial pages

```
docs/
  index.md                      # Landing page (from README + expansion)
  getting-started/
    installation.md              # pip install ucc, platform support
    quickstart.md                # First circuit, compile, simulate, sample
  guide/
    compiling.md                 # Parse -> HIR -> Compile -> Program
    simulation.md                # execute(), get_statevector(), sample()
    gates.md                     # Adapted from design/gates.md
  theory/
    overview.md                  # Adapted from design/overview.md (factored state, pipeline)
    architecture.md              # Adapted from design/architecture.md
  development/
    contributing.md              # Pre-commit, testing, branching
    building.md                  # Adapted from DEVELOPMENT.md
  explorer.md                    # Full-viewport iframe to ./explorer/
  javascripts/
    katex.js                     # KaTeX auto-render init script
```

Content strategy:
- **`index.md`**: Hero section, badges, 5-line code example, feature highlights, links.
- **`getting-started/`**: Practical, copy-paste-friendly. Minimal theory.
- **`guide/`**: Python API walkthrough with code examples. `gates.md` is a formatted
  version of `design/gates.md` showing the supported gate set and Stim decompositions.
- **`theory/`**: Adapted from design docs. Math-heavy, uses KaTeX. Target audience:
  researchers who want to understand the factored state representation.
- **`explorer.md`**: Brief description of the explorer + a prominent link/button to
  `explorer/` which opens the standalone SPA. No iframe — the explorer works best
  full-screen.
- **`development/`**: Extracted from `DEVELOPMENT.md`, reorganized for the docs nav.

### 1.4 KaTeX init script

`docs/javascripts/katex.js`:
```javascript
document.addEventListener("DOMContentLoaded", function () {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
  });
});
```

### 1.5 GitHub Actions: production deploy (`docs.yml`)

New workflow `.github/workflows/docs.yml`:

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4

      # --- Build explorer SPA ---
      - name: Build Wasm module
        run: |
          docker run --rm \
            -u "$(id -u):$(id -g)" \
            -v "${{ github.workspace }}:/src" -w /src \
            emscripten/emsdk:3.1.74 \
            bash -c '
              emcmake cmake -B build-wasm -S src/wasm \
                -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_QUIET=ON && \
              cmake --build build-wasm -j$(nproc)'
          mkdir -p explorer/public
          cp build-wasm/ucc_wasm.js explorer/public/
          cp build-wasm/ucc_wasm.wasm explorer/public/

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: explorer/package-lock.json

      - name: Build explorer
        run: |
          cd explorer && npm ci
          VITE_BASE_PATH=/ucc-next/explorer/ npm run build

      # --- Build MkDocs ---
      - name: Install uv
        uses: astral-sh/setup-uv@v6

      - name: Build docs
        run: uv run mkdocs build --strict

      # --- Merge: copy explorer into mkdocs output ---
      - name: Copy explorer into site
        run: cp -r explorer/dist site/explorer

      # --- Deploy to gh-pages (preserving pr-preview/) ---
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
          keep_files: true  # preserves pr-preview/ on gh-pages
```

Key points:
- `keep_files: true` ensures `pr-preview/` directories aren't wiped.
- Explorer is built with `VITE_BASE_PATH=/ucc-next/explorer/` so asset paths work.
- MkDocs outputs to `site/`, then explorer is copied to `site/explorer/`.
- The combined `site/` is deployed to `gh-pages` root.

### 1.6 GitHub Actions: PR preview (update `pr_preview.yml`)

Replace the current explorer-only PR preview with a **full-site preview**. Any PR
gets a complete docs+explorer preview. The wasm build is already the slow part (~5 min);
mkdocs adds ~10 seconds.

The updated `pr_preview.yml` build job adds these steps after the existing wasm+explorer
build:

```yaml
# Same wasm + explorer build steps as today, but with PR base path for explorer:
- name: Build explorer
  run: |
    cd explorer && npm ci
    VITE_BASE_PATH=/ucc-next/pr-preview/pr-${{ github.event.number }}/explorer/ npm run build

# Then build mkdocs and merge:
- name: Install uv
  uses: astral-sh/setup-uv@v6

- name: Build docs
  env:
    SITE_URL: https://unitaryfoundation.github.io/ucc-next/pr-preview/pr-${{ github.event.number }}/
  run: uv run mkdocs build --strict

- name: Copy explorer into site
  run: cp -r explorer/dist site/explorer

# Upload site/ as artifact instead of explorer/dist
```

Benefits:
- Contributors editing docs get a live preview link in the PR comment.
- Explorer changes are previewed in context of the full site.
- One preview system instead of two.
- Preview URL shows the real site structure: nav, theme, math rendering.

### 1.7 Justfile recipes

```just
# Serve docs locally with live reload
docs-serve:
  uv run mkdocs serve -a 0.0.0.0:8000

# Build docs to site/
docs-build:
  uv run mkdocs build --strict
```

### 1.8 Update `.gitignore`

Add `site/` to `.gitignore` (MkDocs build output).

---

## Phase 2 — Future Enhancements (Out of Initial Scope)

These are deferred but worth noting:

- **API Reference (auto-generated)**: Use `mkdocstrings[python]` to auto-generate
  API docs from Python docstrings. Requires adding docstrings to `__init__.py` exports.
  Blocked on: the C++ bindings (`_ucc_core`) don't have Python-visible docstrings yet.
- **Versioned docs**: `mike` for versioned documentation (v0.1, v0.2, etc.).
  Not needed until there are actual releases.
- **Search indexing of theory PDFs**: `design/theory.pdf` could be linked but not
  indexed. Not a priority.
- **Notebook integration**: `mkdocs-jupyter` to render `.ipynb` tutorials inline.
  Good for when tutorial notebooks exist.
- **Custom domain**: If `docs.ucc.dev` or similar is desired.

---

## Estimated Effort

| Task | Estimate |
|------|----------|
| 1.1 Dependencies | 5 min |
| 1.2 mkdocs.yml | 10 min |
| 1.3 docs/ content (8 pages) | 60-90 min |
| 1.4 KaTeX script | 5 min |
| 1.5 CI workflow | 15 min |
| 1.6 Justfile + gitignore | 5 min |
| **Total** | **~2 hours** |

---

## Decisions

1. **Color scheme**: `deep purple` + `amber` for now; can change later.
2. **Theory docs scope**: `overview.md` and `architecture.md` only. The opcode math spec
   (`phase1_math_spec.md`) is internal compiler detail — defer to development/contributor
   docs later.
3. **Explorer embedding**: Nav link to dedicated `/explorer/` page (no iframe). The
   explorer is a full SPA that works best standalone.
4. **Docs deploy independence**: Deploy independently on push to main. Branch protection
   already ensures CI passes before merge, so anything on main is tested. No need to
   couple the docs workflow to the test workflow.
