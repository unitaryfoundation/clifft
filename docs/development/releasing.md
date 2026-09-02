<!--pytest-codeblocks:skipfile-->

# Releasing

Clifft uses [setuptools-scm](https://github.com/pypa/setuptools-scm) to
derive the package version from git tags. The release workflow builds
wheels for Linux (x86_64, aarch64), macOS (arm64), and Windows (amd64), then publishes to
PyPI via trusted publishers.

Linux wheels use the exact Clang version pinned by `CLIFFT_LINUX_WHEEL_CLANG_VERSION` in the
release workflow, the matching `lld` linker, and ThinLTO. Update that pin deliberately after
building and benchmarking representative Linux wheels; do not replace it with a floating
`latest` toolchain.

## Versioning

The version is determined automatically from git tags:

- **Tagged commits** (e.g., `v1.2.3`): version is `1.2.3`
- **Release-candidate tags** (e.g., `v1.3.0rc1`): version is `1.3.0rc1`
- **Development builds** (e.g., `1.2.4.dev3+g1a2b3c4`): the latest tag,
  commit distance, and hash determine the version. After `v1.3.0rc1`, the default
  setuptools-scm progression is `1.3.0rc2.devN+gHASH`.

There is no hardcoded version in `pyproject.toml`. The git tag is the single source of truth.

## Release candidates

Use a release candidate when the package must be installed from PyPI for validation before the
final release. Complete the changelog and docs-home preparation described below before tagging the
first candidate. Prepare the final release section, such as `## [1.3.0]`, rather than a separate rc
section; the GitHub prerelease uses that section for its notes. `git-cliff` writes the preparation
date into that heading, so update it to the actual publication date before creating the final tag.

Optionally run the manual TestPyPI workflow first with a unique development version such as
`1.3.0rc1.dev1`. Do not use `1.3.0rc1` for that smoke run: TestPyPI files are immutable, so the
tag-triggered workflow would skip the smoke artifacts instead of uploading the tagged artifacts.

Tag the prepared commit using the canonical Python release-candidate form:

```bash
git tag v1.3.0rc1
git push origin v1.3.0rc1
```

The tag builds and tests the release artifacts, publishes them to TestPyPI and PyPI, and creates a
GitHub prerelease. It does not publish versioned docs, update `stable`, or refresh the root
Playground. Install the published wheel explicitly for validation:

```bash
python -m pip install --only-binary=:all: "clifft==1.3.0rc1"
python -c "import clifft; print(clifft.__version__)"
```

After validation, update the existing final changelog section and make documentation-only changes
without prepending a duplicate section. `git-cliff` ignores rc tags, so an unreleased preview still
covers every commit since the previous final release. If runtime, packaging, compiler, or build
inputs change, increment the candidate tag (for example, `v1.3.0rc2`) and repeat validation.
Otherwise, create the final tag using the normal process below.

## Release process

Follow every step for a release made without a candidate. When finalizing a validated candidate,
update the already-prepared changelog and docs instead of generating them again.

### 1. Test on TestPyPI (optional but recommended)

Verify that wheels build and install correctly by running the release workflow manually.
Manual dispatch always publishes to TestPyPI only — it cannot publish to PyPI.

1. Go to **Actions** > **Release** > **Run workflow**
2. Select the branch (usually `main`)
3. Enter a unique `test_version` used only for this TestPyPI upload
4. Wait for builds to complete, then verify the exact version:

    ```bash
    TEST_VERSION=1.3.0.dev1
    pip install --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ "clifft==$TEST_VERSION"
    python -c "import clifft; print(clifft.__version__)"
    ```

### 2. Update the changelog

Generate the new release section using [git-cliff](https://git-cliff.org/):

```bash
git cliff --unreleased --tag vX.Y.Z --prepend CHANGELOG.md
```

This prepends only the unreleased changes since the previous tag and preserves
older hand-edited release sections. Do not use `-o CHANGELOG.md` for routine
releases unless you intentionally want to regenerate the entire changelog.
Generate this section before the first release candidate. After an rc tag exists, edit the prepared
section directly to avoid prepending a duplicate; rc tags do not truncate unreleased previews.

Keep each changelog paragraph and list item on one physical line, and use blank
lines only to separate Markdown blocks. The release workflow copies the
changelog section verbatim, and GitHub Releases renders ordinary line endings
inside prose as visible line breaks. Let the editor wrap long lines visually.

Write documentation links in changelog entries (and the README) with the
versioned `stable/` prefix, e.g.
`https://unitaryfoundation.github.io/clifft/stable/guide/strong-simulation/`.
The docs site only redirects at its root, so unversioned deep links 404; the
GitHub release notes extracted from the changelog inherit whatever links are
written here. The root `playground/` and `bench/` URLs are the intentional
unversioned exceptions.

Review, edit if needed (add the release summary, fix typos, clarify entries,
remove noise), then commit:

```bash
git add CHANGELOG.md
git commit -m "docs: update changelog for vX.Y.Z"
```

### 3. Update the docs home page

Update the "What's New" section in `docs/index.md` with a short,
editorial summary of the release. Keep this section user-facing and
curated rather than generated directly from the changelog. Link to the
most relevant new documentation or tutorial, and include a link to the
full changelog.

### 4. Tag and push

```bash
git tag vX.Y.Z
git push origin main vX.Y.Z
```

### 5. CI runs automatically

The tag push triggers the release workflow:

1. **Build** — sdist and wheels for all platforms
2. **Publish to TestPyPI** — dry run on the test index
3. **Publish to PyPI** — the real release (only on tag push)
4. **Create GitHub Release** — extracts release notes from `CHANGELOG.md`
5. **Publish versioned docs** — updates the exact version, `stable`, and the
   root stable Playground

If any step fails, subsequent steps are skipped.

### 6. Verify

```bash
pip install clifft==X.Y.Z
python -c "import clifft; print(clifft.__version__)"
```

Check that the [GitHub Release](https://github.com/unitaryfoundation/clifft/releases)
was created.

Then verify the hosted docs and Playground:

- `https://unitaryfoundation.github.io/clifft/<version>/` exists.
- For the newest release, `https://unitaryfoundation.github.io/clifft/stable/`
  shows the new release docs.
- `https://unitaryfoundation.github.io/clifft/playground/?url=<raw-stim-url>`
  still loads a remote circuit.

## Documentation versions

Documentation is versioned separately from package builds. Pushes to `main`
publish unreleased documentation to
`https://unitaryfoundation.github.io/clifft/dev/`. Final release tags publish
the exact release version, update the `stable` docs copy, and refresh the root
stable Playground at `https://unitaryfoundation.github.io/clifft/playground/`.
Release-candidate tags do not publish documentation.
If a lower SemVer patch line is released after a newer stable version, the
workflow publishes the exact version docs but leaves `stable` and the root
Playground unchanged.

Each docs version includes its own Playground build, so examples and docs match
the selected Clifft version. The root `/playground/` path is a stable
compatibility URL for externally shared links. Dev Playground links are
unreleased and should not be used as permanent public links.

## Changelog maintenance

The changelog is generated from conventional commit messages using
git-cliff with the config in `cliff.toml`.

Use conventional commit prefixes: `feat:`, `fix:`, `docs:`, `perf:`,
`refactor:`, `test:`, `build:`, `ci:`. Commits with `chore:` and `style:`
are excluded from the changelog.

Install git-cliff locally:

```bash
# macOS
brew install git-cliff

# Or via cargo
cargo install git-cliff
```

## Prerequisites (one-time setup)

These steps are needed once when setting up the repository:

1. **PyPI trusted publisher**: On [pypi.org](https://pypi.org), configure a
   trusted publisher for `clifft` (owner: `unitaryfoundation`, repo: `clifft`,
   workflow: `release.yml`, environment: `pypi`).

2. **TestPyPI trusted publisher**: Same on [test.pypi.org](https://test.pypi.org)
   with environment `testpypi`.

3. **GitHub environments**: Create two environments in repo settings:
    - `pypi` — optionally add required reviewers for production releases
    - `testpypi` — no restrictions needed
