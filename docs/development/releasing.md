<!--pytest-codeblocks:skipfile-->

# Releasing

Clifft uses [setuptools-scm](https://github.com/pypa/setuptools-scm) to
derive the package version from git tags. The release workflow builds
wheels for Linux (x86_64, aarch64), macOS (arm64), and Windows (amd64), then publishes to
PyPI via trusted publishers.

## Versioning

The version is determined automatically from git tags:

- **Tagged commits** (e.g., `v0.2.0`): version is `0.2.0`
- **Development builds**: version is `0.2.1.dev3+g1a2b3c4` (tag + commit distance + hash)

There is no hardcoded version in `pyproject.toml`. The git tag is the single source of truth.

## Release process

### 1. Test on TestPyPI (optional but recommended)

Verify that wheels build and install correctly by running the release workflow manually.
Manual dispatch always publishes to TestPyPI only — it cannot publish to PyPI.

1. Go to **Actions** > **Release** > **Run workflow**
2. Select the branch (usually `main`)
3. Wait for builds to complete, then verify:

    ```bash
    pip install --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ clifft
    python -c "import clifft; print(clifft.__version__)"
    ```

### 2. Update the changelog

Generate the changelog using [git-cliff](https://git-cliff.org/):

```bash
git cliff --tag v0.2.0 -o CHANGELOG.md
```

Review, edit if needed (fix typos, clarify entries, remove noise), then commit:

```bash
git add CHANGELOG.md
git commit -m "docs: update changelog for v0.2.0"
```

### 3. Tag and push

```bash
git tag v0.2.0
git push origin main v0.2.0
```

### 4. CI runs automatically

The tag push triggers the release workflow:

1. **Build** — sdist and wheels for all platforms
2. **Publish to TestPyPI** — dry run on the test index
3. **Publish to PyPI** — the real release (only on tag push)
4. **Create GitHub Release** — extracts release notes from `CHANGELOG.md`

If any step fails, subsequent steps are skipped.

### 5. Verify

```bash
pip install clifft==0.2.0
python -c "import clifft; print(clifft.__version__)"
```

Check that the [GitHub Release](https://github.com/unitaryfoundation/clifft/releases)
was created.

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

4. **Initial version tag**: After migrating to the new repo, push the first
   tag to establish the version baseline:

    ```bash
    git tag v0.1.0
    git push origin v0.1.0
    ```
