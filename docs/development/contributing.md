<!--pytest-codeblocks:skipfile-->

# Contributing

Clifft is developed at [unitaryfoundation/clifft](https://github.com/unitaryfoundation/clifft) on GitHub.

## Reporting Issues

- **Bugs:** Use the [bug report template](https://github.com/unitaryfoundation/clifft/issues/new?template=bug_report.yml) with a minimal reproducer.
- **Feature requests:** Use the [feature request template](https://github.com/unitaryfoundation/clifft/issues/new?template=feature_request.yml).

## Development Workflow

1. Fork the repository and create a feature branch
2. Make your changes with atomic, [conventional commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `test:`, `docs:`)
3. Run pre-commit checks before committing
4. Open a pull request against `main`

## Contributor Agreement

By submitting a pull request, you confirm that:

- The contribution is your original work (or you have the right to submit it).
- You license it under this project's [Apache-2.0 license](https://github.com/unitaryfoundation/clifft/blob/main/LICENSE).

The PR template includes a checkbox for this — please check it when opening your PR.

## AI-Assisted Contributions

We welcome AI-assisted contributions. If you use AI tools (Claude, Copilot,
ChatGPT, etc.) to help write code:

- **Review all generated code** before submitting. You are responsible for the
  correctness, security, and quality of the contribution.
- **Include an `Assisted-by:` git trailer** in your commit message identifying
  the tool and model used:

    ```
    feat: add new optimization pass

    Assisted-by: Claude (Sonnet 4.6) <noreply@anthropic.com>
    ```

- The human author remains the commit author. The AI tool is credited via the
  trailer, not `Co-authored-by`.

## Code Quality

We use pre-commit hooks to enforce formatting and linting:

```bash
# Install pre-commit hooks (runs automatically on git commit)
uv run pre-commit install

# Run all checks manually
uv run pre-commit run --all-files
```

### C++

- **Formatter:** clang-format
- **Standard:** C++20
- **Namespace:** All code in `namespace clifft { ... }`
- **Comments:** Explain *why*, not *what*. Omit if the code is self-explanatory.

### Python

- **Linter/Formatter:** Ruff
- **Type checker:** mypy (strict mode)
- **Python version:** 3.12+

## Running Tests

=== "Python"

    ```bash
    uv run pytest tests/python/ -v
    ```

=== "C++"

    ```bash
    cmake -B build -DCMAKE_BUILD_TYPE=Debug
    cmake --build build -j
    ctest --test-dir build --output-on-failure
    ```

=== "Both"

    ```bash
    just py-test
    just test
    ```

## Code Coverage

```bash
# Python coverage
just py-cov

# C++ coverage (requires lcov)
just cpp-cov

# Both
just cov
```

HTML reports are generated at `coverage/python/index.html` and `coverage/cpp/index.html`.
