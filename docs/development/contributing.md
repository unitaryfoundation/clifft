<!--pytest-codeblocks:skipfile-->

# Contributing

UCC is developed at [unitaryfoundation/ucc-next](https://github.com/unitaryfoundation/ucc-next) on GitHub.

## Development Workflow

1. Fork the repository and create a feature branch
2. Make your changes with atomic, [conventional commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `test:`, `docs:`)
3. Run pre-commit checks before committing
4. Open a pull request against `main`

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
- **Namespace:** All code in `namespace ucc { ... }`
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
