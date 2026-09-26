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

Users that do not follow these steps, and in particular fail to do a human pre-review of any AI generated content including issues and pull requests, may be subject to a ban from the repo.

## Code Quality

We use pre-commit hooks to enforce formatting and linting:

```bash
# Install pre-commit hooks (runs automatically on git commit)
uv run pre-commit install

# Run all checks manually
uv run --frozen --only-group dev pre-commit run --all-files --show-diff-on-failure
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

## Writing Tests

Choose tests that validate the behavior affected by your change. Reuse existing
shared tests where applicable. Use `sampling_mode` for ordinary/survivor sampling,
`importance_sampling_mode` for forced-fault sampling, or `noncomp_sampling_api`
for leakage/loss trajectories. These fixtures test features across supported
modes and let new modes inherit applicable tests.

Check that each test exercises the behavior it claims to cover. Configuration
alone may not establish this: optimization can remove relevant work, and
execution policies can select a different path.

When testing parallel execution, provide enough work for multiple workers to
contribute without asserting a particular scheduling outcome. Keep resource use
bounded.

Mark measured high-cost sampling or fixture cases with Catch2 `[expensive]`
or `@pytest.mark.expensive` so CI runs their full workload in optimized builds;
retain small representative checks in Debug.

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

### GPU behavioral tests

The `sampling_mode` fixture includes HIP and CUDA in FP64 with automatic tier
selection. Each compiled program retains its own sampler, with a 65-shot batch
limit to exercise batching and partial batches with bounded workspace. CPU and
GPU modes share the behavioral assertions; identical random rows across
backends are not required. Forced-fault sampling, noncomputational trajectories,
and the CPU compiler-profile matrix keep their own fixtures.

GPU cases skip when their backend or device is unavailable. On a GPU machine,
build the corresponding [HIP](hip-backend.md#hardware-and-source-build) or
[CUDA](cuda-backend.md#hardware-and-source-build) extension and install the test
dependencies with `uv sync --frozen --only-group dev` before building. Then run
serially to avoid competing GPU workspaces:

```bash
# AMD: shared FP64 cases plus the existing focused HIP tests.
git rev-parse HEAD
uv run --no-sync pytest tests/python --require-gpu=hip \
    -k 'hip-fp64 or test_experimental_hip' -v -rP --durations=20

# NVIDIA: shared FP64 cases plus the existing focused CUDA tests.
git rev-parse HEAD
uv run --no-sync pytest tests/python --require-gpu=cuda \
    -k 'cuda-fp64 or test_experimental_cuda' -v -rP --durations=20
```

`--require-gpu` fails before testing if the requested backend has no available
device. The adapter's execution checks print backend/device information,
precision, and the actual tier for narrow and five-coordinate states; `-rP`
includes this output in the report. Preserve the revision and report when
sharing hardware results. Existing focused GPU tests retain their precision,
tier, replay, and workspace checks; automatic selection on these small shared
cases does not establish coverage of every GPU tier or FP32.

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
