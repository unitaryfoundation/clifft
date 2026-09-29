<!--pytest-codeblocks:skipfile-->
# Testing Strategy

Clifft combines focused C++ tests with Python integration tests. C++ tests
localize failures in parsing, compilation, planning, and execution. Python tests
check public behavior against analytic expectations and independent simulators.
Exact state and record checks complement statistical checks of noisy circuits.

## Independent References

* **Qiskit Aer** checks small unitary circuits, including non-Clifford gates,
  across compiler profiles. Statevectors are compared up to global phase;
  exact record probabilities and sampled joint distributions check correlations
  that measurement averages can miss. See the
  [statevector oracle](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_qiskit_aer.py)
  and [compiler conformance tests](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_compiler_conformance.py).
* **Stim** checks Clifford gate semantics and noisy detector/observable
  statistics. Probability-one errors give exact regression cases; stochastic
  comparisons use shot-noise bounds. See the
  [Clifford oracle](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_stim_statevector_oracle.py)
  and [statistical tests](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_statistical_equivalence.py).
* **Analytic and small dense references** check leakage/loss trajectories,
  transition probabilities, and correlations between records and final states.
  See the [noncomputational oracle tests](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_noncomp_oracle.py).

Structured circuits make failures easier to diagnose: mirror circuits test
reversibility, entangled circuits expose correlation errors, and repeated
growth and measurement exercise active-state reuse. Random circuits supplement
these targeted cases. C++ tests also compare kernels and tableau operations
against simple references; [Tableau Conventions](tableau-conventions.md) defines
the algebraic contract.

## Shared Sampling Behavior

Shared fixtures run the same behavioral assertions across supported execution
configurations. New shared tests inherit the existing configurations, and new
configurations inherit the applicable tests. Coverage includes ordinary and
postselected sampling, forced-fault sampling, and noncomputational trajectories.

Compiler-profile comparisons, exact-query APIs, large benchmarks, and focused
resource, dispatch, and cross-mode comparisons retain their own configurations.
These tests complement the shared behavioral suite. Configuration alone does
not prove that a particular execution path ran: representative cases must keep
the relevant work after optimization and exercise the intended runtime path.
See [Writing Tests](contributing.md#writing-tests) for fixture selection and
contribution guidance.

## CI Coverage

CI exercises CPU instruction sets, supported platforms, and WebAssembly.
Debug builds check internal invariants; optimized builds cover release behavior
and expensive workloads. Sanitizers check memory errors, undefined behavior,
and data races, while coverage jobs report which code the suites exercise.

Small representative cases remain in Debug. Measured high-cost cases marked
`expensive` run in optimized CI; local runs include them by default. The
[CI workflow](https://github.com/unitaryfoundation/clifft/blob/main/.github/workflows/ci.yml)
and [C++ test configuration](https://github.com/unitaryfoundation/clifft/blob/main/tests/CMakeLists.txt)
define the exact build and test selections.

## Running the Tests

With the development environment and C++ build prepared:

```bash
just py-test
just test
```

To exclude expensive cases, as the CI Debug jobs do:

```bash
uv run pytest tests/python/ -m "not expensive" --durations=20
ctest --test-dir build --output-on-failure --label-exclude expensive
```

See [Running Tests](contributing.md#running-tests) for direct build/run commands
and [Code Coverage](contributing.md#code-coverage) for coverage reports.
