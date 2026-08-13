<!--pytest-codeblocks:skipfile-->
# Testing Strategy

Clifft uses a layered testing strategy. Fast C++ unit tests validate individual compiler and runtime components, while Python integration tests compare full-system behavior against independent simulation oracles and statistical expectations.

This split mirrors Clifft's architecture. The compiler pipeline is tested for
deterministic correctness: parsing, Clifford absorption, HIR construction,
coordinate planning, expression lowering, and executable-plan preparation
should produce reproducible results. The sampling layer is tested
statistically: noisy circuits and detector outputs are compared against
independent references within shot-noise bounds.

Because Clifft is an exact simulator for near-Clifford fault-tolerant circuits, the tests emphasize both sides of the system: exact basis and frame transformations, and correct stochastic behavior under noise, measurements, detectors, and observables.

## Core Primitives and the Stim Contract

Clifft relies on Stim for stabilizer tableau operations used in Clifford-frame tracking and Pauli rewinding. We do not duplicate Stim's test suite or independently re-test its underlying GF(2) tableau algebra.

Instead, Clifft maintains contract tests for the specific Stim semantics that the compiler depends on. The dedicated suite ([`tests/test_stim_contract.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_stim_contract.cc)) checks the expected Heisenberg rewinding behavior, Pauli-string conventions, and tableau conjugation rules used by the front end.

These tests act as a compatibility tripwire. If an upstream Stim change alters an API behavior or convention that Clifft relies on, the contract suite should fail close to the source of the mismatch rather than producing a harder-to-debug compiler or runtime error later in the pipeline.

## Structured and Random Circuit Oracles

Random circuit fuzzing is useful for finding edge cases, but it is not sufficient on its own. Deep random circuits can produce output distributions and dense states whose errors are difficult to diagnose locally. Clifft therefore combines random fuzzing with structured circuit families whose expected behavior is known analytically.

* **Mirror circuits ($UU^{\dag} = I$):** We generate deep, entangling circuits with a bounded number of non-Clifford gates and append the exact inverse circuit. The final state must return to `|00...0⟩`. These tests exercise active-state expansion, non-Clifford phase handling, measurement-free reversibility, and normalization behavior. With optimization enabled, related tests check that the compiler can recognize and eliminate cancelling non-Clifford structure in these cases ([`test_peephole_oracle.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_peephole_oracle.py)).

* **Structured compiler stress tests:** We generate circuit families designed to exercise specific parts of the compiler and executor:
    * **Commutation tests:** circuits that force non-Clifford operations through chains of commuting and anti-commuting Pauli structure, stressing HIR rewrites and scheduling.
    * **Coordinate and expression tests:** CNOT/CZ fan-out patterns that verify multi-qubit Pauli products map to the intended active coordinates and affine dependencies.
    * **Active-state lifecycle tests:** circuits that repeatedly introduce and remove active degrees of freedom, stressing active-array growth, compaction, and accumulated scale-factor handling.

* **Random fuzzing:** Dense random Clifford+T circuits are used to shake out
  edge cases in coordinate planning, prepared active operations, and the
  routing of physical correlations through the symbolic frame.

All procedural generators are centralized in [`utils_fuzzing.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/utils_fuzzing.py).

## External Cross-Validation Oracles

End-to-end Python tests compare Clifft against independent references whenever practical. These tests validate the full compiler-to-symbolic-executor path rather than isolated implementation details.

* **Statevector equivalence with Qiskit Aer:** For small circuits, we extract Clifft's frame-factored state representation and expand it into a dense $2^n$ state vector. We then compare this state against the same circuit simulated by Qiskit Aer using a strict fidelity threshold ([`test_qiskit_aer.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_qiskit_aer.py)). This checks that Clifft's non-Clifford phase handling and frame reconstruction agree with an independent dense-state simulator up to global phase.

* **Statistical equivalence with Stim:** For purely Clifford noisy circuits, Clifft should reproduce the detector and observable statistics produced by Stim. We run surface-code-style extraction circuits for many shots in both simulators and require each detector and logical observable marginal to agree within a binomial shot-noise bound ([`test_statistical_equivalence.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_statistical_equivalence.py)). This validates Clifft's ahead-of-time handling of stochastic noise, measurements, detectors, and classical record logic in the Clifford regime.

* **Deterministic trajectory tests:** To test individual noisy trajectories without relying on statistical convergence, we inject deterministic Pauli errors such as `X_ERROR(1.0)` into entangled circuits. Clifft's detector and observable outputs are then compared directly against Stim's frame-tracking sampler ([`test_detector_oracle.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_detector_oracle.py)). These tests check that rewound frames, prepared symbolic actions, and detector updates produce the expected classical outcomes.

## Layer-by-Layer C++ Unit Testing

The C++ core is unit-tested with `Catch2`. These tests target individual layers of the compiler and runtime so that failures can be localized before reaching the full Python integration suite.

* **Parsing and AST:** [`test_parser.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_parser.cc) validates conversion from text to `clifft::Circuit`, including `REPEAT` unrolling and supported Stim-like syntax.

* **Front end:** [`test_frontend.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_frontend.cc) checks Clifford absorption, Heisenberg rewinding, and extraction of the Pauli masks passed into HIR.

* **Symbolic planning and lowering:** `test_sampling_planner.cc`,
  `test_sampling_plan.cc`, and `test_sampling_executor.cc` cover coordinate
  selection, affine dependencies, plan validation, prepared actions, and
  execution boundaries.

* **Active-state kernels:** `test_sampling_kernels.cc` and the focused
  rotation, measurement, and instrument suites compare scalar and SIMD
  implementations across width and mask boundaries.

The legacy backend and SVM suites remain as private differential oracles while
the implementation is being removed. Passing them does not make their Python
types or bytecode part of the production API.

## Running the Tests

We use `pytest` for the Python oracles and `CTest` for the C++ units. You can run the test suites locally using the provided `just` shortcuts.

=== "Python"

    ```bash
    uv run pytest tests/python/ -v
    # Or using just:
    just py-test
    ```

=== "C++"

    ```bash
    cmake -B build -DCMAKE_BUILD_TYPE=Debug
    cmake --build build -j
    ctest --test-dir build -E Bench --output-on-failure
    # Or using just:
    just test
    ```

    `-E Bench` skips the [bench] performance cases. Run them
    explicitly with `ctest --test-dir build -R Bench` when collecting
    timing data.

To generate HTML coverage reports for both layers of the application to ensure new features are thoroughly exercised:

```bash
just py-cov    # Generates Python coverage report
just cpp-cov   # Generates C++ coverage report (requires lcov)
just cov       # Runs both
```
