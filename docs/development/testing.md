<!--pytest-codeblocks:skipfile-->
# Testing Strategy

Clifft uses a layered testing strategy. Fast C++ unit tests validate individual compiler and runtime components, while Python integration tests compare full-system behavior against independent simulation oracles and statistical expectations.

This split mirrors Clifft's architecture. The compiler pipeline is tested for deterministic correctness: parsing, Clifford absorption, HIR construction, Pauli localization, bytecode generation, and active-state updates should produce exact, reproducible results. The sampling layer is tested statistically: noisy circuits and detector outputs are compared against independent references within shot-noise bounds.

Because Clifft is an exact simulator for near-Clifford fault-tolerant circuits, the tests emphasize both sides of the system: exact basis and frame transformations, and correct stochastic behavior under noise, measurements, detectors, and observables.

## Core Primitives and the Stim Contract

Clifft relies on Stim for stabilizer tableau operations used in Clifford-frame tracking and Pauli rewinding. We do not duplicate Stim's test suite or independently re-test its underlying GF(2) tableau algebra.

Instead, Clifft maintains contract tests for the specific Stim semantics that the compiler depends on. The dedicated suite ([`tests/test_stim_contract.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_stim_contract.cc)) checks the expected Heisenberg rewinding behavior, Pauli-string conventions, and tableau conjugation rules used by the front end.

These tests act as a compatibility tripwire. If an upstream Stim change alters an API behavior or convention that Clifft relies on, the contract suite should fail close to the source of the mismatch rather than producing a harder-to-debug compiler or runtime error later in the pipeline.

## Structured and Random Circuit Oracles

Random circuit fuzzing is useful for finding edge cases, but it is not sufficient on its own. Deep random circuits can produce output distributions and dense states whose errors are difficult to diagnose locally. Clifft therefore combines random fuzzing with structured circuit families whose expected behavior is known analytically.

* **Mirror circuits ($UU^{\dag} = I$):** We generate deep, entangling circuits with a bounded number of non-Clifford gates and append the exact inverse circuit. The final state must return to `|00...0⟩`. These tests exercise active-state expansion, non-Clifford phase handling, measurement-free reversibility, and normalization behavior. With optimization enabled, related tests check that the compiler can recognize and eliminate cancelling non-Clifford structure in these cases ([`test_structural_oracles.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_structural_oracles.py), [`test_peephole_oracle.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_peephole_oracle.py)).

* **Structured compiler stress tests:** We generate circuit families designed to exercise specific parts of the compiler and VM:
    * **Commutation tests:** circuits that force non-Clifford operations through chains of commuting and anti-commuting Pauli structure, stressing HIR rewrites and scheduling.
    * **Parity/localization tests:** CNOT/CZ fan-out patterns that verify multi-qubit Pauli products are localized correctly and that fused bytecode evaluates parities as intended.
    * **Active-state lifecycle tests:** circuits that repeatedly introduce and remove active degrees of freedom, stressing active-array growth, compaction, and accumulated scale-factor handling.

* **Random fuzzing:** Dense random Clifford+T circuits are used to shake out edge cases in greedy Pauli localization, bytecode generation, and the routing of physical correlations through the virtual frame.

All procedural generators are centralized in [`utils_fuzzing.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/utils_fuzzing.py).

## External Cross-Validation Oracles

End-to-end Python tests compare Clifft against independent references whenever practical. These tests are intended to validate the full compiler-to-VM path rather than isolated implementation details.

* **Exact state-vector equivalence with Qiskit Aer:** For small circuits, we extract Clifft's frame-factored state representation and expand it into a dense $2^n$ state vector. We then compare this state against the same circuit simulated by Qiskit Aer using a strict fidelity threshold ([`test_qiskit_aer.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_qiskit_aer.py)). This checks that Clifft's non-Clifford phase handling and frame reconstruction agree with an independent dense-state simulator.

* **Statistical equivalence with Stim:** For purely Clifford noisy circuits, Clifft should reproduce the detector and observable statistics produced by Stim. We run surface-code-style extraction circuits for many shots in both simulators and require each detector and logical observable marginal to agree within a binomial shot-noise bound ([`test_statistical_equivalence.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_statistical_equivalence.py)). This validates Clifft's ahead-of-time handling of stochastic noise, measurements, detectors, and classical record logic in the Clifford regime.

* **Deterministic trajectory tests:** To test individual noisy trajectories without relying on statistical convergence, we inject deterministic Pauli errors such as `X_ERROR(1.0)` into entangled circuits. Clifft's detector and observable outputs are then compared directly against Stim's frame-tracking sampler ([`test_trajectory_oracle.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_trajectory_oracle.py)). These tests check that rewound frames, bytecode execution, and detector updates produce the expected classical outcomes.

## Layer-by-Layer C++ Unit Testing

The C++ core is unit-tested with `Catch2`. These tests target individual layers of the compiler and runtime so that failures can be localized before reaching the full Python integration suite.

* **Parsing and AST:** [`test_parser.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_parser.cc) validates conversion from text to `clifft::Circuit`, including `REPEAT` unrolling and supported Stim-like syntax.

* **Front end:** [`test_frontend.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_frontend.cc) checks Clifford absorption, Heisenberg rewinding, and extraction of the Pauli masks passed into HIR.

* **Pauli localization and backend lowering:** [`test_backend.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_backend.cc) tests localization of Pauli products into virtual single-qubit axes and verifies the virtual Clifford updates emitted by the backend.

* **Schrödinger VM array operations:** [`test_svm.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_svm.cc) bypasses the compiler and directly constructs localized VM instructions. These tests check the active-state array kernels, parity evaluation, and low-level update rules used during bytecode execution.

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
    ctest --test-dir build --output-on-failure
    # Or using just:
    just test
    ```

To generate HTML coverage reports for both layers of the application to ensure new features are thoroughly exercised:

```bash
just py-cov    # Generates Python coverage report
just cpp-cov   # Generates C++ coverage report (requires lcov)
just cov       # Runs both
```
