<!--pytest-codeblocks:skipfile-->
# Testing Strategy

Clifft employs a two-tier testing strategy: fast C++ unit tests to validate individual compiler pipeline mechanics, and extensive Python-based oracles for full-system integration and quantum physics validation.

Because Clifft is an exact simulation engine designed for Fault-Tolerant Quantum Computing (FTQC), the test suite strictly enforces absolute deterministic correctness for basis transformations and bounds statistical deviations for noise sampling.

## Core Primitives & The Stim Contract

Clifft relies heavily on [Stim](https://github.com/quantumlib/Stim) for stabilizer frame tracking and Pauli rewinding. Because `stim` is heavily battle-tested by the quantum error correction community, we intentionally **do not** write tests verifying its underlying GF(2) mathematics.

Instead, we explicitly validate **our assumptions** about Stim's APIs. We maintain a dedicated contract test suite ([`tests/test_stim_contract.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_stim_contract.cc)) that asserts specific Heisenberg rewinding behaviors and tableau conjugation rules. If an upstream change in Stim ever subtly alters these semantics, this suite acts as an immediate tripwire, preventing mysterious upstream breakages in the Clifft Front-End.

## Verifying Quantum Phenomena (Structured vs. Random)

Random circuit fuzzing is a staple of quantum simulator testing, but it has a major flaw: deep random circuits quickly scramble amplitudes into uniform white noise. This makes it mathematically difficult to prove that deep, uniquely quantum phenomena—like complex phase accumulation or destructive interference—are actually being simulated correctly, as opposed to just outputting random noise.

To guarantee physical accuracy, Clifft relies heavily on **Structured Testing** alongside random fuzzing:

* **Mirror Circuits ($U U^\dagger = I$):** We construct deep, highly entangled circuits, insert a bounded number of non-Clifford `T` gates, and immediately append the exact analytical inverse of the circuit. Mathematically, the final state vector must perfectly return to $|00\dots0\rangle$. This proves that the VM's active array expansion, `T`-gate phase tracking, and floating-point renormalization are perfectly reversible. Furthermore, when the AOT optimizer is enabled, it proves that complete analytical `T`-gate annihilation occurs, collapsing the peak active dimension to exactly 0 ([`test_structural_oracles.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_structural_oracles.py), [`test_peephole_oracle.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_peephole_oracle.py)).
* **Structured Topologies:** We generate specific circuit topologies designed to trigger complex compiler behaviors:
    * **Commutation Gauntlets:** Tests the HIR pass manager by forcing `T` gates to slide past deep chains of commuting and anti-commuting Pauli barriers.
    * **Star-Graph Honeypots:** Generates massive CNOT/CZ fan-outs to verify the `MultiGatePass` fused bytecode loops evaluate parities correctly.
    * **Breathing Memory Lifecycles:** Injects and measures `T`-state qubits repeatedly to stress the VM's array compaction and ensure the continuous scale factor $\gamma$ doesn't drift into IEEE-754 underflow limits.
* **Random Fuzzing:** Dense random Clifford+T circuits are used specifically to shake out edge cases in the greedy Pauli localization pass and ensure the generated $\mathcal{O}(N)$ bytecode correctly routes physical correlations.

All procedural generators are centralized in [`utils_fuzzing.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/utils_fuzzing.py).

## External Cross-Validation Oracles

To ensure Clifft's novel architecture produces universally correct results, we validate the end-to-end Python API against independent, third-party physics oracles.

* **Exact State Vector Equivalence (Qiskit):** We extract the VM's factored state representation ($|\psi\rangle = \gamma U_C P |\phi\rangle_A$) and geometrically expand it into a dense $2^n$ state vector. This array is compared via a strict fidelity check ($>0.9999$) against the exact same circuit simulated by **Qiskit Aer** ([`test_qiskit_aer.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_qiskit_aer.py)). This guarantees our non-Clifford phase math matches the rest of the industry.
* **Statistical Equivalence (Stim):** Because Clifft computes stochastic noise models and classical logic Ahead-Of-Time, we must prove our gap-sampling distributions match standard Monte Carlo execution. We run complex surface code extraction rounds for millions of shots in both Clifft and Stim. We then require the marginal firing probability of every QEC detector and logical observable to agree with its expected value within a 5σ binomial shot-noise bound. ([`test_statistical_equivalence.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_statistical_equivalence.py)).
* **Exact Trajectories:** To verify our Heisenberg-rewound virtual frames trigger the exact same classical detector cascades as a physical laboratory, we inject $100\%$ deterministic Pauli errors (e.g., `X_ERROR(1.0)`) into entangled topological states. We verify that Clifft's hardware-agnostic bytecode flips the exact same deterministic detector bits as Stim's frame-tracking sampler ([`test_trajectory_oracle.py`](https://github.com/unitaryfoundation/clifft/blob/main/tests/python/test_trajectory_oracle.py)).

## Layer-by-Layer C++ Unit Testing

Beneath the end-to-end Python oracles, the C++ codebase is rigidly unit-tested using `Catch2`. Because the pipeline is broken into independent stages, each stage is tested in isolation:

* **Parsing & AST:** [`test_parser.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_parser.cc) validates lexical conversion of text to `clifft::Circuit`, unrolling of `REPEAT` blocks, and multi-qubit syntactic sugar.
* **Front-End:** [`test_frontend.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_frontend.cc) validates mathematical absorption of physical Cliffords and rewound mask extractions.
* **Pauli Localization:** [`test_backend.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_backend.cc) feeds massive, random `stim::PauliString` masks into the Back-End's localization pass, asserting that the resulting sequence of virtual CNOT/CZ gates perfectly isolates the mask to a single active or dormant virtual bit.
* **VM Array Math:** [`test_svm.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_svm.cc) completely bypasses the compiler. We manually construct localized `Instruction` opcodes and directly mutate a dummy `SchrodingerState` to assert the raw C++ array loops and hardware `popcount` routines correctly evaluate quantum superposition.

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
