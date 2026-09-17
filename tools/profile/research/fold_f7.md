# Full f7 reconstruction and compiled logical blocks

The complete reconstructed cultivation sequence now runs through injection,
d3 and d5 checks, regular d5-to-d7 growth, both verified d7 checks, and terminal
syndrome postselection. It uses 99 physical qubits and 221 non-Clifford gates.
The native code-boundary sampler retains at most four coherent monomial terms
acting on two logical amplitudes, rather than a dense physical residual.
No production Clifft compiler, executor, or API has changed.

This is a paper-guided circuit with explicit reconstruction choices, not an
authors' circuit artifact or a reproduction of their rare logical-error results.
The checked-in [noisy circuit](../fixtures/fold_cultivation_f7.stim) uses p=0.001
and Clifft's extended Stim syntax, including CCZ and EXP_VAL. Every detector
must be postselected to obtain the contract studied here.

## Circuit provenance and choices

The sequence, injection, fold assignments, and verified cat schedules follow
[Takada, Bartlett, and Williamson, Appendix A](https://arxiv.org/html/2609.16929v1).
The 99-qubit and 221-non-Clifford counts match the paper. The earlier d3-to-d5
construction remains as documented in the [composition study](fold_composition.md).
The existing f3/f5 exported circuits are byte-for-byte unchanged.

The new regular d5-to-d7 growth transcribes the four CNOT layers in Fig. 2 of
[Higgott et al., Optimal local unitary encoding circuits for the surface code](https://quantum-journal.org/papers/q-2021-08-05-517/).
Coordinates are transposed to the reconstruction's X/Z convention. The old d5
patch is embedded centrally; 44 fresh data qubits are reset, 22 receive H, and
four disjoint local layers contain 22, 22, 18, and 18 CNOTs. Independent Stim
checks verify every target stabilizer and logical X, Y, and Z transport. The
block compiler separately certifies signed stabilizer and logical transport.

The full sequence is:

1. Inject into rotated d3, morph to regular d3, and measure its syndrome.
2. Perform two d3 fold checks with their cat preparation and decoding.
3. Grow to rotated d5, morph to regular d5, and measure its syndrome.
4. Perform two d5 fold checks.
5. Grow directly to regular d7 and measure its syndrome.
6. Perform two d7 fold checks, followed by a noisy d7 syndrome round.
7. Apply an ideal syndrome round and report logical X/Y/Z expectations.

Elementary H, CX, T, T_DAG, and native CCZ gates have depolarizing noise after
execution. Resets have X noise afterward and Z measurements have X noise
beforehand. There are 2,205 physical noise locations; no idle noise is included.
Sequential stabilizer extraction reuses one ancilla. The within-stabilizer
N/W/E/S CNOT order is an explicit reconstruction choice. Its noisy behavior
need not match an unavailable authors' implementation.

The terminal ideal syndrome and logical probes retain the earlier research
contract. This covers postselected cultivation with final error detection;
it does not implement the paper's final-error-correction decoder or its final
fictitious logical-measurement circuit. No fault-distance-seven certificate or
statistically useful rare logical-error rate has been established for this
particular gate ordering.

## The f7 verification outcomes

Fig. 19(c) prepares eight main cat qubits and six verification qubits. Acceptance
requires the verification records to be all zero **or** all one. The circuit
therefore uses five adjacent-record equality detectors per preparation, not six
individual zero-record detectors.

Construction certifies that the ideal preparation produces GHZ8 tensor GHZ6.
Any fixed Pauli history in this Clifford circuit commutes to a tensor-product
Pauli on the output. Thus the registers remain separable, and the two flag
strings remain equiprobable complements. Both strings are uniform exactly when
the propagated flag-flip mask is zero or 63. Their conditioned main-cat density
matrices coincide; summing their probabilities contributes no extra factor of
one half. This is classical marginalization of measured records, not coherent
addition of measurement branches.

The native executor checks that mask using precomputed parity maps. The
independent CH reference follows the all-zero flag record and rescales its
amplitudes by sqrt(2) at each of the two f7 preparations. Qiskit Aer separately
checks equality of the two conditioned density matrices for the ideal state
and selected Pauli faults. A uniform flag flip accepts; an isolated flag flip
rejects. The two accepted records are marginalized, not explicitly emitted.

## Representation and compilation

The [compiled block design](fold_blocks.md) extends without new per-shot
planning. Growth certificates, Pauli responses, cat decoder amplitudes, and
contraction addresses are computed before execution. Physical faults choose
syndrome sectors using fixed parity maps. At actual code boundaries, up to four
monomial overlaps reduce the state back to two logical amplitudes and a Pauli
frame. No projection is inserted before the physical syndrome measurement.
A fully traversed f7 history uses at most 13 overlaps.

The native diagnostic now supports 85 data coordinates with 128-bit masks,
8,192 fault-bit slots, and 2,851 complex contraction slots. f7 uses 6,199 encoded
fault-bit positions. The contraction scratch alone is 45,616 bytes; this is not
a whole-process memory measurement. Constant tables and other fixed arrays
also occupy memory. GCC/Clang 128-bit integers are a research portability
constraint. Hot execution and physical-fault sampling perform no allocation,
throwing validation, tableau evolution, or topology planning.

Compilation is once per circuit. No per-shot compiler or history cache is
needed. The exporter still consumes the structured reconstruction; it is not
an arbitrary-circuit recognizer. A production path needs eligibility checks
for injection, code transport, supported noise, cat preparation, and fold
structure, followed by a cheap decline to ordinary Clifft for unsupported
circuits. Composition across a live generic residual remains future work.

## Validation and performance

| Diagnostic | Full f7 reconstruction |
| --- | ---: |
| Native sampled attempt at p=0.001, median | 45.41 us |
| Sampled attempts | 100,000 |
| Accepted attempts | 12,667 |
| Native fixed-history evaluation, median | 58.69 us |
| Native whole-process maximum RSS | 4,980 KiB |
| Current scalar Clifft planned peak active width | 44 |
| Dense complex-double coefficients at width 44 | 256 TiB |

Sampling time is the median of five trials of 20,000 attempts. It includes
sampling every physical noise location, evaluating the conditioned state,
drawing acceptance from its norm, and accumulating accepted logical probes.
Compilation and I/O are excluded. The fixed-history mixture deliberately
includes costly surviving hooks; its timing is not natural-noise throughput.
The executable used GCC 13.3, O3/native instructions without fast-math, one thread
on the unpinned EPYC VM, after tests had completed. Maximum RSS comes from
`/usr/bin/time` for that benchmark process, including loaded constant tables.

The same exported circuit reaches 44 active coordinates in the current scalar
Clifft planner. Sampling was deliberately skipped at the configured width cap;
256 TiB is the theoretical coefficient-array size, not an attempted allocation
or measured process RSS. There is no measured f7 Clifft runtime and no claimed
f7 speedup ratio. The earlier f5 timing comparison remains in the prior report.

Validation comprises 115 fixed histories compared with the independent
phase-preserving CH/tableau reference: 64 natural histories at p=0.001, 32 stress
histories at p=0.01, ten paired hooks, one logical-error tail, two selected
nonzero growth sectors, and six ideal/flag/high-coordinate diagnostics.
Maximum acceptance error is 1.12e-16; maximum conditional X/Y/Z error is
2.23e-16. Native probability and unnormalized probes agree with the Python
block result to 2.23e-16. Both nonzero growth-sector cases accept with probability
1/4; the logical-error tail retains negative X and Y. The selected cases are
diagnostics without natural-noise statistical weights.

The complete research suite passes 48 tests, including Stim growth/cat checks,
Aer checks, phase-sensitive coherent overlaps, and generated native execution.
All 115 native fixtures and 500 sampled attempts also pass address and undefined
behavior sanitizers. Leak detection was disabled because LeakSanitizer cannot
run under this sandbox's tracing; this run does not claim leak coverage.

These results make the representation promising for a large-residual,
practically motivated circuit family. They do not establish best-in-class
performance against other simulators, arbitrary-circuit applicability, or exact
agreement with the authors' circuit. The next useful comparisons are another
backend on this same artifact and a recognizer with explicit eligibility and
fallback behavior. Five-round coherent memory remains a generic control.

## Reproduction

Use the Python environment described in the composition study, with Cirq Core
for independent reference validation and a GCC/Clang C++20 compiler. Run timing
without concurrent benchmark or test work. Generated programs include the native
research header by absolute path.

```sh
python tools/profile/fold_cultivation.py --output /tmp/f7-circuits
cmp /tmp/f7-circuits/reconstructed_f7_p0.001.stim tools/profile/fixtures/fold_cultivation_f7.stim
OPENBLAS_NUM_THREADS=1 python tools/profile/fold_blocks.py --distance 7 --histories 64 --stress-histories 32 --validate --output /tmp/f7-validation.json
OPENBLAS_NUM_THREADS=1 python tools/profile/export_fold_blocks.py --distance 7 --output /tmp/f7-blocks.cpp
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic /tmp/f7-blocks.cpp -o /tmp/f7-blocks
/tmp/f7-blocks 100 20000
build-research/profile_structure tools/profile/fixtures/fold_cultivation_f7.stim 1 22 1
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_*.py'
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer /tmp/f7-blocks.cpp -o /tmp/f7-checked
ASAN_OPTIONS=detect_leaks=0 /tmp/f7-checked 1 100
```

The profiler command intentionally caps sampling at 22 active coordinates;
it completes planning and declines the 44-coordinate allocation. Raw results,
validation rows, and artifact hashes are in `fold_f7_data.json`.
