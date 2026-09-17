# Compiled blocks for the complete f5 reconstruction

Follow-up: the [full f7 reconstruction](fold_f7.md) now extends this result
through regular d7 growth and both verified d7 checks. The measurements below
describe the original f3/f5 experiment and its smaller native capacities.

**The representation now gives a substantial full-protocol result.** A standalone
native prototype samples the reconstructed noisy f5 circuit in a median of
17.73 microseconds per attempt, including physical fault generation, all-zero
postselection, and accepted-shot logical X/Y/Z expectations. The current scalar
Clifft diagnostic takes 364.66 milliseconds per attempt on that same input.
This is about 20,600x in this specific comparison, with compilation excluded
from both. It is not a comparison against other simulators or an established
best-in-class result.

This is still the [paper-guided reconstruction](fold_composition.md), not the
authors' Reg5/f7 artifact. Its noise locations, syndrome CNOT order, and terminal
ideal syndrome plus expectation probes have the limitations documented there.
At the time of this experiment, the full f7 protocol was unimplemented.
No production Clifft API or executor has changed.

## What now runs without per-history planning

The previous [fixed-contraction experiment](fold_contraction.md) covered isolated
overlaps between ideal code boundaries. This prototype connects those kernels
through injection, noisy syndrome extraction, verified cat preparation/readout,
both pairs of fold checks, and d3-to-d5 growth.

The state at a complete code boundary consists of two complex logical amplitudes
and a Pauli frame. Between boundaries, up to four coherent monomial operators
act on those amplitudes. This is eight implicit logical-basis components; it
does not assert four stabilizer components in the previous reference's basis.
No small terms are truncated and no branch is sampled as a classical mixture.

There are three reductions after the initial d3 syndrome round: the syndrome
following growth, the last noisy d5 syndrome, and the terminal ideal syndrome.
A fully traversed history needs at most nine monomial overlap evaluations:
four at the first reduction, four at the second, and one at the last. Many
histories reject earlier. Plan geometry and all fault dependencies are fixed.

### Clifford noise and injection

For each Clifford region, construction propagates symbolic Pauli bits once.
The resulting bit-parity maps give the final Pauli frame and which ideal
measurement outcomes correspond to the requested noisy all-zero record.
Execution evaluates these maps; it never evolves a tableau or discovers a
dependency. Reset and measurement locations retain their distinct noise rules.

The injection T gate has a certified, separate ideal |+> input. An X component
of an incoming Pauli error is a stabilizer of that ideal input and can be removed
before T up to a history-wide phase. Its Z component is retained. This permits
a compiled Pauli response through this particular injection; the rule is not
applied to an arbitrary T input. Tests cover X/Y/Z faults on both sides.

History-wide phases of stochastic Pauli errors can be discarded. Relative
phases of coherent fold branches and logical components cannot, and are kept.
This distinction depends on the stipulated stochastic Pauli noise model.

### Noisy code boundaries and growth

The d3 check pair is followed by growth before the next complete syndrome
measurement. Inserting a d3 code projection immediately after the checks would
be incorrect: a leaked component can survive a later nonzero syndrome sector.

Construction conjugates each d5 stabilizer back through the ideal growth
isometry. It certifies that the result is a signed product of d3 stabilizers
and Z stabilizers of freshly prepared qubits. The same certificate verifies
transport of both signed logical axes. This yields fixed maps from d3 syndrome
bits to d5 records, a left inverse, and constraints rejecting incompatible
record strings. Gaussian elimination happens only while building these maps.

At execution, the physical fault bits select an ideal syndrome sector. A
precomputed Pauli representative D moves that sector to the zero-syndrome code:

```text
P_s = D P_0 D^dagger
new logical amplitudes = sum_j weight_j <code|D^dagger C_j|code> old amplitudes
new physical frame = propagated D combined with the region's output fault frame
```

The existing fixed contraction evaluates each matrix element. The propagated
frame remains available for the next fold block; it is not silently corrected
away. Two adversarial histories explicitly select nonzero sectors through
growth, then apply legal physical faults that return those components to the
code. Both accept with probability 1/4 and match the independent reference.
An early code projection would lose these contributions.

### Cat checks

Construction certifies the prepared GHZ stabilizers and deterministic flag,
and tabulates the small decoder's amplitudes. At execution, preparation faults
select the two cat strings and their relative sign. A fixed instruction stream
computes the core's diagonal phases, mirror-pair enable bits, and Pauli flips.
Decoder fault parity maps choose the appropriate matrix-element row. The two
weighted operators are composed with the incoming terms before any actual code
boundary is applied.

## Native execution and compilation

`fold_blocks.py` is an allocating Python correctness prototype. All its tableau,
code transport, GF2 basis, and contraction-order work occurs in constructors.
An automated test disables those planners after construction and executes an
encoded history successfully.

`export_fold_blocks.py` emits a circuit-specific C++ translation unit with
constant parity maps, contraction addresses, decoder tables, and action streams.
`fold_blocks_native.h` executes those constants using fixed-capacity arrays and
scalar bit/complex arithmetic. Its ordinary execution and fault-sampling loops
perform no dynamic allocation or throwing validation. This is an experimental
code-generation route, not a proposed requirement to invoke a C++ compiler from
Clifft's public API. The constants could instead populate a validated plan.

The native diagnostic is explicitly bounded to f3/f5: 41 data coordinates,
four monomial terms, 603 complex contraction slots, and 8,192 fault-bit slots.
The exporter checks these capacities before generation. f5 uses 2,579 independent
encoded fault-bit positions for 933 physical noise locations; coincident Pauli
components at the same boundary can share a bit. All 933 physical noise trials
are still sampled independently. Uniform nonidentity depolarizing Paulis use
integer rejection sampling, and Bernoulli draws use the prescribed 53-bit rule.

This gives a concrete compile-once design for this circuit family. Measured
Python plan construction was about 0.18 seconds for f5; a separate compilation
of the generated native program took 1.67 seconds with GCC 13.3. These are
one-run setup diagnostics. There is no recompilation for a shot or fault history,
and no cache keyed by previously observed histories.

## Validation and measurements

The same 214 fixed histories were evaluated by the compiled-block prototype
and the independent phase-preserving CH/tableau reference:

- f3: 64 natural histories at p=0.001, eight paired hooks, one logical-error
  tail, and 32 stress histories at p=0.01.
- f5: 64 natural histories at p=0.001, ten paired hooks, one logical-error
  tail, two selected nonzero growth sectors, and 32 stress histories at p=0.01.

Maximum acceptance discrepancy was 1.12e-16; maximum conditional logical-probe
discrepancy was 2.23e-16. Every native fixed-history fixture also agrees with
the Python block result to 2.23e-16 in probability and unnormalized probes.
The logical-error tails retain negative X/Y expectations, so surviving states
are not automatically replaced by the target T state. The sector and hook
strata are targeted diagnostics, not weighted error-rate samples.

The prior independent dense Qiskit and Aer checks remain in the research suite.
The current suite has 44 tests, including compilation and execution of an
exported native program. The full native f5 fixture set and 500 sampled attempts
also ran under address/undefined-behavior sanitizers without reported errors.
Floating point is retained; neither exact integer arithmetic nor an arbitrary
precision guarantee is claimed. Conditional probe reporting uses the reference's
1e-12 acceptance threshold, without truncating the underlying state.

| Diagnostic | f3 | f5 |
| --- | ---: | ---: |
| Native fixed-history evaluation, median | 2.56 us | 13.70 us |
| Native sampled attempt at p=0.001, median | 3.87 us | 17.73 us |
| Sampled attempts | 100,000 | 100,000 |
| Accepted attempts | 78,779 | 41,606 |
| Current scalar Clifft, fresh comparison | not rerun | 364.66 ms/attempt |

Native results are medians of five trials of 20,000 sampled attempts each. The
fixed-history timing is a separate stress-mixture diagnostic, not natural-noise
throughput. Sampling includes generating every physical Pauli fault, evaluating
the conditional state, drawing acceptance from its norm, and accumulating logical
probes for survivors. It does not retain rejected records or expose arbitrary
measurement-outcome sampling. All accepted measurement records are known zero.

The baseline is `profile_structure`'s existing single-thread scalar executor,
with one warm-up and 128 timed attempts, all detectors postselected and the same
three terminal `EXP_VAL` instructions. It accepted 53 of 128 attempts, consistent
with the native acceptance fraction. Both exclude plan creation, allocation,
and output I/O from hot timing. Native code used O3/native instructions without
fast-math; the baseline used the existing Release/native build. Measurements
ran sequentially on the unpinned EPYC VM. The denominator is attempted shots,
including discarded shots, for both implementations.

The large difference is attributable to avoiding the 22-active-qubit residual
on this encoded workload, not to reduced shot count or new SIMD kernels. The
comparison has not surveyed all public Clifft sampling configurations, parallel
modes, or competing simulators. No statistically useful rare logical-error rate
was measured. Raw results, validation rows, and artifact hashes are recorded in
`fold_blocks_data.json`.

## Decision and next work

This warrants continuing the code-boundary representation as the leading
candidate. It now satisfies the basic performance test on a complete,
practically motivated large-residual reconstruction. Its applicability is
narrower than a general exact MPS backend, and that restriction is part of why
the implementation stays manageable.

Before production integration, separate the generated constants from the native
diagnostic and specify the eligibility certificate for an input Clifft circuit.
The current prototype starts from the structured reconstruction; it does not
recognize arbitrary user circuits. Unsupported injection, code transport, noise,
cat preparation, or fold graph must decline before dispatch. Generic Clifft
composition, including any conversion at a live residual boundary, remains open.

The next external-validity target is full f7 and the authors' exact Reg5/f7
artifacts, followed by a controlled many-shot comparison against other relevant
backends. The earlier d7 overlap result alone does not establish full f7 cost.
Five-round coherent memory remains a fallback/control workload; one-round
memory and few-shot setup advantages remain outside this study's objective.

## Reproduction

Use the offline environment from `fold_composition.md`, including optional
Cirq Core for independent validation, and a C++20 compiler. Generated programs
include the repository's native research header by absolute path.

```sh
python tools/profile/fold_blocks.py --distance 5 --histories 64 --stress-histories 32 --validate --output /tmp/blocks-f5.json
python tools/profile/export_fold_blocks.py --distance 5 --output /tmp/blocks-f5.cpp
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic /tmp/blocks-f5.cpp -o /tmp/blocks-f5
/tmp/blocks-f5 100 20000
```

Repeat with distance 3 for the smaller control. With the prior reconstructed
circuits and profiler build:

```sh
build-research/profile_structure /tmp/clifft-fold-cultivation/reconstructed_f5_p0.001.stim 128 22 1
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_*.py'
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer /tmp/blocks-f5.cpp -o /tmp/blocks-checked
ASAN_OPTIONS=detect_leaks=0 /tmp/blocks-checked 1 100
```
