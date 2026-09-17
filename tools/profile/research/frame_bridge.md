# Measuring through a compiled Clifford fault frame

The measurement boundary from [the fault-frame study](fault_frames.md) can be
handled exactly using precomputed coefficient substitutions. The result is the
same two weighted data monomials already used by compiled logical blocks. It
does not require a tensor contraction over cat qubits, a runtime tableau, or
discarding cat-data corrections.

This study implements that bridge in the standalone native sampler and compares
complete attempted shots, including physical-noise generation and retained
survivor outputs. It covers the original full f7 reconstruction and the held-out
shuffled-syndrome/changed-growth implementation from the adapter study. No
production dispatcher or public API changes are included.

**Decision: stop pursuing this bridge as a separate execution direction.** It
closes the supported measurement gap correctly, but gives no demonstrated
throughput improvement or additional circuit coverage. Keep the original
logical-block fold evaluator. Retain the fault-frame algebra, bridge prototype,
and surviving cat-data fault cases as research evidence and regression coverage.

## How the bridge works

In the ideal core, the two uniform computational cat strings select data
operators U_0 and U_1. Both operators are compiled once, with their phases. For
the current folds U_0 is identity and U_1 is the fold Clifford. Cat preparation
faults initialize the shared Clifford fault frame E; the core's fixed descriptor
updates it as in the previous study.

To contract the actual cat measurement, substitute the ideal cat string b into
E's quadratic phase polynomial. A data-data CZ remains a data CZ. A cat-data CZ
becomes a data Z when b=1. Cat-only linear and quadratic terms become branch
phases. The cat X part chooses the actual decoder-table column. For decoder
record r, the data Kraus operator is

```text
K_r = (1/2) sum over b=0,1 of D[r, b*all_cat_bits xor x_cat] E_b U_b.
```

Here D is the existing precomputed cat-decoder sign/zero table, and E_b is the
restricted data monomial with all constant phases retained. Preparation flags
and decoder faults use the same compiled linear maps as the original sampler.
No separate data-only exit rule is needed for this supported measurement.

This identity holds as an operator on arbitrary data inputs. In particular,
the data cannot be projected into its code prematurely. The sampler composes
these monomials with the incoming coherent terms and waits for the existing
certified code boundary before contracting to two logical amplitudes. It still
uses at most four coherent terms between the two checks.

The bridge compiles all edge routes, data/cat coordinate maps, ideal operators,
and decoder behavior. Native execution only updates and substitutes coefficient
values. Fixed-capacity frame and monomial storage is allocation-free; no hot
exception, topology discovery, or adaptive basis selection is added. A mixed
plan with ordinary branches for the smaller checks and bridge branches for f7
also passes every full-circuit fixture, demonstrating preplanned fallback.

## Exercising the actual measurement gap

Failing the previous sufficient data-only rule does not by itself imply a
cat-data CZ: cat-only Paulis also fail that rule. The 24 selected single-fault
exit failures per circuit all reject in the complete reference. They are useful
operator checks but insufficient surviving-history coverage by themselves.

Ten additional legal histories put data X faults just after corresponding CCZ
gates in the first and second f7 checks. The first check leaves a data-only X
correction, which passes through cat decoding and preparation. Conjugating that
X by the second CCZ creates a cat-data CZ; the second X cancels the data flip.
At the second measurement the common correction is precisely one cat-data CZ.
Every such history has full-circuit acceptance probability 1/4 in both the
original and held-out circuits.

This closes a meaningful gap in the previous screen: the changed measurement
must be handled on nonzero accepted branches. The Python test verifies the
correction's exact structure and survival, and the independent CH/Stim reference
checks full-circuit acceptance and all unnormalized logical XYZ probes. The
largest reference discrepancy across all 68 new boundary histories is
5.56e-17.

## Validation

- An independent three-qubit Aer instrument test compares full data Kraus
  matrices for 64 noisy preparation/core/decoder histories. It checks arbitrary
  coherent data inputs, including interleaved gates that produce cat-data
  phases. Only a whole-history global phase discarded by the pre-existing
  Clifford noise maps is aligned; relative branch phases and norms are checked.
- Python compares complete branch operators for every supported single Pauli
  placement in preparation, core, and decode at distances 3, 5, and 7, plus
  dense fault histories. Weight signs are normalized into monomial phases before
  exact comparison.
- Each native circuit variant checks 3,792 fault-history branch operators and
  another 768 cases with arbitrary incoming data Paulis. The latter propagate
  the incoming Pauli into E and contract its cat-data phases directly, then
  compare with composing the original branch operator with that same Pauli.
- Each variant checks 96 full-circuit fixtures, including the previous 62
  histories and 34 new boundary cases. The bridge, ordinary evaluator, mixed
  fallback evaluator, and Python expected results agree within 2.23e-16.
- Full survivor comparisons retain all 350 measurements, 348 detectors, and
  five probes per accepted shot: logical XYZ and two physical Pauli probes
  with an intervening terminal Clifford. Measurement/detector rows and survivor
  counts match exactly; floating probe differences must stay below 2e-12.
- The native bridge, incoming-frame contraction, fixtures, and sampler pass
  ASan, UBSan, and LeakSanitizer. The Clifft core library linked by the probe is
  the existing release library; the new inline research kernels are instrumented.

The existing reusable plan worker is rebuilt against the modified research
headers for the regression suite. The new native executables are also used by
two explicit Python tests, so a generated program that accidentally runs only
the baseline cannot pass those tests.
All 115 research tests pass in 212.0 seconds. Repository formatting, lint,
type, and hygiene hooks pass before committing.

## Full sampling comparison

The two paths share the same noise sampler, code-boundary contraction schedules,
output adapter, RNG algorithm, and per-trial seed. Only the fold evaluator
changes. The research headers expose a templated evaluator seam; ordinary calls
retain the existing behavior. Both paths preallocate survivor output storage,
and both allocate that storage within the measured sampling call.

Each trial attempts 100,000 shots at p=0.001. The evaluator order alternates
between trials. Builds and correctness tests finish before timing, and the two
circuit variants run sequentially on the VM. Generated code uses GCC with
`-O3 -march=native`, without fast-math. Circuit certification and one-time data
planning are outside the timer. Per-shot fault generation, quantum acceptance,
contractions, and all retained rows/probes are inside it.

The adjacent raw JSON records every timing repetition and output check. These
are sustained full-sampler measurements; the approximately 0.93 us/core frame
update from the previous study omitted the measurement bridge and cannot be
used as an end-to-end speedup claim.

| Median cost per attempted shot | Original f7 | Held-out growth and syndrome implementation |
| --- | ---: | ---: |
| Existing logical-block evaluator | 45.40 us | 45.70 us |
| Fault frame plus measurement bridge | 46.15 us | 46.11 us |

Original-circuit trial ranges are 45.35-47.46 us for the existing evaluator and
46.01-46.51 us for the bridge. Held-out ranges are 45.24-46.02 us and
45.68-46.13 us. Bridge medians are approximately 1.7% and 0.9% higher, respectively.
With three trials on an unpinned VM, these small differences are not a claim
of a precisely established performance penalty. They show no practical gain
that would justify the extra evaluator. There is no best-in-class comparison.

Across the two variants, 600,000 attempted histories are compared between the
two methods (1.2 million total sampler attempts). They produce the same 75,589
survivors per method, with 26,456,150 measurement values and 26,304,972 detector
values compared exactly. The largest difference among 377,945 retained probe
values is 1.12e-16. Original per-trial survivor counts are 12,667, 12,580, 12,586;
held-out counts are 12,553, 12,604, 12,599.

## Scope and continuation

This bridge supplies an exact derivation and implementation of the supported
measurement, not a new general state representation. It returns the same two
monomials as the original logical-block evaluator. Broader recognition is still
limited by the current code geometry, injection/growth certificates, protocol
skeleton, and terminal postselection contract. No authors' circuit artifact,
final-error-correction decoder, or arbitrary live-state handoff is added.

The bounded study's decision rule was to retain a second execution direction
only if complete measurements showed a practical throughput or coverage benefit.
That condition is not met. The previous small-state storage result remains
valid; the frame's measurement contraction recovers the structure already used
by logical blocks, so it does not establish an independent superior simulator.
Other frames, a different protocol family, or a stronger CAMPS implementation
are not ruled out by this comparison.

The next useful coverage target is an actual different MSC benchmark
schedule or gadget structure, with gate-derived boundary certificates; more
permutations of the existing reconstruction would not establish that coverage.
Five-round circuits and large residual states remain the targets. One-round
memory, few-shot startup, and SIMD optimization remain outside this work.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 python tools/profile/study_frame_bridge.py --output /tmp/frame-bridge.cpp --metadata /tmp/frame-bridge-export.json --independent
OPENBLAS_NUM_THREADS=1 python tools/profile/study_frame_bridge.py --held-out --output /tmp/frame-bridge-held.cpp --metadata /tmp/frame-bridge-held-export.json --independent
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -I src -I build-research/generated /tmp/frame-bridge.cpp build-research/src/clifft/libclifft_core.a -o /tmp/frame-bridge
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -I src -I build-research/generated /tmp/frame-bridge-held.cpp build-research/src/clifft/libclifft_core.a -o /tmp/frame-bridge-held
/tmp/frame-bridge 100000 3
/tmp/frame-bridge-held 100000 3
CLIFFT_FRAME_BRIDGE=/tmp/frame-bridge CLIFFT_FRAME_BRIDGE_HELD_OUT=/tmp/frame-bridge-held OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_fold_frame_bridge.py'
```

Use the existing NumPy/Stim/Cirq/Aer research environment and scalar release
core library. See [frame_bridge_data.json](frame_bridge_data.json) for results.
