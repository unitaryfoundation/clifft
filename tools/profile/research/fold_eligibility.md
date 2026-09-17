# Compiled logical blocks: eligibility and output composition

The bounded integration experiment supports a narrow, useful entry point:
recognize a complete certified cultivation family before HIR optimization,
bind its physical noise probabilities, and produce Clifft's existing
`SamplingSurvivorResult`. It preserves the logical-block speed at f7 while
returning full accepted measurement records and terminal Pauli expectations.
It does not change any production code, public API, or executor invariant.

This follows [the alternatives comparison](alternatives.md) and the
[full f7 reconstruction](fold_f7.md). The circuits remain paper-guided
reconstructions, not authenticated author artifacts. The physical mechanism is
still two logical amplitudes and a Pauli frame at certified code boundaries,
with short coherent sums and precomputed contractions inside fold checks.
The new work establishes how to recognize and expose that restricted method.

## Eligibility contract

`export_fold_recognizer.py` generates three trusted family certificates and
their existing offline contraction kernels. `fold_recognition.h` matches an
actual parsed Clifft `Circuit` against these certificates. It never treats
stage comments, filenames, or hashes as evidence of eligibility.

| Input or requested behavior | Decision |
| --- | --- |
| Full reconstructed f3/f5/f7 schedule | Accept |
| Consistent injective relabeling, including sparse qubit indices | Accept |
| Comments, coordinate metadata, TICK, parser aliases, equivalent batching or REPEAT expansion | Accept after Clifft parsing |
| Same physical channels at certified sites, with independently chosen probabilities in [0,1] | Accept |
| Omitted physical channels | Accept as zero probability |
| CCZ written using Clifft's exact parser decomposition | Accept, provided no extra internal noise is inserted |
| Noiseless named one/two-qubit Clifford prefix on recognized wires | Accept: certificate proves every wire is reset before first use |
| Noiseless named one/two-qubit Clifford suffix on data, with interleaved EXP_VAL probes | Accept through compile-time probe transformation |
| Ordinary survivor sampling, all detectors postselected to expected zero | Accept, with or without retained rows |
| Changed internal gates, wiring, syndrome order, extra physical noise, or different CCZ decomposition | Decline conservatively |
| Partial/nonzero postselection, observables/decoding, fixed fault weight, ordinary unpostselected records, likelihood/replay, instruments | Unsupported |
| Suffix measurements, noise, non-Clifford gates, or ancilla probes | Decline; requires additional output semantics or live-state execution |

The matching is deliberately conservative. Semantically equivalent gate
reorderings are not generally recognized. It is a recognizer for three
certified schedules with parameter and boundary variations, not a general
gadget-discovery algorithm. Existing MSC corpus circuits are different
schedules and decline. That is a coverage gap, not evidence that their
physics cannot support logical-block simulation.

The request adapter in this standalone program only models ordinary survivor
sampling and explicit negative controls. Unsupported public API entry points
would need to bypass this candidate in a production dispatcher. No such
dispatcher is installed here.

## Output semantics and the representation boundary

For each physical Pauli history the kernel returns acceptance probability P
and unnormalized logical XYZ. Sampling performs an independent Bernoulli
acceptance draw, then emits the conditional survivor row. Every ordinary
postselected measurement is zero. In f7 each of the two six-bit verification
groups is an independent fair choice between 000000 and 111111, conditioned
on acceptance. The earlier GHZ8 tensor GHZ6 certificate and Aer checks justify
this reconstruction even under Pauli faults: the two flag strings have the
same conditional data state. The implementation samples both possibilities.

All detector outputs are zero, and no observable columns exist under this
contract. `keep_records=false` returns only counts, including no `exp_vals`,
matching the existing survivor API. With records enabled, measurements,
detectors, and expectations use the existing row-major result type. A separate
deterministically seeded record RNG makes the retained-record option leave
the physical fault and acceptance draws unchanged. Seed replay is defined
within this experimental backend, not against generic Clifft's RNG stream.

For a suffix Clifford C and requested physical Pauli O, planning computes
`C^-1 O C`. If it anticommutes with a code stabilizer, its expectation is zero.
Otherwise signed stabilizer elimination reduces it to logical I, X, Y, or Z.
The hot sampler uses only that precomputed component and sign. This includes
probes at intermediate points in the Clifford suffix and preserves physical
Pauli signs; it does not assume that C preserves the code space.

All matching, tableaux, elimination, and probability binding happen before
sampling. Coefficients, history, and maximum output storage are allocated
before the shot loop. The hot path contains no allocation, exceptions, or
topology discovery. Completed result vectors shrink after sampling, as output
storage rather than live executor state. The native program is single-threaded.

A live residual-state handoff is a different architectural task.
`Executor::resume` requires the continuation to preserve active-coordinate
meaning/order, live coefficients, symbols, records, and RNG; it is not an
arbitrary encoded-state importer. This experiment neither uses that API for a
different purpose nor adds an alternate runtime tableau interpreter. Whole
circuit routing can share the result type without solving representation
handoff. A future prepared-plan variant or live handoff needs an explicit
architecture proposal before implementation.

## Measurements

Raw process results, input hashes, compiler information, and source hashes are
in [fold_eligibility_data.json](fold_eligibility_data.json). Timings use the same
VM, native scalar C++ kernels, and existing Release Clifft core as the preceding
study. The new recognizer is compiled without fast-math so finite-probability
validation remains meaningful.

Sampling uses three separate runs of 100,000 attempts per family/output mode,
physical noise p=0.001, and seed 19331. The sample timer includes executor setup,
output allocation/zeroing, conditional flag reconstruction, and completed-row
resizing; it excludes file parsing, recognition, JSON emission, and process
startup. RSS is process maximum, including the family catalog and outputs.
This is sustained-sampling measurement, not a few-shot latency study.

| Circuit/output | Median us/attempt | Range us/attempt | Median process RSS KiB |
| --- | ---: | ---: | ---: |
| f5, counts only | 19.74 | 19.71-20.05 | 6,612 |
| f5, full survivor rows | 19.65 | 19.37-20.01 | 38,176 |
| f7, counts only | 45.62 | 45.39-46.63 | 8,104 |
| f7, full survivor rows | 46.15 | 46.13-46.80 | 78,580 |

The same seed gives 41,606 f5 and 12,667 f7 survivors in every output mode and
run, matching the preceding logical-block/proxy comparison. Small timing
differences between retained rows and counts-only are run variation, not
evidence that retaining records is faster. Full output preallocation costs
O(attempts times record width), even though the internal state is small. A
bounded streaming API could reduce that storage, but is outside this study.

Median recognition plus probability/probe binding was f5 0.250 ms, f7 0.550 ms.
This cost is paid once per parsed circuit. Reference-catalog setup ranged
from 2.26 to 8.77 ms per process and is reported separately from parsing
the supplied circuit.

All eight checked corpus circuits other than one-round circuits decline and
then successfully compile through the existing trace/optimization/planning
path. No dense state is allocated for this fallback check. In particular,
five-round coherent d5 remains width 13. A late mismatch consisting of an
otherwise eligible f7 circuit followed by T also declines and retains the
ordinary width-44 plan. Fallback is safe routing, not a promise that generic
Clifft can afford to execute that width.

Corpus rejection costs ranged from 1.10 to
10.35 us on already parsed inputs with the catalog prepared.
Their subsequent ordinary compilation took 0.76-23.63 ms, excluding
parsing. The late f7 mismatch cost 0.547 ms to reject versus
129.73 ms for subsequent ordinary compilation.

Catalog construction parses the embedded reference circuits and prepares
code-elimination bases once per process. These costs are separate from
recognition and should be cached if integrated. The heavy contraction plans
are generated and compiled offline for the three families; their generation
and C++ compilation are not included in these online timings. No per-shot
compilation is needed for the supported family/noise parameters. A new schedule
still needs a new certificate/plan, so these results do not establish that
compile-once is best for arbitrary circuits.

## Validation and reproduction

The complete research suite passes 69 tests, including 12 new native-recognizer
tests. They cover parser-equivalent syntax, sparse relabeling, omitted and
nonuniform noise, exact CCZ decomposition, rejection of internal CCZ noise,
changed schedules and unsupported requests, reset-erased prefixes, ordinary
fallback compilation, deterministic output, both f7 flag groups, and selected
physical flag-readout faults. Up to eighty generated Clifford/probe checks per
distance compare against Stim conjugation and known signed code operators.
The existing suite retains its independent Aer/CH physical-state checks and
fixed-history native-kernel checks. ASAN/UBSAN passes all 12 recognizer tests
with leak detection disabled; the new research code is instrumented, while
the linked pre-existing Clifft core is the ordinary Release build.

Use the research Python environment containing Stim, NumPy, Qiskit Aer, and
Cirq. With `build-research` containing the existing scalar Clifft core:

```sh
OPENBLAS_NUM_THREADS=1 python tools/profile/export_fold_recognizer.py --output /tmp/recognizer.cpp
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -I src -I build-research/generated /tmp/recognizer.cpp build-research/src/clifft/libclifft_core.a -o /tmp/recognizer
/tmp/recognizer tools/profile/fixtures/fold_cultivation_f7.stim 100000 19331 1 100 ordinary
CLIFFT_RECOGNIZER_NATIVE=/tmp/recognizer CLIFFT_PROXY_NATIVE=/tmp/proxy-build/profile_proxy OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_*.py'
```

Arguments are circuit path, attempts, seed, keep-records flag, recognition
repetitions, and request mode. Repeat timing runs three times; use keep-records
0 for counts only. `fallback` compiles a declined circuit normally without
sampling it. `fixed`, `records`, `partial`, and `nonzero` exercise unsupported
request contracts. Input parsing errors remain errors rather than successful
recognition. At most 1,000 attempts emit detailed row diagnostics.

Build the sanitizer executable with `-O1 -g -fsanitize=address,undefined
-fno-omit-frame-pointer` instead of the optimization flags, and set
`ASAN_OPTIONS=detect_leaks=0`. The recognizer certificate references its trusted
family catalog; the catalog must outlive the certificate and sampling call.

## Decision and next research step

Proceed with this restricted representation, but establish useful coverage
before production dispatch. The immediate next experiment should replace
whole-schedule matching with certificates for independently varied syndrome
and growth schedules, beginning with a second f7 schedule. Certify code-space
and noise-transfer properties from gates, reuse the existing contraction
kernel only where those properties hold, and test against the independent
CH/Aer references. This directly tests whether the method is a useful class of
simulations rather than a fast implementation of one generator.

Keep that work bounded: if adapting a second schedule requires extensive new
special cases or runtime topology analysis, retain the current method as a
specialized research backend and revisit Clifford-frame/tensor alternatives.
Do not build a general state-switching framework first. The large-fold frame
study and direct noise-averaged contraction remain open independent directions.
There is still no measured generic-Clifft f7 runtime or best-in-class claim.
