# Logical-block simulation across changed syndrome schedules

The representation survives two changes to the full f7 syndrome schedule with
no changes to the native executor, contraction geometry, or state capacity.
The offline compiler now verifies what each syndrome gadget does from its
gates, instead of assuming the reconstruction's stabilizer order. Both changed
circuits agree with the independent coherent reference on 115 histories each.

This follows [the eligibility study](fold_eligibility.md). It closes one
specific coverage gap: the block planner can certify and compile different
syndrome schedules without a hand-derived implementation for each schedule.
It does not yet discover those regions in arbitrary parsed circuits or close
the artifact, growth-schedule, decoding, and live-state handoff gaps.

## Circuits and why the changes matter

The two variants alter every syndrome stage, including the initial d3 stage,
the rounds following growth, and the terminal ideal round:

- `reverse_support`: reverse the order of the CNOTs within each stabilizer's
  ancilla measurement circuit.
- `reverse_checks`: also reverse the order of the complete stabilizer checks,
  so measurement record order and the sequence of X/Z checks change.

All other stages remain fixed. The circuits still have 99 physical qubits,
221 non-Clifford gates, and 2,205 physical noise sites. Each keeps exactly the
same code stabilizers and ideal postselection contract, but the noisy physical
process is not assumed equivalent to the original circuit.

For example, the first d7 X check originally couples ancilla 85 to data
0, 1, 7. An ancilla X fault after the first CNOT spreads to data 1 and 7.
Reversing that check to 7, 1, 0 makes the corresponding fault spread to data
0 and 1. Independent Stim propagation verifies this difference. Reordering
whole checks also changes which subsequent measurements can detect a fault.
Each variant is therefore validated against its own gates and noise history.

The chosen reorderings are controlled experiments, not recommended hardware
schedules. No claim is made that they retain fault distance seven or have the
same logical-error rate. Growth, injection, cat circuits, and fold graphs were
held fixed to isolate syndrome-order dependence.

## Gate-derived certificate

`fold_schedule.certify_syndrome` splits a declared syndrome region into ancilla
reset/unitary/measurement gadgets and verifies the following before execution:

1. Each gadget resets and measures one non-data ancilla and touches only that
   ancilla and the declared data. The currently admitted gate alphabet is H/CX.
2. Conjugating the ancilla Z readout backwards produces a positive data Pauli
   times the ancilla's initial Z stabilizer.
3. The entire Clifford tableau equals the canonical nondemolition measurement
   of that data Pauli, up to an irrelevant overall phase.
4. The region measures every stabilizer of the declared code basis exactly
   once, in whichever order the supplied gates actually use.

Step 3 matters: obtaining the right readout observable alone does not prove
that the circuit leaves the data unchanged apart from projection. A test adds
an H on an unrelated data qubit before readout; it leaves the pulled-back
observable unchanged but fails the full-tableau certificate.

The resulting ordered stabilizers replace the previous implicit canonical
ordering when building syndrome-sector maps and constraints through growth.
Symbolic Pauli propagation regenerates the physical fault maps from the actual
gate sequence. The existing growth certificate still verifies signed code and
logical transport. Nonzero syndrome sectors retain their Pauli representatives
and coherent contributions; they are not replaced with an early code projection.

The region certificate does not contain a list of the two allowed permutations.
A held-out, randomly shuffled f7 schedule is also compiled directly from its
gates and checked against the coherent reference. The variant generator is
only a source of controlled circuit inputs.

The research `Protocol` constructor now accepts a supplied structured
reconstruction, checking its stage contract and fixed injection/morph prefix.
`export_fold_recognizer.py --schedule ...` generates a catalog for the selected
variant using the same exporter and native sampler. This remains an offline
structured-input route: the C++ recognizer still matches the resulting complete
schedule. An old catalog correctly declines the new schedule, and vice versa.
There is no production dispatcher or arbitrary-circuit region finder.

## Correctness and performance

Each variant's 115 fixed histories consist of 64 natural-noise histories at
p=0.001, 32 stress histories at p=0.01, ten paired hooks, a logical-error tail,
two selected nonzero growth sectors, and six ideal/flag/high-coordinate cases.
Both variants match the phase-preserving CH/tableau reference with maximum
probability error 1.12e-16 and conditional logical-probe error 2.23e-16. The
nonzero-sector cases still accept with probability 1/4. These are diagnostic
strata, not weighted estimates of rare logical-error rates.

The same 115 fixtures per variant also match native execution to 2.23e-16.
The reverse-checks native fixture set plus 2,500 sampled attempts passed
ASAN/UBSAN with leak detection disabled. Aer independently verifies both
measurement-outcome Kraus matrices for the X/Z extraction gadgets and their
different support sizes. The complete research suite passes 76 tests, with
seven new schedule tests, including missing/duplicate/wrong stabilizers,
unwanted data actions, a shuffled schedule, and catalog eligibility boundaries.

Both variants retain at most four coherent monomial terms, 13 contractions on
a full traversal, 6,199 encoded fault bits, and the existing 2,851 complex
contraction scratch slots. Native execution has no new descriptors or runtime
tableaux, planning, allocation, or exceptions. Circuit-specific parity maps and
syndrome maps are the parts that change. Offline Python construction with the
new checks took approximately 1.8 seconds per f7 variant in individual setup
diagnostics; these are not online recognition timings.

| Schedule | Median us/attempt | Range us/attempt | Survivors per 100,000 |
| --- | ---: | ---: | ---: |
| original | 45.07 | 44.87-45.27 | 12,667 |
| reverse_support | 45.44 | 45.05-45.79 | 12,665 |
| reverse_checks | 47.80 | 47.16-47.98 | 12,610 |

Sampling comparisons use three independent process runs per circuit, each with
100,000 attempted shots at p=0.001 and seed 19331, retaining all survivor rows
and logical XYZ probes. Each repetition uses the same seed for timing stability;
the three runs do not supply 300,000 independent statistical histories.
Compilation, circuit parsing, and online matching are outside the sampling
timer. Setup and output allocation, physical-fault generation, acceptance,
conditional flag records, and probe output are inside it. Runs are sequential,
after tests and native builds have finished, on the unpinned VM.

Ordinary Clifft still plans peak active width 44 for both changed circuits,
so generic dense sampling was skipped. There is no measured generic-Clifft f7
speedup ratio. All nine catalog/circuit combinations were checked: the three
matching pairs accept and the six mismatched pairs decline.

Raw histories, certificates, source/input hashes, native checks, and timings are
in [fold_schedule_data.json](fold_schedule_data.json). The raw native correctness
and sanitizer diagnostics include incidental timing fields; those fields were
not used for performance comparisons.

## What remains and the next bounded step

There is now evidence for a reusable syndrome-boundary compiler rather than
just a changed hard-coded schedule. The evidence applies to sequential,
single-ancilla nondemolition measurements of the declared stabilizer basis;
simultaneous/interleaved extraction, different generator bases, and other
measurement instruments are not covered by this certificate.

The whole-circuit recognizer still requires a generated catalog entry. A useful
next integration experiment is to locate syndrome regions in parsed circuits
between unchanged certified injection/fold/growth blocks, run this certificate,
and build a plan from the supplied schedule. That would let an unseen supported
syndrome ordering enter without regenerating a catalog executable. It requires
an offline planning adapter, not a new hot representation or live-state handoff.

Keep that separate from the next physics coverage test: an independently varied
growth circuit, with signed code/logical transport and noisy sector behavior
checked against its own reference. The authors' unavailable artifacts and the
different existing MSC corpus circuits remain external-validity targets. This
study does not justify claiming broad corpus coverage or best-in-class speed.

## Reproduction

Use the research environment with Stim, NumPy, Cirq Core, and Qiskit Aer:

```sh
OPENBLAS_NUM_THREADS=1 python tools/profile/study_fold_schedule.py --schedule reverse_support --output /tmp/f7-support
OPENBLAS_NUM_THREADS=1 python tools/profile/study_fold_schedule.py --schedule reverse_checks --output /tmp/f7-checks
c++ -std=c++20 -O3 -march=native /tmp/f7-checks/native.cpp -o /tmp/f7-checks/native
/tmp/f7-checks/native 1 10
OPENBLAS_NUM_THREADS=1 python tools/profile/export_fold_recognizer.py --schedule reverse_checks --output /tmp/f7-checks/recognizer.cpp
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -I src -I build-research/generated /tmp/f7-checks/recognizer.cpp build-research/src/clifft/libclifft_core.a -o /tmp/f7-checks/recognizer
/tmp/f7-checks/recognizer /tmp/f7-checks/circuit.stim 100000 19331 1 100 ordinary
CLIFFT_RECOGNIZER_NATIVE=/tmp/recognizer CLIFFT_SCHEDULE_RECOGNIZER_NATIVE=/tmp/f7-checks/recognizer CLIFFT_PROXY_NATIVE=/tmp/proxy-build/profile_proxy OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_*.py'
```

The original recognizer and proxy executables are generated as described in
the preceding studies. Repeat the native recognizer timing command three times
for each catalog/circuit pair. For sanitizer validation, compile `native.cpp`
with `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` and run it with
`ASAN_OPTIONS=detect_leaks=0` and arguments `1 500`. Ordinary Clifft planning can
be checked with `build-research/profile_structure CIRCUIT 1 0 1`, which avoids
allocating a nonzero-width dense residual.
