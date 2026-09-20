# SOFT d7 physical circuit investigation

## Decision

Follow-up: the [coherent Clifford gadget experiment](clifford_gadget.md) derives
and validates a two-Clifford-term reduction of these physical intervals. Its
offline sampler is not yet competitive with ordinary Clifft, and its dynamic
stabilizer work is not a production execution design.

This is a useful large-state stress input, but **not yet a validated d7
cultivation benchmark**. An independent exact calculation gives a 3/8 firing
probability for detector 69 with every physical noise probability set to zero.
This happens in the 19-data-qubit stage, before the 37-data-qubit stage.
The source needs a protocol/observable audit before its acceptance or logical
error rate can establish a cultivation result. Do not silently repair it or
choose a favorable postselection mask.

Current Clifft already simulates the complete file at about 140 attempts/s on
one core, with an 8 MiB dense coefficient array. The compiled CSS-block
prototype declines it. Numerical trajectory diagnostics find substantial
actual stabilizer nullity inside the expensive interval: approximately 18
versus allocated width 19. After the inverse T layer, nullity returns to one
while width remains 19. Recovering that final compression alone would miss
most of the expensive work.

The evidence supports investigating **compiler-certified physical intervals
within Clifft**, not replacing Clifft's state representation. It does not
justify promoting the current prototype or enlarging its synthetic certificate.
The next missing capability is an exact ancilla-instrument interval certificate
that handles mixed T/T_DAG layers and correlated physical fault effects, with
ordinary state/record handoff. A correct large physical circuit is still needed
to demonstrate that this investment improves cultivation simulation.

## Provenance and protocol scope

The [fixture manifest](../fixtures/soft/manifest.json) pins the exact
[SOFT file](https://github.com/haoliri0/SOFT/blob/686051afe06e28c433fffec1c61686458728ca2e/benchmark/circuit/MSC_circuit_d7_p0.0005.stim),
its distributed Apache-2.0 license and benchmark configuration. SHA-256:
`83760fdb1e71be32c1e1eb83bbbf4176e9d32adb48384b14b818b294e7bf25f5`.
The 57,925-byte, 860-line file was added in commit
`77f1f388f4c2c5d6412c52259a8c9388677b8f9f` ("sync symft with SOFT").
No generator, derivation or earlier artifact provenance was found in the
published history inspected for this file. The upstream circuit manifest
does not list this input. Its benchmark config does, with postselection false.

The input has 80 physical qubit indices, 355 measurements, 344 detectors,
and 127 physical T/T_DAG applications. It starts with noisy injection and
contains growth and ancilla extraction around 7-, 19- and 37-data-qubit T
layers. The latter two stages use mixed T and T_DAG signs, followed later by
their inverse layers. It contains two- and one-qubit depolarization, reset
Pauli errors and measurement readout errors, all at 0.0005. Repeated syndrome
extraction and noise on idle wires remain included. There are no
record-controlled CX operations in this source.

The final operation at line 823 directly measures one all-data Y product
and 36 CSS X/Z checks. Observable 0 is the XOR of contributions at lines
781 and 824, not two independent logical observables. The terminal noiseless
product measurements and their detector comparisons are part of the task.
The file ends there; it does not contain a later escape/storage experiment.
Neither the code-distance label nor the construction's fault distance is
certified by this investigation.

This is distinct from SOFT's explicitly named unverified d7 proxy, and from
the fold-transversal surface-code f7 protocols in the author requests.
Those requests remain useful.

## Validation and the concrete failure witness

Noise removal preserves every physical gate, target and source line, including
noise inside indented REPEAT blocks. Source checksums guard against accidental
replacement of the circuit. The independent exact oracle works backwards
from a requested record parity through Clifford operations, T rotations,
measurement instruments and resets. It uses integer coefficients in
Q(sqrt(2)); there is no numerical coefficient truncation. Stim supplies syntax
parsing only. The propagation code does not call Clifft's planner or executor.

For detector 69, declared at source line 475, the oracle obtains

```text
E[(-1)^D69] = 1/4
Pr(D69 = 1) = (1 - 1/4)/2 = 3/8
```

The [exact result](soft_cultivation_exact.json) records the source hash and
524,288-term peak. Evaluation took about 50 seconds in the initial run.
The bounded oracle is a correctness witness, not a proposed fast simulator.
Its small-circuit checks cover reset and measurement duals, Y signs, and
random three-qubit Clifford+T expectations against Qiskit Statevector and Aer.

Across 768 noiseless Clifft attempts, D69 fired 296 times. Only 126 attempts
had all **raw** detectors zero; 97 of those had observable 0 set. These are
uninterpreted output counts, not accepted cultivation shots or logical-error
estimates. In 768 noisy attempts only five had all raw detectors zero, and
all five had observable 0 set. These small counts are not rare-error evidence.

The S proxy reveals a separate reference convention: raw detector 270 is
deterministically one, and all other noiseless detector and observable bits
are zero. Stim's default detector sampler subtracts this reference. The study
restores it before comparing noisy S-proxy outputs against Clifft. Over
32,768 shots, output marginals and the all-raw-zero event agree within the
seven-standard-error smoke-test tolerance (largest marginal difference
0.00568). This is not a proof of joint-distribution equality or a T-state
oracle. Raw record-to-detector/observable parities are independently checked
with Stim for every full-record T benchmark batch.

A full physical-prefix Aer MPS attempt with 128 shots exceeded its 60-second
budget and produced no validation result. The exact Pauli witness replaces
that attempted check; no full d7 Aer agreement is claimed. The observed
randomness could indicate missing corrections, an inappropriate T conversion,
or an undocumented acceptance/observable contract. The study does not select
one of those explanations without a derivation.

## Current Clifft baseline

Sampling source: `faa53693cef399197e27bd39c325a07acd438c68`, with study additions
at `41a18d45`. The checked upstream main was
`8a00dbe3b102e3ff23edfdce90e49c0feef2b4ad`; its only change from the sampling
baseline was `playground/src/components/KHistoryChart.tsx`. Thus the production
sampling implementation measured here matches upstream main at inspection.

Release native GCC 13.3, AVX-512, AMD EPYC 9554P VM, one worker pinned to logical
CPU 0, scalar batch size 1. Each rate is the median of three calls of 256
attempts after 16 warmup attempts. Compilation is separate; public-API worker
setup and output construction are included. No postselection or reference
normalization is enabled. Aggregate-only uses `sample_survivors` with no mask
and `keep_records=False`; every attempted shot completes.

| Input | Compile time | Full-record attempts/s | Aggregate-only attempts/s |
| --- | ---: | ---: | ---: |
| Zero noise | 9.9 ms | 138.0 | 139.1 |
| Source noise p=0.0005 | 97.0 ms | 140.2 | 138.4 |

Both plans peak at width 19. Coefficients occupy 8 MiB; coefficients plus the
scalar measurement scratch allocation occupy 12 MiB. Those are exact array
sizes, not total RSS. The Python process's high-water RSS, including later
Walsh-transform analysis arrays, was 217 MiB. Immutable plans, symbols,
records, Python and allocator overhead are outside the 12 MiB figure.
The [raw report](soft_cultivation_data.json) preserves timings and diagnostics.

The [historical Ticit benchmark](https://github.com/inmzhang/ticit/blob/main/docs/benchmark.md)
reported 106/108/52 attempted shots/s for Ticit/SymFT/Clifft on an i5-14600KF
at older revisions, also without d7 postselection. Those numbers are provenance
context, not a matched comparison to this VM or a current speedup claim.
Since the postselection contract is unverified, this study does not report
accepted shots/s or benchmark an invented early-rejection mask.

## Where the work and non-stabilizer state occur

The ordinary source-mapped plan estimates 27,992,255 unfused coefficient visits;
88.0% occur at width 19. This is a static work estimate, not measured CPU-time
attribution or a model of fused execution.

| Source region | Estimated fraction of dense visits |
| --- | ---: |
| Forward 37-data mixed T layer, lines 685-686 | 20.3% |
| Ancilla X measurement, line 729 | 3.7% |
| Inverse mixed T layer, lines 776-777 | 68.4% |
| Terminal product measurements, line 823 | 7.5% |

The snapshot tool runs ordinary compiled circuit prefixes and exports the
completed conditional pure state after dispatch. Offline analysis filters
candidate Pauli X translations by probability-magnitude symmetry using Walsh
autocorrelation, then checks all Z phases for each remaining candidate. The
large snapshots had only one or two candidate translations, avoiding a 4^19
scan. The reported nullities use floating-point tolerances; they are numerical
diagnostics, not certified exact ranks. No such discovery runs inside Clifft.

| Prefix endpoint | Compiled active width | Numerical stabilizer nullity |
| --- | ---: | ---: |
| Before 19-data T layer, line 400 | 1 | 1 |
| After forward 19-data T layer, line 404 | 10 | 9 |
| After inverse 19-data T layer, line 471 | 10 | 1 |
| Before 37-data T layer, line 684 | 1 | 1 |
| After forward 37-data T layer, line 686 | 19 | 18 |
| After ancilla measurement, line 729 | 18 | 17-18 |
| After inverse 37-data T layer, line 777 | 19 | 1 |
| Before terminal evaluation, line 822 | 19 | 1 |

Each endpoint was sampled with seeds 71 and 72, both without and with source
noise. Prefixes are independently compiled and sampled: matching seeds do not
assert a shared fault history across different prefixes. This small set does
not cover every fault trajectory. Conditional dormant Pauli frames and
compiler coordinate maps are Clifford operations, so they do not change the
nullity of the exported active pure state.

For the sampled peak states, shared-basis conservatism accounts for roughly
one coordinate, not the gap from 19 to one. Large improvements therefore need
to avoid materializing the high-nullity interval. The return to nullity one
does suggest a useful boundary certificate, but recovering it only after line
777 leaves the preceding 92.5% of estimated dense work already performed.

## Prototype fit and smallest missing capability

The assessed prototype is commit
`ac84c347d1c64c0472bbf00d8b21e78f71817465`, not other attempts from the issue.
Its `try_plan_css_blocks` checks whole-circuit physical width before building:
80 exceeds its 63-data limit and is even, so it returns
`CSS data width outside selected range`. This conclusion follows directly
from its entry condition at `src/clifft/sampling/css/planner.cc:366`; it is
a source-level rejection diagnosis, not a newly timed prototype execution.

Increasing that limit would not address the mismatch. The certificate also
requires ideal encoded preparation, no spectator/ancilla wires, identical
full-data T layers and direct all-Y/full-CSS measurements. This input contains
growth, noisy ancillary CX extraction, correlated two-qubit faults, mixed
T/T_DAG layers, and a terminal instrument with many records. Its small encoded
boundary states do not by themselves certify the intervening instrument.

A defensible next extension would select a 37-data interval after growth,
prove its incoming syndrome/code frame, compile the full ancilla measurement
instrument including physical noise and reported-bit flips, and return its
logical coefficients and complete record/frame effects to the ordinary
executor. Planning remains offline; execution applies fixed descriptors with
preallocated storage. Simply replacing noisy ancillas by an ideal MPP would
change the task. The interval's fault-dependent coherent effects need an exact
certificate and differential noisy-record probabilities before implementation
can be justified as a general improvement.

No production optimization is introduced by this study. The concrete output
is the reproducible failure witness, current baseline, source-cost profile,
and trajectory diagnostics. They sharpen the next experiment without claiming
a speedup or working fault-distance-7 cultivation result.

## Reproduction

```bash
cmake --build build-study --target profile_cultivation snapshot_cultivation -j 4
.venv/bin/python tools/profile/soft_cultivation_study.py \
  --profiler build-study/profile_cultivation \
  --snapshot build-study/snapshot_cultivation \
  --output tools/profile/research/soft_cultivation_data.json
.venv/bin/python tools/profile/cultivation_pauli_oracle.py \
  --output tools/profile/research/soft_cultivation_exact.json
.venv/bin/python -m pytest -q tools/profile/test_soft_cultivation_study.py \
  tools/profile/test_cultivation_study.py tools/profile/test_cultivation_aer.py
```

Configure `build-study` with `CLIFFT_BUILD_PROFILER=ON` as in the earlier
[corpus study](cultivation_corpus.md). Run the timing and exact-oracle commands
sequentially. The native snapshot's temporary binary format is host-native
float64 real coefficients followed by imaginary coefficients; the Python
driver consumes it locally and retains only compact diagnostics.
