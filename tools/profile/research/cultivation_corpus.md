# Physical cultivation corpus and baseline

## Decision

Follow-up: the [SOFT d7 investigation](soft_cultivation.md) found a public large
physical T input, but its noiseless outputs fail the expected cultivation
sanity checks. That report adds a baseline and an exact failure witness; it
does not replace the validated small-circuit corpus below.

The released Chan et al. circuits provide a useful physical correctness corpus,
including flags, noisy ancilla extraction and corrected Bell growth. They do
not supply the missing large-state performance target for compiled logical
blocks. Both d5 variants reach only ten active coordinates on current Clifft;
their dense coefficient array is 16 KiB. Do not enlarge the synthetic CSS
certificate on the strength of these measurements.

The next discriminating workload remains the exact Reg3/Reg5 T circuits used
by Hartweg and Pineiro Orioli. Requests for those artifacts and Takada et al.'s
fault-distance-3/5/7 physical circuits have been sent by the project maintainer.
The public Sahay Clifford examples are not substituted for these inputs.

There is a concrete compiler-oriented opportunity to investigate meanwhile:
move provably state-independent postselection ahead of coefficient work. This
study identifies candidate detectors and estimates remaining dense work; it
does **not** implement a new sampling path or report an achieved screening
speedup. Existing early rejection already captures much of the benefit.

## Corpus and validation

The [corpus manifest](../fixtures/cliffordea/manifest.json) pins four immutable
author files and their MIT license at revision
`a62a353e77468a2ff54c24e83133ad65a63e2294` of
[cliffordea](https://github.com/timchan0/cliffordea/tree/a62a353e77468a2ff54c24e83133ad65a63e2294).
The [corpus README](../fixtures/cliffordea/README.md) specifies the separate
S-to-T conversion and the already-applied d5 correction. Nothing is replaced
by ideal preparation or an idealized physical check. Appended noiseless
evaluation operations remain part of the requested task.

All declared detectors must be zero for acceptance. Observable 0 is the
authors' logical-error indicator. The study uses the declared conventions
without automatic syndrome normalization.

- Each complete T circuit gives trivial detector and observable outputs in
  256 zero-noise shots. Focused tests also check their record-to-output mapping
  independently with Stim.
- Each S proxy is compared with independent Stim sampling over 32,768 shots.
  Detector and observable parities agree exactly for supplied Clifft records;
  output marginals and acceptance agree within the stated seven-standard-error
  smoke-test tolerance. This is not a proof of joint-distribution equality.
- Both d3 T circuits are translated to elementary Qiskit operations and checked
  with Aer MPS trajectories, without bond truncation. The checks use 128
  noiseless and 2,048 noisy shots each. They include physical Pauli channels,
  measurements and resets, readout flips, and the terminal product measurements.
  [Raw oracle results](cultivation_aer_data.json) record versions and counts.
- The Aer translator fails on unsupported gates. Its readout construction is
  checked separately to ensure a flipped reported bit leaves the true
  post-measurement data state intact. It uses a parity ancilla, rather than
  changing the data state to implement classical readout noise.
- Fourteen focused tests cover source checksums, reversible gate conversion,
  zero-noise outputs, the native dependency analysis, source mapping and the
  independent measurement expansion.

No independent full d5 T oracle or rare logical-error-rate validation is claimed.
The fixed-fault samples are diagnostics, not enough data to establish fault
distance. Their strata are not combined into an unconditional error estimate.
The unflagged input differs in bytes from the Wan-Zapirain reference; comparing
its logical-error rates requires a separate convention reconciliation.

## Measured baseline

Baseline: Clifft `faa53693cef399197e27bd39c325a07acd438c68`, ordinary compilation
with default HIR passes. Release native GCC 13.3 build, AVX-512, AMD EPYC 9554P
VM, one requested CPU worker, automatic batch policy. Python 3.13, NumPy 2.4.2,
Stim 1.15.0. The report records the Python package version; the CMake and Python
version labels can differ, so the git SHA is the baseline identity.

Each timing is the median of three public-API calls of 16,384 attempts following
64 warmup attempts. Compilation is reported separately. File loading and
analysis are outside the sampling timer; worker setup and output construction
are inside it. No competing oracle run was active during these measurements.
The fixture circuit model is unchanged at `p=0.001`.

| Circuit | Physical qubit index span | Peak active width | Full records, us/attempt | Early rejection, us/attempt | Early-path acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 unflagged | 15 | 4 | 0.39 | 0.89 | 68.8% |
| d3 flagged | 17 | 4 | 0.42 | 0.92 | 64.0% |
| d5 unflagged | 42 | 10 | 17.47 | 5.39 | 14.3% |
| d5 flagged | 55 | 10 | 17.79 | 4.88 | 10.6% |

Full records use `sample`; early rejection uses `sample_survivors` with all
detectors selected and `keep_records=True`. Consequently these are comparisons
of user workflows, including their batching and output-volume differences,
not an isolated measurement of the early-exit kernel. In particular, the
automatic survivor path is scalar; early rejection is not beneficial for
these very small d3 programs at this noise rate.

At fixed fault weight 5, d5 unflagged uses 17.18 us/attempt for full records and
2.83 us with early rejection. The flagged variant uses 17.66 and 4.07 us.
Acceptances at that weight are about 1.99% and 5.42%. Comparing equal fault
weights between circuits with different noise-site counts is not comparing
their unconditional physical error rates.

The [raw measurements](cultivation_corpus_data.json) include zero-noise runs,
fixed weights 3 and 5, per-repeat timings, counts, circuit hashes, noise-site
counts, compilation latency and accepted attempts per second. No GPU,
external-simulator performance comparison, or best-in-class claim is included.

## What is expensive

`profile_cultivation` uses the existing production HIR pipeline and semantic
planner with source provenance retained. It records width changes and the
existing `predicted_dense_passes` estimate per action. Python checks that its
peak width agrees with the public compiled program.

Flagged and unflagged d5 have the **same estimated dense work by active width**:
64,077 coefficient visits in the unfused plan, of which 57,344 occur at width
10. Flags add noise/records/control overhead but do not enlarge this residual.
The main sources are the physical d5 T layers and the terminal evaluation.

31,560 of the d5 visits, about **49%**, map to source lines after the last noisy
instruction: the appended noiseless logical measurement and code projection.
This is part of the benchmark's output contract. Removing it would change the
task. A future terminal-instrument optimization must reproduce its joint
measurement/output behavior. The corresponding fraction is about 56% for d3.

These are static unfused coefficient-visit estimates, not CPU-time attribution,
RSS measurements, or a proof that half of elapsed time is removable. Fusion,
SIMD, classical work and rejection alter the relationship to time.

The present CSS-block prototype does not recognize these physical intervals:
their noisy preparation, ancillary extraction and growth are outside its
certificate. The fact that the terminal evaluation contains a full projection
does not certify intervening noisy syndrome rounds as identical boundaries.

## Screening opportunity and its limits

The diagnostic expands derived affine expressions and record snapshots,
including parity cancellation. It classifies a detector as state-independent
only when its expanded expression depends entirely on presampled Pauli faults
or symmetric, state-independent readout flips. Quantum branch dependencies
decline. Instruments are explicitly unsupported. This is an analysis of the
current plan, not a complete search for all possible algebraic reductions.

| Circuit | State-independent detectors | Attempts rejected by those detectors at p=0.001 | Estimated extra dense-work saving beyond current early rejection |
| --- | ---: | ---: | ---: |
| d3 unflagged | 16 / 20 | 29.6% | 9.6% |
| d3 flagged | 18 / 22 | 34.3% | 11.7% |
| d5 unflagged | 93 / 107 | 83.9% | 36.6% |
| d5 flagged | 108 / 122 | 88.2% | 40.8% |

The last two columns are calculated from complete sampled records. For each
shot, the current early-rejection work estimate stops at the first fired
detector in **plan execution order**. The hypothetical screened estimate gives
zero dense work to shots caught by the state-independent subset, then uses
the same early-rejection estimate for remaining shots. Screening evaluation,
additional noise draws and lost batching are not charged, so this is an
opportunity estimate, not a performance prediction. The static work model is
the retained-noise plan, including when applied to the zero-noise samples.

An 88% rejection fraction therefore does **not** imply an eightfold speedup over
current Clifft. Most rejected shots are already cheap. For natural p=0.001
sampling, even eliminating the estimated extra dense work would only reduce
that work component by roughly 37-41%. At fixed weight 5, the extra dense-work
savings are larger: about 72% unflagged and 57% flagged.

A useful next experiment is a small precompiled screening pass that reuses
these affine dependencies and preserves the sampling contract. Before changing
production execution, measure the cost of evaluating the screening expressions
and preparing any additional readout faults. A win must beat the existing
early-rejection workflow, especially its roughly 5 us/attempt d5 baseline.
This study does not introduce runtime topology analysis or change executor
invariants.

## Reproduction

From the repository root, prepare the ordinary Python development environment:

```bash
uv sync --frozen --group dev
cmake -S . -B build-study -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCLIFFT_BUILD_TESTS=OFF -DCLIFFT_BUILD_PROFILER=ON
cmake --build build-study --target profile_cultivation -j 4
.venv/bin/python tools/profile/cultivation_study.py \
  --profiler build-study/profile_cultivation \
  --output tools/profile/research/cultivation_corpus_data.json
.venv/bin/python tools/profile/validate_cultivation_aer.py \
  --output tools/profile/research/cultivation_aer_data.json
.venv/bin/python -m pytest -q tools/profile/test_cultivation_study.py \
  tools/profile/test_cultivation_aer.py
```

Run the timing harness and Aer sequentially. The source-map diagnostic can also
be invoked directly on a generated T circuit to inspect the full TSV action
trace. JSON results retain the source line numbers and width/work data without
duplicating long symbolic expressions.
