# Complete native folded-MSC attempts

Follow-up: at the user's request, the [complete f7 extension](folded_protocol_f7.md)
was tested before the automatic fallback/consolidation proposed below. It
samples complete reconstructed f7 attempts with independent history checks;
the original f3/f5 measurements and next-step proposal are retained here.

Date: 2026-09-20. The fixed-contraction prototype now samples complete
reconstructed f3/f5 attempts, including a real Clifft prefix, fresh physical
faults, coherent folded checks, noisy syndrome extraction, final fictitious
error detection (FED), and the final folded logical measurement. At f5 and
p=0.001 it is about 90-94 times faster than ordinary Clifft with PR 471's
unbounded scheduler enabled. This supersedes the kernel-only timing boundary
in the [previous experiment](folded_contraction.md).

This is an opt-in profiling implementation, not an automatically selected
public Clifft backend. It accepts the explicitly annotated reconstructed
family. It neither demonstrates arbitrary-circuit coverage nor a complete
f7 cultivation protocol. The [reconstruction caveats](folded_msc_f5.md),
including the unresolved author gate-order convention and absence of
decoder-based final error correction or escape/storage, still apply.

## What executes per attempt

`compile_folded_protocol.py` prepares one reusable bundle. The native
`sample_folded_protocol` target then:

1. Executes the physical prefix using Clifft's existing planner and executor.
   Compiler-produced expectation actions certify that every reachable
   handoff is a signed CSS code space with one logical qubit and product-Z
   ancillas. The certification uses symbolic active-Pauli projections, not
   a test of only the noiseless input. Physical logical X/Y/Z expectations
   provide the two complex logical amplitudes.
2. Draws fresh faults at every remaining physical noise site, including
   depolarization after T, CX and CCZ gates and cat preparation/readout noise.
   It updates preallocated unary and pair phase tables for the four coherent
   paths through the two checks. Branch translations and the absence of
   branch-dependent classical records were established by the compiler.
3. Samples the cat outcomes and X syndrome using the existing fixed
   contraction plans. It retains both logical amplitudes and their phase.
4. Applies compiler-precomputed Pauli and record effects through the actual
   noisy post-check circuit and noiseless final syndrome extraction. It
   binds and samples the final physical folded logical measurement, retaining
   all visible records and hidden reset outcomes.

Early rejection occurs in the ordinary prefix and after the folded block's
cat records and syndrome records. Both paths reject on the same detector
set. The prototype sometimes performs more work before rejecting; the
timings include that cost. All hot storage is allocated during construction;
the hot methods are `noexcept`, use deterministic Xoshiro draws, and apply
preplanned arithmetic. No per-attempt Python, tableau evolution, fault
propagation analysis, or topology planning occurs. Production Clifft source
and Stim are unchanged.

## Matched baseline and timings

Both paths execute in the same native binary and use the same physical
circuit and noise model. The build merges study base
`41a18d455e855650f3c5dffa75db301f67ff2ded` with PR 471 head
`8f69b3c305303483e5dd33dcf4d13dbbebae1fa9`. PR 472 adds documentation, not
another runtime optimization. The default optimization pipeline runs first;
the active-width scheduler then runs explicitly, either with its default
bounded search or with an unlimited search budget.

AMD EPYC 9554P, GCC 13.3 native Release build, one worker pinned to CPU 0.
Each number is the median of three batches. At f5 the batches contain 256
hybrid attempts and 64 ordinary attempts; f3 uses 4,096 and 65,536. Warmups,
compilation, bundle loading, and trace printing are outside the timer.
Prefix execution, fault draws, binding, contractions, records, final
evaluation, and acceptance counting are inside it. Verification workers ran
on other cores, so these are representative timings, not an isolated-machine
microbenchmark. Raw batches and hashes are retained.

| Family | Scheduler | Rejection | Ordinary/attempt | Hybrid/attempt | Ordinary / hybrid |
| --- | --- | --- | ---: | ---: | ---: |
| f3 | unbounded | full histories | 4.19 us | 186 us | 0.022x |
| f3 | unbounded | early | 3.53 us | 164 us | 0.021x |
| f5 | bounded | full histories | 384 ms | 3.71 ms | 103x |
| f5 | bounded | early | 208 ms | 2.08 ms | 100x |
| f5 | unbounded | full histories | 336 ms | 3.72 ms | 90x |
| f5 | unbounded | early | 196 ms | 2.08 ms | 94x |

The f3 regression is substantial: ordinary Clifft is about 45-47 times
faster. A seamless integration must retain ordinary execution for this
case. The benefit demonstrated here is the larger reconstructed f5 workload,
not a general improvement across the existing small-circuit corpus.

Preparing the f5 bundle in Python takes about 5.55 s once. Native bundle
loading, prefix compilation and executor construction take about 19 ms;
ordinary unbounded compilation and executor construction take about 211 ms.
Including bundle preparation, the measured f5 advantage amortizes after
roughly 30 early-rejected attempts, excluding Python startup. The compiler
cost is material for one-shot use and should be cached or reduced in an
integration. No offline per-fault history cache is used.

## Correctness evidence

- 144 fresh complete histories: 24 each for f3/f5 at p=0, 0.001 and 0.03.
  The audit reconstructs each physical Pauli fault and forces the complete
  visible and hidden record string through physical Clifft replay. The
  difference between full and prefix log probabilities agrees with the
  native suffix probability to at most 2.67e-15. All noiseless cases accept
  with logical output zero.
- 24 of those histories also agree with the independent coherent-Clifford
  oracle, with maximum absolute log-probability error 1.78e-15. On 12 f3
  fault realizations, fresh dense-Aer trajectories additionally verify that
  oracle to 4.00e-15. Aer draws its own measurement records; these are not
  described as forced replays of the native records.
- A separate f3 frequency check uses 196,608 complete attempts per path at
  p=0.001. Acceptance is 0.791402 for the hybrid and 0.792043 for ordinary
  Clifft, a difference of 0.50 pooled standard errors. This supplements
  conditional-probability checks with a fresh-noise distribution check.
- Twenty focused tests pass, including complete physical histories at both
  distances, a compiler rejection test for a measurement that distinguishes
  coherent cat branches, analytic logical projectors, brute-force factor
  sums, native Born statistics, and the existing contraction/native-terminal
  tests. Formatting, lint, type and file-hygiene checks pass.

These are simulator correctness and throughput checks. They do not establish
the protocol's claimed fault distance or estimate its rare logical-error
rate. Importance sampling remains a separate statistical requirement for
efficiently measuring rare failures; faster ordinary attempts do not remove
that requirement.

## Memory

The f5 fixed contraction plans still have approximately 1.77 MiB of numeric
payload. One approximately 154.5 KiB input buffer is rebound for every shot,
instead of retaining the previous kernel experiment's 24 bound histories.
The Clifft prefix peaks at width 8, while the ordinary full circuit peaks at
width 22 and requires a 64 MiB coefficient array per lane.

Fresh shell-launched process measurements put the hybrid's high-water RSS
before ordinary-plan construction at 9.61 MiB, and total process peak at
12.92 MiB even including ordinary-plan compilation without its executor.
Constructing and running the ordinary executor in addition raises the process
peak to 80.89 MiB. The latter holds both implementations and is not presented
as isolated ordinary-only RSS. Process peaks include executable/runtime and
loader overhead, so they are not per-lane payload estimates.

The timing driver's Python-launched processes inherit a Python high-water
mark in `ru_maxrss`. Consequently, their `native_peak_rss_kib` values are
not usable as sampler memory measurements; the separate shell-launched
measurements in the context artifact are the memory evidence above.

## Next steps retained

1. Make the f5 mechanism a reviewable Clifft extension: identify eligible
   regions from compiler IR, carry the signed-CSS certificate and preplanned
   record/phase actions into an executor operation, and choose it only when
   a conservative work estimate beats ordinary execution. Unsupported
   circuits must continue through the existing path. The present generator
   supplies stage annotations and geometry; automatic recognition is still
   missing.
2. Extend the reconstructed family to f7 and first audit geometry, growth,
   verified-cat acceptance and plan size. Support its 85 data qubits without
   the current physical-mask limit. Preserve the f5-to-f7 logical-state
   continuation instead of reverting to a dense f5 prefix. The preliminary
   f7 scope estimate of 65,536 entries is not a timed complete protocol and
   does not prove manageable total memory or runtime.
3. Compare with author circuits when available, reconcile gate order and
   final-evaluation conventions, and only then add importance sampling for
   logical-error estimates. No author-identical full protocol, f7 success,
   or decoder/escape result is claimed by this milestone.

## Artifacts and reproduction

- [Complete-attempt timings and source hashes](folded_protocol_benchmark.json)
- [Fresh faults, records and probability checks](folded_protocol_validation.json)
- [Build provenance, memory and frequency check](folded_protocol_context.json)

Build `sample_folded_protocol`, `sample_folded_checks` and
`replay_cultivation` with `CLIFFT_BUILD_PROFILER=ON`. For the optimized
comparison, use the isolated PR-471 merge described in
[the baseline report](scheduled_baseline.md). The harness refuses to run a
non-off scheduler mode when built without that pass.

```bash
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=tools/profile uv run --frozen --group dev --with cirq-core==1.6.1 \
  python tools/profile/benchmark_folded_protocol.py \
  --sampler /tmp/clifft-pr471-baseline/build-study/sample_folded_protocol \
  --bundles /tmp/folded-protocol-bench --output /tmp/folded-protocol-timings.json
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=tools/profile uv run --frozen --group dev --with cirq-core==1.6.1 \
  python tools/profile/audit_folded_protocol.py --schedule unbounded \
  --sampler /tmp/clifft-pr471-baseline/build-study/sample_folded_protocol \
  --reference /tmp/clifft-pr471-baseline/build-study/replay_cultivation \
  --output /tmp/folded-protocol-validation.json
```

The native CLI is `sample_folded_protocol bundle shots traces seed early
schedule baseline_shots`. Use `early=0` for trace validation. For memory,
launch it directly from a shell with `/usr/bin/time -f %M`, first with
`baseline_shots=0` and then `baseline_shots=1`, rather than from the Python
compiler process. Tests are `test_folded_protocol.py`,
`test_folded_check_contraction.py`, `test_native_terminal.py` and
`test_gadget_contraction.py` under `tools/profile`.
