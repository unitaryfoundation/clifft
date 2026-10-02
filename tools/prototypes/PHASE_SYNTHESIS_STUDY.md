# Parity-preserving phase synthesis

This study compares active-width and sampling outcomes on the 1,037 circuits in
`clifft-triorthogonal-bench-minimal.zip`. It starts from the state-aware phase
prototype at `bde4df717fd36d66ac941a55c3e6811bfda4a403` and adds the experimental
`preserve_parities=True` option. Existing defaults are unchanged.

## Finding

The synthesis itself is small and useful. Replacing the existing synthesis
unconditionally is a poor policy: after squeezing and scheduling, it improves
peak width on 23 circuits and worsens it on 13, despite reducing T count on 40.
Choosing by T count or by the phase pass's local width/work score still produces
12 final-width regressions.

A simpler placement is promising: append the new synthesis after the existing
phase, squeeze and scheduling passes. Its existing width/work guard then sees
the circuit we intend to execute. This lowers peak width on 22 circuits and
dense work on 32, with no structural regressions in this corpus. No additional
round of scheduling was needed for those peak-width gains.

| Approach | Lower final peak | Higher final peak | Lower dense work | Higher dense work |
| --- | ---: | ---: | ---: | ---: |
| Replace early synthesis | 23 | 13 | 27 | 19 |
| Compile both and choose by final peak then work | 23 | 0 | 27 | 0 |
| Append projected synthesis after scheduling | 22 | 0 | 32 | 0 |

The two-candidate choice is an offline study comparison, not an implemented
production selection policy. The late cleanup requires one additional compiler
pass and naturally retains the input when its existing guard rejects a rewrite.
It misses some gains available before scheduling, particularly in noisy inputs.
Neither strategy increases the number of circuits reaching peak width one or
two; the gains reduce the widths and work of larger remaining problems.

## Construction

The existing pass finds coordinates `(u, v)` where `v` spans the Clifford
kernel. Consequently, modulo eight,

```
f(u, v) = f(u, 0) + q(u, v)
```

where `q` is a diagonal Clifford polynomial: even linear coefficients, quadratic
coefficients divisible by four, and no cubic terms. This is an operator identity;
the removed dependence is retained in the Clifford frame.

Keep the original weighted parity list and evaluate each parity at `(u, 0)`.
Each input term contributes at most one output parity. Combine duplicates modulo
eight and recover `q` by polynomial subtraction. The resulting T count cannot
exceed the input list's odd-weight term count. This avoids expanding each cubic
monomial independently, while reusing the existing core reduction and Clifford
frame. The original synthesis can still outperform projection on individual
inputs, so projection is not a universal replacement.

For example, circuit 0125 has 62 input T rotations. The old synthesis proposed
714 rotations on its 18-variable core and was rejected. Projection uses 53;
final scheduled peak falls from 20 to 18. Circuit 0230 illustrates the opposite
scheduling outcome: T count falls from 68 to 59 but final peak rises from 4 to
20 if projection replaces the early synthesis. Late cleanup instead keeps peak
four and removes one rotation.

## Measurements

Both methods run in the same Release binary, with assertions enabled, native
AVX-512, GCC 13.3, on an AMD EPYC 9554P VM. Rank cap is 32 and known stabilizers
are enabled. All designated detectors are postselected and syndrome normalization
is disabled. The old arm exactly reproduces the previous study's per-circuit
peak widths and T counts.

Dense work uses the compiler's structural cost model: `2^k_after` for a dense
rotation and `2^k_before` for an active measurement. Scalar and purely symbolic
operations contribute zero. This is a proxy for execution cost, not an exact
hardware model; the report also records actual sampling.

Every one of the 46 structurally changed early-projection circuits was sampled.
The final-width/work choice selects 27 projected candidates, all of which had
faster measured sampling: 1.28x to 11,880x, median 5.17x. The other 19 candidates
were slower. These are affected-case results, not a corpus-wide speedup estimate.

All 32 circuits changed by the late cleanup were sampled separately. Twenty
improved by 4.38x to 10,385x. The other twelve had measured median improvements
of only about 0.5%-7.7%, which should not all be treated as reliable wins on a
shared machine. The late cleanup does not capture the early projection's large
gains on 0139 and 0647, but does retain the large gains on 0486 and 0518.

Sampling used one thread, batch size one, seven alternating repeats, a 0.12-second
calibration target and at most 1,000,000 attempted shots per repeat. Compilation
and warmup are excluded; sampler setup and result collection are included.
Separate acceptance probes passed every shot in both arms on the 46 changed
noiseless inputs.

The study also times compilation on all ten clifft-bench inputs and eight canary
inputs, with and without scheduling, using four ABBA cycles and equal iteration
counts. Projection in place of the existing synthesis has similar compile cost
on those inputs and changes no final metadata. Late cleanup after scheduling
also changes no metadata there. On those noisy workloads, conservative noise
ordering checks can prevent the additional pass from proceeding; the small cost
there should not be extrapolated to eligible noiseless phase blocks.

Compilation was also timed on all 46 affected corpus circuits. For the late
cleanup, additional compilation had a median of 0.45 ms and reached 40.2 ms on
the most expensive case. For the twenty cases with clear sampling gains, the
measured compile cost is recovered after roughly 1-263 attempted shots (median
16), using the observed per-shot savings. The small sampling gains have much
less certain break-even estimates. Compiling both full candidates instead adds
0.39-49.0 ms on the 27 cases selected by final width/work.

Noisy probes cover circuits 0125, 0139, 0230, 0481, 0625 and 0741 under dephasing
after every unitary, depolarization after the final T layer, and depolarization
after every unitary, at probability 0.001. Early projection retains useful gains
in the first two models but also retains scheduling regressions. Late cleanup
helps only two of the eighteen probes. Full gate-by-gate depolarization produces
no additional changes in this probe set. These are structural results; noisy
throughput was not benchmarked.

## Validation and reproduction

- 1,044 non-expensive C++ tests pass, including exhaustive small polynomial
  evaluations and sparse 32- and 64-variable projection checks.
- 1,002 Python phase and shared conformance tests pass with projection enabled;
  100 unavailable backend cases skip.
- Independent Aer checks pass on 400 unitary circuits and 160 measurement/feedback
  circuits for each placement. Early projection rewrote 396/400 and 160/160;
  late projection rewrote 102/400 and 27/160. Maximum errors are below 5e-15.
- Existing phase tests include independent noisy joint-record and forced-fault
  checks. GPU execution, a new WASM build and the expensive suite were not run.

Build this branch and ensure Python imports that extension, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/prototypes/bench_phase_synthesis.py \
  /path/to/clifft-triorthogonal-bench-minimal /tmp/phase-synthesis-results \
  --sample-changed
```

The harness emits raw per-stage structural metrics, paired sampling records and
summary counts. It is smoke-tested against four representative corpus entries.
The corpus archive SHA256 is
`52cc0b3893c0b18da3d7fa32c611445c3d9f8fdcd0578323ac10a8e72824f77b`.
Raw study logs, compilation timings and the original experiment scripts are
provided in the separate study artifact.

## Relationship to synthesis literature

[Reed-Muller decoding](https://arxiv.org/abs/1601.07363),
[TODD and TOOL](https://arxiv.org/abs/1712.01557), and
[FastTODD and TOHPE](https://arxiv.org/abs/2407.08695) provide more general
T-count optimization methods. They were not implemented or benchmarked in this
study. The parity-preserving construction addresses our specific expansion
problem without that search machinery. The scheduling regressions demonstrate
why better T count alone would not settle whether a more elaborate method is
worthwhile for Clifft.
