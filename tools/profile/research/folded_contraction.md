# Native folded-check contractions and logical continuation

Follow-up: the [complete native sampler](folded_protocol.md) implements the
integration tasks below, including fresh faults and a compiler-certified
prefix. It measures complete-attempt f5 speedups against PR 471. This report
retains the earlier kernel-only experiment and its original timing boundary.

Date: 2026-09-20. The two noisy folded checks now have a fixed-contraction
sampling kernel that returns a coherent logical state after syndrome
measurement. This removes the terminal-only restriction of the previous
gadget experiment at the numerical-kernel level. Complete-history and
continuation checks pass on the reconstructed f3 and f5 circuits.

This is not yet a complete native protocol sampler. Incoming states and
physical faults are bound to local numeric tables offline. The native timer
measures sampling those tables and returning the logical amplitudes. It
excludes prefix sampling, drawing new faults, binding their tables, and
executing the remaining physical protocol. No end-to-end speedup against the
[PR 471 baseline](scheduled_baseline.md) is claimed.

## Representation

At the incoming boundary the data occupy a signed CSS code space with one
logical qubit. Ordinary Clifft evaluates the physical prefix. Offline
expectation probes recover the CSS signs, logical Bloch vector, and product
Z ancillas for the histories used in this experiment.

Each folded check uses disjoint single-qubit factors on the diagonal and CZ
factors on mirror pairs. Fixing a cat's computational branch makes the
physical operations monomial: they permute computational bits and multiply
them by local phases. The compiler keeps both cat branches coherent. Two
checks give four paths, each retaining physical T, T_DAG, CX and CCZ noise
locations, cat faults, flag records and reset outcomes.

The four paths' bit permutations differ by a known code-space translation.
Fixed binary duals express that translation in CSS-generator and logical
coordinates. The local phase factors have one or two binary parity inputs.
They are contracted over the X-stabilizer variables with a fixed elimination
order and precomputed gather addresses. No fault-dependent Gaussian
elimination or stabilizer synthesis is required by the native kernel.

For syndrome sampling, paired amplitude contractions identify the unobserved
Fourier variables. The compiler builds a separate fixed plan for every
prefix of measured X checks. The native code first samples the two cat
Hadamard outcomes, then the X syndrome, and finally evaluates both complex
logical amplitudes. Their relative phase is retained. The offset, syndrome,
and normalized two-component state together specify a continuation.

The research implementation consists of:

- `../folded_check_contraction.py`: fixed one-/two-parity factor plans,
  offline physical-path binding, marginal probabilities and amplitudes.
- `../sample_folded_checks.cpp`: preallocated native arithmetic, deterministic
  Xoshiro sampling and logical-state output. It uses the existing contraction
  executor with four-entry local tables; older two-entry clients are unchanged.
- `../audit_folded_contraction.py`: Clifft prefix extraction, complete-history
  checks, and validation through noisy CSS extraction.

All native vectors and contraction buffers are allocated during construction.
The hot methods are `noexcept` and perform only fixed indexed arithmetic and
RNG draws. They do not evolve tableaux, discover dependencies, resize buffers,
or invoke Python. The bundle loader validates dimensions, local phases and
incoming normalization before sampling. Production Clifft code is unchanged.

## Verification on physical circuits

The audit uses the existing saved full-protocol histories, including their
actual prefix records and physical fault realizations. It does not replace
the incoming state with an ideal logical state.

For each history it:

1. Replays the physical prefix in Clifft and checks that the boundary has
   definite CSS signs, product ancillas, and a pure logical state.
2. Binds both noisy folded checks, retaining their coherent four-path sum.
3. Propagates the fixed Pauli faults through the subsequent physical CSS
   extraction offline, preserving visible and hidden record flips and the
   final data frame. This includes the noisy post-checks and noiseless final
   syndrome extraction, not just ideal projectors substituted for noisy gates.
4. Evaluates the final physical folded logical measurement and combines its
   probability with the prefix and block probabilities.

All 24 f3 histories agree with their saved independent dense-Aer probabilities
to a maximum absolute log-probability difference of 5.45e-15. All 24 f5
histories agree with the previously validated coherent-Clifford reference to
1.78e-15. The f5 faults span p=0, 0.001, 0.01 and 0.03 and include rejected
protocol histories.

The native kernel then generates 48 new block histories for each distance,
cycling through the 24 bound input/fault cases. Each sampled block is carried
through the actual noisy post-checks and final syndrome circuit and compared
with full physical Clifft replay:

| Check | f3 maximum error | f5 maximum error |
| --- | ---: | ---: |
| Conditional log probability | 3.56e-15 | 1.78e-15 |
| Continued logical X/Y/Z expectations | 7.78e-16 | 1.12e-15 |
| Root-outcome normalization | 6.67e-16 | 5.56e-16 |
| Marginal probability versus returned amplitude norm | 2.23e-16 | 2.23e-16 |

The native trace's root index refers to the internal post-Hadamard bits;
physical readout flips are included when reconstructing its circuit records.
Checking all three logical expectations detects relative-phase errors that
probability agreement alone could miss.

Fifteen focused tests pass. New independent checks compare two-parity factor
sums against exhaustive enumeration, compare both distances' clean checks
with analytic logical projectors, verify a 2,048-sample native Born
distribution and its returned logical phases, and reject oversized leaves
before allocating them. Existing native-gadget and contraction tests remain
green after the local-table extension. Formatting, lint, type and hygiene
checks pass.

## Kernel cost and memory

AMD EPYC 9554P, CPU 0, one worker, GCC 13.3 native Release build. Medians of
three batches of 256 kernel samples after 16 warmups:

| Block | X variables | Largest table | Kernel time/sample | Fixed plan numeric payload |
| --- | ---: | ---: | ---: | ---: |
| f3, two checks and X syndrome | 6 | 16 complex entries | 149 us | 66.0 KiB |
| f5, two checks and X syndrome | 20 | 256 complex entries | 3.43 ms | 1.77 MiB |

The payload includes amplitude/marginal scratch, product buffers, leaf-index
tables, gather arrays and output indices. It excludes container overhead and
bound inputs. At f5, one incoming state/fault case's 32 local-table terms use
about 154.5 KiB including their coefficients; the diagnostic holds 24 such
cases for its fixed workload. A fresh-fault sampler should reuse one such
buffer instead of caching histories. No total-RSS comparison is claimed.

The sum of elimination table entries visited across the f5 marginal plans is
23,792 before accounting for coherent term pairs and both logical sectors.
The largest individual table is small, but total work is not just 256
operations. A preliminary scope-only calculation at distance seven gives a
largest table of 65,536 entries. This is not a generated or timed complete f7
protocol and does not establish polynomial scaling.

Offline binding takes a median 1.74 ms for f3 and 4.91 ms for f5 in these
cases. These Python costs are excluded from the native timings and cannot be
silently ignored in an end-to-end claim. The f3 kernel is already much slower
than ordinary optimized Clifft; a future integration needs an ordinary
fallback for small circuits. The f5 kernel cost is encouraging relative to
the remaining full-protocol cost, but the two timing boundaries differ.

## Remaining integration work

The next milestone is a complete-attempt sampler using these fixed plans:

1. Compile and certify the generalized signed-CSS handoff from the Clifft
   prefix. The present audit verifies individual boundaries numerically;
   production dispatch requires compiler certificates covering all histories.
2. Compile physical noise-site effects and record dependencies once. Draw
   fresh faults and fill the preallocated phase tables natively for every
   attempt. The present audit performs binding and Pauli propagation offline.
3. Return the sampled logical state, CSS signs, data frame and actual records
   to the planned continuation, including final logical evaluation and
   consistent early rejection.
4. Time complete attempts against PR 471, including all prefix, noise,
   binding, record and continuation work. The measured optimized f5 baseline
   remains approximately 208 ms per early-rejected attempt at p=0.001.

These are integration tasks for extending Clifft. Replacing the offline
binding with per-shot Stim/Cirq or replanning would violate the repository's
execution contract and is not an acceptable implementation.

## Reproduction

[f3 data](folded_contraction_f3.json) and [f5 data](folded_contraction_f5.json)
retain source/binary hashes, all complete-history checks, native sample
traces, continuation errors, payload accounting and timing batches.

```bash
cmake --build build-study --target sample_folded_checks replay_cultivation -j2
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=tools/profile uv run --frozen --group dev --with cirq-core==1.6.1 \
  python tools/profile/audit_folded_contraction.py --distance 5 \
  --output /tmp/folded_contraction_f5.json
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=tools/profile uv run --frozen --group dev --with cirq-core==1.6.1 \
  python -m pytest -q tools/profile/test_folded_check_contraction.py \
  tools/profile/test_native_terminal.py tools/profile/test_gadget_contraction.py
```

Use `--distance 3` for the independent small-protocol audit. The raw data's
native times are kernel measurements; keep that qualification when reusing
them in plots, reports or comparisons.
