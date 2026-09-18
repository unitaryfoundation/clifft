# Fixed factor contractions for coherent CSS syndrome sampling

The bounded scaling test is positive. CSS syndrome sampling can use fixed
factor-elimination plans instead of enumerating the entire X-generator
support. The implementation retains all coherent cross terms, handles physical
Pauli frames, and computes the final logical amplitudes. It composes with the
complete original d3/d5 Python sampler without changing its record contract.

On synthetic 37-data-qubit blocks, the native conditional syndrome kernel is
about 159 times faster than the sparse/Fourier reference in this experiment.
A 61-data-qubit block also fits comfortably in the factor worker, although its
sparse support would have a billion entries per logical component. These are
block results, not full MSC7 simulation or a speed comparison against Clifft.

## Representation and fixed sampling order

Let `B u` be an X-stabilizer product, with one binary coordinate per independent
X check. A positive CSS logical basis is a uniform superposition of these
products, with the logical-X translate supplying the other logical basis state.
Each diagonal/bit-flip monomial maps that basis into one Z-syndrome coset.

After accounting for the physical frame and the coset representative, a term's
amplitude has the form

```
a_t(L) * product_q omega ** (c_t,L,q * parity(support_q & u)) / sqrt(2**r)
```

Here `L` is the physical logical-Z bit, `r` is the X-check rank, and each
`support_q` is the set of X generators touching physical data wire `q`.
Both the coefficients and the coset label depend on the fixed fault history;
the factor scopes do not. Ancilla overlaps form a small Gram matrix between
terms, so the contraction keeps interference instead of replacing a coherent
sum with a classical mixture. Different Z-syndrome cosets are orthogonal.

For a partial X-syndrome, bra and ket coordinates are equal on unmeasured
checks and independent on measured checks. Each measured bit contributes two
unary sign factors. Physical phase factors on the bra and ket retain their
original small scopes. The contraction is an average over `r + m` variables
for a prefix of `m` measured checks, giving the joint prefix probability.

Choose the measurement order to equal a compiled elimination order of the
single-copy factor graph. For each measured variable, eliminate its bra and
ket copies before moving to the next variable. Then eliminate the unmeasured
shared suffix. Each copied variable has the neighbors its corresponding
single-copy variable would have; eliminating both copies adds the same
remaining neighbor clique. This bounds the prefix contraction width by the
single-copy induced width. The compiler checks that bound for every prefix.

The sampler first draws the Z-syndrome coset, then samples X outcomes from
successive prefix masses. There is one fixed plan per prefix. It does not
recompute an elimination order, discover a dependency, or run a tableau per
shot. Final logical amplitudes use a single-copy phase sum with unary syndrome
signs, followed by the precomputed syndrome-dual correction convention.

This is still exponential in the graph's induced width. It is not a polynomial
algorithm for arbitrary circuits or arbitrary CSS codes. Dense generator
support, coupled phases outside this certificate, or excessive compiled tables
must decline. The implementation imposes an explicit table-entry budget.

## Coverage

The original MSC controls retain the same gates, noise histories, frames, and
outputs as the preceding study. No extra code projection is inserted between
gadgets.

- All 95 stored original histories match terminal norms and complex logical
  amplitudes from the sparse contraction, including relative phase conventions.
- Exhaustive small-prefix checks and selected d5 prefixes match the full Fourier
  distribution for arbitrary complex logical inputs and coherent weights.
- Full original d3/d5 sampling works with factor norms, syndrome sampling, and
  final contraction substituted for the sparse terminal implementation. Eight
  newly sampled stress histories match the independent coherent reference and
  complete records/detectors/observables.
- Odd diagonal phases also pass original-gate Aer checks on seven data qubits.
  The factor method itself does not require even/Clifford phases; the large
  independent stabilizer oracle uses the even-phase subset.
- The scaling family uses CSS supports extracted from Stim's generated
  `color_code:memory_xyz` check-extraction geometry. Its 7, 19, 37, and 61 data
  wires are code blocks, not authored complete cultivation circuits. The
  compiler separately checks independence, commutation, and logical conventions.
- Four synthetic inputs per size include arbitrary logical superpositions,
  physical X/Z frames, four coherent Clifford monomials, and complex weights.
  Two inputs put all four terms in one coset; two use two cosets. This avoids
  testing only mutually orthogonal terms or ideal zero syndromes.
- Independent Cirq CH states plus Stim tableaux check the initial coset mass,
  an intermediate prefix, and the full selected syndrome: 48 comparisons across
  the four sizes. Maximum relative error is below 4.8e-16. No large dense state
  is constructed for this oracle.
- Forty-eight native sampled syndromes agree with the Python contraction, with
  maximum relative error below 1.8e-15. Eight also pass independent CH projection
  at the native-selected syndrome, with maximum relative error below 5.6e-16.
- Tests disable sparse-basis construction and topology compilation while
  sampling the large blocks. Native allocation instrumentation checks the hot
  path; ASAN/UBSAN covers the 37- and 61-qubit factor workers. Leak detection is
  disabled because it is incompatible with the VM's tracing sandbox.

All 55 MSC regression tests pass, including the seven new factor tests.
Required repository checks are recorded in the branch checkpoint. The implementation remains research code; no production
executor or public API is changed.

## Scaling and native measurements

| Data qubits | X rank | Largest joint table | Factor us/sample | Fourier us/sample | Factor coefficient scratch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 3 | 8 | 27.41 | 1.52 | 1,248 B |
| 19 | 9 | 8 | 95.74 | 109.98 | 4,704 B |
| 37 | 18 | 32 | 506.05 | 80393.88 | 11,936 B |
| 61 | 30 | 64 | 1744.79 | not allocated | 23,968 B |

Times are medians of three runs, pinned to the same available CPU. Both native
implementations use `-O3 -march=native`, assertions and allocation instrumentation,
and no fast math. Each run cycles through the same four prepared coherent
inputs. Factor runs use 1,000 samples; Fourier runs use 1,000 samples for 7/19
qubits and 100 for 37 qubits. The 61-qubit Fourier allocation is not attempted.

These measurements include the initial norm/coset calculation, RNG, and the
complete joint CSS syndrome sample. They exclude physical fault sampling,
injection, growth, per-history monomial binding, and final logical-amplitude
recovery. They therefore cannot be compared as whole-circuit shot times with
the previous MSC or f7 workers. The two kernels use the same prebound inputs;
this is not a best-in-class claim against another simulator.

The 37-qubit Fourier worker uses about 32 MiB of coefficient scratch and 8 MiB
of probability scratch; the factor worker uses 11,936 coefficient bytes and
no probability table. The 61-qubit factor worker uses 23,968 coefficient bytes.
The compiled plans are additional storage: all-prefix native gather indices
occupy about 119 KiB and 454 KiB respectively, plus leaves, metadata, and the
small inputs. Total serialized plans are reported in the raw data. Do not
confuse a tiny runtime coefficient workspace with total process memory.

The 61-qubit sparse worker would allocate 128 GiB for four terms' two logical
components and another 32 GiB for its full probability table. The factor
method avoids those arrays rather than making their traversal faster.

Raw results: [factor study](msc_factor_sampling_data.json), including plans,
measurements and native samples; [selection screen](msc_factor_screen_data.json).

## Cheap eligibility and selection screen

There are two separate decisions:

1. **Can the block be represented exactly?** Require a certified positive CSS
   code with the chosen logical convention, bounded coherent diagonal/bit-flip
   terms, known product-spectator behavior, and factor plans within the entry
   budget. Unsupported structure declines during compilation. Final pure
   logical-state extraction additionally requires common spectator states up to
   scalar phase; the probability kernel can retain their general Gram matrix.
2. **Is this a useful execution choice?** Use the ordinary compiler's actual
   active width and the block plan's coefficient/work estimates, not physical
   qubit count. The research screen keeps the measured width-4/10 controls on
   ordinary Clifft, asks for head-to-head full-circuit timing for intermediate
   cases, and flags dense allocations above a 1 GiB study budget for a bounded
   certified-factor trial. This is conservative research triage, not a calibrated
   general routing model or a new production dispatch policy.

The original MSC d3/d5 controls remain at active widths 4 and 10. A separately
constructed planning control prepares an encoded logical magic state, executes
one original T-conjugated Y-product check, and measures the CSS checks. It
plans at widths 4, 10, 19, and 31 for the four block sizes. The 37-qubit control
therefore needs an 8 MiB dense coefficient vector and should be timed in both
complete implementations. The 61-qubit control plans a 32 GiB vector and is
flagged by the study's memory guard. These planning controls are not the
arbitrary four-term kernel inputs, so their widths do not supply a Clifft
speedup denominator for the kernel benchmark.

The existing reconstructed f7 still plans at active width 44, corresponding to
256 TiB for a dense vector. It remains a positive memory-separation control for
its existing fold-block implementation. That code supports a different phase
family and output contract; this CSS sampler does not automatically replace it.

## Decision and next work

Proceed with this factor sampler as the scalable contraction candidate inside
compiled logical blocks. The bounded test supports its memory scaling and
correctness, including complete control-circuit composition. Keep the sparse
implementation as a reference and the original d3/d5 ordinary-Clifft fallback.

The next meaningful test is to lower the factor contractions and final logical
amplitude recovery into the complete native block worker, then benchmark a
fully specified noisy five-check sequence against the original elementary-gate
circuit where the dense baseline fits. Include dynamic physical fault binding,
logical-state handoff between checks, and all outputs in those timings. Use the
37-data-qubit planning control as a manageable bridge; label the expanded
synthetic circuit clearly. Do not report
it as a full MSC7 protocol, infer fault distance, or claim a best-in-class result.
No authors' full MSC7 artifact has been obtained.

## Reproduce

```bash
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -DCLIFFT_MSC_CHECK_ALLOCATIONS tools/profile/msc_factor_native.cpp \
  -o /tmp/msc-factor-native
c++ -std=c++20 -O3 -march=native -I src -I build-research/generated \
  tools/profile/msc_baseline_benchmark.cpp \
  build-research/src/clifft/libclifft_core.a -o /tmp/msc-baseline-benchmark
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_factor_sampling.py \
  --native /tmp/msc-factor-native --baseline /tmp/msc-baseline-benchmark \
  --output /tmp/msc_factor_sampling_data.json
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_factor_screen.py \
  --baseline /tmp/msc-baseline-benchmark \
  --factor-data /tmp/msc_factor_sampling_data.json \
  --output /tmp/msc_factor_screen_data.json
CLIFFT_MSC_FACTOR_NATIVE=/tmp/msc-factor-native \
CLIFFT_MSC_SAMPLER=/tmp/msc-sampler-native \
CLIFFT_MSC_RECORD_PROBE=/tmp/msc-record-probe \
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p 'test_msc_*.py'
```
