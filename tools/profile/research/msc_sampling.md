# Complete static MSC sampling and the native control result

Compiled logical blocks now sample complete original MSC d3/d5 trajectories.
There is no per-shot CH state extraction, gadget topology discovery, tableau
update, or record-equation elimination. This closes the static sampling gap
left by [the injection study](msc_injection.md). It does **not** justify routing
these small controls through logical blocks: the first native implementation
is substantially slower than current Clifft on both.

The implementation remains standalone research code. No production dispatcher,
API, or representation-switching framework changes are included.

## What changed

The gadget compiler propagates computational-bit coordinates once. It writes
fault-dependent flips as parity rows and phases as sums of products of two
parities modulo eight. At runtime, physical X/Z fault bits and the true gadget
outcome select two monomials. Y phases and branch-relative phases are retained;
root faults erased by reset still alter the exposed hidden reset bit. The
compiler also fixes the intervening ancilla measurement and fault schedule.
The old offline binder remains an independent comparison path.

Record sampling has three parts:

- Injection samples seventeen uniform free true-event bits, then evaluates
  pre-eliminated affine constraints and the two logical Choi signs. The weighted
  logical state is normalized before entering cultivation.
- Growth samples the commuting input instrument projector: 34 single-wire
  spectator checks and six checks on the seven-data-qubit input. The coherent
  sum is retained while each conditional probability is calculated. Eighteen
  further uniform bits complete its 155 true events through fixed parity maps.
  Existing logical contraction and frame maps supply the output code state.
- The final two gadgets remain coherent through their actual intervening
  measurements. The terminal sampler groups amplitudes into fixed coset slots
  and applies a Walsh-Hadamard transform along the CSS X-generator coordinates.
  It samples the actual final syndrome jointly, without enumerating 2^18
  sectors or inventing an earlier code projection. Equal coset labels control
  interference between coefficient slots; they do not trigger topology work.

For the nineteen-data-qubit terminal code, each of at most four terms uses
2 x 512 amplitudes. The X-syndrome transform is over 512 entries, and the
Z-syndrome is a precomputed parity of the term's physical bit-flip offset.
Logical amplitudes are summed coherently within each coset before probabilities
are formed. The final logical state is recovered in the already-validated
syndrome/frame convention. Reported readout flips, detectors, observables, and
hidden reset outcomes are all retained. The compiler checks that every true
event has exactly one sampling owner.

The Python implementation allocates and is a correctness reference. The C++
worker allocates plans, records, fault storage, and all coefficient/scratch
arrays before sampling; its shot path uses fixed loops and arrays. Generated
plan layouts are validated before dispatch. Allocation instrumentation and
sanitizer checks apply to the standalone worker, not a production executor.

## Validation

The adjacent [raw study data](msc_sampling_data.json) contains native histories,
quantum trajectory probabilities conditional on those physical faults, final
logical amplitudes, all outputs, reference errors, and repeated timings.
Classical fault-sampling probabilities are not multiplied into the reported
quantum trajectory weights.

- All 95 stored full histories preserve the offline binder's monomial phases,
  product ancillas, and coherent weights. The static gadget test additionally
  checks every allowed single-wire X/Y/Z fault in every recognized region,
  both true outcomes, and mixed physical histories.
- Python-generated full trajectories cover both circuits at nominal and 20x
  noise. Complete weights and final logical density matrices agree with the
  independent coherent reference; Stim independently evaluates output parities.
- All 64 d3 terminal sectors are compared with the existing projector
  contraction for five arbitrary complex logical inputs/fault histories. This
  checks the entire small-block Fourier distribution and its normalization,
  including sectors that an ideal-only sample would not visit.
- The native study checks 64 complete new trajectories: sixteen per circuit at
  each noise scale. Sixteen of these additionally replay the original elementary
  gates in Clifft, with fixed physical faults and explicitly exposed resets.
  The original T gates are retained in that oracle.
- Runtime tests disable the offline binder and compilation/reference entry
  points after plan construction. The native allocation-check build asserts
  zero ordinary new/new[] calls during every shot. ASAN/UBSAN runs cover 100
  attempted shots per circuit; leak detection is disabled because the VM's
  sandbox tracing is incompatible with LeakSanitizer.

For the native study, maximum probability error is 1.12e-15 against the
coherent reference and 4.27e-14 against original elementary Clifft. Maximum
normalized logical-density error is 1.99e-14. Over the 30,000 nominal attempts,
block/Clifft accepted counts are 20,695/20,720 for d3 and 4,394/4,325 for d5;
these aggregate counts are a distribution sanity check, not a replacement for
the trajectory-level comparisons.

The complete MSC suite passes all 48 tests, including the existing original-gate
Aer checks and the native sampler test.
There is still no independent full forty-two-qubit physical-state oracle for
d5; its full state comparison is against the coherent reference, while the
separate elementary Clifft replay checks trajectory probability and records.

## Native throughput

Timing excludes one-time plan construction and loading, includes all original
noise sites and complete unfiltered output sampling, and uses the same CPU for
both contenders. Neither contender postselects away failed attempts. The block
worker additionally computes the final two logical amplitudes and the full
true-trajectory probability; the baseline does not expose those outputs.
These are practical end-to-end worker timings, not equal-instruction kernel
microbenchmarks. Assertions/allocation counting remain on in the block worker.
The baseline links the existing Release, native-CPU, OpenMP-off Clifft build
with `-ffast-math`; the block worker is compiled without fast math.

Medians of three runs of 10,000 attempts per contender:

| Circuit | Block worker us/attempt | Clifft us/attempt | Block/Clifft | Clifft peak width |
| --- | ---: | ---: | ---: | ---: |
| MSC d3 | 31.60 | 1.02 | 30.9x slower | 4 |
| MSC d5 | 1088.79 | 17.46 | 62.4x slower | 10 |

The native worker is an initial scalar implementation, with repeated coherent
norm calculations and general parity-polynomial evaluation. It has not been
optimized. Nevertheless, the relevant negative observation is structural:
current Clifft's peak active widths on these original noisy fixtures are only
4 and 10. Their physical widths of 15 and 42 do not make them large residual
states for Clifft. The new sampler therefore adds block-management work without
removing the large coefficient arrays that motivated this study. These controls
establish composition coverage, not a speed advantage or best-in-class result.

## Decision and next work

Keep ordinary Clifft for the original d3/d5 controls. Do not spend this study on
SIMD or tune this worker to chase those small-state timings. Keep the static
plans, record sampler, and independent replay as coverage infrastructure for
compiled logical blocks.

The reconstructed f7 result remains separate: roughly 45 us per attempted shot
in its earlier native worker, while ordinary Clifft plans active width 44.
Nothing in this MSC control benchmark measures an f7 speedup ratio or proves
equivalence to an authors' complete f7/MSC7 circuit. Its ideal final syndrome
contract and missing authors' artifact remain limitations.

The next bounded study should test a cheap eligibility/cost decision against
actual compiled active width and the block contraction's support/term counts,
using both these negative controls and the existing large f7 reconstruction.
A circuit should qualify because its residual state is expensive for Clifft,
not merely because it has many physical qubits or a recognizable cultivation
pattern. The physics follow-up is to test whether the fixed factor-contraction
schedules used by f7 can also sample these CSS projections without enumerating
the full X-generator support. That support grows exponentially with the number
of independent X checks; the present 512-entry transform alone is not a scaling
argument for MSC7. Compile the factor scopes/elimination order once and preserve
coherent cross terms. Compare with the sparse Fourier reference on d3/d5 before
trying larger code blocks; a larger block-only experiment must remain labeled
as such, not as an authors' full MSC7 circuit.

Then extend output/eligibility coverage at the large-state boundary before
considering production dispatch. The CSS Fourier sampler here is a bounded
additional contraction option, not a general scalable replacement for the
existing factored f7 contraction.

No further user approval is needed for this bounded research work. An actual
change to the repository's architectural invariants would still need review.

## Reproduce

```bash
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -DCLIFFT_MSC_CHECK_ALLOCATIONS tools/profile/msc_sampler_native.cpp \
  -o /tmp/msc-sampler-native
c++ -std=c++20 -O3 -march=native -I src -I build-research/generated \
  tools/profile/msc_baseline_benchmark.cpp \
  build-research/src/clifft/libclifft_core.a -o /tmp/msc-baseline-benchmark
c++ -std=c++20 -O3 -march=native -I src -I build-research/generated \
  tools/profile/msc_record_probe.cpp \
  build-research/src/clifft/libclifft_core.a -o /tmp/msc-record-probe
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_sampling.py \
  --native /tmp/msc-sampler-native --baseline /tmp/msc-baseline-benchmark \
  --probe /tmp/msc-record-probe --output /tmp/msc_sampling_data.json
CLIFFT_MSC_SAMPLER=/tmp/msc-sampler-native \
CLIFFT_MSC_RECORD_PROBE=/tmp/msc-record-probe \
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p 'test_msc_*.py'
```
