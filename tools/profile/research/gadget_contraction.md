# Fixed contractions for the physical terminal gadget

Follow-up: the [native handoff and sampler](native_terminal.md) now executes
complete fresh-noise attempts, including the prefix and parameter binding.
The timings below remain measurements of the earlier isolated kernels.

The interference calculation can be compiled into a small fixed schedule for
the 37-data-qubit gadget in the pinned SOFT input. Sequential sampling also
fits fixed schedules. Neither calculation needs fault-dependent elimination,
tableau evolution, or stabilizer intersection at execution time.

This is a positive result for extending Clifft's existing compiled-block
approach. It is not yet an end-to-end speedup: the experiment still prepares
the incoming code coordinates and physical fault parameters offline, and its
record-generating sampler is Python. No production compiler or executor path
has changed.

The source and its limitations are unchanged from the
[physical input investigation](soft_cultivation.md). In particular, this is
a real public physical T circuit with a large simulation workload, but its
noiseless outputs do not establish a valid distance-seven cultivation protocol.
Results here concern its exact circuit distribution, without postselection.

## What was compiled

The [coherent gadget reduction](clifford_gadget.md) writes each outcome's
operator as two coherent Clifford terms. For the final gadget, those terms
are monomial operators: computational-basis flips with local phases. Its
parity restricts to all-data X; its other factors act on product-state
spectators. After it, the source measures spectators and then all-data Y and
the complete CSS checks, with physical Pauli faults and readout flips retained.

The 18 independent X checks supply local binary support coordinates. A signed
code basis has computational support

```
b = offset XOR (logical_bit * all_data) XOR sum_i x_i * X_check_i
```

and character `(-1)^(incoming_X_syndrome . x)`. The offset represents the Z
syndrome. Each physical qubit touches only a few check coordinates. The
contraction therefore has one two-entry local phase table per data qubit,
plus one character table per check coordinate. Incoming and outgoing
syndromes, mixed T signs, and propagated faults bind table values; they do not
change the graph or elimination order.

For a complete terminal record, each input-logical/Clifford-branch pair needs
one contraction. Four complex contributions are added before taking the
squared magnitude. The output all-Y eigenvector, global Pauli phases, hidden
reset bit, physical spectator results, and readout flips are included. Final
Z syndromes are deterministic given the gadget outcome, boundary, and faults;
inconsistent records get zero probability.

The complete-record schedule has 55 leaves and 18 elimination steps. Its
largest elimination scope has five binary variables: **32 entries**. Across
the steps it processes 238 entries, with 1,012 precomputed gather addresses.
The C++ benchmark allocates its scratch and product arrays once, before any
evaluation. Its ordinary evaluation loop contains arithmetic and fixed
indexed loads, with no allocation, exceptions, or topology discovery.

## Sampling without runtime elimination

A probability for one complete record alone would not establish an efficient
sampler. We also compiled the partial-record probabilities.

Squaring the amplitude introduces two copies of the 18 support variables.
Summing over an unobserved X-check outcome forces the corresponding variables
in the two copies to agree. With `j` observed X checks, the paired contraction
therefore uses `18 + j` variables and a graph determined solely by measurement
order. There are 19 such fixed plans. Their largest elimination scope is eight
variables: **256 entries**. The two branches and two input amplitudes require
ten diagonal/cross-term contractions after conjugate symmetry is used.

The Python sampler chooses the gadget bit, the logical Y bit, and then the
18 X-check bits from these marginals. It reconstructs the deterministic
spectator, Z-check, and hidden-reset records. The current sampler certifies
that spectator measurements are deterministic for the supplied boundary; it
declines other boundaries. All 32 physical histories used here meet that
restriction. It checks normalization over both gadget outcomes, including
their coherent interference.

## Validation

[Raw results](gadget_contraction_data.json) record seeds, complete physical
records, errors, plan sizes, and timings. One plan family is reused throughout:

- Eight prior histories plus 24 fresh histories, spanning zero noise, source
  noise, and a ten-times-noise stress setting. The largest fault count is 77.
  Complete terminal probabilities agree with native full-circuit forced
  replay within `1.4e-14` absolute error, after conditioning on the prefix.
- Four changed records per history: logical Y, an X check, a Z check, and the
  hidden reset bit. All 128 variants agree with native replay, including
  impossible records and 34 variants with appreciably nonzero probability.
- Eight independently generated terminal samples at four prepared boundaries.
  Restoring the original prefix records and replaying the full physical
  circuit agrees within `5e-14` in log probability.
- Small seven-qubit code tests use arbitrary complex logical amplitudes and
  syndromes. They exhaustively compare complete-record probabilities against
  dense physical gate/projector calculations using Qiskit operators, then
  compare every partial marginal against sums of those dense probabilities.
  Faults between measurement and reset and mixed T signs are included.
- Random complex parity-factor tests compare the compiler to direct sums;
  native-kernel tests compare both amplitude and marginal evaluation modes.
  Together with the preceding Aer, Stim, and circuit-study checks, all 39
  focused tests pass.

The boundary adapter independently certifies the common signed CSS checks
and spectator states for each physical history, and extracts both complex
logical amplitudes using the reference CH states. That is a validation and
input-preparation step, not a compiled handoff or an all-history certificate.

## Measured cost and its limits

Single pinned CPU, AMD EPYC 9554P, native Release build; package versions and
the preceding full-sampling baseline are retained in the JSON. The kernels
cycle through 32 prebound histories. Compilation and parameter preparation
are excluded from kernel timings.

| Operation | Observed cost | Meaning |
| --- | ---: | --- |
| C++ complete-record probability | 4.7 us | Four coherent amplitude contractions |
| C++ partial-record probability | 20-50 us | Ten paired contractions, depending on prefix |
| Sum of query costs for a terminal sample | About 0.67 ms | Arithmetic budget derived from measured kernels; not a native sampler timing |
| Python fixed-schedule terminal sampling | About 41 ms | Eight actual conditional samples, preparation excluded |
| Earlier Python CH terminal sampling | About 483 ms median | Eight conditional samples from the preceding experiment |
| Ordinary Clifft full sampling | About 7.2 ms/attempt | Preceding full-circuit baseline, including the prefix |

The query budget charges five zero-prefix queries plus the 18 successive
X-check queries. This covers the maximum query count in this implementation,
but omits RNG, record updates, parameter binding, and the prefix handoff.
It must not be reported as an achieved full-circuit throughput or speedup.
Complete-record kernels were timed three times; the marginal kernel timings
are one batch per plan and should be treated as feasibility measurements.

The complete-record evaluator needs 6,480 bytes of numerical scratch; the
largest marginal evaluator needs 21,424 bytes. All marginal plans together
contain 55,904 gather addresses. These figures exclude plan metadata, input
tables, Python objects, and the prefix state, and are not total RSS claims.

Python planning took about 6 ms for the complete-record code and 61 ms for
the marginal schedules. Median reference-prefix preparation was 429 ms;
boundary certification/amplitude extraction was 1.35 ms; binding a forced
terminal history was 1.88 ms. Those per-history reference costs prevent the
current experiment from being an end-to-end performance improvement.

## Consequence for integration

There is now evidence for a terminal compiled block within Clifft's current
architecture, using the same basic fixed-factor strategy as the selected
prototype. The physical ancilla gadget and the measurement-order marginals
extend its scope; a dynamic stabilizer backend is not required by this
interference calculation.

The next integration target is concrete: at the existing width-one boundary,
certify the CSS code and product spectators from the compiler's symbolic state;
compile the map from its two coefficients and affine symbols to this code's
logical amplitudes and syndrome coordinates; compile physical fault/reset/
readout contributions; and lower the fixed marginal schedules with preallocated
storage. Unsupported boundaries must retain ordinary execution. The current
per-history Stim/Cirq certification cannot be moved into dispatch.

Only after that bridge and a native record-generating loop exist should the
full physical input be benchmarked against ordinary sampling and the small
validated corpus checked for regressions. This result justifies that focused
compiler extension. It does not yet justify broad claims about distance-seven
cultivation, other code families, or arbitrary non-Clifford continuations.

## Reproduction

With the profiling targets enabled:

```bash
cmake --build build-study --target replay_cultivation profile_gadget_contraction -j 4
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --group dev --with cirq-core==1.6.1 \
  python tools/profile/benchmark_gadget_contraction.py \
    --output tools/profile/research/gadget_contraction_data.json
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --group dev --with cirq-core==1.6.1 \
  python -m pytest -q tools/profile/test_gadget_contraction.py \
    tools/profile/test_clifford_gadget.py tools/profile/test_soft_cultivation_study.py \
    tools/profile/test_cultivation_study.py tools/profile/test_cultivation_aer.py
```

Cirq remains an optional research dependency. The benchmark reuses the prior
report's eight reproducible histories and generates the 24 fresh ones itself.
