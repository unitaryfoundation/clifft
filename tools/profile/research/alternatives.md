# Bounded alternatives to compiled logical blocks

**Keep compiled logical-block simulation as the leading integration candidate.**
An independently implemented Pauli proxy is correct on the tested reconstructed
protocols, but slower in this native comparison. A bounded frame search improves
early five-round memory states without making its later residuals attractive
for an exact MPS replacement. Neither result rules out a stronger implementation
or a different circuit family. No production API, executor, or planner changed.

The [checkpoint](checkpoint.md) preserves the logical-block implementation and
its outstanding eligibility/composition work. This study stops at comparison;
it does not start production integration or an automatic backend selector.

## Pauli-proxy construction and output contract

[Takada, Bartlett, and Williamson](https://arxiv.org/html/2609.16929v1) motivate
replacing admissible Clifford-stabilizer protocol measurements by Pauli-proxy
measurements after propagating physical errors. This is measurement-equivalence,
not an equality between the physical magic state and a stabilizer state. The
implementation here specializes that approach to our structured f3/f5/f7
reconstructions; it does not implement the paper's general admissibility test.

`pauli_proxy.py` independently walks elementary physical gates. It retains a
continuing X frame, emits Z components of physical faults, and propagates the
frame through Clifford gates. At T/T_DAG, a continuing X produces S_DAG/S before
the ideal non-Clifford gate is removed. At CCZ, the derivative of its cubic
Boolean phase produces CZ and Z corrections, including the terms from multiple
simultaneous X components. Omitting those corrections is not an exact proxy.
Only a history-wide global phase is discarded.

This preprocessing handles injection, growth, cat preparation, decoding, and
all original physical fault locations. Uniform f7 flag records are selected by
ZZ parity postselection, retaining both uniform outcomes. The unused flag
registers are separable from the main cat for a fixed Pauli history. The proxy
does not supply individual accepted flag records or rejected-record statistics.

After the terminal ideal syndrome projection, the physical output is in the
single-logical-qubit code. Our derived probe adapter evaluates two additional
ideal logical measurements on copies of the proxy state:

```text
Hplus  = (X_L + Y_L) / sqrt(2)
Hminus = (X_L - Y_L) / sqrt(2) = i Z_L Hplus
<X_L> = (<Hplus> + <Hminus>) / sqrt(2)
<Y_L> = (<Hplus> - <Hminus>) / sqrt(2)
```

Each measurement circuit passes through the same fault-propagation/proxy
construction with the remaining X frame. The controlled i in Hminus is an S
gate on its probe ancilla, not a discardable global phase. Logical Z is evaluated
with the continuing frame's sign. These are mapped physical probes, not raw
tableau XYZ values. The ideal proxy has logical X=1 whereas the physical ideal
T state has X=1/sqrt(2); a test explicitly guards this distinction.

The implemented output contract is therefore the same acceptance probability
and conditional logical XYZ expectations as the logical-block benchmark, under
the reconstruction's final-error-detection convention. An arbitrary physical
state handoff, arbitrary probe, generic input circuit, continuous noise, and
final-error-correction decoding remain unsupported.

## Native comparison

The exporter emits physical operation schedules and fixed-fault fixtures.
`pauli_proxy_native.h` constructs a Clifford program for each sampled history
and runs **unmodified external Stim 1.15.0**, fetched through CMake at
`42e0b9e099180e8570407c33f87b4683cac00d81`. The native code is an independent
research baseline, not a new Clifft trajectory executor. Its tableau work and
allocations belong to this external baseline. Adopting such an execution path
inside Clifft would need a separate architecture decision.

| Median attempted-shot cost at p=0.001 | f5 | f7 |
| --- | ---: | ---: |
| Compiled logical blocks, fresh run | 19.85 us | 44.69 us |
| Per-history proxy construction plus external Stim | 40.09 us | 73.54 us |
| Main proxy construction component | 11.12 us | 26.39 us |
| Accepted out of 100,000 attempts, either method | 41,606 | 12,667 |
| Proxy process maximum RSS | 4,852 KiB | 4,912 KiB |

Each number is a median of five trials of 20,000 attempts on the unpinned EPYC
VM, with GCC 13.3 and O3/native instructions without fast-math. Runs were
sequential. Both methods include physical-fault sampling, quantum acceptance,
and logical XYZ accumulation. Both use the same physical-noise iteration order
and deterministic RNG rules; their accepted counts and probes agree. Input I/O
and circuit-independent setup are outside timing. The proxy includes per-history
program allocation, construction, Stim state construction, and probe work. The
construction component measures the main circuit only; probe construction is
included in total time. Tiny timer instrumentation costs are not subtracted.

The fresh f5 baseline uses the current widened native arrays and differs from
the historical 17.73 us measurement. The f7 baseline is also rerun here. The
comparison is about 2.02x for f5 and 1.65x for f7 in favor of logical blocks.
This is not a comparison with the authors' implementation or all Clifford
simulators, and neither candidate has an arbitrary-circuit recognizer.

The proxy is a real alternative: it avoids a dense magic residual and coherent
overlap contractions. Its cost here is rebuilding and executing a physical
Clifford program. Fixed conditional descriptors, a less expensive output
contract, or larger codes could change the ranking; none is measured in this
bounded study. Ordinary per-shot compilation is not automatically superior just
because a generic dense representation is infeasible.

## Independent validation

The proxy agrees with the logical-block reference on 337 fixed histories:
108 f3, 112 f5, and 117 f7. Cases include natural p=0.001 histories, p=0.01
stress histories, paired hooks, nonzero growth sectors, both uniform flags,
high-coordinate faults, and logical X/Y/Z tails. Maximum discrepancy in
acceptance or conditional probes is 2.23e-16. The logical-block references were
previously validated against the independent phase-preserving CH method.

Additional Aer tests check the local T/T_DAG/CCZ propagation identities on
generic inputs, every X-mask pattern, and all four measurement records of a
small two-check magic-state protocol under individual physical Pauli faults.
Native fixtures agree in acceptance and unnormalized probes to 2.23e-16.
The f7 native build, including Stim, passed ASAN/UBSAN on 117 fixtures and 500
sampled attempts. Leak detection is disabled because of sandbox tracing.
These are correctness diagnostics, not a rare logical-error-rate estimate.

## Bounded Clifford-frame and tensor screen

Motivated by the frame dependence discussed in the
[MPS/CAMPS comparison](https://arxiv.org/html/2609.19116v1), `frame_screen.py`
searches coordinate order and Clifford frame on saved residual vectors. It
changes no execution state approximately: every candidate is a unitary Clifford
change of coordinates, and inverse transformation recovers the original vector
to within 1e-15 in this experiment.

For each of three checkpoints from the prior five-round coherent d5 memory
study, sample zero selects among the current order and 32 seeded permutations.
Then four alternating nearest-neighbor sweeps search all 20 two-qubit Clifford
representatives modulo output-local Cliffords. A step minimizes numerical rank,
then entropy, at its cut. The resulting single order/frame is applied unchanged
to three held-out samples at that checkpoint. These are independent
unconditional prefixes, not accepted-shot states or continuous trajectories.

| Checkpoint | Width | Original maximum rank | Framed maximum rank | Framed MPS coefficient estimate | Dense coefficients |
| --- | ---: | ---: | ---: | ---: | ---: |
| 144, earlier | 13 | 16 | 8 | 222 | 8,192 |
| 488, later | 13 | 54-60 | 45-52 | 8,490-9,386 | 8,192 |
| 544, later | 12 | 62-63 | 40-42 | 7,400-7,640 | 4,096 |

Ranges cover training and held-out samples. Rank uses relative tolerance 1e-12;
the raw data also retain 1e-8, 1e-10, and 1e-14 results. At 1e-14 the later framed
maxima are 55-60 and 49-50, respectively. These are numerical diagnostics, not
certificates of exact algebraic rank. The MPS estimate is the standard sum of
2 times neighboring bond dimensions, excluding singular-value, factorization,
and workspace costs. Numerical thresholds can undercount exact required storage;
no state or singular value is actually truncated by this diagnostic.

The selected frames have 7, 28, and 20 two-qubit Clifford representatives.
Ordering plus frame search costs approximately 0.22-0.44 seconds per training
snapshot in this Python diagnostic. Applying the stored frame densely costs
roughly 0.3-1.2 milliseconds per snapshot. These are not optimized MPS execution
times. A reusable compiler-selected frame could amortize search, but this screen
does not measure tensor gate application, conjugated Pauli support, measurement
updates, or frame transitions over a full trajectory.

The outcome is **defer an exact MPS backend for this control**, not reject
frame-improved tensors generally. In particular, this pass does not profile a
new frame on the large fold residuals or reproduce the paper's CAMPS policy.
That large-circuit question remains open. Early-state compression alone is not
enough to justify a backend for sustained five-round sampling.

## Reproduction and decision

Use the research Python environment with Stim, NumPy, Qiskit Aer, and Cirq Core.

```sh
python tools/profile/pauli_proxy.py --distance 7 --output /tmp/proxy-f7.json
python tools/profile/export_pauli_proxy.py --output /tmp/proxy.cpp
cmake -S tools/profile/proxy_native -B /tmp/proxy-build -DCMAKE_BUILD_TYPE=Release -DPROXY_SOURCE=/tmp/proxy.cpp
cmake --build /tmp/proxy-build --target profile_proxy -j4
/tmp/proxy-build/profile_proxy 7 20000
/tmp/proxy-build/profile_proxy 5 20000
CLIFFT_PROXY_NATIVE=/tmp/proxy-build/profile_proxy OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_*.py'
```

CMake also accepts `FETCHCONTENT_SOURCE_DIR_STIM` pointing to an unchanged local
checkout at the pinned revision. For ASAN/UBSAN, use a separate build with
`CMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer` and matching
`CMAKE_EXE_LINKER_FLAGS`; run with `ASAN_OPTIONS=detect_leaks=0`.

Reproduce the logical-block comparison using the f5/f7 export commands in their
reports. Regenerate the coherent-memory snapshots using the pinned circuit and
profiler command in [the first diagnostic](factored_state.md), then run:

```sh
OPENBLAS_NUM_THREADS=1 python tools/profile/frame_screen.py --directory /tmp/d5r5-snapshots --actions 144 488 544 --output /tmp/frames.json
```

Raw timings, all validation rows, frame choices, snapshot hashes, and source
hashes are in `alternatives_data.json`. The complete research suite passes 57
tests with the native proxy test enabled.

Recommendation: resume the logical-block eligibility/composition study when
choosing production work. Retain the proxy as an independently useful baseline
and potential restricted backend. Keep large-circuit Clifford-frame/tensor
research open, but do not let this bounded negative control become a claim that
those representations cannot beat logical blocks on another workload.
