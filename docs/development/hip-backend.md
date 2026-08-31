<!--pytest-codeblocks:skipfile-->

# Experimental HIP Sampling Backend

Clifft has an optional AMD HIP sampling backend for Linux x86-64 and
MI300X-class `gfx942` devices. It is a development target, not a public API or
an automatically selected runtime backend. The default build remains CPU-only.

The semantic boundary is `sampling::SamplingPlan`:

```text
HIR -> SamplingPlan -> CPU ExecutablePlan -> trusted CPU sampling oracle
                    -> private HIP executable -> device interpreter
```

The HIP executable is a flat, backend-specific packing of existing
`SamplingAction` alternatives. It stores host-precomputed Pauli phases,
pairings, active-width transitions, expressions, and noise distributions. The
device therefore executes the plan without performing topology planning or
allocating storage in its dispatch loop.

CPU and HIP lowering share only execution-ready Pauli preparation and result
containers. Their executable layouts, mutable state, dispatch order, and
workspace ownership remain backend-specific.

## Building for gfx942

HIP support is off by default. Enable it explicitly and select the target GPU:

```bash
cmake -S . -B build-hip -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build-hip --target clifft_tests clifft_hip_tests -j
ctest --test-dir build-hip --output-on-failure
```

The exe.dev Linux image exposes ROCm through Clang rather than `hipcc`. Its
equivalent configuration is:

```bash
cmake -S . -B build-hip -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_COMPILER=/usr/bin/clang++-17 \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr \
    -DCMAKE_HIP_ARCHITECTURES=gfx942
```

This produces a `gfx942` HIP binary without requiring a GPU on the build host.

## Current Execution Tier

The first vertical slice uses one HIP thread per shot and preallocates all
coefficient, scratch, symbol, record, and output storage before kernel launch.
It is intentionally restricted to peak active width `k <= 4`, where serial
coefficient work per shot is small.

`sampling::hip::Sampler` uploads an executable once and owns scalar result
metadata plus a reusable, precision-specific workspace; it does not retain a
duplicate host executable. Large requests are divided into bounded batches;
the kernel receives the global shot offset so changing the batch size does not
change a seeded shot's random stream. The free C++ sampling functions remain
convenience wrappers that construct a temporary sampler. Aggregate-only survivor
sampling downloads survival flags and observables, but omits record, detector,
and expectation-value transfers that its caller does not consume. Overlapping
calls on one retained sampler are rejected by the backend.

This tier supports both coefficient formats in one backend:

- FP64 coefficients are the default.
- FP32 coefficients are experimental.
- Coefficient storage and amplitude evolution use the selected precision.
- Probability reductions, normalization-factor calculation, aggregate
  statistics, replay log-probabilities, and `EXP_VAL` results use FP64 in both
  modes.

The interpreter handles rotations, promotions, active and dormant
measurements, affine records and symbols, categorical Pauli noise, asymmetric
readout noise, detectors and postselection, observables, and expectation-value
probes. Fixed-row sampling rejects postselected plans and directs callers to
survivor sampling instead.

The initial backend rejects transition instruments. Leakage and loss,
importance sampling, `sample_k`, asynchronous execution, and multi-GPU
execution are also outside this tier.

## Tests That Run Without a GPU

Ordinary CPU builds compile the host-side HIP lowering and its tests. These
tests check the packed actions, expressions, noise tables, prepared Pauli data,
supported-width checks, and explicit rejection of unsupported plans. Adding a
new `SamplingAction` without handling it in the HIP lowering also fails during
this build.

When HIP is enabled, the ROCm CI job additionally compiles the device code for
`gfx942`, runs the GPU-free HIP conformance cases, and installs the optional
Python extension through the editable developer workflow. The Python boundary
test passes shared HIR from `_clifft_core` into `_clifft_hip` and inspects the
lowered program. Tests that launch kernels report as skipped when no AMD GPU is
visible.

This GPU-free coverage does not establish that kernels launch or produce
correct results on a GPU.

## Tests That Run on an AMD GPU

The hardware tests exercise both FP64 and FP32 coefficient modes. They cover:

- repeatability within one HIP mode for a fixed seed;
- forced measurement outcomes compared with the CPU executor;
- expectation values compared with the CPU executor using explicit
  tolerances;
- asymmetric readout noise and multi-outcome Pauli noise;
- postselection, survivor counting, and retained output rows; and
- statistical comparisons with the CPU for stochastic results.

These tests are intended to run manually on MI300X hardware during the
experimental phase. A supported backend will require regular hardware testing
and a declared ROCm and driver compatibility matrix.

## How Correctness Is Determined

Deterministic outputs and forced measurement branches are compared directly
with the CPU `ExecutablePlan`. Forced replay checks every branch of selected
small circuits, including branch probability, later measurements, detectors,
observables, and expectation values. This catches state-evolution errors that
could be missed by a statistical test.

Stochastic circuits are compared by their observed distributions with
sample-count-aware tolerances. CPU and HIP use separate random-stream domains,
so the same user seed does not correlate their oracle samples. They are not
required to produce identical rows. A fixed HIP mode must remain repeatable for
the same seed, including when a request is divided into different batch sizes.

## Next Work

- Run the complete conformance suite on MI300X hardware.
- Aggregate survivor statistics on the device when full rows are not
  requested.
- Add a cooperative thread-block path using on-chip shared memory through
  approximately `k = 10`, including the cultivation distance-5 fixture.
- Evaluate a global-memory path for larger active widths after the cooperative
  path is validated.

See [HIP Kernel Development](hip-kernel-development.md) for the experimental
Python workflow, source map, and extension checklist.
