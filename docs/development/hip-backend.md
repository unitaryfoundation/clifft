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

## Building for gfx942

HIP support is off by default. Enable it explicitly and select the target GPU:

```bash
cmake -S . -B build-hip -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build-hip --target clifft_tests -j
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

This produces a `gfx942` HIP fat binary without requiring a GPU on the build
host. Tests that need a visible HIP device report as skipped; lowering,
validation, and zero-shot tests still run offline.

## Current Execution Tier

The first vertical slice uses one HIP thread per shot and preallocates all
coefficient, scratch, symbol, record, and output storage before kernel launch.
It is intentionally restricted to peak active width `k <= 4`, where serial
coefficient work per shot is small.

This tier supports both coefficient formats in one backend:

- FP64 coefficients are the default.
- FP32 coefficients are experimental.
- Probability reductions, normalization factors, aggregate statistics, and
  `EXP_VAL` results use FP64 in both modes.

The interpreter handles rotations, promotions, active and dormant
measurements, affine records and symbols, categorical Pauli noise, asymmetric
readout noise, detectors and postselection, observables, and expectation-value
probes. Fixed-row sampling rejects postselected plans and directs callers to
survivor sampling instead.

The initial backend rejects transition instruments. Leakage and loss,
importance sampling, `sample_k`, asynchronous execution, and multi-GPU
execution are also outside this tier.

## Conformance and Next Tiers

Hardware conformance is backend-specific. Tests require repeatability within
each HIP coefficient mode, explicit `EXP_VAL` tolerances against the CPU
`ExecutablePlan`, and CPU-oracle statistical agreement for stochastic noise,
readout noise, postselection, and observables. CPU and HIP execution order need
not match, so same-seed output is not required to be identical across
backends.

The cultivation distance-5 fixture reaches peak active width `k = 10` and is
kept as an explicit boundary test. The next execution tier should assign a
cooperative thread block to each shot and keep coefficients in LDS through
approximately `k = 10`. A persistent global-memory tier through approximately
`k = 18` can follow after that cooperative path is validated on MI300X
hardware.
