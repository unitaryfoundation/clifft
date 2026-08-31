<!--pytest-codeblocks:skipfile-->

# Experimental GPU Execution

!!! warning "Experimental and source-build only"
    GPU execution is not part of Clifft's stable API or published wheels. The
    current HIP backend has a narrow hardware and workload tier, requires an
    explicit source build, and may change without compatibility guarantees.
    It is never selected automatically.

Clifft's stable Python API executes on the CPU. Experimental GPU backends use
separate program and sampler types so backend-specific precision, workspace,
and launch controls do not leak into the stable CPU contract.

## Backend Status

| Backend | Status | Selection | Distribution |
|---|---|---|---|
{% for backend in workflow_contracts['backends'] -%}
| {{ backend['name'] }} | {{ backend['status'] }} | {{ backend['selection'] }} | {{ backend['distribution'] }} |
{% endfor %}

CUDA will receive its own explicit experimental boundary when an implementation
exists. The HIP API and current capabilities should not be interpreted as a
CUDA support promise.

## Current Capability Matrix

| Workflow or feature | Stable CPU | Experimental HIP |
|---|---|---|
{% for capability in workflow_contracts['backend_capabilities'] -%}
| {{ capability['feature'] }} | {{ capability['cpu'] }} | {{ capability['hip'] }} |
{% endfor %}

The current HIP tier uses one GPU thread per shot and targets workloads with
small active states. It supports the existing prepared sampling actions for
rotations, measurements, expressions, supported noise, post-selection,
observables, and expectation-value probes. Unsupported programs are rejected
during lowering rather than silently falling back to the CPU.

## Hardware and Build Requirements

The current documented target is Linux `x86_64` with ROCm and an MI300X-class
`gfx942` device. Hardware validation is still manual. Other AMD architectures
can be development build targets, but they do not yet have the same conformance
coverage.

The ordinary `pip install clifft` package is CPU-only. Build an editable HIP
installation from a checkout:

```bash
git clone https://github.com/unitaryfoundation/clifft.git
cd clifft

uv venv
CMAKE_ARGS="-DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942" \
    uv pip install -e .
```

When the HIP compiler and ROCm root are installed under `/usr`, provide them
explicitly:

```bash
CMAKE_ARGS="-DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_HIP_COMPILER=/usr/bin/clang++-17 \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr" \
    uv pip install -e .
```

The build can compile device code without a visible GPU. Sampling requires a
compatible device at runtime. Check all three states before constructing a
sampler:

```python
from clifft.experimental import hip

print(hip.is_built())
print(hip.is_available())
print(hip.backend_info())
```

- `is_built()` reports whether the optional native extension is installed.
- `is_available()` reports whether the extension loaded and sees an AMD GPU.
- `backend_info()` explains a missing build or load failure and lists visible
  devices when the runtime is available.

Clifft does not yet publish a supported ROCm and driver compatibility matrix.
Treat successful local conformance tests as a requirement for experimental
use.

## Compile and Reuse a HIP Sampler

The experimental facade compiles Stim-compatible text through the shared HIR
and semantic sampling-plan pipeline, then lowers it into a private HIP program:

```python
from clifft.experimental import hip

program = hip.compile("""
    H 0
    T 0
    H 0
    M 0
    OBSERVABLE_INCLUDE(0) rec[-1]
""")

sampler = hip.Sampler(program)
result = sampler.sample(100_000, seed=1234)
print(result.measurements.shape)
print(result.observables.shape)
```

`hip.Program` is not a CPU `clifft.Program`, and neither type is accepted by the
other backend's sampling functions. `hip.compile()` currently accepts
Stim-compatible text; it does not expose the stable CPU `input_format` switch.

Construct one `hip.Sampler` per concurrently active caller and reuse it for
repeated requests. Construction uploads the program and allocates a bounded
workspace on the device that is current at that time. Calls on one sampler are
synchronous and must not overlap.

## Post-Selected Survivors

Compile the detector mask into the HIP program and call the survivor method:

```python
from clifft.experimental import hip

program = hip.compile(
    "H 0\nM 0\nDETECTOR rec[-1]",
    postselection_mask=[1],
)
sampler = hip.Sampler(program)
result = sampler.sample_survivors(
    100_000,
    keep_records=True,
    seed=42,
)

print(result.passed_shots, result.total_shots)
print(result.measurements.shape)
```

`Sampler.sample()` rejects a post-selected program because fixed-row output
cannot represent discarded shots. `Sampler.sample_survivors()` always returns
aggregate survivor metadata and retains per-survivor rows only when
`keep_records=True`.

## Precision, Workspace, and Launch Controls

FP64 coefficient evolution is the default. FP32 reduces coefficient storage
and is a separate experimental precision configuration:

```python
sampler = hip.Sampler(
    program,
    precision="fp32",
    max_batch_shots=16_384,
)
result = sampler.sample_survivors(100_000, seed=42, block_size=256)
print(sampler.allocated_device_bytes)
```

Probability reductions, normalization factors, aggregate statistics, replay
log-probabilities, and `EXP_VAL` outputs remain FP64 in both precision
configurations.

- `max_batch_shots` bounds the retained device workspace. Larger requests are
  divided into synchronous launches that reuse it.
- `block_size` controls HIP launch geometry and must be between 1 and 1024.
- `allocated_device_bytes` exposes retained workspace size for experiments.

These controls are backend-specific. They are not equivalents of CPU
`batch_size`, `threads`, or `thread_layout`.

## Seeds and Cross-Backend Comparison

A fixed seed repeats within the same HIP precision and configuration. Splitting
one request across different retained workspace batch sizes preserves each
seeded HIP row because kernels receive the global shot index.

CPU and HIP use separate random-stream domains. The same user seed does not
produce matching CPU and GPU rows, and FP32 and FP64 should be treated as
separate numerical configurations. Compare deterministic branches directly and
stochastic results statistically.

## Unsupported Workloads

The current HIP backend rejects:

- peak active width above 4;
- transition instruments, leakage, and loss;
- `sample_k()` and other fixed-fault importance sampling;
- the stable exact-probability and state-vector query APIs;
- asynchronous calls and overlapping calls on one sampler; and
- multi-GPU execution.

There is no automatic CPU fallback. Keep a stable CPU path when a production
workflow requires broader capability or compatibility guarantees.

## Developer Documentation

Users experimenting with the backend should start on this page. Backend
contributors should continue with [HIP Backend Internals](../development/hip-backend.md)
and [HIP Kernel Development](../development/hip-kernel-development.md) for the
private executable format, conformance strategy, and kernel extension rules.
