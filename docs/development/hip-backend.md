<!--pytest-codeblocks:skipfile-->

# HIP Backend

!!! warning "Experimental and source-build only"
    The AMD HIP backend is not part of Clifft's published wheels or stable API.
    It supports a limited hardware and workload tier, requires an explicit
    source build, and may change without compatibility guarantees. It is never
    selected automatically.

Use `clifft.experimental.hip` to sample circuits on AMD GPUs. It shares the
CPU backend's compiler and planner and provides separate `Program` and
`Sampler` types.

## Current capabilities

| Workflow or feature | HIP support |
|---|---|
| Ordinary fixed-row sampling | Supported for eligible programs |
| Post-selected survivor sampling | Supported for eligible programs |
| Measurements, detectors, observables, and `EXP_VAL` | Supported |
| Pauli and readout noise | Supported |
| Peak active width | `k <= 30`, subject to available device memory |
| Coefficient precision | FP64 default; FP32 experimental |
| Fixed-fault importance sampling | Not supported |
| Leakage, loss, and transition instruments | Not supported |
| Exact-probability and state-vector queries | Not supported |
| Asynchronous or multi-GPU execution | Not supported |

Unsupported programs are rejected during lowering; there is no automatic CPU
fallback.

### Execution tiers

The backend selects a tier from the program's peak active width, coefficient
precision, and the device's shared-memory limit:

| Tier | Threads per shot | Coefficient storage | Auto selection |
|---|---|---|---|
| `thread_per_shot` | 1 | global memory | `k <= 4` |
| `block_shared` | one block | shared memory | wider states that fit shared memory |
| `block_global` | one block | global memory | otherwise |

In Python, `hip.Sampler(program)` defaults to `tier="auto"`. Pass a tier name
from the table to select it explicitly. `hip.selected_tier(program, precision)`
reports the automatic choice without allocating a workspace; `sampler.tier`
reports the tier in use.

C++ defaults to `ExecutionTier::Auto`. Set `SamplingOptions::tier` or pass a
tier to the `Sampler` constructor or `replay_shot`. Inspect the choice with
`selected_tier(executable, precision)` or `Sampler::execution_tier()`.

An explicit `block_shared` request fails if the program needs too much shared
memory. Explicit `thread_per_shot` requests can run widths above four, subject
to available device memory.

## Hardware and source build

The documented target is Linux `x86_64` with ROCm and an MI300X-class `gfx942`
device. Hardware validation is still manual. Other AMD architectures can be
development targets but do not have the same conformance coverage.

HIP builds require CMake 3.21 or newer and a ROCm toolchain. The commands below
assume the HIP compiler and ROCm root are discoverable automatically unless
their locations are supplied explicitly.

Build an editable installation from a checkout:

```bash
git clone https://github.com/unitaryfoundation/clifft.git
cd clifft

uv venv
CMAKE_ARGS="-DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942" \
    uv pip install -e .
```

If the HIP compiler and ROCm root are installed under `/usr`, provide them
explicitly:

```bash
CMAKE_ARGS="-DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_HIP_COMPILER=/usr/bin/clang++-17 \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr" \
    uv pip install -e .
```

For standalone C++ development, configure and build the HIP target directly:

```bash
cmake -S . -B build-hip -G Ninja \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build-hip -j
```

The build can compile device code without a visible GPU. Sampling requires a
compatible device at runtime:

```python
from clifft.experimental import hip

print(hip.is_built())
print(hip.is_available())
print(hip.backend_info())
```

Clifft does not yet publish a supported ROCm and driver matrix. Treat local
conformance testing as a requirement for experimental use.

## Compile and reuse a sampler

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

`hip.Program` and `clifft.Program` are not interchangeable. `hip.compile()`
currently accepts Stim circuit text and does not expose the CPU
`input_format` switch.

Construct one sampler per concurrently active caller and reuse it. Construction
uploads the program and allocates a bounded workspace on the device that is
current at that time. Calls on one sampler are synchronous and must not
overlap.

For post-selection, compile the detector mask into the HIP program and call
`sampler.sample_survivors()`. Fixed-row `sample()` rejects a post-selected
program. Survivor sampling always returns aggregate counts; set
`keep_records=True` to retain survivor rows.

### Precision and launch controls

FP64 coefficient evolution is the default. FP32 reduces coefficient storage
and is a separate experimental numerical mode:

```python
sampler = hip.Sampler(
    program,
    precision="fp32",
    max_batch_shots=16_384,
)
result = sampler.sample(100_000, seed=42, block_size=256)
print(sampler.allocated_device_bytes)
```

Probability calculations, replay log-probabilities, and `EXP_VAL` outputs
remain FP64 in both modes.

- `max_batch_shots` bounds retained device workspace. Larger requests are
  split into batches. The sampler may reduce this limit to fit available
  memory; `sampler.max_batch_shots` reports the retained capacity.
- `block_size` sets the number of threads per block. The default, 0, selects
  256 for `thread_per_shot` or 64, 128, or 256 for the block tiers. Explicit
  values must be in `1..1024` for `thread_per_shot`, or one of 64, 128, and 256
  for the block tiers.
- `allocated_device_bytes` exposes retained workspace size for experiments.

These controls differ from CPU `batch_size`, `threads`, and `thread_layout`.
A fixed seed reproduces rows for the same HIP precision, tier, and block size,
regardless of workspace batch size. Changing tiers or block sizes can change
rounding and sampled rows. CPU and HIP also use different random streams;
compare their stochastic results statistically.

## Testing

Ordinary CPU builds test HIP plan lowering and validation. HIP-enabled CI
also compiles `gfx942` device code and runs tests that do not need a GPU.

Kernel tests are skipped without a visible AMD GPU, so hardware testing is
still required.

Manual MI300X tests exercise FP64 and FP32 repeatability, forced branches and
expectation values against the CPU executor, noisy distributions,
post-selection, and retained output rows. A supported backend will require
regular hardware testing and a declared ROCm/driver matrix.

See [HIP Kernel Development](hip-kernel-development.md) for implementation
details and contribution instructions.
