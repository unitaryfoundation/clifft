<!--pytest-codeblocks:skipfile-->

# HIP Kernel Development

This guide covers changes to the HIP backend. CPU and HIP share
`SamplingPlan`; HIP's packed actions, workspace, and kernels are private to
the backend.

For build instructions and API usage, see
[HIP Backend](hip-backend.md).

## Build a Developer Installation

The ordinary Python package always contains `clifft.experimental.hip`, but its
native extension is built only when HIP is explicitly enabled. On a ROCm host,
install an editable developer build with:

```bash
CMAKE_ARGS="-DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942" \
    uv pip install -e .
```

On hosts where the HIP compiler is installed under `/usr`, add its location to
the same command:

```bash
CMAKE_ARGS="-DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_HIP_COMPILER=/usr/bin/clang++-17 \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr" \
    uv pip install -e .
```

For C++ iteration, use the standalone build documented in
[HIP Backend](hip-backend.md). Both paths compile device
code offline; only kernel-launch tests require a visible GPU.

## Python Iteration Loop

The Python facade uses the same HIR optimization and `SamplingPlan` boundary as
the CPU backend:

```python
from clifft.experimental import hip

print(hip.backend_info())

program = hip.compile("""
    H 0
    T 0
    H 0
    M 0
    OBSERVABLE_INCLUDE(0) rec[-1]
""")
print(program.inspect())

sampler = hip.Sampler(
    program,
    precision="fp64",
    max_batch_shots=16_384,
)
result = sampler.sample(100_000, seed=1234, block_size=256)
branch = sampler.replay_shot([0])
```

`Program` can be inspected without a GPU.
`Sampler` selects FP32 or FP64 coefficient evolution, uploads the program, and
allocates its bounded workspace on the device current at construction. It is
synchronous and should be reused for repeated calls. Overlapping calls on one
sampler are rejected; use a separate sampler per caller.

## Source Map

| Change | Primary location | Contract |
| --- | --- | --- |
| Support a `SamplingAction` | `src/clifft/sampling/hip/executable_plan.cc` | Exhaustive lowering from the shared plan |
| Change shared Pauli preparation | `src/clifft/sampling/pauli_preparation.h` | Execution-ready geometry consumed by CPU and HIP lowering |
| Change a packed action | `src/clifft/sampling/hip/device_program.h` | Private host/device descriptor |
| Change coefficient evolution | Device half of `src/clifft/sampling/hip/sampler.hip` | FP32 and FP64 interpreter templates |
| Change per-shot memory layout | `coefficient_elements_per_shot` in `device_program.h` and `Sampler::Impl` in `sampler.hip` | Shared host/device sizing and retained workspace |
| Change launch or batching | Host half of `sampler.hip` | Global shot indices and synchronous batches |
| Change the Python experiment | `src/python/clifft/experimental/hip.py` | Typed optional facade over `_clifft_hip` |
| Add conformance cases | `tests/test_hip_sampler.cc` and `tests/python/utils_hip.py` | CPU oracle, replay, and distributions |

The `__HIP_DEVICE_COMPILE__` boundary in `sampler.hip` separates device
interpretation from host ownership and collection. Kernel templates must keep
device-visible explicit instantiations when their launchers are hidden from the
device pass.

## Add or Change an Action

1. Add explicit lowering for the existing `SamplingAction` alternative in
   `ExecutablePlan::lower_action`. The dependent static assertion makes an
   unhandled alternative fail in ordinary CPU builds.
2. Put only execution-ready fields in `detail::Action`. Pauli geometry,
   coordinate changes, and symbolic dependencies belong in planning or
   lowering, not in the kernel.
3. Implement the tag in both `interpret_shots` and `interpret_shots_cooperative`,
   for FP32 and FP64.
4. Add host-only packing assertions to `test_hip_executable_plan.cc`.
5. Add forced-replay or deterministic hardware coverage before relying on a
   statistical comparison.

Do not create a second name for an existing sampling action. The device tag is
private serialization for the HIP interpreter, not another semantic IR.

## Change the Workspace or Add an Execution Tier

`Sampler` allocates the program and workspace before launching a kernel.
Requests larger than `max_batch_shots` reuse the workspace across launches.
Each launch receives a row count and global shot offset. Survivor sampling
with `keep_records=False` skips unused record, detector, and expectation-value
downloads.

The block tiers use `interpret_shots_cooperative`, with one block per shot.
Threads share the shot's symbols, records, and outputs. Use
`CooperativeLane::is_writer()` for these writes and synchronize before
overwriting a value that other threads may still be reading. Pairings, Pauli
phases, and active-width transitions must be prepared before execution.

When changing batching or launch geometry, preserve these invariants:

- no allocation inside a kernel or ordinary dispatch loop;
- the RNG uses the global shot index, so batch size cannot change seeded rows;
- coefficient arithmetic follows the selected precision;
- reductions, normalization, statistics, replay likelihoods, and expectation
  values remain FP64; and
- results agree with the CPU `ExecutablePlan`.

## Add Another AMD Architecture

Select another GPU target at build time:

```bash
cmake -S . -B build-hip -G Ninja \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx950
```

A development binary may contain more than one target:

```bash
cmake -S . -B build-hip -G Ninja \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES="gfx942;gfx950"
```

Measure performance before adding device-specific launch settings. Keep the
action format and interpreter shared across GPU targets.

## Conformance Workflow

Run the C++ backend tests:

```bash
cmake --build build-hip --target clifft_tests clifft_hip_tests -j
ctest --test-dir build-hip --output-on-failure -R HIP
```

For Python API and replay tests:

```bash
uv run pytest tests/python/test_gpu_replay_reference.py tests/python/test_experimental_hip.py -v
```

`tests/python/utils_hip.py` provides exact repeatability, full-row
distribution, and forced-record probability helpers. Prefer forced replay for
small branching circuits because it probes every reachable branch and its
likelihood. Use joint-distribution comparisons for noise and other stochastic
behavior, with both precision modes parameterized. Add later measurements,
detectors, observables, and expectation values after non-diagonal operations so
tests observe the evolved state rather than only the first sampled outcome.
