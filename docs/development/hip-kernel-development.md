<!--pytest-codeblocks:skipfile-->

# HIP Kernel Development

This guide is the handoff point for extending the experimental AMD backend.
The backend is intentionally private below `SamplingPlan`: changes to its
packed actions, workspace, and kernels do not define a cross-backend ABI.

For the experimental user contract and minimal Python workflow, start with
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

`Program` is an immutable host lowering and can be inspected without a GPU.
`Sampler` selects FP32 or FP64 coefficient evolution, uploads the program, and
allocates its bounded workspace on the device current at construction. It is
synchronous and should be reused for repeated calls. Overlapping calls on one
sampler are rejected; use a separate sampler per caller. Its
`allocated_device_bytes` and `max_batch_shots` properties make memory experiments
visible without exposing raw buffers.

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
3. Implement the tag in the interpreter switch for both coefficient types.
4. Add host-only packing assertions to `test_hip_executable_plan.cc`.
5. Add forced-replay or deterministic hardware coverage before relying on a
   statistical comparison.

Do not create a second name for an existing sampling action. The device tag is
private serialization for the HIP interpreter, not another semantic IR.

## Change the Workspace or Add an Execution Tier

`Sampler` owns one uploaded program, scalar result-layout metadata, and one
precision-specific workspace. It does not retain a duplicate host executable.
Allocation is complete before a batch enters the kernel. A request larger than
`max_batch_shots` reuses that workspace, and each launch receives both its local
row count and global shot offset. Aggregate-only survivor requests skip unused
record, detector, and expectation-value downloads; device-side survivor
aggregation remains a separate execution-path extension.

A cooperative path should add a separate kernel and typed launcher, then
dispatch by peak active width. It should not add topology work to the device:
the packed executable already carries pairings, Pauli phases, expressions, and
active-width transitions. Keep the current thread-per-shot path as the small
width reference while the cooperative path uses on-chip shared memory through
approximately `k = 10`.

When changing batching or launch geometry, preserve these invariants:

- no allocation inside a kernel or ordinary dispatch loop;
- the RNG uses the global shot index, so batch size cannot change seeded rows;
- coefficient arithmetic follows the selected precision;
- reductions, normalization, statistics, replay likelihoods, and expectation
  values remain FP64; and
- the CPU `ExecutablePlan` remains the semantic oracle.

## Add Another AMD Architecture

The interpreter has no `gfx942` semantic branch. Select another target at
build time:

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

Add architecture-specific launch traits only after measurement shows a real
difference, such as block size, shared-memory capacity, or wavefront tuning.
Do not duplicate the action format or interpreter solely to name another GPU.

## Conformance Workflow

The C++ suite is the canonical backend conformance layer:

```bash
cmake --build build-hip --target clifft_tests clifft_hip_tests -j
ctest --test-dir build-hip --output-on-failure -R HIP
```

The Python suite provides quick developer probes:

```bash
uv run pytest tests/python/test_experimental_hip.py -v
```

`tests/python/utils_hip.py` provides exact repeatability, full-row
distribution, and forced-record probability helpers. Prefer forced replay for
small branching circuits because it probes every reachable branch and its
likelihood. Use joint-distribution comparisons for noise and other stochastic
behavior, with both precision modes parameterized. Add later measurements,
detectors, observables, and expectation values after non-diagonal operations so
tests observe the evolved state rather than only the first sampled outcome.
