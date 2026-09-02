<!--pytest-codeblocks:skipfile-->

# CUDA Backend

!!! warning "Experimental and source-build only"
    The NVIDIA CUDA backend is not part of Clifft's published wheels or stable
    API. It supports a limited hardware and workload tier, requires an explicit
    source build, and may change without compatibility guarantees. It is never
    selected automatically.

Clifft's CPU implementation is the stable reference. The experimental CUDA
backend shares circuit parsing, compilation, and symbolic planning with it,
then lowers the prepared plan into a private GPU executable. It follows the
same contract as the [HIP backend](hip-backend.md): a backend-specific
`ExecutablePlan` and `Sampler` keep precision, workspace, and launch controls
outside the stable API, and the CPU executor remains the semantic oracle.

## Current capabilities

| Workflow or feature | CUDA support |
|---|---|
| Ordinary fixed-row sampling | Supported for eligible programs |
| Post-selected survivor sampling | Supported for eligible programs |
| Measurements, detectors, observables, and `EXP_VAL` | Supported |
| Pauli and readout noise | Supported |
| Peak active width | `k <= 30`; three execution tiers, see below |
| Coefficient precision | FP64 default; FP32 experimental |
| Fixed-fault importance sampling | Not supported |
| Leakage, loss, and transition instruments | Not supported |
| Exact-probability and state-vector queries | Not supported |
| Asynchronous or multi-GPU execution | Not supported |

Unsupported programs are rejected during lowering; there is no automatic CPU
fallback.

## Execution tiers

Where the HIP backend assigns every shot to one thread, the CUDA backend
selects one of three tiers per program and device:

| Tier | Shot ownership | Coefficient residence | Automatic selection |
|---|---|---|---|
| `ThreadPerShot` | one thread | global memory, one slab per shot | `k <= 4` |
| `BlockShared` | one thread block | opt-in dynamic shared memory | shot fits the device's shared-memory budget |
| `BlockGlobal` | one thread block | global memory, one slab per resident block | otherwise |

In the cooperative tiers every thread of a block evaluates the scalar control
flow (random draws, branch selection, expressions) redundantly from identical
inputs, so branch decisions need no broadcast storage. Coefficient sweeps are
strided across the block, measurement probabilities are tree-reduced through a
fixed shared-memory scratch, and byte outputs are written by lane 0. The
per-shot RNG derives from the global shot index, so the tier, block size, and
concurrency cap cannot change seeded rows within one tier. Different tiers sum
probabilities in different orders, so compare them statistically or through
forced replay rather than row for row.

The shared-memory budget is the device's opt-in limit minus a 16 KB reduction
scratch. On an H100 or H200 (227 KB opt-in) FP64 states fit through `k = 13`
and FP32 through `k = 14`; devices with a 96 KB or 164 KB opt-in cover one or
two fewer coordinates. `selected_tier()` reports the automatic choice for an
executable on the current device without allocating a workspace.

## Hardware and source build

The documented target is Linux `x86_64` with the CUDA toolkit and a
Hopper-class `sm_90` device (H100 or H200), where the backend was validated.
Other architectures can be development targets by setting
`CMAKE_CUDA_ARCHITECTURES`; the interpreter has no architecture-specific
branch, but their shared-memory budgets and conformance coverage differ.

CUDA builds require a CUDA toolkit with C++20 support (CUDA 12 or newer).
Build an editable installation from a checkout:

```bash
git clone https://github.com/unitaryfoundation/clifft.git
cd clifft

uv venv
CMAKE_ARGS="-DCLIFFT_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90" \
    uv pip install -e .
```

For standalone C++ development, configure and build the CUDA targets
directly:

```bash
cmake -S . -B build-cuda -G Ninja \
    -DCLIFFT_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build-cuda -j
```

The build compiles device code without a visible GPU. Sampling requires a
compatible device at runtime:

```python
from clifft.experimental import cuda

print(cuda.is_built())
print(cuda.is_available())
print(cuda.backend_info())
```

## Compile and reuse a sampler

```python
from clifft.experimental import cuda

program = cuda.compile("""
    H 0
    T 0
    H 0
    M 0
    OBSERVABLE_INCLUDE(0) rec[-1]
""")

sampler = cuda.Sampler(program)
result = sampler.sample(100_000, seed=1234)
print(sampler.tier)
print(result.measurements.shape)
```

`cuda.Program` and `clifft.Program` are not interchangeable. `cuda.compile()`
currently accepts Stim circuit text and does not expose the CPU
`input_format` switch. `cuda.selected_tier(program)` reports the tier
automatic selection would pick without allocating a workspace.

The same contract is available from C++:

```cpp
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/cuda/executable_plan.h"
#include "clifft/sampling/cuda/sampler.h"
#include "clifft/sampling/planner.h"

using namespace clifft;

const sampling::SamplingPlan plan = sampling::plan_sampling(trace(parse(R"(
    H 0
    T 0
    H 0
    M 0
    OBSERVABLE_INCLUDE(0) rec[-1]
)")));
const sampling::cuda::ExecutablePlan executable(plan);

sampling::cuda::Sampler sampler(executable);
const sampling::SamplingResult result = sampler.sample(100000, uint64_t{1234});
```

Construct one sampler per concurrently active caller and reuse it. Construction
resolves the execution tier, uploads the program, and allocates a bounded
workspace on the device that is current at that time. Calls on one sampler are
synchronous and must not overlap.

For post-selection, compile the detector mask into the plan and call
`sampler.sample_survivors()`. Fixed-row `sample()` rejects a post-selected
program. Survivor sampling always returns aggregate counts; pass
`keep_records = true` to retain survivor rows.

### Precision and launch controls

```python
sampler = cuda.Sampler(
    program,
    precision="fp32",
    max_batch_shots=16_384,
    tier="auto",
    max_concurrent_shots=0,
)
result = sampler.sample(100_000, seed=42, block_size=256)
print(sampler.allocated_device_bytes)
```

The C++ constructor takes the same values in the same order:

```cpp
sampling::cuda::Sampler sampler(executable,
                                sampling::cuda::CoefficientPrecision::FP32,
                                /*max_batch_shots=*/16384,
                                sampling::cuda::ExecutionTier::Auto,
                                /*max_concurrent_shots=*/0);
```

- FP64 coefficient evolution is the default. FP32 halves coefficient storage
  and widens the shared-memory tier by one coordinate. Probability reductions,
  normalization factors, aggregate statistics, replay log-probabilities, and
  `EXP_VAL` outputs remain FP64 in both modes.
- `max_batch_shots` bounds the retained per-shot output rows. Larger requests
  are split into synchronous launches that reuse the workspace.
- `tier` forces one execution tier for experiments (`"thread_per_shot"`,
  `"block_shared"`, or `"block_global"` from Python); `block_shared` is
  rejected when the program does not fit the device. `Sampler.tier` reports
  the resolved choice.
- `max_concurrent_shots` caps how many shots the cooperative tiers keep
  resident per launch, which bounds `BlockGlobal` slab memory. Zero derives the
  cap from the multiprocessor count and free device memory.
- `block_size` must be a power of two between 1 and 1024. In the cooperative
  tiers it is the number of lanes that share one shot.
- `allocated_device_bytes()` exposes the retained workspace size.

CPU, HIP, and CUDA use separate random-stream domains, so compare
deterministic branches directly and stochastic results statistically.

## Architecture

```text
HIR -> SamplingPlan -> CPU ExecutablePlan -> trusted CPU sampling oracle
                    -> private HIP executable -> device interpreter
                    -> private CUDA executable -> tiered device interpreter
```

The CUDA executable is a backend-specific packing of prepared `SamplingAction`
alternatives with the same shape as the HIP executable: host-computed Pauli
phases and pairings, active-width transitions, expressions, and noise
distributions. The device executes the plan without topology planning or
allocation in its dispatch loop. CUDA and HIP lowering share execution-ready
Pauli preparation and result containers with the CPU backend; their executable
layouts, workspaces, and kernels remain backend-specific.

| Change | Primary location | Contract |
| --- | --- | --- |
| Support a `SamplingAction` | `src/clifft/sampling/cuda/executable_plan.cc` | Exhaustive lowering from the shared plan |
| Change a packed action | `src/clifft/sampling/cuda/device_program.h` | Private host/device descriptor |
| Change coefficient evolution | Device half of `src/clifft/sampling/cuda/sampler.cu` | Lane-strided FP32 and FP64 action bodies shared by every tier |
| Change tier selection or launch | Host half of `sampler.cu` | `resolve_tier`, `resolve_concurrency`, and `Sampler::Impl::launch` |
| Change the Python experiment | `src/python/clifft/experimental/cuda.py` | Typed optional facade over `_clifft_cuda` |
| Add conformance cases | `tests/test_cuda_sampler.cc` and `tests/python/utils_cuda.py` | CPU oracle, replay, tiers, and distributions |

When changing the kernels, preserve these invariants:

- no allocation inside a kernel or ordinary dispatch loop;
- every lane of a block advances the same RNG and takes the same branches;
- a collapse stages through scratch, because lanes write packed outputs while
  other lanes still read overlapping sources;
- reductions, normalization, statistics, replay likelihoods, and expectation
  values remain FP64; and
- the CPU `ExecutablePlan` remains the semantic oracle.

## Testing and contribution boundary

Ordinary CPU builds compile the host-side CUDA lowering tests. They check
packed actions, expressions, noise tables, prepared Pauli data, width limits,
and rejection of unsupported plans. Adding a `SamplingAction` without CUDA
lowering fails during this build.

CUDA-enabled CI additionally compiles `sm_90` device code and runs GPU-free
conformance cases:

```bash
cmake --build build-cuda --target clifft_tests clifft_cuda_tests -j
ctest --test-dir build-cuda --output-on-failure -R CUDA
```

The Python suite provides quick developer probes with the same helpers:

```bash
uv run pytest tests/python/test_experimental_cuda.py -v
```

Kernel-launch tests are skipped without a visible NVIDIA GPU, so this coverage
does not establish runtime correctness on hardware. The hardware suite
exercises FP64 and FP32 repeatability, every tier against the CPU executor on
forced branches and expectation values, cross-tier agreement on a wide
program, noisy distributions, post-selection, retained output rows, and the
cooperative concurrency cap.
