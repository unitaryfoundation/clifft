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
fixed shared-memory scratch, and byte outputs are written by lane 0.

The per-shot RNG derives from the global shot index alone, so a shot draws the
same random stream whatever the batch size and concurrency cap. Rows are
reproducible for a fixed tier, precision, and block size. Rows are *not*
guaranteed to be bit-identical across tiers or block sizes: coefficient sweeps
are lane-strided and probabilities tree-reduce across the block, so both the
tier and the block size change the floating-point summation order, which can
move a probability across a measurement threshold. Compare different tiers or
block sizes statistically, or through forced replay, rather than row for row.

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

## Measured performance

Numbers from one H100 PCIe (80 GB, 114 multiprocessors, driver 570.195.03)
with CUDA 12.8, g++ 13.3, and CMake 3.28 in the
`nvidia/cuda:12.8.1-devel-ubuntu24.04` container, against the production CPU
sampler on the same machine (two AMD EPYC 9554 sockets, 28 vCPUs exposed,
Release build). `threads=0` is every core. The tool that produced them lives
in the tree; it checks each CUDA run's record marginals, pass rate, and
observable rates against the single-thread CPU run under a two-sample
binomial tolerance and reports the largest `|z|` (all rows below stay under
2.4):

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCLIFFT_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCLIFFT_BUILD_BENCHMARKS=ON
cmake --build build-cuda --target clifft_cuda_bench -j
B=./build-cuda/benchmarks/clifft_cuda_bench
$B --shots 100000 --threads 1,0 --precision both tests/fixtures/qv10.stim
$B --shots 20000  --threads 1,0 --precision both tests/fixtures/coherent_d5_r5.stim
$B --shots 100000 --threads 1,0 --precision both --postselect tests/fixtures/cultivation_d5.stim
$B --shots 200000 --threads 1,0 --width-sweep 1..9
$B --shots 200000 --threads 0 --concurrency-sweep 114,228,456,912,1824,3648,7296 \
    tests/fixtures/qv10.stim tests/fixtures/coherent_d5_r5.stim
```

Sampling time is the retained `sample()` or `sample_survivors()` call after
one warm-up call. Construction is program upload plus workspace allocation;
the first sampler in a process also pays CUDA context creation (about
130 ms here), later ones 1 to 4 ms. Device bytes is
`allocated_device_bytes()`.

### Fixtures

| Workload | Backend / tier | Precision | Sampling | Shots/s | Device bytes |
|---|---|---|---|---|---|
| QV-10, width 10, 100k shots | CPU `threads=1` | fp64 | 4164 ms | 24 k | |
| | CPU `threads=0` | fp64 | 288 ms | 348 k | |
| | CUDA auto = `BlockShared` | fp64 | 175 ms | 570 k | 1.4 MB |
| | CUDA `BlockGlobal` | fp64 | 265 ms | 378 k | 87 MB |
| | CUDA `ThreadPerShot` | fp64 | 3851 ms | 26 k | 1.5 GB |
| | CUDA auto = `BlockShared` | fp32 | 163 ms | 615 k | 1.4 MB |
| Coherent QEC d5/r5, width 13, 20k shots | CPU `threads=1` | fp64 | 9428 ms | 2.1 k | |
| | CPU `threads=0` | fp64 | 678 ms | 29.5 k | |
| | CUDA auto = `BlockShared` | fp64 | 204 ms | 98 k | 16 MB |
| | CUDA `BlockGlobal` | fp64 | 403 ms | 50 k | 700 MB |
| | CUDA `ThreadPerShot` | fp64 | 2201 ms | 9.1 k | 3.7 GB |
| | CUDA auto = `BlockShared` | fp32 | 117 ms | 171 k | 16 MB |
| Cultivation d5, width 10, postselected, 100k shots | CPU `threads=1` | fp64 | 269 ms | 372 k | |
| | CPU `threads=0` | fp64 | 24 ms | 4.14 M | |
| | CUDA auto = `BlockShared` | fp64 | 389 ms | 257 k | 1.0 GB |
| | CUDA `BlockGlobal` | fp64 | 371 ms | 270 k | 1.1 GB |
| | CUDA `ThreadPerShot` | fp64 | 79 ms | 1.26 M | 2.5 GB |
| | CUDA `ThreadPerShot` | fp32 | 69 ms | 1.46 M | 1.8 GB |

Where the backend wins: dense fixed-row sampling at width 10 and above. On
QV-10 the automatic tier is 1.6x the 28-core CPU (24x one core); on the
width-13 coherent QEC fixture it is 3.3x in FP64 and 5.8x in FP32, and the
shared tier beats the global tier by 2x because the shot never leaves the
chip.

Where it loses: postselected cultivation. Most shots fail an early detector
and exit after a few actions, so the run is dominated by per-shot overhead
rather than coefficient work. A thread block per shot is the wrong shape for
that, and the automatic choice ends up 16x slower than the 28-core CPU and
5x slower than the forced `ThreadPerShot` tier, which itself is 3x slower
than the CPU. Use `tier="thread_per_shot"` for discard-heavy programs, and
do not expect a speedup on them from this backend as it stands; a tier
policy that accounts for expected discard depth is future work.

### Why `ThreadPerShot` stops at width 4

Synthetic width-`k` family (`k` Hadamard/T pairs, CX layers, rotations,
expectation values before and after a collapsing measurement, full readout),
200k shots, FP64, block size 256, in shots per second:

| Width | CPU `threads=1` | CPU `threads=0` | `ThreadPerShot` | `BlockShared` | `BlockGlobal` |
|---|---|---|---|---|---|
| 1 | 30.8 M | 48.6 M | 312 M | 64.2 M | 63.5 M |
| 2 | 20.3 M | 51.6 M | 179 M | 43.2 M | 41.8 M |
| 3 | 8.5 M | 35.8 M | 104 M | 22.9 M | 22.3 M |
| 4 | 6.8 M | 27.3 M | 69.5 M | 20.1 M | 19.4 M |
| 5 | 4.5 M | 18.0 M | 47.9 M | 18.0 M | 17.4 M |
| 6 | 1.6 M | 11.9 M | 25.5 M | 16.2 M | 15.8 M |
| 7 | 1.1 M | 10.7 M | 12.1 M | 14.7 M | 14.3 M |
| 8 | 634 k | 8.1 M | 6.0 M | 13.3 M | 13.0 M |
| 9 | 357 k | 4.3 M | 3.0 M | 12.0 M | 11.5 M |

`ThreadPerShot` throughput halves with every coordinate because one thread
sweeps the whole state; the cooperative tiers lose only 10 to 20% per
coordinate once the sweep is wide enough to occupy the block. On this
family the crossover is between widths 6 and 7. The cutoff at 4 is
conservative: widths 5 and 6 leave 2.7x and 1.6x on the table on this
device. It stays at 4 for now because the per-thread tier's global slab pool
grows with both the batch size and `2^k` (1.5 GB at width 10 above) while the
shared tier's footprint does not, and because the crossover has only been
measured on this family, in FP64, on one device. Raising it is a
one-constant change (`kThreadPerShotMaxActiveWidth`) once more workloads
are measured.

### Why the concurrency default is 32 blocks per multiprocessor

`max_concurrent_shots` caps how many shots the cooperative tiers keep
resident per launch; the default derives 32 per multiprocessor (3648 here).
Automatic tier, FP64, 200k shots:

| Cap | QV-10 (width 10) | Coherent d5/r5 (width 13) |
|---|---|---|
| 114 (1 per SM) | 277 k | 97.8 k |
| 228 | 435 k | 97.9 k |
| 456 | 520 k | 98.1 k |
| 912 | 540 k | 98.0 k |
| 1824 | 558 k | 98.1 k |
| 3648 (default) | 568 k | 98.1 k |
| 7296 | 574 k | 98.1 k |

At width 10 a shot uses 40 KB of shared memory, so several blocks share a
multiprocessor and throughput keeps improving until roughly 16 to 32
resident blocks per multiprocessor, with 1% left beyond the default. At
width 13 a shot uses 212 KB, one block fills the multiprocessor, and the
cap is irrelevant above 114. The default therefore saturates the shared
tier, and for the global tier it bounds slab memory at 32 slabs per
multiprocessor unless free memory bounds it first.

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
program, noisy distributions, post-selection, retained output rows, the
cooperative concurrency cap, and automatic selection on both sides of the
device's shared-memory boundary (located at runtime through
`selected_tier()`), where forced replay pins the collapse and the state after
it against the CPU and survivor rows are checked for completeness and
alignment. Run it on a machine with a device:

```bash
./build-cuda/tests/clifft_cuda_tests -d yes
uv run pytest tests/python/test_experimental_cuda.py -v
```
