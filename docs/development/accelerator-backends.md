# Accelerator Backend Development

Accelerator support is under active development. It is private implementation
work: Clifft does not yet expose a stable accelerator API, ship CUDA or ROCm
packages, or select a GPU automatically.

The initial goal is to learn from small native CUDA and HIP implementations
without reviving the retired bytecode VM or freezing a common device command
stream prematurely.

## Shared Semantic Input

Every execution strategy starts from `sampling::SamplingPlan`. The planner has
already resolved coordinate changes, active Pauli actions, symbolic
dependencies, output slots, and active-width transitions. Accelerator
preparation may change storage layout, fuse actions, and choose launch
strategies, but it must not perform tableau evolution, commutation analysis,
localization, or dependency discovery at runtime.

The private `accelerator::analyze_plan_requirements` helper gives a backend a
single construction-time check for:

- every semantic action present in a plan;
- presampled noise;
- postselection; and
- the required peak active width.

Each backend compares that inventory with its current implementation and
rejects an unsupported plan before allocating device storage or launching
work. The support decision and error remain backend-owned during the spikes so
we do not freeze a generic capability interface prematurely.

The existing compile profiler prints the same requirement summary for a real
circuit:

```bash
cmake -B build-profile \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCLIFFT_BUILD_PROFILER=ON
cmake --build build-profile --target profile_compile -j

CLIFFT_COMPILE_ITERATIONS=1 \
  CLIFFT_CIRCUIT_FILE=tests/fixtures/cultivation_d5.stim \
  ./build-profile/profile_compile
```

This command does not need a GPU. It lets collaborators compare the semantic
coverage required by cultivation, distillation, synthetic-width, and smaller
correctness circuits before implementing device kernels.

## Provisional Backend Shape

CUDA and ROCm should own their prepared representations and native device
code. During the spikes, each implementation needs only three conceptual
operations:

1. Check whether the backend supports the semantic plan.
2. Prepare backend-specific descriptors, storage, and launch information.
3. Run shots into storage allocated during preparation.

This is a collaboration convention, not a C++ virtual interface or public
ABI. Data layout, kernel decomposition, launch policy, and tuning may differ
between CUDA and ROCm. Common code should be introduced only after both
implementations demonstrate that it is genuinely shared.

The existing CPU executable remains unchanged while the spikes run. The
public Python API continues to use CPU execution.

## Development Workflow

Shared support code and tests land on `main` through small pull requests. CUDA
and ROCm development may use one draft branch per vendor, based on the shared
scaffold. A vendor slice is ready to merge behind a disabled-by-default build
option when:

- host-side preparation and capability tests pass without a GPU;
- device code compiles for every declared target;
- the CPU oracle and a trusted-device smoke test pass;
- unsupported operations return a specific capability error; and
- the ordinary CPU build and Python package are unchanged.

An experimental backend may be disabled in normal builds, but it must be
compiled by an explicit CI job and exercised on a schedule. Code that no
workflow builds or tests should not remain on `main`.

## Working Without a Local GPU

The shared architecture, plan analysis, CPU oracle, and batch sampler can be
developed and tested on macOS. A Linux x86-64 machine without a GPU can also
compile device code for explicit CUDA and AMD targets once the corresponding
toolchain targets exist.

Real hardware remains required for kernel launches, device memory behavior,
synchronization, numerical and statistical correctness, and performance. The
intended loop is local development, GPU-less compile checks, then controlled
smoke and benchmark runs on vendor hardware.

The first AMD target is MI300X (`gfx942`). `gfx950` may be compile-tested but
must not be advertised until it passes on MI350X or MI355X hardware. NVIDIA
correctness and performance targets will be recorded by the CUDA slice rather
than embedded in the shared contract.

## Initial Scope

The first vertical slices target Linux x86-64, one synchronous GPU, and FP64.
They should exercise ordinary sampling, survivor sampling with postselection,
records, detectors, and observables. `EXP_VAL` should be included when the
spike confirms that it fits the initial kernel design.

Leakage and loss, `sample_k`, importance sampling, exact state queries,
multi-GPU execution, asynchronous APIs, FP32, automatic backend selection,
packaging, and a stable public device representation are deferred.
