# Sampling benchmarks

`clifft_benchmarks` contains the CPU performance cases used by benchmark CI.
With `CLIFFT_ENABLE_CUDA=ON`, `clifft_cuda_bench` adds CPU-versus-CUDA sampling
and CUDA sampler construction cases. Both use Google Benchmark.

## CPU canary

The PR canary runs every benchmark registered in `clifft_benchmarks.cc`.
Add a useful workload by implementing and registering it there, using a name
containing only lowercase letters, digits, and underscores. `DISPLAY_NAMES`
in `tools/ci/benchmark_canary.py` provides optional report labels; workloads
without a label appear under their registered name.

Build and run the same suite locally:

```bash
cmake -S benchmarks -B build/canary -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/canary --target clifft_benchmarks -j
./build/canary/clifft_benchmarks \
    --benchmark_min_time=0.5s --benchmark_repetitions=3
```

Use `--benchmark_list_tests=true` to inspect the suite or
`--benchmark_filter='^phase_'` to run only the phase cases.

## External input execution

`--benchmark_filter='^external_inputs_'` compares reusable scalar and packed
execution for identical supplied noise/readout columns in 256-shot batches.
The surface-code case exercises optimized presampled carriers; the repeated
T-gate case exercises active quantum state and ordinary noise propagation.
Preparation, input generation, and copying results out of the executor are
excluded. These are primitive throughput measurements, not end-to-end leak/loss
sampling or deltakit-stim comparisons.

## Noncomputational scalar comparison

`noncomp_fixed.py` compares the internal scalar fixed path with trajectories on
surface-code circuits with inline leakage, full partner depolarization, and
label-independent readout. Both return all records, heralds, and detailed final
status; a final restoring reset erases leaked labels. It reports preparation
separately from repeated fixed sampling, plus active width and classical input
workspace. Trajectory times include their ordinary per-call preparation.

```bash
python benchmarks/noncomp_fixed.py --distances 3 5 --shots 1000
```

Use an optimized installed Clifft build. These timings do not compare against
deltakit-stim or measure packed execution. The workspace metric excludes the
immutable plan and quantum coefficient/scratch buffers.

## CUDA driver

Build with a CUDA toolkit; running the CUDA cases also requires a compatible
NVIDIA GPU:

```bash
cmake -S . -B build-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCLIFFT_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCLIFFT_BUILD_BENCHMARKS=ON
cmake --build build-cuda --target clifft_cuda_bench -j
./build-cuda/benchmarks/clifft_cuda_bench \
    --width-sweep 4..10 --shots 10000 --precision both --threads 1 \
    --benchmark_filter='/sample/' --benchmark_min_time=0.5s \
    --benchmark_repetitions=3 --benchmark_out=cuda-sampling.json \
    --benchmark_out_format=json
```

Use `--benchmark_list_tests=true` to inspect names, or filter by workload,
`/cpu/`, `/cuda/`, `/block_shared/`, `/fp32/`, or `/construct_destroy/`.
Registration and CPU cases work without a GPU; unavailable CUDA cases and
unsupported forced tiers are reported with a skip error. Inspect those errors
before comparing timings.

Workload controls:

| Option | Meaning |
| --- | --- |
| `fixture.stim ...` | Compile fixtures through the production frontend and optimizer |
| `--width-sweep LO..HI` | Add synthetic circuits with each active width in the range |
| `--shots N` | Shots per sampling iteration; default 100000 |
| `--seed S` | Repeatable seed for each call; default 42 |
| `--threads a,b,...` | CPU thread budgets; default `1,0`, where 0 is automatic |
| `--precision fp64\|fp32\|both` | CUDA coefficient precision; default FP64 |
| `--block-size N` | CUDA launch block size; default 256 |
| `--concurrency-sweep a,b,...` | Add automatic-tier cases with explicit concurrency caps |
| `--postselect` | Postselect every detector in fixture workloads and retain survivor rows |

The driver registers automatic selection and all three forced CUDA tiers.
The result label reports the selected tier; counters report active width,
retained device bytes, and the resolved concurrency cap. A cap of zero in the
case name requests automatic selection. Large thread-per-shot or global-tier
cases can exhaust device memory; select a practical shot count and width range.

## Timing boundaries and checks

All cases use wall-clock time because the CUDA sampling API is synchronous.
The `sample` and `sample_survivors` cases time repeated calls using one prepared
executable and, for CUDA, one retained sampler. They include host result
allocation, transfers, synchronization, and result destruction. Survivor
throughput counts attempted shots. CPU cases use the production sampling API,
including its per-call executor and worker setup.

`construct_destroy` times CUDA sampler construction and destruction, including
program upload and workspace allocation. It excludes compilation, CUDA context
initialization, and one-time kernel configuration, which run before timing.
It performs no sampling. The construction case uses a batch capacity of
`min(shots, 65536)`, as does retained sampling.

Before each sampling measurement, a small untimed warmup checks output shapes
and survivor-count consistency. Statistical conformance belongs in
`tests/test_cuda_sampler.cc`; benchmark timings are not correctness evidence.
Use Google Benchmark's repetition, filtering, and JSON output controls for
performance comparisons.
