# Sampling benchmarks

`clifft_benchmarks` contains the CPU performance cases used by benchmark CI.
With `CLIFFT_ENABLE_CUDA=ON`, `clifft_cuda_bench` adds CPU-versus-CUDA sampling
and CUDA sampler construction cases. Both use Google Benchmark.

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
