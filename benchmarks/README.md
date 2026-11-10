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

The canary builds the PR head and the merge-base of the PR with `main`, so a
branch behind `main` is measured against the code it changed, and the report
says how many `main` commits lie beyond the merge-base. A change of at least 5%
is notable and at least 10% is a possible regression or improvement, and either
must repeat in both A/B/B/A pairs with low noise. Rows under 100 us need a 25%
change to count, because code placement alone has historically moved them by
10-20%; smaller changes on those rows are reported as micro-workload changes
without a verdict.

Build and run the same suite locally:

```bash
cmake -S benchmarks -B build/canary -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/canary --target clifft_benchmarks -j
./build/canary/clifft_benchmarks \
    --benchmark_min_time=0.5s --benchmark_repetitions=3
```

Use `--benchmark_list_tests=true` to inspect the suite or
`--benchmark_filter='^phase_'` to run only the phase cases.

`sample_online_loss_surface_d7_64_shots` exercises explicit online execution on
the existing noisy d7/r7 surface-code fixture, with loss at rate 0.001 after each
CX operand and default skip-gate partner behavior. It times prepared sampling
on one worker, including worker setup and output allocation; preparation is
excluded. An untimed check verifies online execution and zero continuations.
This is a regression canary, not a comparison with another simulator.

### Shared survivor sampling

Shared execution can reduce repeated coefficient evolution, but its preparation
cost matters for short jobs. `--benchmark_filter='^(sample|prepare)_shared_'`
keeps first-use preparation, repeated sampler construction, and retained
sampling visible as separate canary rows.
These cases reuse the noisy cultivation fixtures, postselect all detectors,
and request aggregate survivor and observable counts with raw parities.

The d3 sampling case covers packed dormant measurements and rare symmetric
readout. Both d3 and d5 use upfront noise grouping and site-row noise. The d7
sampling case covers incremental sharing and site-sparse noise. Two further
cases measure d5 grouped first-use preparation and repeated sampler
construction. These are fixed configurations for regression monitoring.
Choosing defaults and attributing gains to individual options require separate
comparisons. All use FP64, one worker, capacity 8192, and an initial
reservation of 128 states. Reconstruction is requested for all cases and
selected for d5 and d7; d3 uses its original width-four plan.

Each fixture's `Program` is compiled once per process and reused across
calibration and repetitions. `sample_shared_*` times 65,536 attempts per call
after warming one retained `SurvivorSampler`; compilation and construction are
excluded, while aggregate result allocation and destruction are included.
`prepare_shared_cultivation_d5_first_use_compact_grouped` times one independent
build of the immutable preparation per iteration: the count-only plan, site-row
noise metadata, early checks, and the grouping key. It builds no workers and
excludes Program compilation and reconstruction.
`prepare_shared_cultivation_d5_compact_grouped` times repeated sampler
construction and destruction from the retained `Program`, which reuses the
program's preparation and allocates workers. Neither prepare row runs sampling
in the timed loop. Existing `compile_plan_*` rows measure their own
original-plan workloads separately.

Untimed checks require the expected reconstruction selection, nonzero active
width, entrance rejection, fewer quantum groups than entering shots, and
expression reduction. Grouped cases also check a nonempty noise-sign basis.
Reported group counts come from that fixed-seed validation batch, not from the
timed calls. The `reconstructed` counter reports which plan was selected.
Throughput counts attempted shots. There are no timing assertions.

## External input execution

`--benchmark_filter='^external_inputs_'` compares reusable scalar and packed
execution for identical supplied noise/readout columns in 256-shot batches.
The surface-code case exercises optimized presampled carriers; the repeated
T-gate case exercises active quantum state and ordinary noise propagation.
Preparation, input generation, and copying results out of the executor are
excluded. These are primitive throughput measurements, not end-to-end leak/loss
sampling or deltakit-stim comparisons.

## Noncomputational fixed execution

`noncomp_fixed.py` compares public prepared scalar and packed fixed execution with forced trajectories on
surface-code circuits with inline leakage, full partner depolarization, and
label-independent readout. Both return all records, heralds, and detailed final
status; a final restoring reset erases leaked labels. It reports preparation
separately from repeated fixed sampling, plus active width, actual batch/worker
counts, and retained packed worker bytes. Fixed sampling includes worker setup,
status/input generation, quantum execution, and all output materialization.
Trajectory times include their ordinary per-call preparation.

```bash
python benchmarks/noncomp_fixed.py --distances 3 5 --shots 1000 --batch-sizes 1 65 auto
python benchmarks/noncomp_fixed.py --distances 7 --shots 10000 --threads 4 --skip-trajectories
python benchmarks/noncomp_fixed.py --distances 3 --shots 1000 --event LOSS
```

Use an optimized installed Clifft build. Calls alternate forward/reverse order
and report medians; there are no timing assertions. `--variants` includes
`clifford`, `s`, `t`, and `rotation`. These timings do not establish deltakit-stim
parity. Retained packed worker bytes include executor state, external inputs,
and driver scratch; they exclude the immutable plan, result arrays, and allocator
bookkeeping. The scalar classical workspace metric excludes executor storage.

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
