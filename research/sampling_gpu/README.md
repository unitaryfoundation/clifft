# GPU benchmarking for the symbolic sampler (current main)

Re-runs the 2026-07/08 GPU study against the SamplingPlan runtime that
replaced the legacy SVM. Two artifacts:

1. **Action census** (`census.py` -> `census.md`): what the new planner
   actually emits for the old study's circuit corpus - action mix, width
   profiles, on-chip residency bands. This calibrates everything below.
2. **CUDA backend + benchmark** (`../../src/clifft/sampling/cuda/`,
   `bench_main.cc`): a CUDA device interpreter for lowered sampling plans
   with three execution tiers, raced against the production CPU sampler.

## What the census says (2026-08-24, clifft 0.8.1.dev23)

- Action streams are 4-13x shorter than the legacy bytecode: Cliffords are
  fully absorbed at compile time; what remains is Pauli rotations,
  promotions, and active measurements.
- Measurement + promotion dominate: 75-95% of visit-weighted dense work on
  most corpus circuits (legacy census: 30-73%). Rotation-dominated
  workloads still exist (QAOA 92%, high-T random 82%).
- Width profiles match the legacy peaks (the planner did not change the
  physics): conveyor/cultivation-class circuits live at w <= 11, i.e.
  fully inside every on-chip band; hidden-shift/QAOA-class sit at w 14-16.

Consequence for the backend: the measurement round (block reduction +
collapse) is the primitive to optimize, not gate throughput; and the
shared-memory tier's coverage of real workloads is decided at w <= 13
(Hopper 227 KB) vs w <= 11 (CDNA3 64 KB LDS).

## The CUDA backend

Mirrors the HIP vertical slice (`codex/hip-sampling-vertical-slice`): same
flat descriptors, same lowering, same action semantics, same per-shot RNG
derivation (`shot_seed.h` domain 0x01). New here: cooperative
block-per-shot execution - the architecture the 2026-08 H200 study
measured as the winner for clifft's widths - in two variants:

- `BlockShared`: the whole shot (split re/im + half-width scratch +
  reduction scratch) resident in opt-in dynamic shared memory. FP64 fits
  w <= 13 on sm_90 (196 KB + reduction at block 256).
- `BlockGlobal`: same kernel, state in a per-block global slab; grid is
  bounded by free device memory and shots loop over blocks.
- `ThreadPerShot`: the HIP slice's tier, kept for tiny widths and
  cross-checks.

Scalar control flow runs redundantly on every thread from identical
inputs (including the RNG), so branch decisions need no broadcasts; byte
outputs are written by lane 0 only. Collapse always stages through
scratch: the sequential in-place compaction is not parallel-safe.

## Running on a GPU box

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCLIFFT_BUILD_GPU_BENCH=ON -DCLIFFT_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j --target bench_sampling
uv run python research/sampling_gpu/dump_corpus.py
./build/research/sampling_gpu/bench_sampling --shots 200000 --threads 1,0 \
    research/sampling_gpu/corpus/*.stim > results.csv
```

Every tier runs for every circuit (unavailable tiers report and skip);
the auto-selected tier is starred in the CSV. `--threads 1,0` gives
single-thread and all-cores CPU baselines from the production sampler -
the honest opponent, since cross-shot and intra-shot CPU parallelism now
ship on main.

Validation happens before timing (`--validate N`, default 64): GPU record
rows are replayed through the CPU executable's
`record_log_probabilities` (every row must be reachable), and CPU-vs-GPU
per-record marginals are compared under a binomial tolerance. Exact
row-for-row equality is NOT expected: reduction order and noise-draw
scheduling differ between backends.

Without a CUDA toolkit (this laptop):

```sh
cmake -B build -DCLIFFT_BUILD_GPU_BENCH=ON && cmake --build build --target bench_sampling
./build/research/sampling_gpu/bench_sampling --shots 2000 corpus/foo.stim  # CPU rows
./research/sampling_gpu/check_cuda_syntax.sh  # clang host+device passes
```

## What to measure first (H200 or similar)

1. Tier race per corpus circuit at the best block size: does BlockShared
   reproduce the ~5x over per-action execution at w <= 13 that the
   microbench measured, now on real compiled programs?
2. GPU-vs-CPU at matched shots: the microbench's valley (CPU competitive
   at w 14-17, GPU winning ~15x below and ~6x above) should reappear with
   the census-measured action mix; the measurement-dominated mix found by
   the new census favors the in-kernel tiers further.
3. Block-size sweep (64..512) for the cooperative tiers; the reduction is
   a power-of-two tree, so block size must be a power of two.
4. FP32 vs FP64 on BlockShared: FP32 doubles the shared-memory width
   budget (w <= 14), relevant to the CDNA3 port where LDS is 64 KB.

## Files

- `census.py`, `census.md` - action census (this directory's calibration)
- `circuits.py` - corpus generators (copied from the legacy study)
- `dump_corpus.py` - writes the corpus as .stim files for the bench
- `bench_main.cc`, `CMakeLists.txt` - benchmark driver
- `check_cuda_syntax.sh`, `cuda_stub/` - CUDA-less syntax checking
- Backend sources: `src/clifft/sampling/cuda/{device_program.h,executable.h,executable.cc,sampler.h,sampler.cu}`
