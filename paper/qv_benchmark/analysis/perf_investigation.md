# QV Benchmark: qsim vs UCC Performance Investigation

## Hardware
- CPU: Intel Xeon Platinum 8259CL @ 2.50 GHz (Cascade Lake)
- 2 vCPUs, single-threaded benchmarks (OMP_NUM_THREADS=1)
- L1d: 32 KiB, L2: 1 MiB, L3: 35.75 MiB
- ISA: AVX-512F, AVX-512DQ, AVX-512BW, AVX-512VL, AVX2, BMI2, FMA
- RAM: 7.4 GB DDR4

## Summary

At N=22, qsim is ~15x faster than UCC (0.84s vs 13.0s). The root cause
is a combination of three factors:

1. **Single-precision float (dominant):** qsim uses `float` (4B per component),
   UCC uses `double` (8B). This halves memory traffic per sweep and doubles
   SIMD throughput (8 vs 4 amplitudes per AVX-512 register).

2. **L3 cache residency at N=22:** qsim's 32MB statevector fits in the 35.75MB
   L3 cache. UCC's 64MB statevector spills to DRAM. This causes a 68.7% L1
   cache miss rate for UCC vs 9.3% for qsim (7.3x more misses).

3. **Gate fusion reduces sweep count:** qsim fuses adjacent 1-qubit gates into
   2-qubit matrices, reducing the number of full-array sweeps. When data is
   L3-resident, this costs almost nothing (extra FLOPs are free when bandwidth
   is plentiful), explaining why disabling fusion had <3% impact.

The 15x gap is stable across N=22,23,24 because even when both simulators
are DRAM-bound, qsim processes 2x more amplitudes per instruction and
transfers half the bytes per sweep.

## Test 4: Cache Cliff Mapping (N=22, 23, 24)

### Statevector Sizes vs L3 Cache (35.75 MiB)

| N  | qsim (float) | UCC (double) | qsim fits L3? | UCC fits L3? |
|----|--------------|--------------|----------------|--------------|
| 22 | 32 MB        | 64 MB        | Yes            | No           |
| 23 | 64 MB        | 128 MB       | No             | No           |
| 24 | 128 MB       | 256 MB       | No             | No           |

### Timing Results (3 seeds, seconds)

| Sim  | N  | seed=42 | seed=43 | seed=44 | Average |
|------|----|---------|---------|---------|---------|
| qsim | 22 | 0.838   | 0.850   | 0.825   | 0.838   |
| qsim | 23 | 2.019   | 2.046   | 1.950   | 2.005   |
| qsim | 24 | 4.494   | -       | -       | 4.494   |
| ucc  | 22 | 12.828  | 12.752  | 12.382  | 12.654  |
| ucc  | 23 | 29.780  | 28.186  | 29.583  | 29.183  |
| ucc  | 24 | 67.138  | -       | -       | 67.138  |

### UCC/qsim Ratio

| N  | Ratio |
|----|-------|
| 22 | 15.1x |
| 23 | 14.6x |
| 24 | 14.9x |

**Key finding:** The ratio is stable at ~15x, NOT shrinking when both are
DRAM-bound. This rules out the L3 cache cliff as the primary explanation
for the gap. The dominant factor is the 2x SIMD width + 2x memory
footprint advantage that persists regardless of cache level.

### Scaling Analysis

| Simulator | N transition | Slowdown factor |
|-----------|-------------|----------------|
| qsim      | 22 -> 23    | 2.39x          |
| qsim      | 23 -> 24    | 2.24x          |
| ucc       | 22 -> 23    | 2.31x          |
| ucc       | 23 -> 24    | 2.30x          |

qsim's 2.39x slowdown from N=22 to N=23 (vs the expected 2x from data
doubling) confirms the L3 cache cliff: at N=23, qsim's 64MB statevector
spills out of L3 for the first time.

## Test 5: perf stat Analysis

### Process-wide Hardware Counters (N=22, seed=42)

| Counter               | qsim          | UCC           | Ratio  |
|-----------------------|---------------|---------------|--------|
| Wall time (sim only)  | 0.84s         | 13.0s         | 15.5x  |
| Total cycles          | 9.86B         | 34.76B        | 3.53x  |
| Total instructions    | 10.99B        | 28.43B        | 2.59x  |
| IPC                   | 1.11          | 0.82          | 0.74x  |
| L1-dcache loads       | 1.61B         | 1.58B         | 0.98x  |
| L1-dcache misses      | 149M (9.3%)   | 1087M (68.7%) | 7.3x   |
| Branch misses         | 29.0M (3.1%)  | 6.6M (0.65%)  | 0.23x  |

### Simulation-Only Estimates (subtracting Python overhead)

- Python+imports+circuit overhead: ~3.0s, ~8.25B instructions
- **qsim simulation:** ~2.74B instructions in 0.84s = 3.26 GIPS
- **UCC simulation:** ~20.2B instructions in 13.0s = 1.55 GIPS
- qsim does ~7.4x fewer instructions than UCC for the same circuit

### Critical Finding: L1 Cache Miss Rate

UCC's 68.7% L1 miss rate is catastrophic. The CPU stalls on nearly every
memory access, waiting for data from L2/L3/DRAM. This is a direct
consequence of the 64MB statevector exceeding L3 cache.

qsim's 9.3% L1 miss rate indicates excellent cache behavior -- the 32MB
statevector stays resident in L3 and sequential sweeps have good L1
reuse within each cache line.

### perf annotate: Hot Instructions (UCC)

| % Cycles | Address | Instruction                      | Description            |
|----------|---------|----------------------------------|------------------------|
| 20.98%   | 8c3bc   | vmovapd (%rax),%zmm0             | AVX-512 load (U2 path) |
| 18.19%   | 8c3c2   | vpermilpd $0x55,%zmm1,%zmm3      | Complex multiply setup |
| 8.64%    | 8bdc6   | add $0x4,%r11                    | Loop counter (stalled) |
| 6.35%    | 8bb3a   | add $0x40,%rax                   | Pointer advance (stalled) |
| 5.40%    | 8bb45   | vmovapd %zmm7,-0x40(%rax)        | AVX-512 store          |

39% of all cycles are spent on just two instructions: a load and the
subsequent permute. The load stalls because the data is not in L1 or L2.

### perf annotate: Hot Instructions (qsim)

qsim's hot loop at 0x4b910-0x4bc5a processes fused 2-qubit gates:
- 20 vmovaps loads/stores (single-precision)
- 64 vfmadd/vfnmadd/vmulps (FMA operations)
- 5 pdep (index scatter)
- 32 vbroadcastss (matrix element broadcasts)

The loop body is extremely compute-dense: 64 FMA ops per iteration vs
20 memory ops. This is ideal for hiding memory latency.

## Work Comparison

| Metric                  | qsim     | UCC      |
|-------------------------|----------|----------|
| Circuit ops             | 2,684    | -        |
| Bytecode instructions   | -        | 3,703    |
| Array-touching ops      | 2,684    | 3,637    |
| Frame-only ops          | -        | 44       |
| Peak active qubits      | 22       | 22       |
| Bytes per amplitude     | 8        | 16       |
| Amplitudes per AVX-512  | 8        | 4        |
| Statevector size        | 32 MB    | 64 MB    |

UCC has 35% more array-touching operations than qsim circuit ops.
This is because UCC's Heisenberg compiler converts some Clifford
gates into frame operations but also introduces EXPAND and measurement
ops that qsim handles differently.

## Compiler Flags

- **qsim (pip):** GCC 10.2.1, `-O3 -mavx512f -mbmi2` (from CMakeLists.txt)
- **UCC (scikit-build):** `-O3 -march=native -mtune=native -ffast-math -mavx512f -mavx512dq`

UCC has a slight flag advantage with `-march=native` and `-ffast-math`.
Compiler flags are not a factor in the performance gap.

## Test 6: In-Register AVX-512 Prototype for Axis 0

See `/tmp/test6_final.cpp` for a working prototype demonstrating:

1. **`swap_adj_cx()`**: Swaps adjacent complex<double> pairs within a
   __m512d register using `_mm512_permutexvar_pd` with indices
   {2,3,0,1,6,7,4,5}. This is the key primitive for axis==0 operations.

2. **`h_axis0()`**: Hadamard butterfly using swap + blend:
   - swap adjacent pairs
   - compute sum and (swapped - original)
   - blend even (sum) and odd (diff) positions
   - scale by 1/sqrt(2)

3. **`u2_axis0()`**: Generic 2x2 unitary via coefficient-per-slot vectors:
   - Build alternating coefficient vectors for self/swap terms
   - Two complex multiplies + add

4. **`cnot_c1_t0()`**: CNOT with target axis 0 via masked blend.

5. **`phase_axis0()`**: Phase rotation via u2 specialization.

All 5 tests pass. The key intrinsic is `_mm512_permutexvar_pd` which
performs an arbitrary 64-bit lane permutation in a single cycle.

## Recommendations

### Phase 1: Float Precision (Highest ROI)
Template the SVM on precision type. At N=22, this halves the statevector
to 32MB (L3-resident) and doubles SIMD throughput. Expected speedup: 4-8x.

### Phase 2: Axis 0/1 SIMD Paths
Plug the scalar holes using `_mm512_permutexvar_pd` as demonstrated in
Test 6. Expected speedup: ~20% (reclaims the scalar path overhead).

### Phase 3: Gate Fusion (Optional)
Fuse sequential single-axis ops into compound U2 matrices before sweeping.
Less impactful when data is cache-resident but helps at larger N.

### Phase 4: SoA Layout (Deferred)
Separate real and imaginary arrays. Significant refactor for moderate gain
once float + axis-0 fixes are in place.
