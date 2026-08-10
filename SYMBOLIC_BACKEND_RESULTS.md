# Clifft symbolic vs legacy vs SymFT: four-way measurement

Measured comparison of four sampling implementations on one host, one pinned
core. Clifft legacy and Clifft symbolic come from the **same build** of PR #288's
head, so no build difference separates them. SymFT is the commit pinned by the
`arxiv-stim-clifft-monitor#9` study.

Every number here was measured in this campaign. No ratio is composed from an
earlier study, and no circuit was substituted for another.

## Headline: attempted-shot throughput, normalized to Clifft legacy

`1.00x` = Clifft legacy. Higher is faster. Medians of the per-sample rates.

| Circuit | max k | Clifft legacy | Clifft symbolic | SymFT single-shot | SymFT batched |
|---|---:|---:|---:|---:|---:|
| `surface_d7_r7` | 0 | 355.2k/s — **1.00x** | 464.0k/s — **1.31x** | 211.2k/s — **0.59x** | 3.270M/s — **9.20x** |
| `cultivation_d3` | 4 | 808.6k/s — **1.00x** | 1.095M/s — **1.35x** | 755.4k/s — **0.93x** | 2.604M/s — **3.22x** |
| `cultivation_d5` | 10 | 70.8k/s — **1.00x** | 134.6k/s — **1.90x** | 118.3k/s — **1.67x** | 160.9k/s — **2.27x** |
| `distillation` | 5 | 105.5k/s — **1.00x** | 536.6k/s — **5.09x** | 288.5k/s — **2.74x** | 1.772M/s — **16.80x** |
| `coherent_d3_r3` | 7 | 366.1k/s — **1.00x** | 390.8k/s — **1.07x** | 380.6k/s — **1.04x** | 525.0k/s — **1.43x** |
| `coherent_d5_r1` | 12 | 15.4k/s — **1.00x** | 9.0k/s — **0.59x** | 34.0k/s — **2.21x** | 30.5k/s — **1.98x** |
| `coherent_d5_r5` | 22 | 1.608/s — **1.00x** | 3.562/s — **2.21x** | 55.254/s — **34.36x** | 55.232/s — **34.34x** |
| `regime_k12_l512` | 12 | 21.6k/s — **1.00x** | 35.8k/s — **1.66x** | 12.1k/s — **0.56x** | 13.4k/s — **0.62x** |

Max k agreed exactly between Clifft and SymFT on all eight circuits, so no row
is confounded by one tool compiling a different active width.

## Median raw time and shot counts

Shot counts are sized per cell so one sample lands near 2 s; throughput is the
comparable quantity. Sampling time only — compilation is excluded and reported
separately below.

| Circuit | Arm | Shots | Median sample time | Attempted shots/s | vs legacy |
|---|---|---:|---:|---:|---:|
| `surface_d7_r7` | Clifft legacy | 683,641 | 1.925 s | 355,237/s | 1.00x |
| `surface_d7_r7` | Clifft symbolic | 932,464 | 2.010 s | 464,011/s | 1.31x |
| `surface_d7_r7` | SymFT single-shot | 400,975 | 1.898 s | 211,242/s | 0.59x |
| `surface_d7_r7` | SymFT batched | 6,233,766 | 1.907 s | 3,269,690/s | 9.20x |
| `cultivation_d3` | Clifft legacy | 1,648,553 | 2.039 s | 808,562/s | 1.00x |
| `cultivation_d3` | Clifft symbolic | 2,254,882 | 2.059 s | 1,094,954/s | 1.35x |
| `cultivation_d3` | SymFT single-shot | 1,465,008 | 1.939 s | 755,425/s | 0.93x |
| `cultivation_d3` | SymFT batched | 5,198,143 | 1.996 s | 2,604,159/s | 3.22x |
| `cultivation_d5` | Clifft legacy | 137,661 | 1.945 s | 70,769/s | 1.00x |
| `cultivation_d5` | Clifft symbolic | 262,486 | 1.950 s | 134,631/s | 1.90x |
| `cultivation_d5` | SymFT single-shot | 245,021 | 2.071 s | 118,300/s | 1.67x |
| `cultivation_d5` | SymFT batched | 370,695 | 2.304 s | 160,900/s | 2.27x |
| `distillation` | Clifft legacy | 209,213 | 1.984 s | 105,461/s | 1.00x |
| `distillation` | Clifft symbolic | 1,056,625 | 1.969 s | 536,565/s | 5.09x |
| `distillation` | SymFT single-shot | 588,683 | 2.040 s | 288,544/s | 2.74x |
| `distillation` | SymFT batched | 3,656,072 | 2.064 s | 1,771,758/s | 16.80x |
| `coherent_d3_r3` | Clifft legacy | 724,541 | 1.979 s | 366,146/s | 1.00x |
| `coherent_d3_r3` | Clifft symbolic | 787,770 | 2.016 s | 390,764/s | 1.07x |
| `coherent_d3_r3` | SymFT single-shot | 794,837 | 2.089 s | 380,590/s | 1.04x |
| `coherent_d3_r3` | SymFT batched | 1,010,680 | 1.925 s | 524,953/s | 1.43x |
| `coherent_d5_r1` | Clifft legacy | 31,296 | 2.036 s | 15,375/s | 1.00x |
| `coherent_d5_r1` | Clifft symbolic | 18,334 | 2.029 s | 9,037/s | 0.59x |
| `coherent_d5_r1` | SymFT single-shot | 68,268 | 2.007 s | 34,013/s | 2.21x |
| `coherent_d5_r1` | SymFT batched | 63,005 | 2.068 s | 30,472/s | 1.98x |
| `coherent_d5_r5` | Clifft legacy | 3 | 1.866 s | 2/s | 1.00x |
| `coherent_d5_r5` | Clifft symbolic | 7 | 1.966 s | 4/s | 2.21x |
| `coherent_d5_r5` | SymFT single-shot | 92 | 1.665 s | 55/s | 34.36x |
| `coherent_d5_r5` | SymFT batched | 103 | 1.865 s | 55/s | 34.34x |
| `regime_k12_l512` | Clifft legacy | 41,328 | 1.912 s | 21,613/s | 1.00x |
| `regime_k12_l512` | Clifft symbolic | 73,712 | 2.059 s | 35,805/s | 1.66x |
| `regime_k12_l512` | SymFT single-shot | 24,239 | 2.003 s | 12,102/s | 0.56x |
| `regime_k12_l512` | SymFT batched | 27,148 | 2.025 s | 13,406/s | 0.62x |

## All individual samples

Four process-level samples per cell, arm order rotated each round so no
implementation always runs first. Each entry is `sample time s / attempted shots per s`.

**`surface_d7_r7`**

| Arm | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|---|---|---|---|---|
| Clifft legacy | 1.950 s / 350,649 | 1.892 s / 361,386 | 1.950 s / 350,634 | 1.900 s / 359,826 |
| Clifft symbolic | 2.040 s / 457,042 | 1.980 s / 470,980 | 2.040 s / 457,024 | 1.893 s / 492,589 |
| SymFT single-shot | 1.892 s / 211,888 | 1.904 s / 210,596 | 1.969 s / 203,691 | 1.840 s / 217,966 |
| SymFT batched | 1.949 s / 3,197,878 | 1.940 s / 3,213,289 | 1.864 s / 3,344,013 | 1.874 s / 3,326,092 |

**`cultivation_d3`**

| Arm | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|---|---|---|---|---|
| Clifft legacy | 1.989 s / 828,931 | 2.036 s / 809,624 | 2.057 s / 801,364 | 2.042 s / 807,501 |
| Clifft symbolic | 2.051 s / 1,099,290 | 2.095 s / 1,076,075 | 2.059 s / 1,094,961 | 2.059 s / 1,094,947 |
| SymFT single-shot | 1.943 s / 754,030 | 1.936 s / 756,821 | 1.921 s / 762,452 | 1.956 s / 748,998 |
| SymFT batched | 2.000 s / 2,598,656 | 1.992 s / 2,609,662 | 2.038 s / 2,551,125 | 1.991 s / 2,610,566 |

**`cultivation_d5`**

| Arm | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|---|---|---|---|---|
| Clifft legacy | 2.006 s / 68,617 | 1.908 s / 72,138 | 1.944 s / 70,797 | 1.946 s / 70,741 |
| Clifft symbolic | 1.989 s / 131,968 | 1.907 s / 137,633 | 1.912 s / 137,293 | 2.002 s / 131,113 |
| SymFT single-shot | 2.076 s / 118,043 | 2.067 s / 118,557 | 2.043 s / 119,952 | 2.179 s / 112,433 |
| SymFT batched | 2.216 s / 167,305 | 2.309 s / 160,545 | 2.299 s / 161,256 | 2.340 s / 158,405 |

**`distillation`**

| Arm | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|---|---|---|---|---|
| Clifft legacy | 1.975 s / 105,936 | 1.984 s / 105,438 | 2.005 s / 104,346 | 1.983 s / 105,485 |
| Clifft symbolic | 1.931 s / 547,113 | 1.966 s / 537,427 | 1.983 s / 532,732 | 1.972 s / 535,703 |
| SymFT single-shot | 2.042 s / 288,268 | 2.121 s / 277,552 | 2.024 s / 290,870 | 2.038 s / 288,820 |
| SymFT batched | 2.029 s / 1,801,970 | 2.099 s / 1,741,547 | 2.026 s / 1,804,424 | 2.128 s / 1,718,012 |

**`coherent_d3_r3`**

| Arm | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|---|---|---|---|---|
| Clifft legacy | 1.986 s / 364,789 | 1.972 s / 367,502 | 1.953 s / 370,977 | 2.007 s / 360,971 |
| Clifft symbolic | 2.016 s / 390,689 | 2.016 s / 390,838 | 2.007 s / 392,578 | 2.051 s / 384,114 |
| SymFT single-shot | 2.211 s / 359,439 | 2.057 s / 386,430 | 2.050 s / 387,660 | 2.121 s / 374,750 |
| SymFT batched | 1.934 s / 522,711 | 1.901 s / 531,717 | 2.652 s / 381,124 | 1.917 s / 527,195 |

**`coherent_d5_r1`**

| Arm | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|---|---|---|---|---|
| Clifft legacy | 2.022 s / 15,481 | 2.050 s / 15,270 | 2.059 s / 15,203 | 2.017 s / 15,516 |
| Clifft symbolic | 2.045 s / 8,966 | 2.001 s / 9,164 | 2.030 s / 9,032 | 2.028 s / 9,042 |
| SymFT single-shot | 2.006 s / 34,039 | 2.004 s / 34,072 | 2.009 s / 33,987 | 2.009 s / 33,977 |
| SymFT batched | 2.073 s / 30,391 | 2.062 s / 30,552 | 2.092 s / 30,118 | 2.054 s / 30,673 |

**`coherent_d5_r5`**

| Arm | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|---|---|---|---|---|
| Clifft legacy | 1.943 s / 2 | 1.831 s / 2 | 1.215 s / 2 | 1.901 s / 2 |
| Clifft symbolic | 2.155 s / 3 | 1.932 s / 4 | 1.531 s / 5 | 2.000 s / 4 |
| SymFT single-shot | 1.862 s / 49 | 1.532 s / 60 | 1.646 s / 56 | 1.685 s / 55 |
| SymFT batched | 1.894 s / 54 | 1.855 s / 56 | 1.779 s / 58 | 1.875 s / 55 |

**`regime_k12_l512`**

| Arm | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|---|---|---|---|---|
| Clifft legacy | 1.924 s / 21,475 | 1.924 s / 21,478 | 1.872 s / 22,073 | 1.900 s / 21,748 |
| Clifft symbolic | 2.079 s / 35,451 | 2.039 s / 36,159 | 2.023 s / 36,436 | 2.102 s / 35,062 |
| SymFT single-shot | 2.018 s / 12,014 | 1.975 s / 12,274 | 1.989 s / 12,184 | 2.016 s / 12,021 |
| SymFT batched | 2.007 s / 13,527 | 2.009 s / 13,510 | 2.041 s / 13,302 | 2.077 s / 13,068 |

## Compilation (excluded from the comparison above)

| Circuit | Clifft legacy | Clifft symbolic | symbolic / legacy | SymFT single | SymFT batched |
|---|---:|---:|---:|---:|---:|
| `surface_d7_r7` | 8.1 ms | 459.5 ms | 56.6x | 172.2 ms | 176.7 ms |
| `cultivation_d3` | 0.8 ms | 18.9 ms | 23.9x | 17.1 ms | 17.3 ms |
| `cultivation_d5` | 4.8 ms | 595.8 ms | 124.7x | 563.7 ms | 562.3 ms |
| `distillation` | 3.3 ms | 28.2 ms | 8.6x | 18.4 ms | 18.6 ms |
| `coherent_d3_r3` | 0.6 ms | 2.2 ms | 3.5x | 4.3 ms | 4.6 ms |
| `coherent_d5_r1` | 1.5 ms | 7.9 ms | 5.4x | 9.5 ms | 9.9 ms |
| `coherent_d5_r5` | 3641.5 ms | 3709.1 ms | 1.0x | 99.7 ms | 100.8 ms |
| `regime_k12_l512` | 4.1 ms | 70.8 ms | 17.1x | 1466.6 ms | 1449.6 ms |

Clifft compile time is the `compile()` call (HIR passes, lowering, and for legacy
the default bytecode passes). SymFT compile time is `Circuit()` construction plus
`compile_counts_sampler()`.

Two inversions in this table are worth noting, since the headline comparison
excludes compilation entirely:

- On `coherent_d5_r5` Clifft spends ~3.6 s compiling against SymFT's ~0.10 s, and
  both Clifft arms sample only a few shots per second — so for a small job on that
  circuit compilation dominates total time for Clifft and does not for SymFT.
- On `regime_k12_l512` the direction reverses: SymFT spends ~1.45 s compiling
  against Clifft legacy's 4 ms.
- `cultivation_d5` is the largest symbolic-over-legacy compile ratio measured here
  (~125x), though Clifft symbolic and SymFT compile that circuit in about the same
  wall time (596 ms vs 564 ms).

## SymFT configuration and product components

Both SymFT arms use the identical active-component policy: the library's own
cost gate, with no override (the Python API exposes none). `active_components`
below is what the gate actually selected.

| Circuit | single: batch_size | single: components | batched: batch_size | batched: components |
|---|---:|---:|---:|---:|
| `surface_d7_r7` | 0 | False | 2048 | False |
| `cultivation_d3` | 0 | False | 2048 | False |
| `cultivation_d5` | 0 | False | 32 | False |
| `distillation` | 0 | False | 1024 | False |
| `coherent_d3_r3` | 0 | False | 256 | False |
| `coherent_d5_r1` | 0 | True | 8 | True |
| `coherent_d5_r5` | 0 | True | 1 | True |
| `regime_k12_l512` | 0 | False | 8 | False |

`batch_size` 0 in the single-shot arm means the parameter is inert on that
backend. The batched arm's size is SymFT's automatic choice,
`min(2048, max(1, 32768 / 2^k))`, which is why it shrinks as k grows and reaches
1 at k=22 — i.e. the batched arm does no cross-shot packing at all on
`coherent_d5_r5`. Product components engaged on exactly the two circuits where
the gate selected them, identically in both arms.

## Output sanity checks

Every run consumed its outputs. Discard fraction is the primary cross-arm check.

| Circuit | Clifft legacy | Clifft symbolic | SymFT single | SymFT batched | matched? |
|---|---:|---:|---:|---:|---|
| `surface_d7_r7` | 0.9126 | 0.9121 | 0.9125 | 0.9121 | yes |
| `cultivation_d3` | 0.3135 | 0.3133 | 0.3131 | 0.3131 | yes |
| `cultivation_d5` | 0.8556 | 0.8561 | 0.8556 | 0.8555 | yes |
| `distillation` | 0.9206 | 0.9203 | 0.9202 | 0.9202 | yes |
| `coherent_d3_r3` | 0.9101 | 0.9104 | 0.6263 | 0.6271 | **NO — see caveats** |
| `coherent_d5_r1` | 0.0702 | 0.0717 | 0.0723 | 0.0724 | yes |
| `coherent_d5_r5` | 1.0000 | 1.0000 | 1.0000 | 0.9951 | yes |
| `regime_k12_l512` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no postselection |

**`coherent_d5_r5` discards essentially every shot in all four arms.** That row
therefore measures how fast each implementation *rejects* a shot at its first
failing detector, not how fast it simulates a full k=22 trajectory. It is a valid
comparison of the postselected workload as specified, but it should not be read as
a dense-kernel result.

Logical-error counts are reported in the raw JSON but are only comparable on the
seven single-observable circuits: on `distillation` Clifft counts a shot as a
logical error when any of its 5 observables mismatches, while SymFT scores
observable 0 only.

## Provenance

| Item | Value |
|---|---|
| Clifft commit | `c705906b476520146821264cb77b65836db13cc8` (PR #288 head, branch `codex/issue-280-direct-rotation-avx512`) |
| Clifft version string | `0.7.1.dev35+gc705906b4` |
| Clifft build | `SKBUILD_CMAKE_BUILD_TYPE=Release`, `CLIFFT_CPU_BASELINE=native` |
| Clifft flags | `-O3 -DNDEBUG -march=native -mtune=native -ffast-math` |
| Clifft runtime ISA | `avx512` |
| SymFT commit | `e8fd41806b8caa4cb96d84dabbba278ff70c960c` (haoliri0/SOFT, the commit pinned by monitor#9) |
| SymFT build | `pip install -e ./python`, `SYMFT_PY_NATIVE` default on |
| SymFT flags | `-std=c++20 -O3 -fvisibility=hidden -march=native`; AVX-512 kernels in a separate TU (`-mavx512f -mavx512dq -mfma`), runtime dispatch |
| Compiler | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| Python | Python 3.12.3 (both venvs) |
| CPU | AMD EPYC 9554P 64-Core Processor |
| OS / kernel | Ubuntu 24.04.3 LTS / 6.12.90 |
| CPU affinity | `taskset -c 3`, every sample a fresh process |
| Threads | `OMP_NUM_THREADS=1`, `clifft.set_num_threads(1)`, SymFT `threads=1` |
| Samples per cell | 4, rotated arm order (round r starts with arm r mod 4) |
| Timing | `time.perf_counter()` around the sampling call only, after a discarded warm-up call |

Circuit provenance — `clifft-paper` commit `db7dc9f13a2c2854690e92390c779048a1ac1400`:

| Circuit | SHA-256 |
|---|---|
| `surface_d7_r7` | `30d1940101d70e05a63f0d2f877756ffaeaba7e8e17e94cd2ea40fce04b99583` |
| `cultivation_d3` | `90a7d841e003e5ee38137cd9a3eb6529bb552e49c424bc6b0932a27d97cdb41f` |
| `cultivation_d5` | `c2b4566917bd9bf27a5705284dac02700ef0dcc7c03c91066670db376d633a6d` |
| `distillation` | `188bd53c48dbc21f840fb297df6f41c61f5bad6a856bba621f00ff42078921c1` |
| `coherent_d3_r3` | `87d1308c83894e87c60aeb2dc31b74be89b3460a951929951f2c3ac92606827d` |
| `coherent_d5_r1` | `2707188abe8912f693fe4f910db8c4b6bd795c71a8fba38cc153da90ee5910b8` |
| `coherent_d5_r5` | `54088bbd5f06b441596e414f9fa99d8eeaff8a4a1a862b911e0ad40092c5e549` |
| `regime_k12_l512` | `f7ca68d1d1bffda5c6a0837ba25d6bd5dde01132040e4eff3ffd0c6a9571395d` |

`regime_k12_l512` is generated, not from the paper corpus: it is the k=12, L=512
point of the output-relevant regime-map generator on
`arxiv-stim-clifft-monitor@paper/2607.28600-symft-cpu-bench`, reproduced verbatim.

## Contract

- Postselected circuits (all except `regime_k12_l512`): every detector postselected
  on both tools. Clifft compiles with `postselection_mask=[1]*num_detectors` and
  `normalize_syndromes=True`, then calls `sample_survivors(keep_records=False)`.
  SymFT uses `compile_counts_sampler(postselect_detectors=True, observable=0)`.
- `regime_k12_l512` has no detectors; no postselection on either side.
- Throughput is **attempted** shots per second in all cells.
- Clifft legacy uses `clifft.compile` / `clifft.sample_survivors`; Clifft symbolic uses
  `clifft.experimental.compile` / `clifft.experimental.sample_survivors`. Same process,
  same extension module, same build.

## Caveats

1. **Detector-reference semantics are not reconciled**, per the measurement request.
   Clifft normalizes detectors against the compiled reference; SymFT treats raw
   parity as the detection event. On circuits with a nonzero reference this changes
   how many shots abort early, and therefore attempted-shot throughput. The discard
   table above marks which rows diverge; on those rows the Clifft-vs-SymFT ratio is
   not a like-for-like comparison. The Clifft-legacy-vs-Clifft-symbolic comparison is
   unaffected — both use identical semantics.
2. **`-ffast-math` asymmetry.** Clifft's Release build enables it; SymFT's does not.
   Neither was modified for this campaign.
3. **Logical-error counts are not comparable on `distillation`**, where Clifft counts
   a shot as a logical error if any of its 5 observables mismatches, while SymFT
   scores observable 0 only. Throughput is unaffected.
4. **Shared-tenancy VM.** The host is an 8-vCPU slice of an EPYC 9554P, not a
   dedicated machine. Per-cell repeat spreads are reported above so the reader can
   judge stability directly.
5. Shot counts differ per cell by design (each sized for ~2 s). Throughput is
   shot-count independent; raw times are reported alongside for auditability.
6. **`coherent_d5_r5` is the statistically weakest cell.** Clifft is slow enough
   there that a ~2 s sample is only 3 shots (legacy) or 7 shots (symbolic), so the
   per-sample rate is coarsely quantized and its dispersion is correspondingly
   wide — the four legacy samples span 1.5–2.5 shots/s around a 1.6 median. Treat
   that row's ratios as order-of-magnitude, not two-significant-figure.
