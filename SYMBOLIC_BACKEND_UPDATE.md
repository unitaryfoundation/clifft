# Symbolic sampling backend: where we are

Informal status update on [#280](https://github.com/unitaryfoundation/clifft/issues/280)
(benchmarking the symbolic-coordinate backend against the legacy SVM), plus how
both compare to SymFT.

**Convention used below: speedup relative to clifft legacy (SVM) on the same
circuit, same build, same host. `1.00x` = parity with legacy, higher = faster.**
#280 reports the reciprocal (symbolic ÷ legacy *time*), so e.g. `1.33x` here is
`0.754x` there.

## Headline

Three weeks ago the symbolic backend was ~10x *slower* than legacy on the
output-heavy QEC workloads. After four merged optimizations (#281 expression
registers, #282 scalar rotation specialization, #283 constant-rotation fusion,
#284 AVX-512 fused U4) it is at parity or ahead on most of the corpus, and the
open #288 (direct-rotation AVX-512) closes most of what is left.

## 1. clifft symbolic vs clifft legacy

| Circuit | k | First measured | Current `main` | With #288 (in review) |
|---|---:|---:|---:|---:|
| surface d7 r7, aggregate | 0 | 0.09x | **1.33x** | — |
| cultivation d3, aggregate | 4 | 0.10x | **1.16x** | — |
| distillation, aggregate | 5 | 2.20x | **5.13x** | — |
| coherent d3 r3, aggregate | 7 | 0.63x | 0.88x | **1.09x** |
| coherent d5 r1, aggregate | 12 | — | 0.37x | **0.59x** |
| regime k=12, L=512 | 12 | 0.79x | **1.30x** | **1.69x** |
| QV-10 raw, 10 seeds | ~10 | 0.11x | **0.97x** (median) | ~1.0x (0.89–1.30, 3 seeds) |
| QV-20 raw, 10 seeds | ~20 | 0.10x | **1.34x** (median) | 1.20–1.37x |
| noncomputational d17 r5, lossless | 0 | — | **1.38x** | — |
| noncomputational d17 r5, low leak | — | 0.10x | 0.10x | — |

*"First measured" is the 2026-08-07 triage at `3fdafa4` for most rows; the QV
and noncomputational rows were first measured 2026-08-08 at `b80225c` (i.e.
already after #281/#282). "Current `main`" mixes `b80225c` (surface,
cultivation, distillation, noncomputational) and `48b84ff`/`cbb6a1e` (coherent,
regime, QV) — the earlier rows were screened for regression after #283/#284 but
not fully re-timed. One pinned core of an AMD EPYC 9554P, AVX-512, Release,
medians of 3 balanced blocks, ~5% noise floor established by A/A controls.*

Two things worth calling out:

- **Compile time is still a real gap on output-heavy circuits.** Surface d7 r7
  compiles in 447 ms symbolic vs 9.3 ms legacy (48x). QV-20 compile is at exact
  parity (~3.06 s both). So it's specifically the output/suffix-transformation
  path, not a general frontend problem.
- **The noncomputational low-leak row is a compile problem, not an execution
  problem.** That API compiles continuations internally, and ~58% of its time is
  continuation compilation. Same planner work item as the surface compile gap.

## 2. clifft vs SymFT

Numbers from the reproduction campaign in
[`arxiv-stim-clifft-monitor#9`](https://github.com/unitaryfoundation/arxiv-stim-clifft-monitor/issues/9)
(SOFT `e8fd418` built from source, AVX-512; clifft 0.7.0; same EPYC 9554P host,
one pinned core; reference-neutralized circuits so both tools implement
identical detector semantics). Same convention: **× vs clifft legacy**.

| Workload | k | clifft symbolic | SymFT single-shot | SymFT batched (default) | SymFT GPU (paper only) |
|---|---:|---:|---:|---:|---:|
| pure-Clifford / k=0 counts | 0 | 1.33x † | — | **~20–50x** | — |
| cultivation d3 | 4 | 1.16x | 1.00x | **3.41x** | ~130x |
| cultivation d5 | 10 | — | 1.56x | **1.74x** | ~40–60x |
| distillation | 5 | **5.13x** | 2.84x | **18.2x** | — |
| coherent d3 r1 | 5 | — | 0.95x | 2.34x (1.94x ‡) | — |
| coherent d3 r3 | 7 | 0.88x (1.09x w/ #288) | — | 3.40x (2.08x ‡) | — |
| coherent d5 r1 | 12 | 0.37x (0.59x w/ #288) | — | 3.29x (1.87x ‡) | 145M shots/s claimed §|
| coherent d5 r5 | 22–24 | — | — | 194x (46x ‡) | — |
| regime k=12, L=512 | 12 | **1.30x** (1.69x w/ #288) | — | **0.5x** | — |

† clifft's surface d7 r7 is the k=0 class; SymFT's own benchmark set uses
different (surface-code) pure-Clifford circuits, so this is a class comparison,
not the same circuit.
‡ vs the k-fixed clifft arm. Part of the coherent-family gap was a clifft
planner miss (terminal phase rotation not eliminated through an interposed
noise op), worth 1.2–4.4x; PR #239 addressed most of it, so the as-measured
ratios against 0.7.0 are upper bounds.
§ Whole RTX 4090 (450 W) vs one core of a 20-core Xeon Gold 5218R, and the
checked-in GPU config can't reproduce this entry as written — treat the GPU
column as directional only. The GPU-vs-clifft figures are our CPU ratios
composed with the paper's stated GPU-over-SymFT-CPU factors (38.6x at d3,
24.3x at d5).

### What actually explains the SymFT gap

The attribution work in #9 decomposed it: **cross-shot batching is the dominant
mechanism, not the engine.** SymFT's single-shot backend is 0.95–1.17x vs clifft
on cultivation d3 and coherent d3 r1 — i.e. dead even. The win comes from
evaluating affine sign words 64 shots at a time (92% packing efficiency to 64
lanes measured on a batch-size sweep). It's worth ~3x on cultivation-class,
~6x on distillation-class, and up to ~50x on k=0 Clifford counts workloads —
and **nothing** at k≳10, where every shot does full dense work that can't be
packed.

Which is why the regime row inverts: at k=12 SymFT is 0.5x clifft legacy, and
clifft symbolic is 1.30x (1.69x with #288) — so symbolic is ~2.6–3.4x ahead of
SymFT in that regime. The two exceptions where SymFT's engine genuinely wins are
composite cultivation dense work (1.8–2.6x, profiled to SoA layout + no per-shot
Clifford stream) and coherent d5 r5, where its product-component factorization
is worth ~10x on its own.

## 3. What's next

1. **#288** (direct-rotation AVX-512) — in review; moves coherent d3 r3 ahead of
   legacy and cuts coherent d5 r1 by 36%.
2. Low-pivot / pivot-4 direct rotation kernels — now the largest remaining hot
   path on the real coherent QEC workloads (52% of cycles for 37% of visits).
3. Planner / continuation compilation — the shared cause of the surface compile
   gap and the noncomputational low-leak row.
4. Measurement probability + collapse kernels (~19% on coherent d5 r1) after
   rotation coverage is resolved.
5. Separately from #280: a **counts-only batched sampling path** is the one
   structural thing SymFT has that we don't. It's gated work — material at k≲8
   on stream-dominated workloads, worthless at k≥12.

## Caveats

- Everything above is single-threaded, one pinned core, AVX-512. No AVX2
  implementation of the fused/direct kernels yet, so these numbers do not
  describe AVX2 users.
- The clifft and SymFT campaigns are separate runs at different commits on the
  same host family; cross-table composition is indicative, not a controlled
  measurement.
- SymFT ratios were measured against clifft 0.7.0 legacy, before the symbolic
  optimization sequence landed.
