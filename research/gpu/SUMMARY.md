# GPU acceleration of clifft: what we learned, and what to do next

**Date:** 2026-08-07 · **Evidence:** `microbench/RESULTS.md` (Run 2a, H200 SXM,
FP64) · **Visualisation:** [`architectures.html`](architectures.html)

---

## The two designs, in plain terms

Both run thousands of Monte-Carlo shots of the same compiled circuit on one
GPU. They differ in a single choice: **who owns a shot.**

**Lockstep SoA (our proposed design).** Stack B shots into one `B × 2^k`
tensor. Each bytecode instruction becomes one kernel launched across the whole
batch: "apply H to axis 3" runs for every shot at once, then the next
instruction, and so on. Every shot is at the same instruction at the same
moment — which clifft's compiler guarantees, since the active-rank trajectory
is fixed at compile time.

**MIMD per-shot (clifft-cuda's design).** Give each shot its own team of
threads (a CUDA block) and let it walk the entire program alone, start to
finish, in one kernel launch for the whole run.

The bet we made was **coalescing**: in SoA, neighbouring threads work on
neighbouring shots at neighbouring addresses, which is how you reach a GPU's
rated bandwidth. MIMD, we argued, structurally cannot do this.

**That bet was wrong, for a reason that is obvious in hindsight.** Coalescing
is a 128-byte phenomenon — a warp of 32 threads needs one contiguous cache
line. In MIMD, a block's threads stride side by side through *their own shot's*
slice, which at k=16 is 1 MB of contiguous memory: thousands of perfectly
coalesced lines. Cross-shot adjacency was never needed.

Worse, batching **destroys** something valuable. A circuit hits the same 2^k
amplitudes over and over — enormous temporal locality. MIMD exploits it: load
the shot into the SM's on-chip scratchpad once, run the whole program there,
write back once. SoA structurally cannot: between two instructions on shot *i*
it sweeps every other shot in the batch, so shot *i* is long evicted and must
be re-read from HBM. Same arithmetic, **22× the memory traffic** on our
schedule — and memory is the speed limit.

The one thing SoA has that MIMD doesn't is **fine-grained parallelism**. Its
unit of work is a thread; MIMD's is a whole shot. That only matters when there
aren't many shots.

---

## Who wins where (measured, H200, identical schedule/state/RNG)

| regime | winner | margin | why |
|---|---|---|---|
| **k ≤ 13** — shot fits on-chip (≤ 227 KB) | **MIMD-shared** | **~5×** | whole shot lives in SRAM; HBM goes idle |
| **k = 14–24** — both stream from HBM | **tie** | 0.9–1.1× | both saturate the same bus; coalescing bet refuted |
| **k ≥ 26** — memory caps B below ~2× SM count | **SoA / per-op** | **3–6×** | too few shots to fill 132 SMs; MIMD blocks starve |

The boundary variable is **B, not k**: the SoA/MIMD ratio is ~6.0 at B=8, ~3.2
at B=32, and 1.0 by B=256, essentially independent of k. Memory couples it to
k, because big shots mean few shots.

**clifft's target regime (k < 20) sits entirely in the first two rows.**

### Recommended backend

One MIMD engine with a compile-time dispatch rule:

1. `2^k · 16 B` fits in shared memory → **MIMD, on-chip** (the 5× win)
2. else `B_max = HBM / (16 · 2^peak_rank) ≥ ~2 × SMs` → **MIMD, global**
   (ties SoA, but simpler: one launch, no per-instruction launch tax, and
   postselection discards exit for free)
3. else → **per-op kernels** (the SoA path, already written and validated)

`peak_rank` is known at compile time, so this is a one-line decision at load.

---

## The decision that is *not* yet made

Two separate questions were on the table. Only one is answered.

- **Which architecture?** Settled. More GPU hours will not move it.
- **Is a GPU backend worth building at all?** **Still open, and at real risk of
  coming back "no".**

Our GPU-vs-CPU ratio is 9–12×, but against a 16-vCPU x86 VM — not a serious
opponent. The precommitted no-go threshold is **< 2× vs a tuned server CPU**,
and our own data says the true ratio probably lands near it.

**The CPU is already doing MIMD's trick.** At k ≤ 16 a shot is ≤ 1 MB, so on
CPU it is L2-resident: each core loads its shot into cache once and runs the
whole program there — precisely what makes MIMD-shared win on the GPU, except
the cache hierarchy does it for free. The evidence is in our own CSVs:

| measurement (k=16) | throughput | implies |
|---|---|---|
| single-op rows (1 thread — clifft only threads at k ≥ 18) | 1.7 Gamp/s = 54 GB/s | one core's cache bandwidth |
| `batchedreal` (16 threads) | **12.9 Gamp/s = 414 GB/s** | **7.6× on 16 threads** |

414 GB/s is far above any DDR bus on a 16-vCPU VM slice — the batched workload
is running out of cache, and it scales with **cores**, not DRAM bandwidth.
Extrapolating that scaling to a real server CPU:

| CPU | scaling efficiency | projected shots/s | GPU/CPU |
|---|---|---|---|
| 64-core x86 | 100% | 25,500 | **2.4×** |
| 64-core x86 | 70% | 17,800 | 3.5× |
| 72-core Grace | 100% | 28,700 | **2.1×** |
| 72-core Grace | 70% | 20,100 | 3.1× |

So the honest expectation is **2–3.5×**, brushing the 2× stop line. Factor in
perf-per-watt (H200 ~700 W vs a server CPU ~300 W) and clifft-cuda's own
calibration (1.6× per node, a *loss* per watt), and "the CPU wins" is a live
outcome, not a formality.

**This is cheap to settle and must be settled before any port work begins.**

---

## Next steps, in order

1. **Measure the CPU thread-scaling curve first — ~$1, ten minutes.** Restart
   the existing H200 box and run `bench_cpu` at
   `OMP_NUM_THREADS = 1, 2, 4, 8, 16`. If `batchedreal` scales linearly to 16
   threads, the projection above holds and the go/no-go is effectively decided
   on paper; if it flattens early (DRAM-bound), the GPU case is stronger than
   the table suggests. Either way it costs a dollar and needs no new hardware.
2. **Then the real baseline on a many-core server** (~$2, one hour). Grace on
   GH200 is ideal but capacity-blocked; a 64-core x86 node is available now and
   arguably more representative of what users run — it also closes the
   "no AVX-512 baseline" gap. Run **real clifft**, not just the model kernels,
   on the same circuits. This produces the binding go/no-go verdict.
3. **If it clears the bar:** port to FP64 MIMD, starting from clifft-cuda
   (Apache-2.0, ~2000 lines, currently FP32 with `peak_rank ≤ 19`). Work:
   complex128 amplitudes, the shared-memory path for k ≤ 13, lift the rank cap
   via the global path, and the dispatch rule above. Deliverable order stays as
   scoped: batched forced-replay → noisy sampling → general `sample()`.
4. **Close the modelling gaps** the benchmark left open — early-discard
   (postselection), per-shot noise ops, `SWAP_MEAS_INTERFERE`. All three favour
   MIMD further, so they can only improve the case.
5. **If it does not clear the bar:** write down "CPU wins per watt" and stop.
   The experiment cost ~$8 and produced a decision either way.

---

## Do cuStateVec or cuStabilizer have a role?

**cuStateVec — no.** Three independent reasons, any one sufficient:

- **No headroom to buy.** Our hand-written kernels already run at the HBM
  roofline (H at 154 Gamp/s = 4.9 TB/s measured). These ops are
  memory-bound; no library beats the memory bus.
- **Architectural mismatch.** The value is in tiers 1–2, where the whole
  program runs inside one kernel. cuStateVec is a host-called, per-operation
  library — structurally the losing design there. It could only serve tier 3,
  the rare fallback.
- **Custom kernels are unavoidable anyway.** `EXPAND`, `EXPAND_T` and X-basis
  `INTERFERE` have no cuStateVec primitive.

*Revisit only if* clifft ever targets multi-GPU distributed statevectors at
k > 30, where cuStateVec Ex handles the inter-GPU comms. That is far outside
the current k < 20 profile.

**cuStabilizer — no, and it is a different product.** It accelerates
Clifford-only Pauli-frame simulation (~1000× vs Stim). clifft's entire value is
the *non*-Clifford active block, which cuStabilizer cannot represent.

One tempting idea, checked and rejected: our opcode census found frame ops are
23–77% of real instruction *streams*, so why not offload them? Because they are
~0% of the *work* — they touch no amplitudes, cost O(1) per shot, and in the
MIMD design are already free in-kernel bit operations. Accelerating them buys
nothing.

The one genuine case is circuits that compile to **zero active rank** — our
census found `surface_d5_r6` compiles to 967 instructions with `peak_rank = 0`,
i.e. pure frame bookkeeping. A GPU stabilizer engine would crush those. But
that is Stim's and cuStabilizer's home turf, with no clifft differentiation.
Keep it a watch item only if cuStabilizer grows T-gate support.
