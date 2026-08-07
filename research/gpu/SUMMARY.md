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

## The go/no-go: measured on real hardware

Both questions are now answered from measurement, not projection.

- **Which architecture?** Settled above.
- **Is a GPU backend worth building?** **Yes for most of the range — but the
  cheapest large win is not a GPU at all** (next section).

Run 3 (`results/2026-08-07-cpu48/`) put the CPU side on a **48-core Intel Xeon
6975P-C** (Granite Rapids, 2 MB L2/core — the same per-core cache as the chip
Run 2a extrapolated from, so the mechanism is preserved).

### The 8-core projection held

| k | 48-core CPU (measured) | H200 (MIMD) | GPU/CPU | Run 2a projected |
|---|---|---|---|---|
| 12 | 319,802 | 4,660,343 | **14.6×** | 11.0× |
| 16 | 32,135 | 61,581 | **1.9×** | 1.8× |
| 18 | 2,594 | 15,213 | **5.9×** | 2.3× |

Scaling to 48 cores: **37.2× at k=16** (78% efficiency), 26.6× at k=12,
and only 16.9× at k=18.

### The GPU's advantage is a valley, not a slope

That k=18 number is the surprise, and it completes the mechanism. Each side has
a cache that either holds a shot or doesn't:

| | GPU scratchpad (227 KB) | CPU L2 (2 MB/core) | result |
|---|---|---|---|
| **k ≤ 13** | holds the shot | holds the shot | GPU's on-chip memory is far faster → **14.6×** |
| **k = 14–17** | too small | holds the shot | only the CPU gets the trick → **1.9×** |
| **k ≥ 18** | too small | too small | both stream; HBM ≫ DDR → **5.9×** |

So the CPU is competitive only in a narrow band — k = 14–17, where a shot fits
2 MB of L2 but not 227 KB of shared memory. Below and above it, the GPU wins by
6–15×. Our opcode census puts real circuits on both sides: `magic_conveyor` (11)
and `hidden_ccz_t4` (12) sit in GPU territory, `hidden_shift_np8` (16) and
`iqp_n028` (14) in the valley, `rand_cliffT_n20` (20) back in GPU territory.

---

## The finding that outranks the GPU question

The cross-check against **real clifft** on the same 48-core box exposed
something worth more than any of the above:

| | k=16 throughput on 48 cores |
|---|---|
| real `clifft.sample()` | **1.17 Gamp/s** |
| microbench model, shot-parallel | **65.2 Gamp/s** |

clifft's sampling loop is **serial across shots** (`svm/svm.cc:218`), and its
only parallelism — `parallel_for` in `svm/svm_internal.h` — is gated on
`active_k >= kMinRankForThreads` (18). **For k < 18, which is clifft's entire
target regime, `sample()` runs single-threaded no matter what
`get_num_threads()` reports.** The 96 threads on that box were idle.

Shots are embarrassingly parallel — independent state, independent RNG stream,
disjoint output slices. Parallelising that outer loop is a contained change to
one file, and the measured payoff on this hardware is **~37× at k=16**, versus
1.9× for moving the same workload to an H200.

**Do this first.** It is cheaper than a GPU backend, it benefits every user
without new hardware, and it changes the baseline that any future GPU decision
is measured against.

## Next steps, in order

1. **Parallelise `sample()` across shots** (CPU, one file). ~37× at k=16 on a
   48-core box, no new hardware, benefits every user. Needs a decision on
   per-shot RNG streams and therefore on the reproducibility contract — the
   same decision a batched GPU backend would have forced anyway.
2. **Re-measure the GPU case against that new baseline.** Everything in the
   table above already assumes a shot-parallel CPU, so the GPU ratios stand —
   but the *urgency* of a GPU backend drops once the CPU is 37× faster than
   clifft is today.
3. **Then, if you build the GPU backend:** port to FP64 MIMD, starting from clifft-cuda
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
