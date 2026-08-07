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

## The go/no-go: measured, and it is close to a "no"

Two questions were on the table. Both are now answered.

- **Which architecture?** Settled (above). More GPU hours will not move it.
- **Is a GPU backend worth building?** **Only for small active ranks.**

We measured CPU thread-scaling directly (`results/2026-08-06-h200/cpu_scaling.csv`).
The test box is a slice of an **Intel Xeon Platinum 8468**: 16 vCPU = **8
physical cores** + SMT, 2 MB L2 per core, 105 MB shared L3.

### The workload scales with cores, not with DRAM bandwidth

| k | 1 core | 8 cores | scaling | per-core efficiency | SMT adds |
|---|---|---|---|---|---|
| 12 | 11,598 | 70,338 | 6.1× | 76% | +29% |
| 16 | 755 | 5,807 | **7.7×** | **96%** | +13% |
| 18 | 140 | 1,114 | **8.0×** | **100%** | +8% |

Near-perfect linear scaling on physical cores. This confirms the mechanism:
each core keeps its 1 MB shot in its own 2 MB L2 and runs the whole program
there — the CPU does MIMD-shared's trick for free. A DRAM-bound workload would
have flattened (clifft's own earlier cross-check saw 2.6× on 11 cores at k=20,
where the state no longer fits cache).

### Projected to the full chip this VM is a slice of (48-core Xeon 8468)

| k | per core | 48-core CPU | H200 (MIMD) | GPU/CPU | working set vs 105 MB L3 |
|---|---|---|---|---|---|
| 12 | 8,792 | 422,000 | 4,660,343 | **11.0×** | 3 MB — fits |
| 16 | 726 | 34,800 | 61,581 | **1.8×** | 48 MB — fits |
| 18 | 139 | 6,700 | 15,213 | 2.3×* | 192 MB — **spills** |

\* the k=18 figure flatters the GPU: at 48 cores the working set exceeds L3,
so the real CPU number would be lower — but for the same reason the projection
there is the least trustworthy.

**At k=16 a single 48-core server CPU lands within 1.8× of an H200 — under the
precommitted 2× stop line. A dual-socket box (96 cores) reaches 69,700 shots/s
and simply beats the GPU (0.88×).**

### Why: aggregate L2 bandwidth is the same order as HBM

At k=16 one core sustains 47 GB/s against its L2. Forty-eight of them is
**2.26 TB/s** — versus the H200's measured **4.9 TB/s** of HBM. That factor of
~2 *is* the entire GPU advantage in this regime, and it is a factor of 2 before
accounting for power (700 W vs ~350 W) or price.

The GPU only pulls away when *it* gets to play the same cache trick: at k ≤ 13
the shot fits in the SM's 227 KB scratchpad, the whole program runs on-chip,
and the lead jumps to **11×**.

### Verdict

**Build the GPU backend only if the target workloads compile to
`peak_rank ≲ 13`; above that, a modern server CPU is already competitive.**
`peak_rank` is known at compile time, so this is a dispatch decision, not a
guess. Our opcode census shows real circuits on both sides of that line —
`magic_conveyor` (11) and `hidden_ccz_t4` (12) qualify; `iqp_n028` (14),
`qaoa_ring_n16` (15) and `hidden_shift_np8` (16) do not.

## Next steps, in order

1. **Confirm the projection on a real many-core box** (~$2, one hour). The
   48-core numbers above are extrapolated from 8 cores; the scaling is clean
   (96–100%) and the cache-capacity check holds at k ≤ 16, but L3 contention
   and memory-controller limits at 48 cores are not measured. Run **real
   clifft**, not just the model kernels, on the same circuits — that also
   closes the "no AVX-512 baseline" gap. This is the last cheap step before
   committing engineering time.
2. **Decide by regime, not globally.** If the workloads that matter compile to
   `peak_rank ≲ 13`, build it — the 11× is real and the port is small. If they
   sit at k = 14–20, the honest answer is that clifft's CPU backend is already
   near a GPU's performance per socket, and the effort is better spent
   elsewhere (the CPU kernels, or reducing peak rank in the compiler — which
   would *also* move workloads into the GPU-favourable band).
3. **If you build it:** port to FP64 MIMD, starting from clifft-cuda
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
