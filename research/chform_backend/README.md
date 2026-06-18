# Low-rank stabilizer-decomposition backend (residual-backend prototype)

A working, validated prototype of the "stab-rank back-end" that would replace
clifft's dense active block, plus the demonstration that the **composition**
(clifft's measurement-driven reduction → low-rank stabilizer decomposition of the
residual) keeps the stabilizer rank bounded per measurement episode.

## What it is

`engine.py` maintains the state as a low-rank superposition of stabilizer states
`|psi> = sum_j term_j` and exposes the operations the active block sees:

- **Clifford gates** (H, S, X, Y, Z, CX, CZ, SWAP) — applied to every term; rank χ unchanged.
- **T / R_z** — non-Clifford diagonal, applied by splitting each term into its
  `q=0` and `q=1` computational-basis projections (each a stabilizer state):
  `T_q|s> = P0|s> + e^{iπ/4} P1|s>`. This **branches** the rank (×2 per T unless
  the qubit is already in a definite basis state).
- **Measurement** — projects every term onto the sampled outcome, drops
  zero-support terms, then **recompresses** (see below) to collapse the rank.

`chi` (the number of terms) is the stabilizer rank — the cost driver.

### Term representation: two interchangeable backends

The engine is written against a `Term` interface (`term.py`), so the per-term
store is swappable without touching χ, the branching, or the measurement logic:

- **`DenseTerm`** — a dense `2^k` statevector. Trivially correct; the original
  prototype and the validation oracle.
- **`CHForm`** (`chform.py`) — the **real CH-form stabilizer tableau**
  (`F, G, M, γ, v, s, ω`; Bravyi–Browne–Calpin–Campbell–Gosset–Howard,
  arXiv:1808.00128, Sec 4.1; PDF in `research/refs/`). `O(k^2)` bits per term
  instead of `2^k` complex amplitudes — the genuine memory compression. Clifford
  gates are `O(k)`, the Hadamard is `O(k^2)` (Prop. 4 desuperposition), a single
  amplitude is `O(k^2)` (Eq. 56), and single-qubit Z-projection (for T-branching
  and measurement) is the paper's projective-gate routine.

Select with `LowRankState(n, backend="chform")` (default `"dense"`). χ — the
cost driver — is identical for both; only per-term storage and gate cost differ.
For the conveyor `w=8` case each term drops from 4096 B (dense) to 232 B (CH-form).

> Convention note: the paper conjugates by `U_C^{-1}` (Eq. 43), so every update
> rule follows that convention exactly. An earlier from-scratch derivation in the
> opposite (`U_C`) convention validated its C-type gates but was discarded in
> favour of the verbatim paper algorithm once the reference was available.

## Validation

`python -m research.chform_backend.test_chform` — **gate-by-gate** CH-form vs
the dense oracle (the README's "validate it against, gate by gate"):

- amplitude formula on hand-built states (err `~3e-17`);
- the full Clifford set `{H,S,S_DAG,X,Y,Z,CX,CZ,SWAP}` on 400 random circuits
  (err `~6e-15`) — exercises the Hadamard desuperposition end to end;
- single-qubit projection vs dense on 400 states incl. zero-support cases
  (err `~3e-15`);
- the `G F^T = I` and `‖φ‖ = 1` tableau invariants.

`python -m research.chform_backend.validate` — the **whole engine**, both backends:

- **Exact** vs a direct dense reference on 25 random Clifford+T circuits
  (dense `0.00e+00`, CH-form `~3e-15`).
- **Machine-precision** vs `clifft.get_statevector` on 12 circuits
  (`~1e-16`, up to global phase) — confirms gate/index/rotation conventions
  match clifft.

## The demonstration (`python -m research.chform_backend.demo`)

1. **Unitary Clifford+T** (no measurement): χ grows monotonically to exactly
   `2^(#T)` (16, 256, 1024 for 4/8/10 T's) — the naive, un-reduced rank.

2. **Conveyor** (a width-`w` register re-used across `R` measured mini-IQP
   rounds): χ peaks at `2^w` each round and **collapses back to 1** when the
   round is measured out. For `w=8, R=6` (total T = 48): **peak χ = 256 = `2^8`,
   not `2^48`.** The rank is bounded by *one round's* magic, exactly mirroring
   the static profiler's `mblk` (per-episode magic) finding — now realized in an
   actual low-rank engine.

## Key implementation lesson: recompression is load-bearing

The first version had **no recompression** — measurement only dropped
zero-support terms. Result: χ did **not** collapse on measurement. The surviving
terms stay linearly dependent but distinct, so χ kept growing to `2^(total T)`
and the conveyor blew up (`2^48`). **The rank collapse is not automatic; it
requires active recompression.** A real residual stab-rank backend *must*
recompress on measurement — it is the load-bearing operation, not an optimization.

The recompression here (`recompress_dedup`) merges terms that are scalar
multiples of one another — exact, and enough to collapse a disentangled round to
rank 1. It is a low-effort stand-in for true **sparsification** (the random-walk
methods that reach the `2^{0.228 t}` stabilizer-extent bound). Without
sparsification, within a single episode χ still peaks at `2^t` (the trivial
decomposition), not `2^{0.228 t}`.

(Bug found and fixed along the way: dedup keyed on raw bytes failed to merge
identical one-hot vectors because of signed zeros, `-0.0` vs `0.0`, from complex
multiplies on zero entries; normalized by adding `0.0`.)

## The free-Clifford figure

`clifford_term_ops` = Σ over Clifford gates of χ-at-that-time = the per-term
Clifford updates a **pure** stab-rank backend must perform. clifft's symbolic
frame evolves the Clifford part of *dormant* axes for ~free, so this count is
precisely the work the composition removes relative to a pure stab-rank simulator
— the constant/poly advantage identified in the profiling phase (the exponent is
matched by any measurement-aware stab-rank sim; the Clifford-frame saving is
clifft's genuine edge).

## Scoped next increments

1. **CH-form term store** — ✅ **done** (`chform.py`, `term.py`). A real CH-form
   stabilizer state (F/G/M/γ/v/s/ω) with Clifford/Hadamard/projector updates and
   the Eq. 56 amplitude, `O(k^2)` memory/term instead of `2^k`. Validated
   gate-by-gate against the dense oracle and clifft to machine precision
   (`test_chform.py`, `validate.py`). Still materialises for the cross-term sum
   in measurement and for dedup — see (2a).
2. **Sparsification** — ✅ **done** (`engine.py`, `bench_sparsify.py`). Two parts:
   - **Low-extent magic decomposition.** T-branching now uses the minimal-extent
     Clifford pair `T = αI + βS` (both Clifford) instead of the `|0>/|1>` split.
     `|α|+|β| = 2^{0.114}` per T, so the whole-state extent `‖c‖₁² = 2^{0.228 t}`
     — the `0.228` is literally `log₂(|α|+|β|)²`. The `|0>/|1>` split cost `√2`
     per T (extent `2^t`), which sparsification *cannot* improve; this is the fix
     that makes sparsification pay off. Still exact when un-sparsified (validated
     to machine precision vs the dense oracle and clifft).
   - **`sparsify(k)`** — the BGH unbiased importance-sampling estimator
     (`E‖ψ−ω‖² = (‖c‖₁²−1)/k`). `bench_sparsify.py` confirms: extent grows at
     exactly `2^{0.228 t}`; measured sparsification error tracks the bound to ~1%;
     the estimator is unbiased (averaging → ψ_true). So `k ~ 2^{0.228 t}/δ²` terms
     suffice — the rank collapses from the exact `2^t` to the extent scale, at the
     cost of clifft's exactness (approximate, like all stab-rank sampling).
3. **Native CH-form inner product + norm estimation** — ✅ **done**
   (`norm_est.py`, `test_normest.py`). The non-materialising primitives:
   - **Lemma 3** `<φ|φ_A>` (CH-form vs equatorial state) — validated vs dense
     to machine precision.
   - **Lemma 4** `O(n³)` exponential sum `Σ_x i^{x B x^T}` — validated *exactly*
     (`0.00e+00`) vs brute force.
   - **Lemma 2** norm estimator `‖ψ‖² ≈ avg 2^n |<φ_A|ψ>|²` over random
     equatorial `A` — no `2^n` vector anywhere.
   With these, `engine.amplitude(x) = Σ_a ω_a <x|φ_a>` (`O(χ n²)`) and the norm
   estimate give output probabilities `P(x)=|<x|ψ>|²/‖ψ‖²` with the whole pipeline
   materialisation-free. `sparsify` also got a `O(n²)` tableau-key duplicate merge
   (was a `2^n` `canonical_key`), and `LowRankState(..., sparsify_budget=B)` keeps
   χ at the extent scale during evolution.
4. **Single-shot tensor-product magic injection** — ✅ **done**
   (`engine.inject_magic_layer`, tested in `test_chform.py`). For product magic
   (e.g. IQP's `|T>^n` after `H^n`), each branch-string picks `{I,S}` per qubit
   *independently*, so we **importance-sample `k` branch-strings directly** —
   no `2^t` built. The estimator is unbiased (validated: `max|avg<x|ω> − <x|ψ>|
   = 1.7e-3`) with `‖ω‖² ≈ 1 + (extent−1)/k` (validated exactly at small n, e.g.
   n=10: 1.011 vs 1.013 predicted), so `k ~ 2^{0.228 n}/δ²` — **no factor of t**,
   vs streaming's `~t·2^{0.228 n}/δ²` (variance accumulates over the ~t
   mid-circuit sparsify steps — that's what made the earlier streaming run coarse,
   `‖ω‖²≈2`). For product magic this is strictly better *and* faster.

### The past-clifft benchmark (`bench_vs_clifft.py`) — magic-sparse IQP

`H^n ; T^{±} on each qubit ; CZ chain ; H^n`, single-shot magic injection.
clifft must hold the `2^k` active block (`k = peak_rank = n` here, no
measurements to reduce it); the CH-form backend holds `~2^{0.228 n}` terms of
`O(n²)` bits and never materialises a `2^n` vector.

- **A — Correctness** (n ≤ 10, where clifft's `get_statevector` runs): single-shot
  `P(x)` matches clifft's exact probabilities to `1.4e-3 … 1.2e-2`, with
  `‖ω‖² ≈ 1.01–1.02` (δ=0.2).
- **B — Crossover**: clifft `2^k·16B` = 16 MB (n=20) → 16 TB (n=40) → 1 PB (n=46)
  → 16 EB (n=60), and clifft **cannot even compile at n≥64** (`k≥63` hard wall).
  CH-form memory over the same range: 33 KB → 2.8 MB → 9 MB → 143 MB → 305 MB.
- **C — Actual runs past the wall**: single-shot at n=40/46/50 builds
  `k ~ 2^{0.228 n}/δ²` terms directly (δ=0.4): **3477 terms / 17 MB / 15 s**,
  **8975 / 58 MB / 40 s**, **16889 / 129 MB / 86 s** — where clifft needs
  **16 TB / 1 PB / 16 PB**. Accurate (`‖ω‖² ≈ 1+δ²`, per A and the exact test),
  not coarse, and *faster* than the streaming attempt (n=40: 15 s vs 65 s) — the
  single-shot fix removed the streaming t-factor that made the earlier run a
  ~100%-error approximation.

*Honest scope.* The **memory/representability win is real and the result is now
accurate**: the `2^{0.228 n}` vs `2^n` separation is demonstrated, and single-shot
sampling reaches it at the optimal budget `2^{0.228 n}/δ²`. The remaining limit is
**wall-clock constant factors** — pure-Python per-gate cost makes n≈50 take
minutes, and norm *estimation* over many terms (`χ·samples·n³`) is the heaviest
step. A compiled / bit-packed-F₂ backend (the paper's qiskit-aer `CHSimulator`
is exactly this) closes that; nothing algorithmic remains.

5. **Clifford frame / composition** — ✅ **done** (`clifford_frame.py`,
   `test_clifford_frame.py`, `bench_conveyor.py`). The original bet: factor
   `|state> = F·(Σ term)` with
   `F` a single global Clifford frame (like clifft's symbolic layer), so Cliffords
   are absorbed into `F` for free instead of re-applied to every term — the
   composition's *unique* edge over pure stab-rank (a constant/poly factor, not the
   exponent).
   - **`CliffordFrame`** — inverse-conjugation tableau (`F⁻¹X_qF`, `F⁻¹Z_qF`),
     `O(n)`/gate Clifford absorption, `conj_Z(q)` Pauli. Validated vs dense
     (`6.7e-16`), symplectic basis preserved.
   - **`LowRankState(..., frame=True)`** — Cliffords → frame (free); `T`/`Rz` →
     branch about the Pauli `P'=F⁻¹Z_qF` at the **optimal `{I,S}` extent (`1.0824/T`,
     verified)**: each diagonal Clifford `D` is applied as `F⁻¹D_qF = W†D_jW`
     (`W` reduces `P'→+Z_j`, sign-normalised by appending `X_j`; functional calculus
     commutes with conjugation, so this is exact). Same coefficients as non-frame →
     same `2^{0.228t}` extent. Measurement → Pauli measurement of `P'` on the
     residual (reduce `P'→Z_j`, reuse `CHForm.project`, `W†` back), `F` untouched;
     readout folds `F` into the terms once.
   - **Validated**: frame engine == plain engine on unitary Clifford+T (`3.3e-15`)
     and on measured Clifford+T with forced outcomes (`5.6e-15`); terms stay
     CH-form. **Free-Clifford result: per-term Clifford ops plain = 45704, frame =
     0.**
   - **Conveyor op-count experiment** (`bench_conveyor.py`): R measured mini-IQP
     rounds with feedforward (rank collapses each round, like clifft's `k→0`).
     Frame == plain (forced outcomes, `4.7e-15`). Per-term Clifford ops, plain
     (measurement-aware stab-rank) vs frame: **473 vs 0** (w=4), **2850 vs 0**
     (w=6), **15411 vs 0** (w=8) — the frame absorbs all of them into 53/78/111
     `O(n)` updates. This is the composition's unique edge over the *best*
     stab-rank baseline, demonstrated: a **constant/poly factor (free Cliffords),
     not the exponent** — exactly what the profiling phase identified.
   - **Honest scope**: this is the *op-count* (work) win, language-independent. It
     is not an exponent win (a measurement-aware stab-rank sim matches the
     `2^{0.228n}` rate), and the profiling caveat stands — no natural circuit was
     found where the residual is *also* magic-sparse at `k>30` (connected circuits
     were either saturated or collapsed to tiny `k`). Two carry-forwards: the
     frame magic branch is single-Pauli `I/P'` (extent `1.707/T`; optimal `1.17/T`
     needs the Clifford `√P'`), and measurement probability still materialises the
     residual at validation scale (norm estimation / a C++ port remove that).

6. **C++ / bit-packed port** — ✅ **done** (`../chform_cpp/`). Standalone C++20,
   no clifft/stim/network deps; `clang++ -std=c++20 -O3`. Turns the validated
   `2^{0.228n}` / op-count wins into wall-clock by removing the pure-Python
   per-gate interpreter overhead.
   - **`chform.hpp`** — bit-packed CH-form term (F/G/M rows in `uint64` words),
     all Clifford gates incl. the H-desuperposition, `amplitude` (Eq. 56),
     projection. Ported from the validated Python; validated vs an in-C++ dense
     oracle (`test_chform.cpp`): full Clifford set `7.99e-15`, projection `5.33e-15`.
     Single-term gate throughput **~66× at n=30, ~18× at n=60** over Python.
   - **`bench_iqp.cpp`** — engine driver (low-rank sum + single-shot magic
     sampling + amplitude). Exact engine (full branch enumeration) == dense IQP at
     n≤10 (`1.45e-15`). Past-clifft IQP wall-clock (δ=0.4, accurate):

     | n | k | build | CH mem | clifft `2^n` | Python |
     |---|---|---|---|---|---|
     | 40 | 3477 | 0.05s | 3.6 MB | 16 TB | 15 s |
     | 50 | 16889 | 0.28s | 22 MB | 16 PB | 86 s |
     | 60 | 82029 | 2.2s | 128 MB | 16 EB | — |
     | 66 | 211728 | 6.7s | 683 MB | 1 ZB | — |
     | 72 | 546497 | 18s | 1.9 GB | 64 ZB | — |

     ~300× over Python at n≤50, and reaches **n=72 in 18 s / 1.9 GB** — past
     clifft's `k≥63` compile wall, where the dense `2^n` block would be 64 ZB.
   - **`norm_est.hpp`** — norm estimation (Lemmas 2/3/4) for output probabilities
     `P(x)=|<x|psi>|^2/‖ψ‖^2`. Validated vs brute force / dense (`test_normest.cpp`):
     exp_sum **`0.00e+00`** (exact), `<phi|phi_A>` **`4.97e-16`**, norm estimate
     **`3.0%`** (L=6000). Full non-materialising pipeline at the frontier:
     `‖ω‖²` ≈ 1.31 (n=40) / 1.20 (n=46) vs target `1+δ²=1.16` (within the L=15
     estimator + single-realization variance).
   - **Bit-packed F₂ algebra**: `real_exp_sum` (Lemma 4's GF₂ core) rewritten with
     word-level row XORs, in-place, **zero per-step heap allocation** (the scalar
     version reallocated an `O(n²)` matrix every recursion step — the real
     bottleneck); `inner_equatorial`'s `K=Gᵀ(A+J)G` flattened with word-level
     set-bit iteration over `G`. Validated identical (exp_sum still exact vs brute
     force). **Norm-est ~3.3× faster**: n=46 `7.2s → 2.2s`, n=40 `2.3s → 0.7s`.
     (Not 64×: at n≤64 the GF₂ rows are a single word, so the win is alloc-removal
     + word ops.)
   - **`K` mod-4 GF₂ decomposition** (`K = Gᵀ·Off·G + Gᵀ·Diag·G`): off-diagonal
     `mod 2` via bit-packed GF₂ bilinear forms, diagonal `mod 4` via popcount sums,
     and `sK` via `z=(A+J)(Gs)` in `O(n²)` — **eliminates the `O(n³)` full matmul**.
     Validated identical vs dense (`4.97e-16`). *Honest:* **no measured wall-clock
     change at n≤46** — the cost there is dominated by the other `O(n²)` terms
     (the `Off`/`J` build, the `G`-transpose, `exp_sum`), not the matmul; the
     asymptotic `n³→n²` win only shows at larger n. Kept for that asymptotic
     benefit; it is correct, just not faster at the sizes benchmarked here.
   - **Remaining (optional)**: nothing structural left — the open items are pure
     performance at larger n (the `O(n²)` `Off`/`J` build and `exp_sum` are now the
     norm-est floor) and the literal clifft-binary coupling (sense 3 of #5).
