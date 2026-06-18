# (k, t_live) profiling for the stabilizer-rank decomposition idea

Instrumentation to decide whether replacing/augmenting clifft's **dense active
block** (a `2^k` statevector) with a **low-rank stabilizer-rank decomposition**
would extend clifft's reach. See `../../memory/` and the project notes for the
full research thread; this directory is the empirical de-risking step.

## The question

clifft's hard wall is `2^k`, where `k` = peak active dimension
(`Program.peak_rank`). Stabilizer-rank methods cost `~2^(alpha * t)` in the
T-count `t` instead (`alpha = 0.228` sampling / `0.5` strong-sim). The bet:
does `k` grow large while the *live* magic content stays small? If so, a
residual stabilizer-rank decomposition of the active block helps.

## What it measures (and why a static walk is exact)

The active block is a **stack**: `OP_EXPAND*` push axis `k` (`active_k++`),
active measurements (`OP_MEAS_ACTIVE_*`, `OP_SWAP_MEAS_INTERFERE`, and `_FORCED`
variants) pop axis `k-1` (`active_k--`), both unconditional. So the entire
`active_k` trajectory is determined by the opcode **sequence**, independent of
measurement outcomes (this is also why clifft can size the `2^k` allocation at
compile time). `analyzer.py` replays that stack offline over a compiled
`clifft.Program` — no hot-loop instrumentation, no per-ISA duplication.

Metrics per circuit (all log2-costs are exponents):

| field | meaning |
|---|---|
| `k` (`peak_active_k`) | peak active dimension; dense cost `= 2^k`. Cross-checked `== Program.peak_rank`. |
| `Ttot` (`n_nonclifford_total`) | total T-injections in the circuit. |
| `mblk` (`peak_t_in_episode`) | peak T-injections folded into one active episode (reset when `k` hits 0). Upper bound on the block's magic content. |
| `sat` (`magic_saturation`) | `mblk / k`. **`< 4.4` ⇒ residual stab-rank beats dense**; `>> 4.4` ⇒ block is magic-saturated, rank `~2^k`, no win. |
| `dense` | `k` |
| `gSR` | `0.228 * Ttot` — whole-circuit stabilizer rank (sampling). |
| `rSR` | `min(k, 0.228 * mblk)` — stab-rank on the active residual. **Capped at `k`** because the block lives in `2^k` dimensions. |

A built-in **validation** asserts the reconstructed `peak_active_k` equals
`Program.peak_rank` for every circuit; a mismatch means the opcode model is
incomplete and the run exits non-zero.

## Running

```bash
python -m research.stabrank_profiling.profile               # full suite + table
python -m research.stabrank_profiling.profile --json out.json
python -m research.stabrank_profiling.profile --no-optimize  # naive lowering
python -m research.stabrank_profiling.profile --no-fused-magic  # treat U2/U4 as Clifford
```

(Requires clifft importable — built here into `.venv-research`.)

## Findings (first run, default suite)

1. **No Clifford inflation (0/17 circuits).** Plain `OP_EXPAND` essentially
   never appears: clifft keeps Clifford superposition in the symbolic frame and
   only expands the dense block for genuine magic. **The original hypothesis —
   that `k` is inflated by Clifford structure a stab-rank decomposition could
   strip for free — is empirically false.** `k` is built from magic.

2. **clifft already beats whole-circuit stabilizer rank on T-heavy circuits**
   (gSR worse than dense in 8/17). Example: `rand_cliffT_n28_t15` has `Ttot=184`
   but clifft compiles it to `k=28`; a pure stab-rank simulator pays
   `0.228*184 = 42`, worse than clifft's dense `28`. clifft's measurement-driven
   reduction (`Ttot=184 → k=28`) is more powerful than paying for every T. This
   is a point in clifft's favour vs. a stab-rank simulator, not against it.

3. **The squeeze pass *is* the active-subspace reduction.** With `--no-optimize`,
   hidden-shift compiles to `k=24`; with the default passes it collapses to
   `k=1`. The optimizer is doing exactly the "reduce the non-Clifford core via
   measurement reordering" that the research idea wanted to feed a stab-rank
   decomposition — and it already shrinks the residual to near-nothing on
   measurement-rich circuits.

4. **Residual stab-rank wins only for magic-sparse blocks (`sat < 4.4`),** i.e.
   low T-density / shallow circuits where each magic injection mostly creates a
   fresh dimension (`mblk ≈ k`) so the extent decomposition `0.228*k < k` holds —
   a ~2–4× *exponent* reduction. At high T-density the block saturates
   (`mblk >> k`), its rank approaches the full `2^k`, and stab-rank buys nothing
   over dense. This is the standard stabilizer-rank tradeoff, correctly bounded.

## Targeted sweep findings (`--sweep`: IQP + CCZ hidden shift)

The default suite never hit large-`k`/low-saturation, so `large_k_sweep()`
engineers it: **IQP** circuits (`H^n . diagonal . H^n`, one T per fresh qubit)
push `k` up at low saturation; **CCZ hidden shift** (7 T's per gadget) is the
high-saturation counterpoint. Results:

5. **The revival regime exists.** IQP lands at **`k = 46–60` with `sat = 2.0`**,
   where residual/stab-rank gives a **>2× exponent reduction** (e.g. `k=60` →
   `rSR=27`, `k=46` → `rSR=21`). These `k` are *past clifft's practical memory
   wall* (~30) and near its **hard compile wall (`k=63`**, the `1ULL<<k` guard).
   clifft simply cannot run them today; a stab-rank backend could.

6. **The squeeze pass halves `k` but keeps saturation low.** Naive IQP gives
   `k=n, sat=1.0`; with passes, `k=n/2, sat=2.0`. It is *also* what makes
   `n=92` compilable at all (`k=46` vs the `k=63` wall) — the optimizer buys
   ~2× in `k`, stab-rank could buy a further ~2× in the exponent.

7. **CCZ hidden shift is magic-saturated (`sat=4.7`) → dense wins**, and
   whole-circuit stab-rank is *worse* than dense (`gSR=48 > 45=k`). Confirms the
   `sat > 4.4` boundary: the faithful BGH-style benchmark is the case clifft's
   dense block already beats stab-rank on.

## Composition-advantage demo (`--demo`: feedforward magic conveyor)

`magic_conveyor` chains `R` fresh `w`-qubit mini-IQP rounds (`H; feedforward;
T's; CZ; H; M`), each measured out so the active block collapses (`k -> 0`)
between rounds. Classical feedforward (`CX rec[-k]`) links them into ONE
connected computation, so a naive up-front `|T>^(R*w)` decomposition pays for
every T (`gSR = 0.228 * R*w`), while the per-episode magic stays `mblk = w`.

8. **The composition beats naive global stab-rank, unboundedly and flatly.**
   `conveyor_rR_w24`: as `R` grows 4→8→16, `Ttot` and `gSR` grow linearly
   (`2^22 → 2^44 → 2^88`) while `dense` and `rSR` stay **flat** (`2^10`, `2^5.5`).
   The composition cost is independent of round count; naive global stab-rank
   grows without bound. This is the measurement-driven-reduction signature.

9. **A circuit where the composition is the UNIQUE feasible method.**
   `conveyor_r12_w128` (n=1536): per-round `k=49`, `Ttot=1536`, `mblk=128`.
   - `dense = 2^49` — **infeasible** (past the dense wall; clifft cannot run it).
   - `gSR   = 2^351` — **absurd** (naive global stab-rank on 1536 T's).
   - `rSR   = 2^29` — **feasible** (residual stab-rank on each round's magic-sparse
     `2^49` block, which holds only 128 T's → rank ~`2^29`, ~5e8 terms, within
     BGH's demonstrated reach).
   The composition is the only method of the three that runs. **This is the
   circuit that demonstrates the composition advantage.**

### The fairness caveat (read this before celebrating)

The demo beats *naive* (measurement-unaware, up-front) stabilizer rank. It does
**not** beat a *measurement-aware* stab-rank simulator on the exponent: such a
simulator also projects on each round's measurements and drops its rank back to
`~2^(0.228*mblk)`, matching `rSR`. So the **exponent** advantage shown here is
against the naive baseline, not the best one.

clifft's genuine, unique edge over the *best* stab-rank baseline is **not at the
exponent level** — it is in constant/poly factors this profiler does not measure:

- **Free Clifford evolution.** Each conveyor round has ~`w` H + ~`w` CZ +
  feedforward ≈ `2w+` Clifford gates. A stab-rank simulator pays `O(chi * k^2)`
  per Clifford gate (`~2^29 * 128^2` here), times every Clifford gate, times
  every round. clifft absorbs all of them into the symbolic frame for ~free.
  For Clifford-DOMINATED circuits (the whole near-Clifford premise) this is the
  decisive saving.
- **Exact, automatic measurement collapse** (a dense array halving) versus
  stab-rank's expensive, lossy re-compression (sparsification) to realize the
  same rank drop.

So the composition is genuinely valuable, but its advantage over state-of-the-art
stab-rank is a **constant/poly factor (Clifford handling), not an exponent.**
Proving *that* needs an operation-count / wall-clock comparison of
(clifft-frame + residual stab-rank) vs (pure stab-rank) — counting the Clifford
gates each pays for — which is the next experiment, not this exponent profiler.

### Honest conclusion for the bet

The compelling regime is **real and reachable**: magic-sparse circuits (IQP /
low-T-density) land at `k = 46–60` with `sat = 2.0`, past clifft's dense wall,
where stab-rank converts an infeasible `2^60` into a feasible `~2^27`. For those
circuits clifft today OOMs (or refuses, at `k>=63`); a stab-rank backend would
run them. **This revives the idea** — for *magic-sparse* circuits, at the cost
of clifft's exactness (approximate sampling).

**But one caveat sharpens the scope of the *novel* part.** On these IQP circuits
`gSR == rSR`: they are single-episode (no `k->0` collapse), so `mblk ≈ Ttot` and
the win comes from **plain (whole-circuit) stabilizer rank**, *not* from the
"residual after clifft's reduction" composition specifically. The composition's
unique value (`residual << global`) only appears when measurement-driven
collapse shrinks the per-block T-count below the total — and the circuits that do
that (surface, the main-suite hidden shifts) are either magic-saturated or
collapse to tiny `k` where dense is already fine. **So the plain large-`k`/
low-`sat` stab-rank win is confirmed; the unique advantage of the novel
*composition* is not yet demonstrated** and would need a circuit that *both*
collapses episodes *and* leaves a magic-sparse residual at `k>30`.

## Limitations

- `mblk` overcounts T's measured out mid-episode (episodes reset only at
  `k = 0`), so `rSR` is conservative (pessimistic) — fine for a go/no-go.
- The true stabilizer rank of the block is not computed; `rSR` uses the extent
  estimate capped at the dimension. Confirming actual rank needs a real
  decomposition (the next step if the bet proceeds).
- Fused `OP_ARRAY_U2`/`U4` provenance is unknown; treated as magic by default
  (`--no-fused-magic` to flip). They were rare in this suite.
- The `surface_*` and `magic_injection_*` generators are crude stand-ins; their
  magic is fully absorbed (`k = 0`), itself a finding (terminal-measured,
  phase-only T's cost nothing) but not a stress test.
