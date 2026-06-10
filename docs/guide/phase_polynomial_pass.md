<!--pytest-codeblocks:skipfile-->

# Global T-Count Optimization

Clifft's default `PeepholeFusionPass` cancels and fuses T gates only along local
forward scans. `PhasePolynomialPass` is an experimental opt-in HIR pass that adds
**global** T-count reduction: bounded multiplicative-commutator-relation (MCR)
reordering, then size-capped TOHPE duplicate-and-destroy on commuting T blocks.

The pass operates entirely on existing Heisenberg-IR Pauli masks. It does not
change the parser, bytecode format, or VM. It is disabled by default.

See the [compilation guide](compilation.md) for the standard pipeline.

## Overview

The workflow has three layers:

1. **Baseline** — `PeepholeFusionPass` fuses identical adjacent T axes locally.
2. **Global pass** — `PhasePolynomialPass` runs bounded MCR reordering, then
   size-capped TOHPE on commuting T blocks.
3. **Cleanup** — a second peephole sweep fuses any newly exposed same-axis pairs.

Per-phase passes (`McrTcountPass`, `TohpePhasePass`) exist so evaluation can
attribute savings to MCR vs TOHPE independently.

## Summary

| Phase | Pass | Technique | When it helps |
|---|---|---|---|
| Baseline | `PeepholeFusionPass` | Local same-axis cancel/fuse | Always (default) |
| 1 | `McrTcountPass` | Bounded MCR reordering | Structured T sandwiches separated by Clifford gates |
| 2 | `TohpePhasePass` | TOHPE on commuting blocks | Large commuting T clusters with duplicate parities |
| Full | `PhasePolynomialPass` | MCR then TOHPE | Both effects; run between peephole sweeps |

The evaluation below compares peephole-only compilation against the global pass
on 34 circuits. MCR is the main win on small structured regression circuits.
TOHPE can remove T gates on specific commuting-heavy artifacts (for example a
surface-code magic-state layer). On typical Qiskit-transpiled arithmetic and
chemistry circuits, peephole already reaches the minimum T count in this
prototype.

## Why this is valid on Clifft HIR

After `parse()` and `trace()`, each `T_GATE` carries Pauli generator masks
(`destab_mask`, `stab_mask`) in the Heisenberg frame. Clifford gates conjugate
these masks during compilation; the optimizer reasons about commutation and
fusion with symplectic GF(2) bit operations, not explicit unitaries.

A **commuting T block** is a contiguous run of `T_GATE` ops where every pair
commutes. Inside such a block, T gates contribute a **phase polynomial** over
GF(2): each gate is one parity column. Valid optimizations are column reorderings
or GF(2) transforms that preserve the polynomial modulo Clifford corrections.

The pass is sound because:

- **MCR rewrites** use `can_swap`, which respects measurement and classical
  dataflow barriers, and keep only rewrites that strictly reduce T count after
  fusion.
- **TOHPE rewrites** apply only inside commuting blocks; destroyed pairs cancel;
  net Clifford residuals fold into the frame via `apply_virtual_s_downstream`.
- **Equivalence** is checked with statevector tests where T count changes; T
  count never increases on the full evaluation corpus.

### Phase 1: MCR reordering

Four T gates on Pauli axes $P,Q,R,S$ satisfy the multiplicative commutator
relation when $T_P T_Q T_R T_S = (-1) I$ (same T direction). In GF(2) this
requires the Pauli product to be identity and the total phase mod 4 to be 2.

The implementation searches bounded contiguous T windows for reorderable
quadruples that expose same-axis pairs for peephole fusion.

### Phase 2: TOHPE

On each commuting block the pass builds a parity table from HIR masks and applies
TOHPE duplicate-and-destroy: find a GF(2) column transform that merges and
cancels a pair while satisfying the TOHPE constraint system, solved here by
GF(2) RREF.

### Prototype limits

- TOHPE is skipped when `num_qubits > 32` (single `uint64_t` column encoding).
- TOHPE block size is capped at 48 gates.
- MCR search: lookahead $\leq 16$ T gates, window span $\leq 64$ Clifford ops.

Clifft builds on its existing `PeepholeFusionPass`, `commutation`, and
`t_fusion` modules. Phase 2 implements TOHPE directly on HIR masks rather than
a separate virtual-Clifford extract.

## Usage

```python
import clifft

pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())
pm.add(clifft.PhasePolynomialPass())
pm.add(clifft.PeepholeFusionPass())
pm.run(hir)
```

To measure each phase in isolation, keep the peephole bookends:

```python
pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())
pm.add(clifft.McrTcountPass())   # or TohpePhasePass()
pm.add(clifft.PeepholeFusionPass())
pm.run(hir)
```

## Evaluation

### What we measure

Every circuit is compiled with the same pipeline shape:

```
PeepholeFusionPass  →  [optional phase pass]  →  PeepholeFusionPass  →  count T
```

Four variants are compared:

| Variant | Middle pass | Question it answers |
|---|---|---|
| Baseline | none | How many T gates does default peephole leave? |
| MCR-only | `McrTcountPass` | How much does phase 1 alone save? |
| TOHPE-only | `TohpePhasePass` | How much does phase 2 alone save? |
| Full | `PhasePolynomialPass` | How much does MCR + TOHPE together save? |

All T counts are **final** gate counts after both peephole sweeps. A positive
`saved` value means the pass removed that many T gates relative to the baseline.

### Circuit corpus (34 total)

| Group | Count | Examples |
|---|---:|---|
| MCR regression | 9 | `toggle_sandwich`, `kicked_xy_block`, pair-block sandwiches |
| Synthesis | 5 | QFT layers, Toffoli chains |
| Arithmetic | 6 | Ripple-carry adders, multiplier, comparator |
| Algorithms | 3 | Grover oracle, CCX/MCX, modular-exp fragment |
| Chemistry | 3 | H2, Ising, Heisenberg Trotter steps |
| Compiled stress | 7 | Factored Clifford+T (see [benchmark guide](benchmark.md)), dense random Clifford+T |
| Surface code | 1 | [`circuit_d3_t_gate_p0.001.stim`](circuits/circuit_d3_t_gate_p0.001.stim) |

Unit tests use the small MCR regression set. The broader corpus above answers
how the pass behaves on workloads Clifft is likely to see beyond toy examples.

The harness prints all 34 circuits. The table below shows eight **selected**
circuits that summarize the pattern.

### Before/after T counts (selected circuits)

| Circuit | Kind | peephole_T | full_T | saved | mcr_saved | tohpe_saved |
|---|---|---:|---:|---:|---:|---:|
| `toggle_sandwich` | MCR regression | 6 | 2 | 4 | 4 | 0 |
| `kicked_xy_block` | MCR regression | 5 | 3 | 2 | 2 | 0 |
| `two_disjoint_pair_blocks` | MCR regression | 10 | 6 | 4 | 4 | 0 |
| `rc_adder_cdkm_4bit` | Arithmetic | 32 | 32 | 0 | 0 | 0 |
| `rc_adder_vbe_4bit` | Arithmetic | 32 | 32 | 0 | 0 | 0 |
| `ccz_mcx_3q` | Algorithm | 7 | 7 | 0 | 0 | 0 |
| `random_ct_6q_d150` | Random Clifford+T | 12 | 12 | 0 | 0 | 0 |
| `surface_d3_t_gate` | Surface code | 29 | 29 | 0 | 0 | 0 |

How to read one row:

- `peephole_T` — baseline T count after one peephole sweep.
- `full_T` — T count after `PhasePolynomialPass` between two peephole sweeps.
- `saved` — `peephole_T - full_T`.
- `mcr_saved` — T removed by `McrTcountPass` alone vs baseline.
- `tohpe_saved` — T removed by `TohpePhasePass` alone vs baseline.

On the sandwich rows, `saved == mcr_saved` and `tohpe_saved == 0`: MCR reorders
T gates so peephole can fuse them; TOHPE adds nothing. On the adder, random, and
surface-code rows, all three savings are zero: peephole already optimal in this
prototype.

### Real-world example: magic-state cultivation (in corpus, no extra gain)

The distance-3 T-gate magic-state cultivation circuit from the SOFT benchmark suite
([`circuit_d3_t_gate_p0.001.stim`](circuits/circuit_d3_t_gate_p0.001.stim)) is
included as a realistic near-Clifford fault-tolerant workload (15 qubits, 29 T
gates, noise, detectors — see the
[importance-sampling tutorial](importance-sampling.md)).

| Stage | T count | What happened |
|---|---:|---|
| After peephole only | 29 | Baseline after local fusion |
| After `TohpePhasePass` | 29 | No additional reduction |
| After full `PhasePolynomialPass` | 29 | MCR also inactive on this input |

TOHPE only runs on contiguous commuting T blocks separated by HIR barriers
(measurements, noise, phase rotations, etc.). On this circuit the surviving
blocks do not expose duplicate parities under the current size caps, so the pass
matches peephole. Including it shows the evaluation covers real FTQC inputs even
when the prototype does not beat the baseline.

### Full corpus totals

| Corpus | Circuits | T (peephole) | MCR saved | TOHPE saved | Full saved |
|---|---:|--:|--:|--:|--:|
| MCR regression | 9 | 60 | 16 (26.7%) | 0 | 16 (26.7%) |
| Real-world / algorithmic | 25 | 69,300 | 0 | 0 | 0 |
| Combined | 34 | 69,360 | 16 (0.02%) | 0 | 16 (0.02%) |

These totals aggregate every circuit in the harness. They are benchmark points
from the current prototype limits, not universal guarantees.

## Interpreting the results

### Where the pass helps

**MCR regression circuits** (`toggle_sandwich`, `kicked_xy_block`, pair blocks):
peephole scans forward only; T gates sit on different Pauli axes behind Clifford
sandwiches. MCR swaps quadruples that satisfy the commutator relation so
same-axis pairs become adjacent and fuse. This is the only regime where the
prototype currently beats peephole on the evaluation corpus.

### Where the pass does not help (in this prototype)

Transpiled arithmetic, chemistry Trotter steps, factored Clifford+T circuits,
dense random Clifford+T tests, and the surface-code magic-state circuit all
show zero additional savings. Peephole plus Clifft's front-end already reach
the minimum T count within the current MCR window and TOHPE size caps.

TOHPE blocks are bounded by the same HIR barriers as MCR windows (measurements,
noise, phase rotations, etc.). Optimizations never merge T gates across those
seams.

### Prototype verdict

The pass is semantically sound and worth keeping as an opt-in optimizer for
structured MCR-family compilation artifacts. It is **not** yet justified as a
default pipeline pass: real-world benchmarks show no gain beyond peephole in this
prototype, and TOHPE must respect HIR barriers to preserve unitary semantics.
Follow-up work would lift the 32-qubit TOHPE cap, widen MCR windows, and compare
against standalone T-optimizer baselines on larger benchmark suites.

## Reproducing evaluation

The full 34-circuit corpus, per-phase breakdown, and optional statevector checks
are produced by the evaluation harness:

```bash
uv run python docs/guide/scripts/phase_poly_evaluation.py
uv run python docs/guide/scripts/phase_poly_evaluation.py --check-equiv
```

Exact T counts depend on the Clifft version and Qiskit transpiler seeds baked
into the harness. The qualitative split in the tables above is stable across
re-runs. Use `--check-equiv` for statevector checks when T count changes and
qubit count $\leq 8$.

## References

The MCR and TOHPE phases follow published T-count optimization methods. MCR-style
regression circuits follow the op-T-mize commuting-block benchmark family.

- Amy, Mosca, Roetteler (2018): phase-polynomial synthesis —
  [Quantum Sci. Technol. 4, 015006](https://doi.org/10.1088/2058-9565/aad604)
- de Meijer, Duca, Duncan (2021): multiplicative commutator relations —
  [arXiv:2109.06445](https://arxiv.org/abs/2109.06445)
- Vandaele et al. (2024): TOHPE algorithm —
  [arXiv:2402.18347](https://arxiv.org/abs/2402.18347)
- op-T-mize benchmark family:
  [PennyLane blog](https://pennylane.ai/blog/2025/01/optimizing-with-op-t-mize-dataset)
- Evaluation harness:
  [`docs/guide/scripts/phase_poly_evaluation.py`](scripts/phase_poly_evaluation.py)
