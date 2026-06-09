# Experimental Global T-Count Pass (Issue #40)

Prototype opt-in HIR passes exploring whether **global** T-count reduction is
feasible on Clifft's Pauli-string Heisenberg IR without changing the parser,
bytecode, or VM.

## Theory

Clifft's front-end absorbs Clifford gates into `U_C` and represents
non-Clifford effects as `T_GATE` ops on virtual Pauli axes. A commuting block
of such ops is simultaneously diagonalizable and forms a **phase polynomial**
over GF(2). The `(x, z)` masks of each `T_GATE` supply the gate-synthesis
matrix directly; Clifford overhead is already in the frame, so the HIR is a
natural substrate for phase-polynomial optimization.

This prototype implements two complementary techniques from the op-T-mize /
T-count literature:

1. **MCR reordering** — bounded multiplicative-commutator-relation swaps on
   contiguous T windows expose same-axis fusion opportunities that
   `PeepholeFusionPass` cannot reach with local forward scans alone.
2. **TODD (size-capped)** — on commuting clusters within T windows, build the
   GF(2) parity table and apply a capped TODD subset (Lempel-X2 style from
   TOpt) to reduce odd-parity terms, re-emitting fewer `T_GATE` ops and
   folding even-parity Clifford residuals into the frame via virtual S
   absorption.

### References

- [Optimizing with the op-T-mize dataset](https://pennylane.ai/blog/2025/01/optimizing-with-op-t-mize-dataset)
- Selinger et al., phase-polynomial / TODD methods for T-count optimization
- Maintainer direction in [issue #40](https://github.com/unitaryfoundation/clifft/issues/40)

## Usage

Not in the default pipeline. Intended sequence:

```python
import clifft

pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())
pm.add(clifft.GlobalTcountPass())
pm.add(clifft.PeepholeFusionPass())
pm.run(hir)
```

Per-phase evaluation:

```python
pm.add(clifft.McrReorderPass())   # MCR only
pm.add(clifft.ToddPhasePass())    # TODD only
```

## Evaluation

```bash
uv run python tools/eval/run_global_tcount_eval.py
uv run python tools/eval/run_global_tcount_eval.py --check-equivalence
```

The harness reports baseline, peephole-only, MCR-only, TODD-only, and full-pass
T counts on MCR-style and synthesis-style benchmark circuits.

## Conclusion (prototype)

The pass reduces T count on structured MCR benchmarks and some synthesis
fragments while preserving statevector semantics on tested circuits. Production
follow-up would need larger benchmarks, configurable size caps, and maintainer
review of TODD correctness bounds on mixed commuting/non-commuting windows.
