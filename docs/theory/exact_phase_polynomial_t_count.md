# Exact Phase-Polynomial T-Count Pass

`ExactPhasePolynomialTCountPass` is an experimental, opt-in HIR pass for bounded
T-count minimization. It is intentionally small: it optimizes only contiguous
commuting `T_GATE` blocks whose independent Pauli-axis rank is at most four.

The pass is motivated by the phase-polynomial view of Clifford+T optimization,
where T-count reduction can be framed as changing the odd coefficients of a
phase polynomial while leaving only Clifford residual phase behind. Amy and
Mosca connect T-count minimization with Reed-Muller decoding, and later TODD
and TOHPE work develops scalable heuristic and polynomial-time reductions for
larger phase-polynomial circuits:

- Matthew Amy and Michele Mosca, [T-count optimization and Reed-Muller codes](https://arxiv.org/abs/1601.07363).
- Luke Heyfron and Earl Campbell, [An Efficient Quantum Compiler that reduces T count](https://arxiv.org/abs/1712.01557).
- Vivien Vandaele, [Lower T-count with faster algorithms](https://arxiv.org/abs/2407.08695).

This pass is not a TODD, TOHPE, or MCR implementation. MCR is a separate
multi-Pauli commutation primitive for reordering non-pairwise-commuting
sequential Pauli computations, as described by Mori, Hakoshima, and Fujii in
[Nontrivial multi-product commutation relation toward reducing T-count in
sequential Pauli-based computation](https://arxiv.org/abs/2509.20052). The exact
pass instead stays inside already-commuting HIR blocks and exhaustively searches
the small local phase-polynomial space.

## HIR Model

For a commuting block, the unsigned `(x, z)` masks of the T axes span a vector
space over GF(2). A basis of rank `r` gives coordinates `a` for each Pauli axis,
and a T or T-dagger gate contributes a coefficient in `Z_8`:

```text
T(P_a)      -> +1 * bit(P_a)
T_DAG(P_a)  -> -1 * bit(P_a)
```

Clifft's HIR Pauli masks can carry a sign bit, and products of commuting
unsigned Pauli generators can also produce a negative representative, for
example `XX * ZZ = -YY`. The pass therefore treats each term as affine in the
chosen coordinate basis. If the original axis is the negative of the basis
product, it adds the coefficient to the tracked global phase and flips the
coordinate coefficient. This is necessary for exact `global_weight` preservation.

Concretely, take generators `g0 = XX` and `g1 = ZZ`. Their coordinate product
`g0 * g1` is `-YY`. If the input term is instead `T(+YY)` with coefficient one,
then `bit(+YY) = 1 - bit(-YY)`. The phase contribution is therefore `1 -
bit(-YY)`: the pass records `1` in `constant_phase` and stores coefficient `-1`
on the coordinate parity. The same normalization is applied in reverse when the
pass emits an operation whose coordinate product is negative: it emits the
unsigned Pauli with the flipped coefficient and adds the emitted constant. At
the end, `global_weight` is multiplied by the eighth root for the input affine
constant, any residual constant, and any emitted-axis normalization constant.

## Search And Acceptance

For rank `r <= 4`, the pass builds the exact truth table of the original phase
function over the `2^r` basis assignments. It then enumerates candidate odd
parity representatives with fewer T terms than the original block, trying both
T and T-dagger signs for each selected parity.

A candidate is accepted only if the residual phase function is Clifford. The
residual is converted to algebraic normal form over `Z_8` and must satisfy:

- no nonzero monomials of degree three or higher;
- linear coefficients are even;
- quadratic coefficients are either zero or four.

Accepted odd terms are emitted as `T_GATE` or `T_DAG`; the Clifford residual is
emitted as `PHASE_ROTATION` operations, and any constant residual phase is
absorbed into `global_weight`. If the total emitted HIR operation count would
exceed the original block length, the pass skips the rewrite. The pass is
registered with `default_enabled = false`.

The quadratic residual lowering uses only identities over the coordinate bits of
the commuting Pauli basis, so it is not specific to Z-basis parities. For bits
`x_i` and `x_j`,

```text
4*x_i*x_j = 2*(x_i xor x_j) - 2*x_i - 2*x_j   (mod 8).
```

The implementation therefore emits coefficient `2` on the parity coordinate
`x_i xor x_j` and subtracts `2` from the two linear coefficients. The parity
coordinate maps back to the product Pauli for the two commuting generators; if
that product is signed, the affine sign normalization described above accounts
for the constant shift and coefficient flip.

## Evaluation

The reproducible evaluator is `tools/bench/exact_phasepoly_tcount.py`. It runs
`trace -> PeepholeFusionPass -> ExactPhasePolynomialTCountPass` and prints a
Markdown table. It also includes conservative `.qc` and Clifford+T OpenQASM 2.0
importers for external benchmark files:

```bash
python tools/bench/exact_phasepoly_tcount.py
python tools/bench/exact_phasepoly_tcount.py --qc-dir path/to/qc_corpus
python tools/bench/exact_phasepoly_tcount.py --qasm-dir path/to/qasm_corpus --skip-unsupported
```

The evaluator also has an explicit exposure mode:

```bash
python tools/bench/exact_phasepoly_tcount.py --collect-t-blocks
```

That inserts `TGateBlockCollectionPass` between peephole fusion and the exact
decoder. The collection pass is also opt-in and default-disabled. It only uses
adjacent swaps approved by Clifft's HIR commutation checker, and only pulls a T
gate into a block when it commutes with every T gate already in that block. It
does not change T count by itself; its purpose is to test whether useful
rank-capped blocks are hidden behind safely movable operations. On the current
built-in fixture set, this mode reports zero collected blocks and zero moved T
gates, so it does not change the real-world fixture results below.

The QASM importer accepts declarations and common Clifford+T gate names, ignores
barriers, and rejects measurements, custom gates, and parameterized gates instead
of silently changing circuit semantics.

The positive cases below are unit-test fixtures built directly in HIR so that
the intended phase-polynomial structure is unambiguous.

| Circuit family | T before | T after | Notes |
|---|---:|---:|---|
| Rank-4 Reed-Muller zero word, all 15 nonzero Z parities | 15 | 0 | Exact zero phase function. |
| Rank-4 zero word plus one repeated T parity | 16 | 1 | Removes the 15-term zero word. |
| Rank-4 unsigned product word with affine Pauli signs | 15 | 0 | Emits six Clifford residual rotations and preserves global phase; checked by dense matrix equivalence. |
| Rank-4 signed product zero word plus one signed odd term | 16 | 1 | Normalizes the signed odd term to an unsigned `T_DAG` plus `exp(i*pi/4)` global phase; checked by dense matrix equivalence. |
| Rank-4 zero word separated by commuting phase rotations | 15 | 0 | `TGateBlockCollectionPass` first exposes the block using safe adjacent swaps; the exact pass then removes it; checked by statevector equivalence. |
| Rank-5 all nonzero Z parities | 31 | 31 | Skipped by the rank cap. |
| Noncommuting T block | 3 | 3 | Skipped by the commuting-block guard. |

The pass also has one source-level integration win using gates Clifft already
parses and traces:

| Source circuit | Traced T | After peephole | After exact pass | Notes |
|---|---:|---:|---:|---|
| Complete `CCZ` hypergraph on 4 qubits, `CCZ 012`, `013`, `023`, `123` | 28 | 8 | 7 | Source text is parsed, traced, peephole-optimized, then reduced by this pass. |

Related source-level families show the pass boundary:

| Source family | Result after peephole and exact pass | Reason |
|---|---|---|
| Complete `CCZ` hypergraph on 3 qubits | 7 -> 7 | A single CCZ is already optimal for this pass. |
| Complete `CCZ` hypergraphs on 5, 6, and 7 qubits | 20 -> 20, 20 -> 20, 63 -> 63 | The exact pass is rank-capped at four. |
| `CCX` ladders on 3 through 7 qubits | No reduction beyond peephole | The traced phase structure is sparse for this decoder. |

The pass was also run after `PeepholeFusionPass` on the repository fixture and
tutorial circuits:

| Circuit | Traced T | After peephole | After exact pass | Blocks considered | Blocks optimized |
|---|---:|---:|---:|---:|---:|
| `tests/fixtures/cultivation_d5.stim` | 72 | 72 | 72 | 5 | 0 |
| `tests/fixtures/qv10.stim` | 0 | 0 | 0 | 0 | 0 |
| `tests/fixtures/target_qec.stim` | 0 | 0 | 0 | 0 | 0 |
| `docs/guide/circuits/circuit_d3_s_gate_p0.001.stim` | 0 | 0 | 0 | 0 | 0 |
| `docs/guide/circuits/circuit_d3_t_gate_p0.001.stim` | 29 | 29 | 29 | 4 | 0 |

As an external smoke benchmark, the QASM importer was also run on a local
checkout of Qiskit's `test/benchmarks/qasm` fixtures with unsupported
parameterized/custom-gate files skipped and imported circuits capped at 5000
gate lines:

```bash
python tools/bench/exact_phasepoly_tcount.py \
  --qasm-dir ../qiskit/test/benchmarks/qasm \
  --skip-unsupported --max-imported-gates 5000
```

Supported Clifford+T/QASM cases under that cap produced:

| Circuit | Traced T | After peephole | After exact pass | Blocks optimized |
|---|---:|---:|---:|---:|
| `20QBT_45CYC_.0D1_.1D2_3.qasm` | 0 | 0 | 0 | 0 |
| `53QBT_100CYC_QSE_3.qasm` | 0 | 0 | 0 | 0 |
| `54QBT_25CYC_QSE_3.qasm` | 0 | 0 | 0 | 0 |
| `depth_4gt10-v1_81.qasm` | 63 | 39 | 39 | 0 |
| `depth_4mod5-v0_19.qasm` | 14 | 8 | 7 | 1 |
| `depth_mod8-10_178.qasm` | 147 | 87 | 87 | 0 |
| `time_cnt3-5_179.qasm` | 70 | 60 | 60 | 0 |
| `time_cnt3-5_180.qasm` | 210 | 136 | 136 | 0 |

`hwb12.qasm` imported to 171482 gate lines and was excluded by the gate cap; a
20-second temporary evaluator run did not complete, so it is not counted as
evidence for or against this pass.

This is the expected boundary for the prototype: it can prove exact wins on
dense small commuting phase-polynomial structure and one source-level complete
`CCZ_4` circuit, plus one bounded Qiskit Clifford+T benchmark file, but the
existing Clifft fixtures do not currently expose such rank-capped dense blocks
after tracing and local peephole fusion.

## Reviewer Checklist

- [x] C++ implementation against Clifft HIR, not an external Python-only tool.
- [x] Opt-in pass, default-disabled, with no parser, HIR layout, bytecode, or VM
  changes.
- [x] Clear theory note with citations for phase-polynomial/Reed-Muller,
  TODD/TOHPE context, and MCR contrast.
- [x] Explicitly different from MCR, TODD, and TOHPE PRs: this is a bounded
  exhaustive decoder for already-commuting blocks.
- [x] Tests cover exact rewrites, skipped blocks, rank cap, noncommuting guard,
  source-level `CCZ_4`, affine Pauli signs, global phase preservation, and
  conservative T-block collection.
- [x] Evaluation includes existing repository fixtures and documents null
  results.
- [x] Reproducible benchmark harness can evaluate built-in source families,
  existing fixtures, supported external `.qc` files, and supported Clifford+T
  QASM files.
- [x] Optional benchmark mode can evaluate whether safe T-block collection
  exposes additional exact-decoder opportunities.
- [x] Bounded external Qiskit QASM smoke results are included, with unsupported
  files and the oversized `hwb12.qasm` case documented.
- [ ] Broader real-world benchmark improvements are not demonstrated here. The
  pass should not be presented as satisfying that part of the bounty bar by
  itself.

## Conclusion

This pass is useful as a correctness-focused prototype and a regression target
for HIR-level phase-polynomial reasoning. It shows that Clifft can express
bounded exact T-count reduction without parser, bytecode, VM, or HIR layout
changes. It is not broad enough to be a production optimizer or a complete
bounty solution by itself.

The most natural follow-up is to keep this exact decoder as a verifier or
small-block cleanup, run the optional collection mode over standard
op-T-mize-style circuits, and compare the resulting table against broader
phase-polynomial reducers. The role of those circuits as benchmarks is
summarized in PennyLane's
[op-T-mize dataset history](https://pennylane.ai/blog/2025/01/optimizing-with-op-t-mize-dataset).



AI Use Disclosure: Code syntax was assisted by Codex model. All handwritten code and accompanying text were thoroughly reviewed and verified by me prior to submission. I retain full responsibility for the final contribution.
