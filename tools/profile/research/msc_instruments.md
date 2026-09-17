# Weighted instruments at the original MSC boundaries

The signed boundary certificates now extend to complete conditional weights and
logical density matrices. Each Clifford interval compiles to a commuting input
projector, record consistency constraints, a power-of-two normalization factor,
and three logical Pauli pullbacks. These describe its action on an arbitrary
incoming coherent state, including interference between gadget terms.

The compiled rule matches all 146 boundary observations in the existing 95
complete d3/d5 histories. The contraction used for this validation still uses
coherent CH states. This is a mathematical specification and an offline
correctness result, not a complete static sampler or new native timing result.

## The weighted update

Let K be the Kraus operator of the interval for fixed physical Pauli faults and
specified true outcomes, including the hidden measurements used to unravel
resets. Compilation produces commuting positive input Paulis A_j, affine signs
s_j, and a nonnegative integer r. When the record constraints hold,

```
P = product_j (I + (-1)^s_j A_j) / 2
K dagger K = 2^-r P.
```

Otherwise K is zero. For an incoming state psi, the joint branch weight is
`2^-r ||P psi||^2`. Divide by `||psi||^2` for the probability conditional on
reaching the interval. Projector contraction must sum coherent terms before
taking the squared norm; treating the gadget branches as a mixture is wrong.

For each output logical axis L_a, compilation also produces an input Pauli B_a
and an affine sign t_a such that

```
K dagger L_a K = 2^-r (-1)^t_a P B_a.
```

The B_a commute with P. Their three normalized expectations give the outgoing
logical Bloch vector. At the previously certified boundaries the actual pure
trajectory has one logical qubit and separated spectators, so this determines
two amplitudes up to a common phase. The study explicitly reconstructs the
weighted ket and compares its density matrix, including the relative phase
seen by logical Y. The earlier output syndrome/frame maps remain responsible
for the physical Pauli frame; their corrections commute with the logical axes.

A common phase can be discarded only after contracting the complete coherent
input for this conditional branch. This calculation does not discard separate
phases of incoming monomial terms.

| Interval | Quantum outcomes including resets | Independent input projectors | Record-only constraints | r |
| --- | ---: | ---: | ---: | ---: |
| d3 injection to first check | 20 | 14 | 3 | 3 |
| d5 injection to first check | 20 | 14 | 3 | 3 |
| d5 growth and syndrome extraction | 155 | 40 | 97 | 18 |

Growth's 40 input projectors factor into six multi-wire code checks on the
incoming seven data qubits and 34 single-wire checks. The inherited spectator
on qubit 5 is not projected by this interval; its state is already certified
by the earlier boundary study. Thus the 42-wire physical interval does not
require a 42-wire dense contraction. This factorization is useful evidence
for a small implementation, but its runtime cost has not been measured.

The r values are normalization factors conditional on the input projector,
not unconditional acceptance probabilities. There can be additional zero
weight because the input state is orthogonal to the selected projector even
when all record-only constraints hold. Hidden reset outcomes also contribute
to these fixed-history probabilities; this is not yet a sampler marginalized
over hidden outcomes.

## Compilation and composition

`msc_instruments.py` expands each reset into a measurement and its conditional
Pauli, adjusts original feedforward record indices, and obtains a complete
stabilizer-flow basis from Stim. Gaussian elimination on output Paulis yields
all input-only flows. Eliminating their input Paulis separates the independent
input projectors from record-only constraints. Reducing the desired output
logical axes against the same flow basis gives the input pullbacks.

For m measurements, k independent input projectors and c independent
record-only constraints, the normalization is `r = m - k - c`. On a maximally
mixed input, each admissible record has probability `2^-(m-c)`, whereas the
input projector has normalized trace `2^-k`. Their ratio fixes the scalar.
Exposing resets is essential to this pure-Kraus argument.

Symbolic physical fault frames propagate once during compilation. A true
measurement outcome shifts by the anticommuting fault parity; a reported
feedforward bit additionally shifts by its readout error. Reset feedback uses
the true outcome. Compiler-produced masks include both effects and the final
Pauli-frame sign. Binding a history evaluates fixed parity rows only. It does
not construct a tableau, discover dependencies, or perform elimination.

The bounded compiler declines non-Clifford intervals, feedback from outside
the interval, and an output logical axis with no input pullback. It is not a
general circuit eligibility API. The CH projector contraction is explicitly
separate from this compiler and remains an offline oracle, so this work does
not add runtime topology planning to the production executor.

## Validation

- All 95 previously recorded full histories, observed at 146 boundaries, match
  the original full-history evaluator. Maximum relative branch-weight error is
  2.23e-16, logical Bloch error 1.12e-16, and weighted density-matrix error
  8.72e-19. This checks complete coherent contraction, not just membership.
- Stim checks the entire input-projector, record-constraint and logical-flow
  basis after independently materializing every legal physical fault direction:
  479,800 signed flows across 4,982 ideal/fault-basis histories. Hidden reset
  measurements are explicit, and original readout-dependent controls are kept.
- Independent dense elementary Kraus operators on two qubits check 448 inputs:
  arbitrary complex superpositions and computational basis states, every
  outcome string, and all combinations of physical X, physical Z and readout
  faults. Both orthogonal input projectors and inconsistent repeated records
  exercise zero probabilities. All three unnormalized logical expectations
  and the branch norm are compared.
- Four new tests also cover the actual interval ranks/full histories,
  inconsistent record rejection with Stim disabled at binding time, and
  decline cases. The combined MSC suite has 26 tests.

The full-history oracle is the prior coherent-stabilizer evaluator. That earlier
study compared d3 with elementary-gate Aer and d5 with elementary-gate Clifft.
This turn adds independent dense small-instrument and Stim flow checks; it does
not close the existing full-d5 external quantum-state reference gap.

Reproduction:

```
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_instruments.py \
  --output tools/profile/research/msc_instruments_data.json
CLIFFT_MSC_RECORD_PROBE=/tmp/msc-record-probe \
  MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p 'test_msc_*.py'
```

## Next bounded step

Replace the CH projector contraction with fixed contractions of the small
monomial sum against the code basis and the single-wire ancilla states. Start
with growth: only six input code checks involve multiple wires, and the
incoming data block has seven qubits. Preserve fault-dependent ancilla signs
and coherent relative phases. Use this weighted-instrument result as the
oracle, including impossible branches, instead of re-deriving the syndrome
schedule in runtime code.

Then compose injection, cultivation, growth and terminal measurements into a
complete static evaluator and compare the full record distribution before
native timing. The original f7 reconstruction remains the large-state
performance target; actual d3/d5 remain composition controls. Neither a new
MSC7 circuit nor an authors' full Reg5/f7 artifact has been obtained.
