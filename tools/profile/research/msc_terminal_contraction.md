# Coherent terminal contraction on the original MSC code blocks

The original MSC final cultivation and terminal logical measurement now contract
as one coherent block, followed by the circuit's actual final code projection.
The calculation returns two weighted logical amplitudes and retains the final
measured syndrome as a physical Pauli frame convention. It applies to the
seven-data-qubit d3 control and nineteen-data-qubit d5 control.

The nineteen-qubit implementation constructs CSS supports directly from the
checks: 512 entries per logical basis state, 1,024 decoder entries total.
Eighteen precomputed syndrome corrections select any of the 262,144 sectors
from this one basis. Compilation does not build a dense nineteen-qubit state
vector or operator, and contraction does not evolve a tableau.

This extends the direct logical-block calculation to the larger final code
support. It is not yet a complete static sampler or a native performance result.

## Keep the two instruments coherent

The final cultivation instrument contributes two monomials of weight 1/2. Its
ancilla measurements contribute scalar overlaps, and intervening physical Pauli
faults update the data monomials and product spectator states. The terminal
T-conjugated logical measurement contributes another two monomials. Composing
them leaves four coherent terms of weight 1/4 before ancilla-overlap factors.

There is no extra code projection between these operations. Each of the four
terms acts on the incoming logical code state and Pauli frame. Only after the
terms have been composed does the contraction select the final measured code
sector and sum the two output amplitudes. Global phases common to a complete
trajectory can be omitted; relative phases between terms cannot.

A synthetic regression demonstrates the distinction: applying S on one data
wire followed by its inverse has probability one at the final positive code
projection. Inserting a positive code projection between them changes the
joint probability to 1/4. This is a counterexample within the supported
monomial family, not a claim that the original circuit is that two-gate example.

The four-term payload retains the true root and terminal outcomes, the hidden
root-reset constraint, physical faults inside and between gadgets, and the
ancilla projection amplitudes. Readout errors remain in the reported record.
There is no record-controlled quantum gate between the two final gadgets in
these fixtures; unsupported intervening gates are declined by the offline
payload builder.

## Sparse code projection

Construction verifies positive CSS checks, one logical degree of freedom, and
the all-data logical X/Z convention. It enumerates the span of the X checks to
obtain |0_L>; applying all-data X gives |1_L>. The sparse decoder maps each
supported computational string to its logical index. There is no dense
2^19-entry coefficient array in the contraction.

Syndrome duals restore the measured check signs while commuting with logical
X and Z. For each of the four monomials, evaluation visits the 1,024 source
entries, applies the incoming frame, monomial phases/flips, and inverse final
syndrome frame, then looks up the positive-code overlap. Thus a d5 contraction
visits 4,096 source entries. This is an operation count for the data sum, not a
runtime or complete memory-use measurement.

All spectators must have the same final product state across coherent terms,
up to scalar phases. Their relative phases are included in the sum. A differing
spectator state declines instead of silently tracing out entanglement. The
actual final ancilla measurements satisfy this condition for the histories
checked. The Python research kernel still allocates small arrays and validates
its inputs; it is not an allocation-free native executor.

## Composition and validation

The isolated terminal calculation matches all 95 previously stored complete
histories: 44 d3 and 51 d5. Both controls include all four pairs of final
cultivation/terminal outcomes. Eleven d3 histories and 36 d5 histories have
nonzero final code syndromes. Maximum relative probability error is 4.16e-14;
maximum density-matrix error normalized by branch probability is 4.08e-14.
These comparisons retain the full logical phase, not only acceptance weights.

For d5, the study additionally feeds the previously implemented growth
contraction's weighted amplitudes directly into this terminal contraction.
It compares the resulting joint probability and final density matrix with the
complete evaluator, starting at the first certified boundary. All 51 d5
histories match, with maximum relative probability error 4.17e-14 and normalized
density-matrix error 4.08e-14. No fresh logical state is extracted from the reference between growth and the final block.
The initial state and fault-bound gadget payloads are still supplied by offline
bridges, as described below.

Independent numerical checks include:

- Dense projector calculations for all 64 d3 syndrome sectors and 51 d5 sectors
  (zero, each of the eighteen individual syndrome directions, and 32 random
  combinations). Every tested synthetic case has nonzero weight. Inputs include
  arbitrary complex logical amplitudes and four coherent diagonal/flip terms.
  The dense reference uses Stim stabilizer preparation for the source state
  and applies all full-vector Pauli projectors, without the sparse decoder or
  its syndrome corrections.
- Two further nineteen-qubit four-term cases checked with Qiskit Aer: eight
  independent statevector evolutions before the dense final projectors. An
  explicit scalar unitary retains each term's global phase after Aer initial
  state assignment; otherwise that test harness would erase relative phases.
- Algebraic checks that every syndrome dual flips exactly its own check and
  commutes with both logical axes. This checks the frame map for arbitrary
  combinations without enumerating 262,144 sector bases.
- Destructive interference, spectator-relative phases, differing-spectator
  rejection, the intermediate-projection counterexample, impossible root reset
  rejection, and execution with Stim/Pauli analysis disabled.

Six new tests bring the combined MSC suite to 37 tests. The dense nineteen-qubit
reference is test-only: it does allocate full statevectors, unlike the sparse
contraction. The Aer comparison covers the data-block monomial evolution, not
the complete 42-wire d5 protocol. The full-d5 external quantum-state reference
gap remains; prior complete d5 comparisons used elementary-gate Clifft.

## What is still missing

The study harness obtains the first logical input from the offline coherent
reference and uses the existing offline gadget binder to turn each fixed
physical-fault history into monomials. The new growth-to-terminal handoff
removes the intermediate state extraction, but it does not replace those
initial/binding bridges. Sampling measurement records, rather than evaluating
specified histories, also remains to be composed into the static evaluator.

The next bounded step is the injection prefix: derive its first weighted
logical state directly from the original circuit, including reset outcomes and
noise, and connect it to the validated growth and terminal contractions. Then
replace the offline gadget fault binding with compiler-produced fixed maps and
validate complete record sampling before native timing. Do not introduce
runtime tableau planning in doing so.

Reconstructed f7 remains the large-state performance target. The original d3/d5
controls add a different physical family and composition coverage; they are
not substitutes for an authors' MSC7/Reg5/f7 artifact or evidence of a measured
speedup against those papers.

Reproduction:

```
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_terminal_contraction.py \
  --output tools/profile/research/msc_terminal_contraction_data.json
CLIFFT_MSC_RECORD_PROBE=/tmp/msc-record-probe \
  MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p 'test_msc_*.py'
```
