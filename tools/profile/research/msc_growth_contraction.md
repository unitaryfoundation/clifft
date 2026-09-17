# Sparse logical-block contraction through original MSC growth

The original d5 circuit's first cultivation gadget and growth interval now
contract directly from two logical amplitudes to two weighted output amplitudes.
The contraction uses sparse code entries, monomial phases, and scalar ancilla
overlaps. It does not evolve a tableau, project a coherent CH state, or discover
quantum dependencies during evaluation.

All 51 stored complete d5 histories match the full-history evaluator. The largest
relative probability error is 4.78e-15 and the largest density-matrix error,
normalized by branch probability, is 2.61e-15. This is a correctness and
representation result. No native performance measurement or complete static
sampler is claimed.

## What is represented

The input at the first certified boundary consists of two logical amplitudes,
a seven-data-qubit Pauli frame, and known single-qubit spectator states. The
first cultivation gadget produces two coherent terms of weight 1/2. Each term
is a diagonal/bit-flip monomial on the data and a product of single-qubit states
on the other wires. Relative phases between terms are retained.

The prior weighted-instrument compiler describes growth as six signed code
projectors on these seven data qubits, 34 single-wire projectors, three logical
Pauli pullbacks, record constraints, and a probability scale of 2^-18. It also
certifies that the output is one encoded qubit on nineteen data wires, with
separated spectators. The earlier boundary plan supplies its physical output
syndrome and Pauli frame.

Qubit 5 is an inherited spectator that growth does not project. The new
contraction requires its state to agree across the coherent terms up to a
scalar phase, and retains that relative phase. If the unprojected spectator
states differ physically, it declines instead of silently tracing out
entanglement and pretending the logical state is pure.

## The direct calculation

Compilation fixes a positive input code basis in the logical convention given
by the growth pullbacks. It also computes six Pauli corrections that select any
of the 64 input syndrome sectors while commuting with both logical axes. Thus
all sectors share one basis; there is no table of independently compiled sector
states.

The source code has eight computational-basis entries in each logical basis
state. For each coherent term, evaluation:

1. Multiplies the 34 single-qubit projection overlaps and the inherited
   spectator's relative scalar phase.
2. Visits the 16 source code entries, applying the input Pauli frame and the
   supplied monomial's flips and phases.
3. Applies the selected syndrome correction and looks up the overlap with the
   fixed two-state input code basis.
4. Adds the resulting two amplitudes coherently across terms, applies the
   signed logical X/Z transport, and multiplies amplitudes by 2^-9.

The final factor is the square root of growth's probability scale; the gadget's
own probability is already present in its coherent sum. The output norm squared
is the probability of the supplied gadget-and-growth history, conditional on
reaching the first boundary. Both hidden reset outcomes and true versus reported
measurement bits retain the conventions of the preceding instrument study.

The decoder table is 2 by 128 complex doubles, or 4 KiB. Only 16 entries are
nonzero. With the original gadget's two terms, evaluation visits 32 source code
entries; the 42 physical wires do not become a dense amplitude state. Python
object overhead, fault binding and the rest of the protocol are not included
in the 4 KiB figure.

This is a bounded seven-wire kernel. Its compiler uses small dense Stim state
vectors to extract sparse basis entries; that implementation should not simply
be extended to larger widths. A larger code should construct its CSS support
and phase data directly. The syndrome-dual method itself does not require
enumerating all syndrome sectors.

## Validation and coverage

The study replays all 51 existing d5 histories, including natural and elevated
noise, root measurement/reset faults, spectator and pair faults, feedforward
readout flips, and the logical-Z tail. It extracts the two-amplitude input at
the first certified boundary and compares the direct gadget-and-growth result
with the full evaluator immediately before the second cultivation gadget.
Fourteen histories have nonzero input frames; the histories cover fourteen
input syndrome sectors. Each comparison checks the full weighted logical
density matrix, including relative phase, as well as probability.

Independent seven-qubit dense projector calculations cover all 64 input
syndromes and all four output logical Pauli corrections: 256 comparisons.
These use arbitrary complex logical amplitudes, Pauli frames, diagonal/flip
monomials, coherent weights and ancilla states. The reference constructs the
source CSS support independently from the terminal check generators and applies
the six full data-projector matrices. It does not use the sparse decoder or its
syndrome corrections.

Additional tests cover exact destructive interference, a relative phase on the
unmeasured spectator, an orthogonal ancilla outcome, inconsistent record
rejection, execution with Stim and basis construction disabled, and conservative
decline for a coupled data/ancilla input projector or differing unprojected
spectator states. Five new tests bring the combined MSC suite to 31 tests.

The full-history reference and the independent small dense calculations have
the same external-validation limits as the prior study: original d3 had an
Aer comparison, while full original d5 was compared with elementary-gate
Clifft. This does not supply an authors' MSC7/Reg5/f7 circuit or close the
full-d5 external quantum-state reference gap.

## Remaining boundaries of this result

`GrowthContraction.contract` is the new fixed contraction. The study harness
still extracts its starting logical state from the offline coherent evaluator
and builds the two gadget terms using the existing offline fixed-fault gadget
binder. Those activities are not a completed production state handoff or a
static native noise-binding path. The Python contraction itself allocates
small arrays and performs validation; a native hot executor would need
preallocated storage and construction-time validation under the existing
architecture rules.

The next useful coverage step is the nineteen-data-qubit final block: retain
the coherent terms through both final cultivation and terminal logical
measurement, and contract only at the actual final code projection. There is
no justified intermediate code projection between those operations. That
will exercise the larger 512-entry logical basis supports while reusing the
same code/ancilla reasoning. Injection and static gadget fault binding must
then be composed before this becomes a complete static evaluator and before
native timing is meaningful.

The reconstructed f7 remains the large-state performance target. These actual
d5 circuits remain composition controls; this result is not a small-shot
startup optimization or evidence of a speedup over another simulator.

Reproduction:

```
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_growth_contraction.py \
  --output tools/profile/research/msc_growth_contraction_data.json
CLIFFT_MSC_RECORD_PROBE=/tmp/msc-record-probe \
  MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p 'test_msc_*.py'
```
