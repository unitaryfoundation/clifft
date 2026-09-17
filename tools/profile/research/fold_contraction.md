# Fixed contraction plans for fold-check overlaps

**Continue toward a compiled code-boundary prototype.** The expensive overlap
operation in the coherent-branch reference can be replaced, for this fold
family, by a small tensor contraction with a fixed schedule. At d7 the largest
joint table has 512 entries; a native diagnostic computes both logical-input
phase sums in about 31 microseconds, with 45,616 bytes of coefficient storage.
No runtime tableau evolution, rank finding, graph planning, or allocation is
needed inside this kernel.

This is an overlap-kernel result, **not a complete cultivation backend or an
end-to-end speedup**. In particular, d7 here means the 85-data-qubit regular-code
fold core. We still do not have the authors' complete Reg5/f7 benchmark circuits,
and have not reconstructed full f7 injection/growth/cat verification. The prior
[complete f5 reconstruction](fold_composition.md) supplies composition evidence,
but its noisy boundaries have not yet been connected to this native kernel.

## Representation and derivation

The paper motivation is the fold structure in Appendix A of
[Takada, Bartlett, and Williamson](https://arxiv.org/html/2609.16929v1), and the
large cultivation workloads discussed in the
[MPS/CAMPS comparison](https://arxiv.org/html/2609.19116v1). The construction below
is our research derivation, not a performance claim from either paper.

Let A contain independent X-stabilizer supports of the regular surface code,
and let l be a physical representative of logical X. The computational logical
basis with zero syndrome is

```text
|b_L> = 2^(-r/2) sum_g |A g xor b l>,    b in {0,1}, g in {0,1}^r.
```

Here r is 6, 20, or 42 for d3, d5, or d7. These are binary variables labeling
products of stabilizer generators, not additional physical or subsystem gauge
qubits. Each physical bit depends on at most two such variables.

For a fixed computational cat string and physical Pauli-fault history, a
completed fold branch is a monomial Clifford operator:

```text
C |z> = omega^p(z) |z xor f>,                  omega = exp(i*pi/4)
p(z) = c + sum_q a_q z_q + 4 sum_(q,t in E) z_q z_t   (mod 8).
```

The linear coefficients are even; E is a subset of the known reflection pairs.
Faults change coefficients, flips, and which pair terms are present, but cannot
introduce a new pair outside that family. Products of these operators stay in
the same family. `compose` retains the constant and linear phase changes caused
by substituting the earlier branch's flips into the later branch's polynomial.

Between ideal zero-syndrome code boundaries, a flip mask with a nonzero
Z-stabilizer syndrome gives a zero matrix. Otherwise it determines the output
logical bit through its commutation with logical Z. The remaining nonzero matrix
element is

```text
<out_L|C|b_L> = 2^(-r) sum_g omega^p(A g xor b l).
```

The bra's uniform positive amplitudes provide this expression; X-syndrome
selection is contained in the phase sum. A physical linear phase becomes a
factor on at most two generator variables. A mirror-pair phase becomes a factor
on at most four. The factor graph includes every allowed pair, even if its
coefficient is zero for a particular history.

A deterministic min-fill planner chooses an elimination order once and emits
all gather addresses. Runtime fills the factors from phase parameters, then
multiplies and sums according to those fixed addresses. There is no per-history
contraction-order search, Gaussian elimination, SVD, or truncation. Intermediate
factor entries use ordinary double-precision complex arithmetic. Initial
Clifford factors are exactly in {1, i, -1, -i}; a possible eighth-root global
phase is applied at the end.

Two consecutive checks must be composed before their next actual syndrome
boundary. Applying a code projection after each would incorrectly discard
components that leave the code space and later return. The implementation and
tests explicitly preserve this distinction. A sum of two branches per check
gives up to four monomial products for a consecutive pair; the measured timing
below is for **one monomial product**, not that entire coherent sum.

## Measured kernel costs

| Regular code | Physical data | Generator variables | Largest joint table | Coefficient bytes | Gather-address bytes | Median time for both input sums |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d3 | 13 | 6 | 16 | 1,584 | 1,536 | 0.52 us |
| d5 | 41 | 20 | 64 | 9,648 | 17,168 | 4.87 us |
| d7 | 85 | 42 | 512 | 45,616 | 122,512 | 30.62 us |

Largest joint table includes the variable being eliminated; the largest
resulting factor has half as many entries. Coefficient storage holds all leaves
and intermediate outputs without lifetime reuse. Address tables can be shared
across shots. These two byte columns exclude leaf parity maps, container
metadata, and benchmark fixtures; they are not total process memory.

Each timing is the median of five trials, each evaluating 97 varied cases
300 times. Both logical-input sums are recomputed every time, including factor
initialization, even when flip routing would reject a branch. Results feed a
printed checksum. There is no memoization by fault history. File loading,
planning, validation, and allocations are outside timing. This is a scalar,
unpinned EPYC VM run, GCC 13.3, Release/O3/native instructions, with fast-math
disabled for this standalone diagnostic. There is no SIMD or threading study.
Raw trials and plan hashes are in `fold_contraction_data.json`.

Excluded costs include sampling faults and outcomes, converting fault bits to
phase parameters, composing operators, coherent branch weights, routing flips,
noisy syndrome/growth handling, and final logical observables. The Python
fixture builder still uses the prior offline per-history operator construction.
Consequently these times must not be divided into the earlier whole-circuit
Clifft timings to advertise a speedup or best-in-class throughput.

## Validation

At each distance, 97 cases cover identity, 32 core branches (including ideal
and synthetic physical Pauli faults), 32 products of consecutive core
branches, and 32 independently randomized admissible phase polynomials. The
fault fixtures are a stress ensemble, not a physical error-rate estimate.

Stim independently synthesizes each complete CSS logical basis state from
its stabilizers. Cirq's phase-preserving CH representation then evaluates every
matrix entry, including complex global phase. The tests check both the routed
code matrix and the raw phase sums, so rejected flip masks cannot hide an
incorrect contraction. All 291 cases agree within the 2e-12 absolute test
tolerance. Small-code cases also match Qiskit dense evolution. Monomial
composition is checked on every computational input for selected d3 cases.
Unsupported widths, non-Clifford linear phases, and pairs outside the compiled
family decline before execution. A separate test catches spurious projections
between consecutive checks.

The native executable validates its input ranges and gather addresses before
execution and checks all outputs against the Python contraction. Maximum
observed discrepancy is 1.58e-16. The full offline research suite passes 38
tests. No production simulator API or execution path changes in this experiment.

## What this changes about the proposed backend

This favors a **small logical state plus coherent fold operators between
certified code boundaries**, using static tensor contractions to reduce each
block back to its logical action. It supplies a concrete answer to one concern
about the prior stabilizer-branch proposal: overlaps need not require dynamic
tableau analysis on every history in this family.

The compile-once question has a positive answer for this kernel. A plan can be
shared across fault histories and consecutive fold products because their
allowed factor graph is unchanged. Per-shot work supplies phase parameters to
the existing plan. This does not settle compilation policy
for arbitrary circuits or approve runtime topology planning elsewhere in Clifft.

The applicability check can be explicit: a recognized CSS code boundary,
transported logical basis, and branch polynomials confined to a compiler-known
pair graph, with an affordable planned contraction width. Unknown structures
should fall back before execution. Conversion of a live generic Clifft residual
into this representation remains a separate integration problem. This result
does not imply that arbitrary circuits have low-width contractions.

The next experiment should connect the complete f5 reconstruction to a reusable
block plan: precompute Clifford propagation of physical fault bits through cat
preparation, growth, and syndrome extraction; derive the actual syndrome-sector
offsets and relative phases; and evaluate each two-check block at its real
measurement boundary. A Pauli-frame representation of nonzero syndrome sectors
looks compatible with the polynomial family, but its complete noisy-boundary
contract still needs derivation and validation. Compare acceptance and logical
X/Y/Z against the existing coherent reference on natural faults, paired hooks,
and accepted logical faults before measuring full many-shot throughput.

Keep full f7 growth and the authors' exact Reg5/f7 artifacts as explicit gaps.
Five-round coherent memory remains a generic control; one-round memory and
few-shot startup gains remain outside the target regime.

## Reproduction

Use the optional offline reference environment described in
`fold_composition.md`; Cirq is required for the large-code independent tests.

```sh
python tools/profile/fold_contraction.py --output /tmp/clifft-fold-contraction
cmake --build build-research --target profile_fold_contraction -j4
build-research/profile_fold_contraction /tmp/clifft-fold-contraction/d3.txt 300
build-research/profile_fold_contraction /tmp/clifft-fold-contraction/d5.txt 300
build-research/profile_fold_contraction /tmp/clifft-fold-contraction/d7.txt 300
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_*.py'
```

The build directory must have `CLIFFT_BUILD_PROFILER=ON`. The versioned text
plan is generated data for this standalone diagnostic, not a supported Clifft
serialization format.
