# Actual MSC gadget coverage

The original d3/d5 MSC benchmark inputs contain a second useful physical gadget
family for compiled logical blocks. Their color-code cultivation checks reduce
to two Clifford monomials, using the same operator representation as the fold
study. This bounded result supports continuing the logical-block direction.
It does not yet provide a sampler for either complete original MSC circuit.

## Corpus and scope

The unmodified, licensed inputs are now checked in under `../fixtures/msc/`,
with source commit, original paths and SHA-256 digests in its README. These
are the actual benchmark controls, not variants of our f7 reconstruction.

| Input | Certified regions | T gates inside these regions | Remaining T gates |
| --- | ---: | ---: | ---: |
| MSC d3 | 2 | 28 / 29 | 1 injection gate |
| MSC d5 | 3 | 90 / 91 | 1 injection gate |

The regions are the d3 cultivation check, the d5 cultivation check where
present, and the final conjugated logical measurement. The largest region has
19 data wires plus 19 ancillas and 38 physical T/T-dagger gates. Two terms
describe its fixed-outcome instrument; no dense 38-qubit state is allocated.

These percentages describe T-gate coverage, not complete circuit coverage or
speedup. d5 remains a correctness/composition control. The reconstructed f7
remains the large-state target. There is still no high-bond memory improvement.

## Why two terms suffice

The cultivation gadget has the form: local diagonal T layer D, CNOT network U,
one root X measurement, reset of that root to |+>, inverse CNOT network, and
inverse T layer. The reset must be included. For true root outcome m its central
Kraus operator is

```
|+><m_X| = (I + X) Z^m / 2.
```

This gives two Pauli branches inside the CNOT network. The final logical test
similarly has central projector `(I + (-1)^m Y_product) / 2`.

For any fixed Pauli fault history, each branch remains a classical affine bit
permutation with phases. The paired CNOT networks cancel the linear permutation;
only bit flips remain. The paired T layers then leave even linear phase
coefficients modulo eight. Each branch is therefore a Clifford monomial.
The diagnostic certifies this cancellation from the supplied gates, rather
than relying on distance labels or a stored operator catalog. It retains
relative phases between the two terms.

Root Pauli faults between measurement and reset are erased, up to a common
trajectory phase. Spectator faults there remain. Readout noise flips the
reported record and must not be confused with the true quantum outcome.
Ancillas prepared in |+> and read in X contract directly: each branch either
matches the requested record with a known sign or contributes zero. This
leaves at most two data monomials, each with weight 1/2.

## Code-boundary evidence

The actual terminal checks define independent commuting CSS generators of
rank 6 on seven data qubits and rank 18 on nineteen data qubits. Stim checks
their consistency and independence. Sparse CSS basis enumeration uses 8 and
512 entries per logical basis vector, respectively.

In the all-data logical-X convention, the terminal check restricts to

```
H_L = [[0, exp(+i*pi/4)], [exp(-i*pi/4), 0]].
```

Its restriction is unitary to within 1.3e-14, establishing preservation of
these ideal code spaces. For the corresponding ideal cultivation check,
contracting the ancillas and applying its outcome-dependent physical Z frame
gives exactly `(I + (-1)^m H_L) / 2`. Outcome one has a nonzero ancilla record
and a single data-Z correction; it cannot simply be discarded as an error.
Each corrected branch is separately checked for code preservation, so a
projected logical matrix is not mistaken for proof of zero leakage.

This certifies ideal local boundaries only. It does not certify the incoming
state after noisy injection or growth, nor every noisy outgoing syndrome sector.

## Independent checks

Seven focused tests cover:

- 18,924 single-Pauli histories over every active wire at every elementary-gate
  boundary across all five actual regions. These include diagnostic locations
  inside bundled source instructions, beyond the physical noise locations.
- 160 twelve-fault histories plus the five ideal histories. Both true outcomes
  and four computational columns are checked for every history: 152,712 column
  comparisons against an independent sparse elementary-gate interpreter,
  with tolerance 4e-12. A single common phase must work across all four columns.
- 16 complete d3-region instrument comparisons against Qiskit Aer, using
  arbitrary entangled inputs and plus-prepared ancillas, with and without
  multiple faults. Projection and reset are applied explicitly, rather than
  reusing the two-branch rewrite. Twenty-four ancilla-record contractions are
  additionally compared to the resulting full states, at tolerance 4e-12.
- Held-out wire permutations, rejection of broken inverse networks, unpaired
  T layers, wrong reset wires and intervening unsupported gates, and both
  ideal code-boundary outcome sectors.

The d5 tests avoid exponential state storage by independently following each
input column through the original gates. They do not constitute a dense
exhaustive check of all 2^38 columns. The algebraic certificate establishes the
operator form; the independent column tests exercise its implementation.

Reproduction:

```
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p test_msc_gadgets.py
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_gadgets.py \
  --output tools/profile/research/msc_gadgets_data.json
```

The raw report contains source hashes, exact region locations, code matrices,
boundary records, correction wires, and numerical errors. No throughput
measurement or native executor extension is claimed here.

## Next bounded milestone

Build a complete fixed-history evaluator for the original d3 MSC input, then
the original d5 input, retaining actual records, detector parity, observable
parity, feedforward and physical noise locations. Certify the intervening
Clifford syndrome and growth maps and their nonzero sectors from the input
gates. The lone injection T can use a small exact boundary treatment. Compare
complete output distributions and selected fault histories independently
before measuring native sustained throughput.

The diagnostic reader is intentionally not a production eligibility checker:
it retains noise and record annotations as source text but does not bind their
semantics or probabilities. It does not validate the whole extended dialect.
Both full circuits still decline the existing fold adapter. A production path
would need a complete parsed-circuit certificate, static fault maps and
contraction plans, and cheap fallback before execution. The current Python
affine propagation is offline research work, not an authorized hot executor.

The locally available original generator was also inspected. Its cultivation
entry points in `cultiv/_construction/_integration.py` explicitly support only
dcolor=3 and 5; other distances raise `NotImplementedError`. Its source is
recorded locally as Strilanc/magic-state-cultivation commit
`871e68ff6df2f75190b1bfd6351459d1b5a037e3`. This does not supply an authors'
MSC7 artifact. A larger generated color-code core would be a labeled synthetic
scaling test, not a replacement for the missing full protocol.
