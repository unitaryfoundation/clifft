# Fixed Clifford frames inside large fold checks

This study screens the physical states *inside* the reconstructed f7 fold checks
without allocating their dense 99-qubit state or Clifft's width-44 residual.
It uses exact stabilizer algebra to establish sufficient MPS and sparse-state
storage bounds. It is a representation feasibility study, not a sampler,
production frame policy, or throughput comparison with compiled logical blocks.

**Outcome: retain compiled logical blocks as the leading implementation.** A
single reusable frame gives small certified storage bounds on ordinary sampled
f7 histories. Legal multiple-hook histories produce real large entanglement in
that same frame, and the conservative affine support needed to cover all tested
histories grows back to 44 directions. These findings support investigating
frames that absorb known fault-dependent Clifford corrections; they do not
justify implementing a general fixed-frame MPS replacement yet.

## Motivation and experiment

[Hartweg and Pineiro Orioli](https://arxiv.org/html/2609.19116v1) show that Clifford
frame choice can make tensor representations useful on difficult cultivation
circuits. Their CAMPS method uses local disentangling; this experiment instead
tests a small set of offline, code-informed frames. It does not reproduce their
algorithm or their circuit artifact. The earlier
[five-round memory screen](alternatives.md) remains a separate negative control.

One ideal trajectory selects a single frame and qubit order for both final
fold checks. The candidates are the physical frame, an inverse preparation
synthesized from canonical stabilizers, and the inverse tableau carried by one
stabilizer branch through ideal Clifford propagation and postselection. Each is
tested in row order and fold-paired order. Selection minimizes the largest MPS
coefficient upper bound across the ideal checkpoints. The chosen frame is then
frozen for every held-out fault history and every sampled point in both checks.
No held-out state influences its selection.

There are 13 checkpoints per check: ten native-operation boundaries, including
inside a T/CX/T_DAG triple, plus three points inside the T/CX decomposition of a
middle CCZ. The all-zero and all-one cat components remain coherent. Up to 16
stabilizer terms are retained at the decomposed checkpoints. No term or singular
value is dropped approximately. The checkpoint state is conditioned on the
preceding prescribed records, not on eventual survival of the full circuit.

## What is certified

For a pure stabilizer state on n qubits, restrict its n independent binary
stabilizer rows to a cut A. Its Schmidt rank is
`2**(rank_GF2(restricted_rows) - len(A))`; this is the standard stabilizer
entanglement calculation described by
[Fattal et al.](https://arxiv.org/abs/quant-ph/0406168).
For a coherent sum of stabilizer states, matrix-rank subadditivity bounds its
Schmidt rank by the sum of those individual ranks, capped by the Hilbert-space
dimension at that cut. Relative coefficients and phases cannot invalidate that
upper bound. They can make the true rank substantially smaller.

The reported MPS storage bound is `sum(2 * chi_left * chi_right)` using these
bond bounds. It counts complex tensor entries, excluding workspaces and frame
storage. This establishes that an exact MPS of at most that size exists at a
checkpoint. It neither constructs that MPS nor measures gate application or
compression cost. A larger upper bound cannot establish a lower bound on the
memory required by another implementation.

An independent affine-support diagnostic restricts each transformed stabilizer
term to its computational-basis support. The rank of its X-generator rows gives
the support size; a deterministic allowed bitstring gives the affine offset.
Taking the span of all term supports and their relative offsets bounds the
whole coherent state's sparse and dense-in-a-subspace storage. Common Pauli
translations are factored out when comparing spans across histories. Thus a
history-wide bit flip alone does not spuriously look like a new active direction.

These calculations use integer binary linear algebra, not SVD tolerances, on
the large states. The CH reference preserves relative phases when obtaining the
states and validating the circuit. Small independent Aer/dense-SVD tests check
the phase expansions, frames, affine bounds, and entanglement calculation.

## Results on the large states

The f7 run considers 62 histories: the ideal training history, 32 natural
p=0.001 histories, eight p=0.01 stress histories, ten single paired-hook
diagnostics, five multiple-hook diagnostics, two nonzero growth-sector cases,
and four selected growth-fault cases. Twenty-eight reach the final checks,
yielding 702 nonempty checkpoints. Ten of the natural histories reach the first
f7 check; two reject before the second. All eight random stress histories reject
earlier. That screening effect is why the deliberately injected multiple-hook
strata are necessary. These are not importance-weighted logical-error estimates.

| Ideal f7 frame and order | Maximum bond upper bound | Maximum MPS coefficient upper bound |
| --- | ---: | ---: |
| Physical frame, row order | 2,560 | 475,184,808 |
| Canonical stabilizer preparation inverse, row order | 1,032 | 13,667,240 |
| Transported inverse tableau, row order | 96 | 479,016 |

The transported frame in row order wins among the six candidates. Fold-paired
ordering is worse here; it is the reconstruction's fold-core labeling, not the
optimized MPS layout from the paper. Training-state extraction takes 1.72 s,
constructing the three frame candidates 0.79 ms, and scoring the six candidates
4.93 s in this Python diagnostic. These timings were collected after tests and
builds finished. No per-shot frame-selection cost is proposed or measured.

At native-operation checkpoints, excluding the extra CCZ-decomposition points,
the ideal maximum upper bond bound is 34. Including those points raises it to
96. Both bounds hold unchanged on the ten natural histories that reach f7 and
the two tested nonzero growth sectors. Single paired-hook histories raise the
corresponding maxima to 68 and 192. Their largest sufficient tensor size is
1,821,096 complex coefficients, versus 479,016 for the ideal checkpoints.

| Multiple-hook diagnostic | Physical Pauli fault events | Maximum sampled bond upper bound | Largest per-history affine cover dimension |
| --- | ---: | ---: | ---: |
| 2 selected controlled pairs | 4 | 320 | 18 |
| 4 selected controlled pairs | 6 | 640 | 22 |
| 8 selected controlled pairs | 16 | 1,280 | 30 |
| 16 selected controlled pairs | 24 | 10,240 | 40 |
| 24 selected controlled pairs | 22 | 9,216 | 36 |

Each selected pair gets cat X faults before and after its CCZ interaction.
Coincident X faults cancel, so the physical event count is not twice the pair
count. Existing fault-layout encoding verifies that every remaining fault is at
a supported physical noise site. The positions are seeded and held out from
frame selection. These are stress diagnostics, not representative draws at
p=0.001. Large numerical storage bounds in this table are never allocated.

The f5 control covers 25 histories, with 23 reaching its checks and 585
checkpoints. It selects the same type of frame and row order. Its maximum ideal
bond upper bound is 40; natural and single-hook maxima are 96, and the
multiple-hook maximum is 512. This control uses the complete two-check
cultivation sequence, not a one-round memory benchmark.

## The difficult states really are entangled

To distinguish real growth from a loose coherent-sum bound,
`fold_frame_certificate.py` intersects the *signed* stabilizer groups shared by
every term. Across a cut, half the rank of the restricted commutation matrix
counts Bell pairs forced by those shared constraints. Any nonzero state obeying
them has Schmidt rank at least `2**bell_pairs`. This is a lower bound on the
coherent state itself, not on its decomposition. Nonzero checkpoint norms are
checked, and dense small-state tests independently verify the calculation.

At the end of the first f7 check, before its cat decode and later syndrome
projection, the bounds on the maximum bond dimension in the fixed frame are:

| History | Certified lower bound | Coherent-sum upper bound |
| --- | ---: | ---: |
| Ideal | 1 | 4 |
| One paired hook | 2 | 8 |
| 2 selected controlled pairs | 4 | 16 |
| 4 selected controlled pairs | 16 | 64 |
| 8 selected controlled pairs | 64 | 256 |
| 16 selected controlled pairs | 1,024 | 4,096 |
| 24 selected controlled pairs | 512 | 2,048 |

The lower bound 1,024 rules out a uniformly small bond such as 96 or 192 in this
particular fixed frame over the tested fault histories. It does not rule out
another frame, a frame depending on the faults, or a better general CAMPS policy.

## Why a small dense subspace is not an immediate substitute

In the selected frame, the ideal and sampled natural states have affine covers
of at most 14 directions and sparse-support upper bounds of 8,194 coefficients.
Single paired hooks raise these to 16 directions and 32,776 coefficients.
But the covers differ across shots: their conservative union, after factoring
out common bit translations, reaches 23 directions already for single hooks
and 44 when the multiple-hook strata are included. This is a union of the
decomposition-derived covers, not a proof that interference cannot reduce the
true support further. It does show that reusing the small ideal cover is not
certified by this method. Retaining each shot's smaller cover through runtime
basis discovery would also violate the current execution invariant.

In addition, conjugated phase generators in the selected f7 frame reach weight
13 and span 94 chain positions. Low tensor storage alone does not establish
inexpensive updates. A native exact evaluator, including frame transitions and
the existing survivor output contract, would be needed for a throughput claim.

## Validation and next step

The full existing-plus-new suite passed 101 tests. The final focused suite
passed all 11 frame tests, including three added after that full run: legal
multiple-hook histories against the logical-block reference, signed group
intersection, and coherent-state entanglement lower bounds. This gives 104
distinct passing research tests. Aer checks unpaired T/T_DAG phases, CCZ
decomposition, partial noisy folds, and frame application. Dense SVD checks
single-stabilizer ranks exactly and brackets coherent sums. Full f3/f7 snapshot
walks preserve the independent CH reference's acceptance and logical probes.

The next useful frame experiment is to remove the known common Clifford hook
correction using descriptors computed offline, then repeat the intermediate-state
bounds and a bounded native update benchmark. This would test whether a simple
fault-conditioned frame can retain the ordinary-history compactness without
runtime tableau evolution or basis discovery. It overlaps the structure already
retained by logical blocks; it needs to demonstrate a benefit in composition or
coverage before becoming a second executor. Generic adaptive CAMPS remains
unmeasured. No production architecture or public API changed in this study.

Raw f7 results and lower-bound certificates are in
[fold_frames_f7_data.json](fold_frames_f7_data.json); the smaller control is in
[fold_frames_f5_data.json](fold_frames_f5_data.json).

## Limits and decision criteria

The screen samples intermediate points; its maxima are not certified maxima
over every elementary gate in a complete trajectory. It includes three internal
points of one CCZ per check, not every intermediate lowered gate. The upper
bounds also depend on the chosen stabilizer decomposition: splitting a CCZ into
T gates can leave more terms even after its final gate, while representing the
same physical state as a shorter whole-gadget decomposition. Neither that term
count nor the upper bond bound is an intrinsic rank of the state.

The frames act on physical states. They are not exports of the precise gauge
used by current Clifft's generic residual. Comparisons with planned width 44
are consequently opportunity indicators, not measured active-width reductions
in Clifft. The state extraction still uses the certified fold decomposition as
an offline oracle; this does not prove a generic compiler can discover the
frame or maintain it cheaply on arbitrary circuits.

Changed frame coordinates also change operation support. A useful small tensor
must be paired with inexpensive updates, measurements, and transitions into and
out of the frame. No per-shot tableau evolution, basis discovery, or topology
planning has been added to Clifft. Such diagnostics are offline research only.
The previous logical-block sampler remains the measured fast implementation.

## Reproduction

Use the existing research environment with Stim, Cirq, NumPy, and Qiskit Aer:

```sh
OPENBLAS_NUM_THREADS=1 python tools/profile/study_fold_frames.py --distance 7 --natural 32 --stress 8 --output /tmp/fold-frames-f7.json
OPENBLAS_NUM_THREADS=1 python tools/profile/study_fold_frames.py --distance 5 --natural 8 --stress 0 --output /tmp/fold-frames-f5.json
OPENBLAS_NUM_THREADS=1 python tools/profile/fold_frame_certificate.py --study /tmp/fold-frames-f7.json --output /tmp/fold-frames-lower.json
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_fold_frames.py'
```

The worker is Python-only and stays outside production execution. No large
dense state is allocated, and no approximate tensor truncation is performed.
One-round memory circuits are not part of this experiment.
