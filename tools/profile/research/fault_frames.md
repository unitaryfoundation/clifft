# Compiled Clifford fault frames

This bounded follow-up removes the known fault-dependent Clifford correction
before applying the fixed frame from [the large-fold study](fold_frames.md).
It gives a positive representation result: all 540 tested native-operation f7
checkpoints have a corrected bond upper bound at most 34, and their combined
conservative affine cover has at most 14 directions. The multiple-hook states
that defeated the fixed frame return exactly to the ideal states.

**Keep compiled logical blocks as the leading complete sampler.** The new
candidate is a **compiled Clifford fault frame**: a small exact correction around
an ideal region, not a general adaptive CAMPS implementation. Its native
bookkeeping is inexpensive, but no tensor executor, complete measurement path,
general Clifft state handoff, or end-to-end speedup has been demonstrated.

## Representation and exact scope

For a region made of CNOT, T, T_DAG, and CCZ gates, interspersed with physical
Pauli faults, write its noisy unitary as

```text
U_faults = E(faults) U_ideal
E |b> = omega**[g + sum(l[q]*b[q]) + 4*sum(e[a,b]*b[a]*b[b])] |b xor x>
omega = exp(i*pi/4), l[q] even modulo 8, e[a,b] binary.
```

The correction contains bit flips, S/Z phases, and CZ edges. It is shared by the
whole coherent state, including both cat branches; it is not chosen separately
for branches or fitted to a trajectory's amplitudes. The identity is valid on
arbitrary input states, including states entangled with spectator qubits.
An incoming correction of the same form is also supported when its possible
edges are included in the compiled input contract. This is narrower than an
arbitrary Clifford frame: its basis permutation is a bit translation, not a
general linear reversible circuit.

There is a simple induction. At a fault, left-multiply E by the Pauli. At an
ideal gate G, replace E by G E G^-1. T conjugating a bit flip adds a linear
Clifford phase. CCZ conjugating bit flips adds only quadratic and lower phases;
its cubic term cancels. CNOT substitutes one binary coordinate into the
quadratic Clifford phase, preserving this representation. For example, cat X
faults surrounding a CCZ leave an unconditional data CZ. Multiple such hooks
can be retained as bits in E instead of entanglement in the residual state.

The compiler fixes every possible quadratic slot and CNOT source/destination
coefficient pair. Execution changes only values in those slots. It performs no
tableau evolution, commutation analysis, localization, support discovery, or
basis selection. The standalone native kernel preallocates all storage and has
no allocations or exceptions in its update loop. For this f7 core there are 121
edge slots, 235 actions, and 320 fault bits. The descriptor frame occupies 576
bytes in the probe, including unused fixed capacity. Building both Python core
plans took about 1 ms; this excludes the pre-existing protocol certification
and fixed-frame selection costs.

Encoded Y uses the existing XZ convention. The physical Y differs by a global
i per fault, which has no effect on the fixed-history probabilities or storage
bounds. The coherent-state inverse application preserves the descriptor's
global and relative phases; the paired-hook exact-equality checks use only X
faults, so no global-phase adjustment is needed.

## Large-state experiment

The experiment reuses exactly the 62 histories and the ideal-trained transported
frame from the prior study: ideal, 32 natural p=0.001 draws, eight stress draws,
ten single paired hooks, five multiple-hook cases, two growth-sector cases, and
four selected growth faults. Twenty-eight histories reach the f7 checks.

This follow-up measures the ten native-operation checkpoints per reached core,
including the interior T/CX/T_DAG points. It excludes the three artificial
CCZ-lowering checkpoints per core from the earlier screen. The 540 checkpoints
are therefore a subset of the previous 702; 34 must be compared with the
earlier *native-operation* bounds, not with the ideal bound 96 that included
lowered CCZ points. These are sampled checkpoint bounds, not maxima certified
over every gate of every trajectory.

Corrections start at the first f7 core. Faults in earlier injection, growth,
syndrome, or cat preparation stay in the incoming physical state. If the
outgoing correction is data-only and all intervening instructions touch only
ancillas, it commutes through the actual intervening cat decoder and preparation
and is carried into the second core. Otherwise the diagnostic leaves that
correction in the physical input and starts a fresh local correction in the
second core; this is bookkeeping in the offline oracle, not a claimed native
fallback or state conversion.

| Quantity across sampled native checkpoints | Corrected result |
| --- | ---: |
| Maximum coherent-sum bond upper bound | 34 |
| Maximum sufficient MPS coefficient count | 36,904 |
| Maximum per-history affine cover dimension | 14 |
| Maximum union of covers at any checkpoint | 14 |
| Maximum sparse coefficient upper bound | 8,194 |

The previous uncorrected union reached 44 directions at a native checkpoint.
Here the corrected union matches the ideal dimension at every sampled native
checkpoint. This is a conservative decomposition-derived cover after removing
common translations, not a proof of the minimal active width of Clifft's
residual, nor a certificate for all physical histories.

At the end of the first f7 core, the multiple-hook upper bounds change as follows:

| Selected hooks | Previous upper bound | Corrected upper bound |
| --- | ---: | ---: |
| 2 | 16 | 4 |
| 4 | 64 | 4 |
| 8 | 256 | 4 |
| 16 | 4,096 | 4 |
| 24 | 2,048 | 4 |

The 16-hook case previously had a signed-stabilizer lower certificate of 1,024.
There is no contradiction: removing E changes the Clifford frame. The stronger
check here is exact coherent equality with the ideal physical state after
correction, at all 300 native checkpoints of the fifteen single/multiple-hook
histories. Canonical stabilizer terms and exact cyclotomic coefficients cancel
identically in the state difference; this is not an SVD-tolerance comparison.

No dense 99-qubit or width-44 state is allocated. Offline CH/Stim inverse-frame
application is used only to validate and profile the representation. The native
timing does not include or propose executing that oracle per shot.

## The boundary limitation

The current cheap sufficient boundary rule requires E to act only on data and
the intervening instructions to act only on ancillas. Native evaluation uses
precomputed masks of forbidden edges and qubits. It does not inspect the state.
All paired-hook examples pass. One of the original natural histories fails at
the first core and rejects before the second core in the physical reference.

Additional core-only coverage includes all 480 single-qubit X/Y/Z placements
at the 160 qubit-locations after ideal core gates, 256 unconditional p=0.001
core noise draws, and 128 dense fault-bit patterns. Every pattern has an exact
correction inside the core. The sufficient exit rule accepts 307/480 single
Pauli patterns and 239/256 natural core samples. These are diagnostic counts,
not full-circuit acceptance rates; the single-qubit patterns are not weighted
by the physical multi-qubit depolarizing model. The natural core draws are not
conditioned on surviving earlier cultivation stages.

A data X fault before a CCZ can leave a cat-data CZ in E. That correction
changes the cat X measurement to a joint observable involving data. Dropping
it, treating it as a classical Pauli record flip, or carrying it unchanged
through the cat H would be wrong. Some declined cases, such as isolated cat
Paulis, admit simpler treatment than this sufficient rule, but that treatment
is not implemented here. Generic H gates can similarly take E outside the
chosen representation.

The descriptor can therefore be useful on arbitrary eligible gate regions,
without needing the reconstructed code or cat layout. It does not guarantee
that the ideal region's state is small, or that its entrance and exit are cheap.
Those are additional requirements for a useful simulator.

## Native update probe

`benchmark_fault_frames.py` exports one precomputed f7 core plan and independent
expected descriptors, plus a six-qubit plan with overlapping CNOT/CCZ gates and
nonempty incoming frames. The native probe checks 926 f7 fixtures and 128 generic
fixtures, as well as the existing logical-block branch outputs for all f7
fixtures. It then alternates seven timing repetitions for each evaluator.

The comparison invokes the existing logical-block `Executor::branches` method;
only its access level in the standalone research header changed. That method
includes cat preparation/decode linear maps and returns two weighted monomials.
The frame probe updates E and evaluates the data-only exit rule. Both consume
the same already-sampled core fault histories, but they produce different
representations and do not perform identical complete work. In particular, the
frame probe does not execute the ideal residual or apply the corrected
measurement. Neither timer includes random fault generation, contractions,
survivor outputs, or full-circuit execution. A ratio between these timers is
not a sampler speedup.

After tests and builds finished, seven alternating repetitions of 200,000
updates each gave the following median times with scalar research code built
using `-O3 -march=native`:

| Already-sampled core histories | Frame plus exit check | Existing weighted branch evaluator |
| --- | ---: | ---: |
| 256 natural p=0.001 core samples | 0.925 us | 1.530 us |
| All 926 diagnostic fixtures | 0.957 us | 1.572 us |

Natural-sample repetition ranges were 0.915-0.950 us and 1.510-1.662 us,
respectively. Both return validated outputs; an opaque compiler barrier prevents
unused output fields from being optimized away. The roughly 0.6 us difference
per final core is small compared with the existing approximately 46 us/attempt
full logical-block result, and the new path still needs measurement handling.
The existing full-sampler measurement remains the relevant throughput result.
No best-in-class claim follows from this experiment.

## Validation and decision

Six new Python tests cover full small-state unitaries against Aer at 32 noisy
prefixes with arbitrary incoming Clifford fault frames, sparse versus complete
precompiled edge universes, inverse-frame coherent phases, paired-hook boundary
acceptance, a measurement-changing boundary counterexample, and unsupported
region/input rejection. The native fixture check additionally exercises CNOT
edge substitutions absent from the specific f7 fold ordering. ASan, UBSan, and
LeakSanitizer pass for the standalone native probe.
The full research suite passes all 110 tests in 206.9 seconds. Repository
formatting, lint, type, and hygiene hooks pass before committing.

This establishes a simple explanation and inexpensive descriptor for the
previous fixed-frame failures. It strengthens the case for a fault-conditioned
representation, but much of the same structure is already present in logical
blocks. Building a generic adaptive MPS executor is not the next step.

The next discriminating study is a **compiled boundary bridge**: retain E through
the cat measurement by compiling its action on the ideal region's small input
space, and compare complete attempted-shot cost with the existing two-monomial
fold path. Start with the same full f7 reconstructions and the adversarial
cat-data corrections that the simple exit rule declines. Require the existing
survivor records and logical/physical probe contract, exact phase validation,
and preplanned fallback. If this is only an equivalent rewriting of current
logical blocks with no coverage or throughput advantage, stop the second
executor direction and retain the algebra as an explanation and possible
region-recognition tool. Extending to different practical protocol families
remains necessary before claiming broader usefulness.

The circuits are still paper-guided reconstructions, not authors' artifacts.
The [CAMPS motivation](https://arxiv.org/html/2609.19116v1) is the sensitivity of
tensor cost to Clifford frames; this experiment uses compiled fault propagation
instead of adaptive disentangler search. No production architecture or public
API changed. One-round memory and few-shot startup are not study targets.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 python tools/profile/study_fault_frames.py --output /tmp/fault-frames-f7.json
OPENBLAS_NUM_THREADS=1 python tools/profile/benchmark_fault_frames.py --output /tmp/fault-frame-native.cpp
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic /tmp/fault-frame-native.cpp -o /tmp/fault-frame-native
/tmp/fault-frame-native 200000
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_fold_fault_frame.py'
```

Use the existing research environment with NumPy, Stim, Cirq, and Qiskit Aer.
Large-state bounds and native results are in
[fault_frames_f7_data.json](fault_frames_f7_data.json).
