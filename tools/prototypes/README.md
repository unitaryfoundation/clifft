# Prepared-state phase-polynomial prototype

`phase_polynomial.py` is an opt-in source rewrite for evaluating a state-aware
optimization before committing to a native HIR pass. It emits ordinary Clifft
circuit instructions and uses the existing compiler, planner and executor.
It does not change the public API or default optimization pipeline.

From the repository root:

```bash
uv run python -m tools.prototypes.phase_polynomial input.stim reduced.stim
```

Or use it before compilation:

```python
import clifft
from tools.prototypes.phase_polynomial import reduce_phase_polynomial

source = "H 0 1\nCX 0 1\nT 1\nCX 0 1\nMPP X0*X1\nMX 0 1"
reduction = reduce_phase_polynomial(source)
passes = clifft.default_hir_pass_manager()
passes.add(clifft.ActiveWidthSchedulePass())
program = clifft.compile(reduction.circuit, hir_passes=passes)
```

The returned `Reduction` includes whether a rewrite was applied, the rejection
reason, initial and retained variable counts, padded deterministic records,
and source/output T counts. When applied, `prepared_state` is the synthesized unitary state
preparation before measurements, for independent statevector checks.

## Supported fragment

The input starts in the simulator's known `|0>` state, prepares independent
variables with initial H gates, and then uses CX/CNOT, SWAP, T/T_DAG, S/S_DAG,
Z and CZ. X-basis measurements and X-product MPP measurements are supported.
Measurements inside the quantum block must be proved deterministic; after the
first unproved measurement, all quantum gates are rejected. Detector and
observable annotations retain their original record references and order.

Noise, noisy readout, feedback, resets, repeat blocks, other measurement axes,
and arbitrary-angle rotations return the entire input unchanged. This version
does not optimize individual regions of a noisy circuit. It also rejects more
than `max_variables` initial H variables (default 32, maximum 64).

This is a prepared-state equivalence, including joint measurement records and
post-measurement physical states, up to global phase. It is not an equivalence
of the full unitary on arbitrary inputs. Physical qubit count is preserved.

## Reduction

1. Track each physical computational-basis bit as a GF(2) linear function of
   the initial H variables. Accumulate the integer phase function modulo eight
   in square-free monomials of degree at most three.
2. At each candidate X measurement, solve for its input-coordinate translation.
   Compute the exact phase difference under that translation. A constant zero
   or four proves the measurement outcome zero or one, respectively. Replace
   that measurement with `MPAD`, preserving its record slot.
3. Retain independent positive check translations that also preserve the final
   phase function. Factor their span and restrict the phase polynomial to a
   complementary set of logical coordinates.
4. Synthesize the smaller phase circuit and a Clifford encoding that recreates
   the original physical state. Apply the original terminal measurements and
   annotations to that state.

All analysis happens before ordinary Clifft compilation. No execution-time
allocation, tableau analysis or topology planning is introduced.

The chosen logical basis is deliberately simple. Resynthesizing a sparse parity
expression as monomials can expand it substantially. By default, the prototype
returns the original input whenever synthesized T count would increase. The
research-only `allow_t_expansion=True` / `--allow-t-expansion` option disables
that guard and can cause large runtime regressions. The guard is a conservative
proxy, not a general guarantee of faster sampling.

## Validation and measurements

Run the independent-oracle and sampling-contract tests:

```bash
uv run pytest tests/python/test_phase_polynomial_prototype.py
```

Run a corpus with the supplied `circuits.csv` format:

```bash
uv run python -m tools.prototypes.bench_phase_polynomial CORPUS OUTPUT \
    --sample-limit 0 --target-seconds 0.02 --repeats 3
```

The harness scans every circuit, verifies output metadata, and writes `scan.json`,
`sampling.json`, `sampling.jsonl` and `summary.json`. Its baseline is the existing
default HIR pipeline plus `ActiveWidthSchedulePass` with default settings.
Sampling uses `sample_survivors`, one CPU thread, batch size one, no retained
records, and the corpus's postselection masks. `--sample-limit 0` benchmarks all
circuits; a positive limit selects a reproducible subset and key examples.
Compilation and source-rewrite costs are reported separately from throughput.

Before a native pass, the remaining work includes choosing better logical bases
for coupled phase functions, replacing the T-count guard with a planner cost
comparison, and defining a representation for noise-dependent Clifford
corrections. This prototype deliberately leaves noisy inputs unchanged.

## Triorthogonal corpus result

The supplied 1,037-circuit corpus was evaluated on an Apple M4 Pro using a
Release Clifft build whose C++ sources match this checkout. Full results and
methodology are summarized in [triorthogonal_results.json](triorthogonal_results.json).

With the default T-count guard, 1,023 circuits were rewritten and 14 were left
unchanged. All accepted rewrites lowered peak active width; the corpus median
fell from 16 to 1. Source/output T counts fell from 159,592 to 2,508.

The full single-thread survivor-sampling sweep measured a median 5,632x speedup
and a 30.3x speedup for equal shots per circuit. No case slowed by more than ten
percent. These are ideal-circuit results against the width scheduler, and they
exclude compilation. Median rewrite-plus-compile time was 7.27 ms, compared
with 3.54 ms for the baseline compiler.

Longer confirmation runs measured approximately 250x on circuit 0005,
3,908x on 0262, 910x on 0689, and 21,598x on 1037. Circuit 1037's peak width
fell from 18 to 1. The three checked fallback cases stayed within three percent
of baseline throughput.

All 95 corpus circuits with at most ten physical qubits were independently
checked against Qiskit Aer statevectors and complete joint record probabilities.
Maximum amplitude error was 1.25e-15 and maximum probability error was 2.23e-16.
The tests also cover randomized supported fragments, collective distinct-parity
cancellation, positive and negative deterministic checks, original record slots
and output annotations, ordinary and survivor sampling, the T-cost guard, and
Pauli-noise fallback against Stim.


## Directional study of the noisy logical core

The next study is in `phase_core_study.py`, with results in
[phase_core_results.json](phase_core_results.json). It goes beyond literal X
translation symmetries: it finds all Pauli stabilizers of the phase state,
including generators with Y and Z factors, and synthesizes a Clifford decoder
around a smaller non-Clifford core. It is offline research code; it does not
change the compiler or implement stochastic sampling.

The exact core size is the stabilizer nullity of the unmeasured prepared pure
state: the variable count minus the rank of its Pauli stabilizer group. This
is the minimum core size under arbitrary Clifford decoding. It is distinct
from Clifft's peak active width, which can be smaller when independent cores
are processed separately. The compression principle and monotonicity under
Pauli measurements are established in
[the stabilizer-nullity literature](https://arxiv.org/abs/1904.01124) and
[Clifford compression](https://doi.org/10.1103/PRXQuantum.6.020324).

All 1,037 circuits were analyzed. Exactly 856 have core size one; 93 have size
two, 34 size three, and 54 size four or larger. The 856 one-core cases are
exactly the cases already synthesized to a single T by the earlier prototype.
There is no additional large group of ideal one-core cases to unlock. One
transversal T layer can implement more than one logical non-Clifford degree
of freedom, so that layer alone does not imply a one-qubit core.

### Why Pauli faults can preserve the core

For the supported initial-H, linear-reversible, diagonal Clifford+T fragment,
a selected Pauli fault path changes physical bits by affine offsets and adds
phase signs. Flipping a T parity changes T to T_DAG; their difference is an
S correction on that parity, hence Clifford. Offset changes to diagonal
Clifford gates also remain Clifford. Consequently the faulty and ideal phase
functions differ by a diagonal Clifford polynomial: even linear coefficients,
quadratic coefficients divisible by four, and no cubic terms. This leaves the
Pauli-stabilizer translation kernel and the non-Clifford core unchanged.

The fixed coordinate encoding is shared across fault paths. The Clifford phase
correction and final physical Pauli offset depend on the selected faults.
It would be incorrect to preserve only the ideal core and discard those
corrections, or to replace noisy syndromes with zero records.

The study also conjugates each X-product check through its complete suffix.
All corpus checks are Pauli-deferrable in the ideal circuit. Adding the fault
Clifford corrections preserves that property. The resulting circuit starts
with the small core and then uses Clifford operations and Pauli measurements;
those measurements can produce nonzero or random syndromes and do not increase
the pure-trajectory stabilizer nullity. This is a whole-fragment identity, not
a claim that the original chronological gate sequence has low width at every
prefix. Because the identity holds per fault path, averaging over the original
Pauli-channel probabilities preserves noisy statistics. The scan samples fault
paths rather than exhaustively enumerating them.

### Validation and reproduction

```bash
uv run python -m tools.prototypes.phase_core_study CORPUS report.json \
    --fault-paths 2 --seed 171309
uv run pytest tests/python/test_phase_core_study.py
```

The full-corpus selected-fault scan covered 2,074 paths with up to 24 inserted
X/Y/Z faults each. Every path retained the same core and coordinate encoding;
all 18,038 checks remained Pauli-deferrable. The tests additionally compare
kernel size with exhaustive Pauli expectations on Aer states and check exact
compression on randomized supported preparations.

Twenty-three corpus circuits with at most six physical qubits and ten records
were independently checked for selected fault paths against complete Aer joint
records and unnormalized post-measurement branch states. Fifteen had random
syndrome outcomes. Maximum record-probability error was 6.11e-16 and maximum
branch-amplitude error after aligning global phase was 9.82e-17. Sixty-four
selected noisy one-core cases, with physical blocks up to 255 qubits, compiled
to peak active width at most one; one needed no active amplitudes at all.
These checks are structural and correctness results, not noisy throughput
benchmarks.

`defer_fixed_faults` emits an ordinary circuit for one already-selected fault
path. It does not parse stochastic noise channels, feedback, resets, general
rotations or additional Hadamards. Its resynthesis has no runtime cost guard
and is a reference construction rather than a production optimizer.

### What production integration still has to resolve

Noise awareness is a design requirement for a future production pass. The
study suggests preserving a fixed magic core while compiling fault-dependent
Clifford corrections and measurement axes. Current HIR conditional operations
express Paulis; general conditional Clifford corrections need an explicit
representation and planner design. No such architecture change is made here.
All topology and dependency analysis would still have to happen before hot
execution.

The symbolic-fault study below tests a bounded, precompiled representation of
those corrections, with stochastic sampling and measured throughput on this corpus.
Independent-core factorization and better basis choices remain relevant for
the 181 cases with core size above one. Extending to noisy regions of current
clifft-bench circuits requires separate entry-state and boundary proofs; the
current results do not establish gains on that suite.


## Symbolic faults and fixed noisy samplers

`symbolic_fault_study.py` compiles fault selectors into modulo-four S powers,
affine CZ controls, final Pauli offsets, and deferred measurement signs and
axes. The magic preparation and linear coordinate encoding are fixed. This
representation avoids enumerating every fault pattern during compilation;
[symbolic_fault_results.json](symbolic_fault_results.json) records the results.
The inputs remain the supported known-zero preparation fragment, with selected
X/Y/Z faults after the initial Hadamards and before terminal MX. Resets,
feedback, additional Hadamards, general rotations and noisy measurement gadgets
are outside this study.

### When one fixed sampler suffices

`affine_record_flips` proves a stronger identity for an eligible model: draw
from one ideal joint record distribution, then XOR compiler-precomputed affine
functions of the fault selectors into its records. It handles both ordinary
Pauli corrections and signed independent T states: on a T state, an S_DAG
correction is Pauli-equivalent to X, up to global phase. It rejects changing
measurement bodies, non-Pauli ancilla corrections, and nonlinear record signs.
It preserves joint syndromes and logical outputs, so postselection must happen
after those record corrections.

The full scan found:

| Independent fault model | Eligible circuits | Eligible one-core circuits |
| --- | ---: | ---: |
| X on every physical qubit immediately before the first T | 742 / 1,037 | 628 / 856 |
| Y on every physical qubit immediately after the last T | 1,037 / 1,037 | 856 / 856 |

The 114 other eligible pre-T cases have several independent magic coordinates.
Eligibility is a symbolic proof, not a finite sample of fault patterns. These
are selected-layer models; they are not noise throughout every gate.

Z-only dephasing at arbitrary supported positions also stays in this fixed
Pauli orbit throughout the fragment. Z faults commute through all the diagonal
gates, and linear reversible gates propagate them as Paulis. Five representative
corpus circuits with 1,353, 2,653, 3,734, 3,476 and 3,340 independent dephasing
sites respectively compiled to peak width one. These placements add a Z fault
on each unitary instruction target after initial preparation. This is a
structural check, without a full-dephasing throughput claim.

`AffineRecordSampler` is a reference wrapper using one existing native compiled
program, NumPy fault draws and parity matrices, followed by classical detector
and observable evaluation. Its chunked driver bounds transient fault-draw
storage. Python orchestration allocates outside native dispatch; it is not a
production executor design. The current planner already has presampled noise
symbols and affine Pauli/record signs, making this class a candidate for compiler
integration. That wiring and its boundary proofs remain to be designed.

### Why arbitrary Pauli noise is harder

The rank of the fault-dependent diagonal Clifford matrix counts independent
changes of Clifford geometry. In the pre-T model it ranges from one to 136;
after the last T it is zero. Counting geometries alone can overstate the
sampling problem: some changes on magic coordinates are Pauli-equivalent, as
in the eligible signed-T models above.

A broader compiler-only scan allows X/Y faults after each primitive unitary
gate target, leaving measurement/reset gadgets noiseless. Its geometry ranks
range from 10 to 136, making enumeration of distinct Clifford shapes impractical
as a general strategy. This is a trajectory/geometry scan, not a stochastic
full-gate throughput benchmark.

We also computed the exact common-Pauli-body size of the unmeasured preparation:
how many quantum coordinates remain after one shared Clifford decoder, allowing
fault-dependent Pauli signs. This intersects the Pauli stabilizer bodies across
all fault selectors without enumerating them. Under the broader gate-target
model, this size equals the original prepared-variable count in all 1,037
circuits, ranging from four to 29. In particular, every ideal one-core case
needs more than one coordinate for that shared prepared-state representation.
Each selected trajectory can still have stabilizer nullity one, with its own
Clifford decoder.

This is not a lower bound on sampling peak width. Earlier Pauli projections,
independent factors, output symmetries or query-specific elimination can make
sampling cheaper than representing the unmeasured state. The result rules out
the simple universal fixed-decoder argument, not every possible noisy k=1
sampler. General fault-dependent Clifford geometry is outside the current HIR
conditional-Pauli representation. Supporting it would require an explicit
architecture decision with all topology analysis completed before hot execution.
No HIR, planner or executor changes are made here.

### Noisy sampling measurements

All comparisons use the current default passes plus ActiveWidthSchedulePass,
one thread and batch size one on an Apple M4 Pro. Five alternating repeats use
separate shot budgets calibrated toward 0.15 seconds, bounded between 16 and
1,000,000 shots. Results are median wall time per attempted shot. They include
fault draws, native sampling, record corrections and postselection, and exclude
compilation and warmup. The candidate is a Python research wrapper.

For independent Y faults with probability 0.001 on every physical qubit just
after the last T:

| Circuit | Fault sites | Current peak | Prototype peak | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 0001 | 128 | 5 | 1 | 1.55x |
| 0005 | 127 | 19 | 1 | 6,288x |
| 0262 | 22 | 15 | 1 | 954x |
| 0689 | 20 | 13 | 1 | 225x |
| 1037 | 127 | 22 | 1 | 51,182x |

For eligible pre-T X models at the same probability, cases 0016, 0002 and 0042
improved by 2.22x, 32,589x and 115,047x respectively, reducing current peaks
4, 20 and 22 to one. These are selected examples, not corpus-wide speedup
estimates. Some expensive baseline runs have only 80 attempted shots across
all repeats; their statistical agreement is a smoke check, with correctness
primarily supported by the symbolic identities and independent exact oracles.

The bounded `BranchBank` experiment separately enumerates at most 256 selected
fault patterns and compiles every leaf before sampling. On five cases with six
selected pre-T X sites at probability 0.03, peak widths 5-18 fell to one, with
speedups 16.8x-19,763x. Its exponential growth makes it unsuitable for production.
It demonstrates the trajectory compression without solving general noise.

The speedups vary because existing active widths vary exponentially in cost,
and because fault draws, record corrections and postselection dominate once
the amplitude core is small. Dense NumPy parity evaluation also penalizes large
physical blocks in this reference driver. Production cost guards and packed
classical record evaluation would matter even for the eligible fixed-sampler
class.

### Validation and next decision

All 49 prototype tests pass; eight GPU variants were skipped. Tests compare
selected paths against Aer joint records and post-measurement branch states,
check common stabilizers by exhaustive Pauli expectations, validate reverse
geometry propagation against an explicit symbolic fault model, and compare
stochastic Clifford models against Stim. The dephasing test includes faults
throughout an interleaved-check example. An additional real-corpus oracle covers
23 small circuits, 46 pre/post-T models and 448 selected fault paths; maximum
full-joint record-probability error is 2.67e-15.

```bash
uv run python -m tools.prototypes.bench_symbolic_faults scan CORPUS OUTPUT
uv run python -m tools.prototypes.bench_symbolic_faults affine CORPUS OUTPUT \
    --placement post --circuits 0001 0005 0262 0689 1037 \
    --target-seconds 0.15 --repeats 5
uv run python -m tools.prototypes.bench_symbolic_faults affine CORPUS OUTPUT \
    --placement pre --circuits 0016 0002 0042 \
    --target-seconds 0.15 --repeats 5
uv run python -m tools.prototypes.bench_symbolic_faults benchmark CORPUS OUTPUT \
    --circuits 0001 0005 0262 0689 1037 --sites 6 \
    --target-seconds 0.15 --repeats 5
uv run pytest tests/python/test_symbolic_fault_study.py
```

The wrappers currently sample independent binary Pauli channels, not categorical
depolarizing channels. The trajectory identities can be averaged using the
original categorical probabilities, but a future driver must preserve their
mutual exclusivity. This study does not establish gains on current clifft-bench
circuits outside the supported entry-state and boundary assumptions.

The production candidate is a guarded Clifford-core compression pass with an
exact affine-noise fast path and unchanged fallback for unsupported models.
The branch bank should remain research code. For broader Pauli noise, the next
directional question is whether measurement-aware elimination removes enough
fault-dependent geometry before a new conditional-Clifford representation is
necessary. Noise-aware compression should be designed around these results,
instead of productionizing the ideal phase-folding prototype independently.


## Native phase-polynomial compiler pass

The native `PhasePolynomialPass` implements the general compiler direction from
these studies. It operates on commuting Pauli rotations in HIR, using the
existing Clifford frame, planner and executors. It is opt in. The Python study
implementations above are historical reference experiments; they are not used
by the native pass or its sampling benchmark.

### Algebra and measurement handling

At a block entry, track signed stabilizers with a fixed positive eigenvalue on
every possible trajectory. Initial Z stabilizers follow from the simulator's
`|0>` input. Unconditioned measurements, conditional Paulis and stochastic Pauli
noise conservatively intersect this known group; a deterministic Pauli fault
conjugates its signs. No relation is inferred from future postselection.

Choose independent actual Pauli rotation axes modulo this group. A dependent
axis is a product of these generators and a proven signed entry stabilizer.
The sign can complement its eigenbit parity and reverse its T angle. This
produces a cubic weighted polynomial `p(x)` modulo eight, valid on the common
entry code. The rewrite does not require the eigenbits to have a uniform
initial distribution.

Solve over GF(2) for translations `a` whose phase difference is
`p(x xor a) - p(x) = c + 4 z.x (mod 8)`, with `c` even. Those directions carry
only Clifford phase structure. A binary basis change puts the remaining
non-Clifford directions first. Synthesize parity phases, retaining at most the
original number of T rotations, and absorb fixed Clifford factors using the
same downstream conjugation helpers as `PeepholeFusionPass`.

Simply stopping at every intermediate measurement limited the first native
version to 606 lower-width circuits. The final version can pull a measurement,
conditional Pauli or expectation value before the pending phase when its exact
conjugate is a Pauli:

```text
U_prefix^dag M U_prefix = omega^(-c) M product(Q_j^z_j)
omega = exp(i*pi/4)
```

Here `c` and `z` come from the full finite difference of the prefix polynomial.
The scalar phase is retained to recover the Hermitian sign. The observer must
also commute with every entry-code constraint used by the whole block, so both
measurement branches remain inside that code. Otherwise the region stops
before it. Random syndrome measurements and classical feedback are supported
when these proofs hold. Record order and references remain intact.

A noise site inside a region is crossed only if every channel with nonzero
probability commutes with all its rotation axes. Noncommuting noise bounds the
region; it is not dropped or relocated. Noise outside the region is preserved
through Clifford conjugation. Categorical probabilities, noise-site order and
full joint record probabilities are preserved. Seeded trajectories can differ
because equivalent circuits consume random draws differently.

Arbitrary-angle rotations and instruments bound regions. The default cap is
32 independent phase variables, with 0-64 supported and zero disabling the
rewrite. A prior scheduling pass that has crossed noise prevents frame-changing
rewrites. This pass belongs before squeezing and scheduling, and is excluded
from the instrument-continuation pipeline because its decisions can depend on
the suffix. No HIR, planner or executor architecture was changed.

### Native corpus results

Results are saved in [native_phase_results.json](native_phase_results.json).
The baseline is `PeepholeFusionPass`, `StatevectorSqueezePass` and
`ActiveWidthSchedulePass`; the candidate inserts `PhasePolynomialPass` after
peephole fusion. Both use the CSV's postselection mask and raw detector signs.
The build uses Release, assertions enabled and NEON on an Apple M4 Pro.

| Metric | Baseline | Native phase pass |
| --- | ---: | ---: |
| Median final sampling peak | 16 | 1 |
| Circuits with final peak one | 0 | 924 |
| T count summed at the phase-pass position | 157,880 | 3,552 |
| Median end-to-end compilation | 3.59 ms | 7.28 ms |

Of 1,037 circuits, 1,003 have a lower final sampling peak, two have a higher
peak, and 32 are unchanged. The pass accepts rewrites in 1,005 cases.
Maximum candidate compilation time in the noiseless scan is 242 ms.

On 69 sampled circuits, summing median seconds per attempted shot estimates a
15.5x speedup for an equal-shot run over that sample. The median individual
speedup is much larger, about 8,260x, because a few unchanged expensive cases
dominate aggregate time. The aggregate estimate is the more useful headline.
These are short measurements on one host, not an exhaustive throughput survey.
Each arm is calibrated toward 0.03 seconds, capped at 200,000 shots, with three
alternating repeats. Expensive baselines sometimes get only one shot per repeat.
Compilation and warmup are excluded; postselection and noise draws are included.

### Noise and existing workloads

The following models use probability 0.001 and ideal measurement/reset gadgets:

- `post_layer`: one depolarizing layer on every physical qubit just after the
  last T in the original circuit.
- `gate_z`: independent Z errors on each affected target after every primitive
  unitary gate, splitting multiple targets into separate gates.
- `gate_depolarizing`: one-qubit or two-qubit depolarizing noise after every
  primitive unitary gate.

| Circuit | Post-T depolarizing peak | Speedup | Gate Z peak | Speedup | Gate depolarizing peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0005 | 19 -> 1 | 59,187x | 10 -> 1 | 70.7x | 12 -> 12 |
| 0262 | 15 -> 1 | 3,424x | 14 -> 1 | 544x | 15 -> 15 |
| 0300 | 16 -> 1 | 49,913x | 16 -> 1 | 2,437x | 16 -> 16 |
| 0689 | 13 -> 1 | 679x | 12 -> 1 | 164x | 13 -> 13 |
| 1037 | 22 -> 1 | 458,390x | 14 -> 1 | 559x | 22 -> 22 |

The full gate-depolarizing examples accept no rewrite and run at approximately
the baseline throughput. The pass remains correct under this noise, but cannot
use fixed entry relations that the faults destroy or cross noncommuting faults.
This confirms that noisy reduction to one is conditional on the noise geometry.
It does not demonstrate a universal noisy one-coordinate representation.

All ten current clifft-bench workloads at commit `566924f` have unchanged final
widths, T counts and action counts, with throughput near 1x. Arbitrary angles,
noise barriers and the absence of matching collective parity structure limit
applicability. Paired compilation measurements range from roughly equal to an
extra 12% on these workloads; the pass skips polynomial analysis when no T gates
remain. There is no demonstrated sampling gain on this existing suite.

### Validation and production limits

The non-expensive C++ regression suite passes all 1,029 cases, including 14
native phase-pass cases with 69 assertions. The selected Python suites pass
187 tests, with 36 GPU cases skipped. Tests cover multiword Pauli masks,
signed stabilizer products, random Clifford layouts, quadratic/cubic cores,
noisy joint records, random intermediate measurements, feedback, non-Pauli
barriers, postselection, raw detectors/observables and CPU execution modes.
Independent references use Qiskit Aer for unitaries and noisy joint records,
and Stim for stochastic Clifford circuits.

Across all 95 corpus circuits with at most ten physical qubits, independent
Aer statevectors agree to a maximum amplitude error of 1.25e-15. Complete joint
record distributions agree with unoptimized native forced-outcome probabilities
to 5.56e-16; that second oracle shares the native executor.

The guard compares peak width and estimated dense work at this pass's HIR
position. Later heuristic scheduling can reverse that comparison: circuits
0204 and 0481 rise from final width four to five, despite reducing the incoming
phase-stage width to six and T count from 26 to 20. Five longer alternating
measurements show approximately 1.43x and 1.37x sampling gains respectively.
The unchanged circuit 0273 is approximately 1x in the same check.

Keep this prototype opt in. Before default enabling, compare complete pipeline
candidates using width, memory and sampling-cost estimates; optimize compilation
cost; and broaden the general/noisy benchmark coverage. More specialized
fault-dependent Clifford methods remain a separate study requiring an explicit
architecture decision. These results support reviewing the native algebra and
measurement pullback together as a general pass, rather than productionizing
the earlier source-specific Python reduction on its own.

```python
passes = clifft.HirPassManager()
passes.add(clifft.PeepholeFusionPass())
phase = clifft.PhasePolynomialPass(max_variables=32)
passes.add(phase)
passes.add(clifft.StatevectorSqueezePass())
passes.add(clifft.ActiveWidthSchedulePass())
program = clifft.compile(source, hir_passes=passes)
print(phase.input_t_count, phase.output_t_count, phase.pauli_pullbacks)
```

After rebuilding the native extension, reproduce the normal compiler/sampler
benchmark and native tests from the repository root:

```bash
uv run python -m tools.prototypes.bench_native_phase CORPUS OUTPUT \
    --existing-bench CLIFFT_BENCH --sample-limit 64 \
    --target-seconds 0.03 --repeats 3
uv run pytest tests/python/test_phase_polynomial_pass.py \
    tests/python/test_pass_docs.py tests/python/test_compile_passes.py \
    tests/python/test_optimization_invariants.py
ctest --test-dir build/phase-native-tests -j8 --output-on-failure -LE expensive
```

## Depolarizing channel covariance and scheduler order

The original native phase study is saved at commit `9f24b16f` on
`codex/native-phase-polynomial-study`. The follow-up lives on
`codex/depolarizing-phase-study`; raw comparisons are in
[channel_covariance_results.json](channel_covariance_results.json).

`channel_covariance_study.cc` is a standalone native compiler experiment. It
uses the existing native passes, planner and ordinary `sample_survivors` for
throughput. Python generates inputs and supplies independent Aer oracles; it
is not a second sampling implementation. No pass is registered and no default,
HIR, planner or executor architecture is changed.

### Channel and labelled-fault semantics

For rotation about Pauli A, a Pauli channel commutes with the rotation at all
angles if every anticommuting error R has the same weight as its partner i*A*R.
Their unsigned bodies differ by XOR with A. Commuting errors impose no further
condition. The tool aggregates equal bodies and requires exact equality of
floating probabilities, conservatively skipping near-equalities.

This allows depolarizing noise on the matching subsystem to cross a T even
though its individual X and Y errors do not commute with T. After Clifford
conjugation, support matters: arbitrary depolarizing sites cannot cross every
rotation. The criterion is checked against independently computed Pauli
transfer eigenvalues in 5,120 one- and two-qubit cases, plus a multiword case.

Two semantic variants deliberately distinguish what is preserved:

- `channel_left` and `channel_right` clear the original logical-noise prefixes.
  They preserve the averaged channel and distributions conditioned on fault
  count, but can change the effect of a particular labelled Pauli fault.
- `labelled_covariant_left` and `labelled_covariant_right` make the same
  permutations while carrying the original logical-noise prefixes. Existing
  planner sign corrections preserve every labelled fault realization.
- `labelled_all_left` and `labelled_all_right` use that machinery to cross any
  Pauli noise site, including channels without covariance.

All variants move only rotations through noise or commuting rotations, stopping
at every other operation. Noise-site order is retained. Twelve tests compare
full joint records with Aer density matrices and enumerate native records for
all small fault patterns. They cover biased channels, asymmetric failures,
conjugated support, arbitrary angles, intermediate measurements and feedback.
The channel-only variant changes one labelled-path probability by 0.5 in a
witness; all labelled variants preserve path probabilities within 2e-14.
The enumeration is an oracle, not a proposed throughput implementation.

### Noisy sampling results

The baseline pipeline is fusion, phase reduction, squeezing and default
active-width scheduling. Each reordered arm adds the movement and another
fusion before phase reduction. Frame-changing passes already skip when original
noise prefixes have become nonredundant. The noise is the prior
`gate_depolarizing` model: probability 0.001 after every primitive unitary,
with ideal measurement/reset gadgets and the original postselection mask.

| Circuit | Baseline peak | Labelled covariant right peak | Sampling speedup |
| --- | ---: | ---: | ---: |
| 0005 | 12 | 10 | 3.21x |
| 0262 | 15 | 15 | 1.01x |
| 0300 | 16 | 16 | 1.02x |
| 0689 | 13 | 13 | 0.99x |
| 1037 | 22 | 13 | 408x |

The averaged-channel and labelled-fault versions have identical widths and
nearly identical throughput here. All five retain their original T counts and
accept no phase-polynomial reduction. Covariance therefore does not recover the
one-coordinate core under full gate-local depolarizing noise in this sample.
The gains come from a different starting order for squeezing and the bounded
scheduler, using the existing fault-preserving representation.

A separate stable block collector checks that this negative phase result is
not merely caused by bubbling each rotation too far. It collects the complete
T layer in 0005, 0300 and 1037. The resulting blocks still exceed both the
32-variable default and the maximum supported 64-variable setting: they have
more than 64 independent axes modulo the fixed signed entry relations that the
pass can prove under noise. No T or peak reduction follows. The other two
examples remain bounded and unchanged. Both collection variants also pass the
joint-record and labelled-path oracles. This diagnoses the current fixed-entry
representation; it does not prove a lower bound on noisy sampling cost.

Timings use one thread, batch size one, calibration toward 0.2 seconds capped
at 200,000 shots, and five alternating repeats. Compilation and warmup are
excluded. Rates are per attempted shot, including postselection and noise.
Circuit 1037 uses five baseline shots per repeat. These are measurements on one
host and five selected noisy examples, not a full noisy-corpus throughput study.

An arbitrary seed can regress severely: moving left raises circuit 0005 from
peak 12 to 18 and runs about 78x slower. A candidate must be compared against
the original complete pipeline. Single compilation measurements for 0300 are
about five seconds at baseline and six seconds for covariant-right movement;
other selected examples are tens of milliseconds. Source parse/trace and the
HIR argument copy are excluded. This experiment is not ready for default use.

### Existing workloads and pass interactions

Reordering also helps two noiseless quantum-volume workloads, which have no T
gates at the phase-pass position. The labelled left arm gives about 1.12x on
qv10 and 1.07x on qv20. Their peaks, HIR operation counts and semantic action
counts are unchanged, but executable action counts fall from 353 to 252 and
from 1,000 to 883. The executor can fuse more adjacent rotations. This benefit
comes from general rotation ordering and prepared-kernel fusion, rather than
phase-polynomial compression or noise covariance.

Most other existing workloads remain near unchanged. The right seed raises the
coherent d5 r5 workload's peak from 13 to 14; unrestricted right movement raises
it to 17. Compiling multiple candidates also costs more. HIR dense-work estimates
can miss the quantum-volume gains, whereas the existing lowered coefficient-
visit estimates reflect the additional executor fusion. Neither estimate is a
complete hardware timing model.

Running the default scheduler twice or three times gives no further peak
reduction on the five noisy examples. Increasing `search_budget` from 16 to 64
with beam eight produces peaks 11, 14, 16, 12 and 20. Beam 32 with budget 64 gives
the original default peaks instead; extra beam width consumes the finite search
budget differently. Larger settings are not uniformly better. Another fusion
pass after squeezing alone does not reproduce the quantum-volume gains.

The next general compiler direction is diverse fault-preserving scheduling
orders with a cost objective that accounts for prepared executor fusion, guarded
by a comparison of complete pipeline candidates. It requires broader noisy
coverage and compilation-cost control. Channel covariance remains a useful
mathematical rewrite criterion, but these results do not justify extending the
phase pass with averaged-channel semantics to recover noisy peak one.
Fault-dependent Clifford compression remains a separate unresolved direction
and would need an explicit architecture decision before implementation.

### Reproduction

With the checkpoint's assertion-enabled static core already built, the study
executable on this macOS OpenMP build is linked as follows:

```bash
mkdir -p build/channel-covariance-study
c++ -std=c++20 -O3 -UNDEBUG -I src -I build/phase-native-tests/generated \
    tools/prototypes/channel_covariance_study.cc \
    build/phase-native-tests/src/clifft/libclifft_core.a \
    -L /opt/homebrew/opt/libomp/lib -lomp \
    -o build/channel-covariance-study/channel_covariance_study
uv run python -m tools.prototypes.bench_channel_covariance CORPUS OUTPUT \
    --binary build/channel-covariance-study/channel_covariance_study \
    --existing-bench CLIFFT_BENCH --target-seconds 0.2 --repeats 5
uv run pytest tests/python/test_channel_covariance_study.py
```

The wrapper's `--scan` collects structural and lowered costs without sampling;
`--settings-scan` compares repeated passes and scheduler settings, and
`--collection-scan` tests block collection at phase-variable caps 32 and 64. Use
`--circuits` to select CSV row numbers. Tests skip if the standalone executable
is absent; `CLIFFT_CHANNEL_COVARIANCE_STUDY` overrides its path.
