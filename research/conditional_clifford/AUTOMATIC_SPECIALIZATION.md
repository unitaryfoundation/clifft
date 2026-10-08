# Automatic conditional simplification: first circuit panel

Date: 2026-10-08.

## Goal and checkpoint

Given an all-zero circuit-entry state, automatically discover algebraic
structure that simplifies complete sampling. The circuit may contain Pauli
noise and measurements. A useful result may be Clifford or retain a smaller
non-Clifford computation. Specialization may depend on sampled faults and
observed measurements; compiling all non-stochastic work once is not a
requirement. Production integration remains a separate design decision.

This study reaches the first breadth checkpoint. The existing algebraic
optimizer already finds useful complete and partial reductions outside BT
when faults are fixed. A constructed measured-state example additionally
needs the observed preparation outcome to become Clifford. However, fresh
whole-circuit compilation is too expensive to make these reductions a useful
general sampling mode. The next question is how much analysis can be shared
while retaining limited per-shot planning, rather than how many more BT
special cases can be recognized.

## Implementation and scope

The new [host](../../tools/profile/automatic_specialization.py) consumes raw
circuit text. It neither identifies circuit families nor takes encoder,
decoder, phase-region, or verification-tail annotations. The panel generator
knows which examples it constructs; the transformation does not.

Three paths are compared:

1. Ordinary default algebraic compilation of the noisy circuit.
2. Sample each independent noise location, replace it with its realized Pauli
   operation or record inversion at the original location, and compile the
   resulting circuit with the same optimizer.
3. Additionally sample the maximal initial Clifford prefix with Stim. Obtain
   its conditional stabilizer state, prepare that state, retain every prefix
   record using MPAD, and compile the remaining circuit. Prefix detector and
   observable declarations and subsequent feedback retain their record slots.

The third path supports one boundary before the first non-Clifford instruction.
It does not support arbitrary continuation from a non-Clifford state at later
measurements. If that initial prefix has no measurements, it makes no change.
Reset-induced hidden outcomes are sampled by Stim when present. Measurements
after the boundary remain live in Clifft. All preparation is derived from the
actual all-zero input, never an assumed all-zero internal boundary.

The host supports independent X/Y/Z errors, depolarization, biased one- and
two-qubit Pauli channels, and symmetric readout flips. Locations may have
different probabilities, including zero and one. A two-qubit channel is one
categorical draw, and an inverted reported measurement does not change its
quantum collapse. Correlation within a two-qubit Pauli channel is supported;
inter-location correlated channels, coherent noise, noncomputational noise,
and tagged instructions are outside this host. There is no history cache,
fault-count truncation, postselection, or enumeration in fresh sampling.

All sampling, tableau work, state reconstruction, and compilation happen in
the Python research host before calling the existing executor. No production
instruction, API, executor contract, or compiler pass changes are made.
Annotations use raw record parities consistently, without recomputing a faulty
reference syndrome for each realization.

## Panel and structural results

The pinned Merlin generator revision is
`097380fac1a3968ca47925146e211fe990f4c396`. Distillation has depolarizing noise
after physical T gates and encoder/decoder CNOTs. BT retains the full synthetic
physical gate/readout noise model and ideal verification tail from the previous
study. Cultivation retains its complete generated circuit noise. Their site
probability is 0.001. The measured-state controls use explicit biased channels
and larger probabilities to exercise both branches.

Direct-X variants replace the ideal verification operation with direct logical
X readouts. They keep actual logical sampling; simply omitting logical outputs
would not test a residual non-Clifford sampling workload. Cultivation d5 was
reserved as a held-out size before implementing the host, and prompted no
algorithm changes. Distillation variants and BT sizes are not counted as
independent families.

Counts below are residual HIR T counts after the existing optimizer, not
literal source gate counts. These are bounded experimental results, not proofs
that every possible history produces the same count.

| Circuit | Ordinary noisy T count | After fixed faults | After faults and prefix outcomes | Active width before -> after faults |
| --- | ---: | ---: | ---: | --- |
| 15-to-1 with direct X readout | 15 | 1 | 1 | 5 -> 1 |
| 15-to-1 with verification | 16 | 0 | 0 | 5 -> 0 |
| Bravyi-Haah k=2 with direct X readout | 14 | 2 | 2 | 5 -> 1 |
| Bravyi-Haah k=2 with verification | 16 | 0 | 0 | 5 -> 0 |
| Cultivation d3 | 29 | 15 | 15 | 4 -> 4 |
| Cultivation d5 | 91 | 53 | 53 | 10 -> 10 |
| BT27 with verification | 401 | 0 | 0 | 33 -> 0 |
| BT27 with direct X readout | 378 | 37 | 37 | 33 -> 8 or 9 |
| BT81 with verification | 1,157 | 671 | 671 | 87 -> 56 |
| BT81 with direct X readout | 1,134 | 648 | 648 | 87 -> 56 |
| Measured parity, with or without later feedback | 2 | 2 | 0 | 1 -> 1, then 0 with prefix outcomes |
| Noncommuting control | 24 | 24 | 24 | 4 -> 4 |

There are 7-16 selected stress histories per case, including identity, faults
spread through the circuit, and multifault cases up to weight 12 when enough
sites exist. Each executable path also receives 32 freshly drawn histories,
with matching fault histories between the two specialization modes. The width
budget is 12; BT81 is inspected but not sampled by this host. Its phase pass
reports capped regions, consistent with the existing 64-variable limitation.
This is a representation obstruction, not evidence that the circuit lacks a
small reduction: the earlier dedicated BT81 construction is entirely Clifford.

The measurement witness prepares two plus qubits, measures Z0*Z1, and applies
T0 followed by T_DAG1. The even-parity branch cancels those phases; the
odd-parity branch has a Clifford phase. The existing fixed-constraint optimizer
retains two T gates when the parity outcome is unknown. Giving it the actual
conditional state exposes the Clifford reduction. A companion case uses the
retained preparation record for feedback after the T gates.

Noise-free versions are included in the JSON. Ordinary compilation already
achieves the distillation and BT27 reductions there. This host adds no new
noise-free optimizer. Further noise-free improvements remain independently
worth investigating, but are not established by conditioning noisy circuits.

Merlin accepts both distillation readout variants. It rejects BT's direct X
readouts with a non-affine-zero-set measurement error, while accepting the
verified BT circuits. It rejects the constructed H-gate controls at parsing.
These are distinct boundaries; CNOT+T syntax alone does not establish Merlin
measurement compatibility.

## Costs and what they imply

Every fresh shot performs new compilation. Timing includes noise drawing,
source materialization, optional prefix sampling and reconstruction, tracing,
optimization, width inspection, lowering, and sampling. Output parity audits
and JSON report construction are excluded. Initial fault-model construction
is reported separately. Ordinary compilation costs and sample batches of 1,
32, and 1,024 shots are retained, so setup and amortization can be compared.

Representative measurements from the pinned single-CPU run:

| Circuit | Ordinary Clifft sampling, us/shot | Fault specialization and sampling, us/shot | Merlin sampling, us/shot |
| --- | ---: | ---: | ---: |
| 15-to-1 with verification | 0.298 | 271 | 16.2 |
| Cultivation d5 | 16.3 | 3,173 | 681 |
| BT27 with verification | Not run: width 33 | 52,416 | 1,470 |

Ordinary sampling uses the median of three 1,024-shot batches after compilation.
Conditional and Merlin sampling use 32 fresh shots; Merlin setup is separate.
These are local diagnostic timings, not a tuned throughput comparison. Including
setup, 32 shots of scored 15-to-1 take about 1.54 ms through ordinary compilation
and 13.66 ms through the fault host. For d5 they take about 18.38 ms and
127.39 ms respectively. With compiled-program reuse, the specialization penalty
is much larger: approximately 910x for scored 15-to-1 and 190x for d5 here.
Generic BT27 specialization is about 36x slower than Merlin in this run.
The earlier dedicated BT host remains a much faster reference.
Per-mode peak memory was not separately profiled in this checkpoint.

Fewer T gates alone are therefore an inadequate success criterion. Cultivation
does not reduce its peak width, and small distillation is already extremely
cheap in ordinary Clifft. An eventual automatic mode should keep ordinary
execution for such cases unless a cheaper conditional representation changes
that comparison. Sampling prefix outcomes adds reconstruction cost and finds
no additional reduction on these protocol examples; its positive result is
currently the measured-state control.

## Correctness evidence and limits

The [validator](../../tools/profile/validate_automatic_specialization.py) records
the following evidence in
[automatic-specialization-validation.json](automatic-specialization-validation.json):

- An exhaustive mixture of 64 positive-probability noise histories checks
  biased Pauli channels, two-qubit categorical faults, inverted Pauli-product
  measurements, readout flips, probability-zero/one noise, MPAD, and feedback.
  Its complete four-record law matches an independent Aer density-matrix
  calculation to below 7e-17. There are also 131,072-shot Stim and 131,072-draw
  categorical probability checks, with unsupported-channel rejection controls.
- For each measured-state control, all eight positive-probability fault
  histories and both preparation outcomes are checked. All 32 conditional
  output laws become Clifford. Their weighted mixtures match independent Aer
  calculations to below 6e-17. Conditional probabilities are checked explicitly;
  observing the same finite samples is not used as an equivalence argument.
- Sixteen distillation histories across four variants check the complete joint
  record law against Aer, including ideal and weight-1/5/12 faults. A guarded
  oracle defers only terminal measurements on qubits that are never used again.
  It obtains the exact reference probabilities, and Clifft replay checks every
  positive-probability record and total probability. The maximum discrepancy
  is below 1.8e-15. Reversing qubit labels preserves these laws and reductions.
- The four-qubit noncommuting control retains all 24 T gates, with statevectors
  and full record distributions checked against Aer for ideal and multifault
  cases, also after wire renaming.
- Cultivation d3/d5, each ideal and with a selected five-fault history, are
  compared with 8,192 independent Merlin shots per backend. Individual records,
  declared parities, and selected correlations pass the moment checks, with
  maximum score below 2.83. These are bug checks, not full-distribution or
  rare-accepted-error equivalence claims.

Fresh timing shots audit record numbering and declared output parities. They
do not statistically validate the entire noisy panel. In particular, the BT
direct-X variants are structural/execution probes without a new independent
large-circuit equivalence check. Prior BT certificates and studies retain their
original scopes; they are not silently extended to these new output laws.

## Next bounded experiment

Do not promote fresh whole-circuit recompilation as a fast sampling mode, or
extend the BT-specific core merely to improve its benchmark in isolation.

The useful next experiment is a shared algebraic analysis of eligible phase
regions: derive support coordinates and fault/outcome-dependent phase responses
once, then update those responses and plan only the reduced computation when
needed. Apply the same representation to a distillation case, BT27, and the
measurement witness; retain non-Clifford remainders and unsupported-region
fallback. No circuit names or enumerated fault-history table should enter the
transformation. Measurement-dependent specialization beyond the initial
Clifford prefix remains a separate, explicit capability to develop.

Evaluate both the cost of specialization and the remaining sampling work.
Require an actual improvement over the current whole-circuit host on a case
where specialization materially lowers simulation cost, while a cost estimate
selects ordinary Clifft on the already-cheap controls. Check the same exact
oracles and report analysis budgets, residual widths, and reasons for fallback.
The BT81 representation ceiling should be recorded separately from runtime
overhead and algebraic inapplicability. Reassess before extending the panel or
changing production execution architecture.

The full panel, hashes, seeds, histories, compiler counters, setup costs, and
timings are in [automatic-specialization.json](automatic-specialization.json).
Reproduction commands are in the [profiling guide](../../tools/profile/README.md#automatic-conditional-simplification-across-circuits).
