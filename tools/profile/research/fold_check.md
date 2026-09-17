# Whole-check Clifford branches: an exact mechanism test

Follow-up: [full encoded f5 reconstruction and composition study](fold_composition.md)
now reaches active width 22 and tests noisy preparation, flags, growth, and
syndrome extraction around these cores.

The study now targets sustained sampling of **large residual states**, especially
full Reg5 cultivation and fault-distance-7 fold cultivation. Few-shot startup
latency is outside this target. The earlier negative product-factor and
per-shot-compilation results still stand for their tested circuits; they do not
rule out representations that retain a whole non-Pauli check.

**Result:** an isolated cat-controlled fold check has an exact, phase-sensitive
two-Clifford-term representation for every fixed Pauli-fault history and every
cat measurement result. The new offline diagnostic validates this identity
against Aer, including conditional output states. This is a reason to continue
the representation study, not a measured cultivation speedup or a new backend.

## Connection to the papers

[Hartweg and Pineiro Orioli](https://arxiv.org/html/2609.19116v1) report a hard
Reg5 workload with peak Clifft active width 22, physical MPS bond dimension 96,
and CAMPS bond dimension 5. Their Clifft timing uses version 0.7.0 and must not
be treated as a current performance baseline. Their frame-dependent compression
is motivation to study both whole-check branches and alternative Clifford
frames, rather than conclude from ranks in Clifft's existing coordinates.

[Takada, Bartlett, and Williamson](https://arxiv.org/html/2609.16929v1) motivate
retaining Clifford errors and non-Pauli checks instead of decomposing everything
into T gates. Appendix A specifies fold-check cores at distances 3, 5, and 7.
This diagnostic reconstructs their controlled factors and control assignments
from Eqs. 183 and 190-194 and Table 4, with abstract data labels. It does not
reconstruct their complete cultivation circuits or implement their specialized
Pauli-proxy simulation algorithm.

[Wan and Zapirain](https://arxiv.org/html/2609.18922v1) motivate grouping errors
by their whole-check response while preserving coherence and correlations;
their all-orders probability contraction is a separate candidate from a
general-purpose sampler. [Chan et al.](https://arxiv.org/abs/2609.17706) show why
internal hook errors and the distinction between actual T circuits and S
proxies matter. A rewrite must preserve those effects, including flag outcomes.

## Our representation experiment

Write the cat state as a coherent sum of its two computational-basis strings.
Every controlled factor in the check applies a Clifford operation to data in
one branch and identity in the other. A Pauli fault on a cat qubit permutes or
phases these strings; it does not create another string. A Pauli fault on data
is a Clifford operation within each branch. Unencoding and measuring the cat
therefore gives the Kraus operator

```text
K_m = (omega^a C_0 + omega^b C_1) / 2,  omega = exp(i pi/4),
```

for a compatible cat syndrome, and zero for an incompatible syndrome. Here the
C terms are data Clifford operators and the phases depend on the physical
fault history and measured result. For a fault-free check this reduces to
`(I + U)/2` or `(I - U)/2`. The two terms must interfere coherently: selecting
one as a classical random branch gives the wrong channel.

The prototype retains phases as integers modulo eight. Each branch is a
computational-basis permutation with a quadratic phase polynomial: X-mask,
linear coefficients modulo eight, CZ edges, and a global eighth-root phase.
Intermediate T gates can make the linear coefficients odd; every completed
branch is checked to have even coefficients and hence be Clifford. Unsupported
operations are rejected before analysis. This is a compact operator representation; it does
not claim that an arbitrary incoming state itself has only two components.

| Check core | Data / cat qubits | Controlled factors | T/T_DAG gates after elementary decomposition | Coherent Clifford terms per compatible result |
| --- | ---: | ---: | ---: | ---: |
| d3 | 13 / 3 | 9 | 38 | 2 |
| d5 | 41 / 5 | 25 | 130 | 2 |
| d7 | 85 / 8 | 49 | 278 | 2 |

The counts exclude injection, growth, syndrome extraction, cat preparation and
verification, and readout. They are not a gate-count speedup estimate. Large
Clifford operations and coherent overlaps still have a cost.

If the incoming state has R stabilizer terms, one check produces at most 2R.
Under the explicit assumption of one two-term T-state injection and otherwise
Clifford operations outside these checks, two, four, and six checks give crude
upper bounds of 8, 32, and 128 terms. Selected Pauli measurements map each
stabilizer term to a stabilizer term or zero, so they do not increase these
bounds. These are algebraic bounds, not measured term counts for the complete
paper protocols. Probability evaluation may require quadratically many
pairwise overlaps; relative phases, cancellations, and merging are essential.

## What was independently checked

`test_fold_check.py` uses Qiskit Aer native CCZ and T-conjugated CNOT operations
as its reference. The reduced three-data/two-cat check has overlapping factors,
so internal Pauli placement matters. A Choi input checks every column of every
conditional Kraus matrix, including preservation of entanglement with a
reference system. The cases comprise the fault-free circuit, every X/Y/Z fault
at every elementary event boundary on every wire (120 cases), and 32 histories with five
fault insertions. All outcomes, including impossible ones, are compared without
renormalization, and the sum of K_dagger K is checked to be identity.

The full d3 core is additionally compared against Aer on a generic dense input,
both without faults and with mixed data/cat faults. Agreement is within 2e-14
absolute amplitude error. Large d5/d7 dense matrices are not built. Their
schedule counts and fault-free Hermitian-involution identity are checked on
100 deterministic random computational-basis columns each.

The test includes faults inside each T-conjugated CNOT, as needed for the
elementary T/CNOT/CCZ noise locations in Takada et al., Appendix A.2. CCZ is a
native operation in that model. Noisy decompositions of CCZ into smaller gates
are not covered. No distribution over faults is sampled here.
Verified cat preparation, noisy unencoding, flag acceptance, and complete
measurement-record bookkeeping remain to be incorporated and validated.

## Native Clifft comparison and its limits

The exporter supplies product-|+> inputs to these isolated cores. These are
mechanism controls, explicitly not encoded cultivation states. With the current
native planner they reach active widths 14 (d3) and 42 (d5); d7 encounters the
planner's limit when a promotion would reach width 60. The d5 circuit is only
compiled, never allocated as a dense state. This demonstrates that gate-level
residual width can miss the two-branch structure on these controls. It does
**not** establish that actual Reg5 has width 42, that full f7 fails compilation,
or that the candidate is faster on either complete protocol. No speedup is
reported. Metadata and circuit hashes are in `fold_check_data.json`.

## Circuit acquisition status

The existing five-round coherent-memory circuits remain useful generic
controls. Existing color-code d3/d5 circuits, including the corrected and
flagged variants already acquired, are useful correctness cases. They do not
cover the requested large fold-cultivation regime.

The public [Sahay source](https://github.com/kaavyas99/MSC_foldedH/tree/9378fd228ba83c592875d155136fcbcd16a45680)
was pinned and its complete Git tree inspected. Its README describes Y-state
sample circuits. `full_circuit(..., cultiv_only=True)` returns after growth to
rotated d5, before a d5 non-Pauli cultivation check; `generate_sv_cirquit` ends
after d3 checks and decode. The Cirq handoff script contains an actual-T d3
conversion, but is not the complete Reg5 target. Increasing `dfinal` does not
make this fault-distance-7 cultivation. Similarly, a distance-7 escape patch
in the Gidney generator is not a distance-7 cultivation stage.

Full Reg5 and full f7 artifacts have **not yet been acquired**. The respective
paper appendices provide reconstruction guidance, but growth, cat verification,
noise locations, terminal decoding, and postselection must all be matched before
claiming reproduction. The core reconstruction here is useful independently;
it does not fill that corpus gap.

## Next experiment and integration decision

1. Finish faithful full Reg5/f7 circuit acquisition or reconstruction, including
   phase conventions, flags, noise placement, and output contract. Cross-check
   smaller instances and published acceptance/logical statistics. Do not use
   a proxy or silently substitute a simpler noise schedule.
2. Carry coherent Clifford branches through repeated checks and intervening
   Clifford growth/syndrome regions in an offline reference. Measure live terms,
   distinct relative Cliffords, cancellations, and overlap work on natural
   fault samples and accepted trajectories. Retain the output state so this
   tests composition, not just a terminal acceptance formula.
3. Compare this with a restricted diagonal-Clifford error frame and with
   fold/code-informed Clifford frame choices for the residual. Choose the
   smallest mechanism that reduces work on the actual large circuits.
4. Only then prototype a reusable execution plan. Judge end-to-end attempted
   and accepted shots per second, memory, and amortized planning cost at large
   shot counts. A favorable core count alone does not justify adoption.

Eligibility can be checked structurally for a delimited cat check with the
supported controlled Clifford factors and Pauli noise model. General rotations,
leakage, incompatible instruments, or an unrecognized cat circuit must decline.
Compile-time refusal can route the entire circuit to existing Clifft without
converting a live state. A hybrid handoff needs a separately measured conversion
cost and a bound on residual width; it is not automatically cheap.

This prototype is offline research, not hot execution. A naive per-shot
stabilizer-branch implementation would perform tableau evolution and dependency
work forbidden by the current executor architecture. Production integration
therefore requires either compiler-precomputed actions/overlap formulas or an
explicitly approved architectural change. Nothing here changes those invariants.

## Reproduction

Using an environment with NumPy, Qiskit, and Qiskit Aer:

```sh
python -m unittest discover -s tools/profile -p test_fold_check.py
python tools/profile/fold_check.py --output /tmp/clifft-fold-core
build-research/profile_structure /tmp/clifft-fold-core/fold_core_d3.stim 64 16 0
build-research/profile_structure /tmp/clifft-fold-core/fold_core_d5.stim 64 16 0
build-research/profile_structure /tmp/clifft-fold-core/fold_core_d7.stim 64 16 0
```

The final command intentionally reports the planner width limit. Do not raise
the allocation cap to sample the d5 mechanism control.
