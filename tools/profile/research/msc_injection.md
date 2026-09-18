# Direct injection and complete MSC fixed-history coefficients

Follow-up: [complete static sampling](msc_sampling.md) replaces the offline
gadget binder and adds full Python/native record sampling. That later study
also records a negative native performance result on these small controls.

The injection prefix now computes its weighted logical state from fixed parity
rows and four precomputed 2-by-2 matrices. This removes the last initial-state
extraction from the coherent reference. Injection feeds the previously validated
growth and terminal contractions, so the coefficient path covers complete
original d3/d5 fixed histories without taking an initial or intermediate CH
state.

All 95 stored histories match complete trajectory weights and final logical
density matrices. Maximum relative probability error is 4.22e-14 and maximum
density-matrix error normalized by trajectory probability is 4.08e-14. These
are complete fixed-history coefficients, not a complete static record sampler.
The existing offline gadget fault binder is still used, and no new native
throughput result is claimed. All weights here are conditional on the specified
physical-fault history; its classical sampling probability is not included.

## Compile the single magic input

Before the first cultivation gadget, each original prefix has exactly one
T-dagger gate. Replace it algebraically with a supplied magic qubit, a CNOT
from the original data wire to that qubit, and a Z measurement of the resource
postselected to zero. Supplying `( |0> + exp(-i*pi/4) |1> ) / sqrt(2)` implements
the original T-dagger divided by sqrt(2). The replacement is a compiler device,
not an additional noisy resource or measurement in the simulated circuit.
The same construction supports a T gate with the opposite magic-state phase.

Everything else in the prefix is Clifford. During construction, prepare the
resource and a reference qubit as a Bell pair instead of inserting its magic
state. Expose the original resets as measurements plus conditional Paulis,
and derive the output stabilizer flows of this Choi circuit. The output code,
all physical spectators and the measured virtual resource are separately
certified. The logical output and reference form a pure Bell-type Choi state,
with independent XX and YY stabilizers in the chosen logical convention.

The signs of these two stabilizers select one of four logical Clifford maps.
All signs and record constraints are affine parities of true quantum outcomes
and original physical fault bits. Neither flow solving nor tableau evolution
is needed during evaluation. Construction declines multiple T gates, prefix
record feedback, an unfixed output degree of freedom, or a Choi state that is
not a full-rank logical Clifford map.

The four maps are stored as sixteen complex doubles, or 256 bytes of coefficient
data. That figure excludes the compiler, circuit, record maps and fault masks.
The existing `InstrumentPlan` now optionally accepts an explicit output-probe
list; injection requests none and constructs its joint Choi probes separately.
Its existing default behavior is unchanged.

## Weights and hidden outcomes

Both prefixes contain 33 original true quantum outcomes, including hidden reset
measurements. The Choi circuit adds one virtual postselection bit and has
sixteen independent record constraints. Its reachable branch probability is
2^-18. Removing the virtual bit leaves sixteen independent constraints on the
33 original outcomes, hence seventeen free bits.

For a normalized signed Choi vector C, the matrix used on the magic input is

```
M = 2 * sqrt(2^-18) * reshape(C),
M dagger M = 2^-17 I.
```

One sqrt(2) factor removes Bell-state normalization; the other removes the
virtual postselected T replacement. Consequently every reachable original
prefix history has probability 2^-17. There are 2^17 such histories for each
fixed physical-fault history, so their probabilities sum to one. Faults shift
the parity constraints and logical signs without changing these ranks.
Inconsistent outcomes return zero weight.

These statements concern trajectories with explicit hidden reset outcomes.
They do not say that the circuit's visible-record distribution is uniform or
that its physical acceptance probability is 2^-17. Marginalizing hidden outcomes
and sampling the actual record contract remain separate work.

The physical code/spectator frame still comes from the earlier signed boundary
maps. Idle future d5 wires retain their physical fault history; the injection
calculation does not silently reset them before growth.

## Complete coefficient composition

`FixedHistoryPlan` in `study_msc_injection.py` computes injection amplitudes,
binds the existing gadget payloads, applies growth when present, and contracts
the final cultivation/terminal pair at its actual final code projection. The
same weighted amplitudes pass directly between blocks. An impossible injection
record returns zero before attempting suffix coefficient work.

The complete coherent evaluator runs separately as a test oracle. Its observer
only compares the independently computed injection and final results with the
reference states; those states never enter the new coefficient calculation.
A test disables the coherent evaluator and CH construction on that path.

## Validation

- All 44 d3 and 51 d5 complete stored histories agree. Injection weights agree
  exactly at floating-point precision, with maximum normalized density error
  7.86e-17. Complete weights and density errors remain below 4.22e-14 and
  4.08e-14, respectively. The prior reference study compared full d3 against
  Aer and full d5 against elementary-gate Clifft.
- Twenty independent original-gate Aer prefix comparisons cover both circuit
  controls, T and T-dagger orientations, X/Y/Z physical faults and mixed stress
  histories. The independent adapter keeps the original T gate and every
  measurement/reset, without the Choi or virtual-resource rewrite.
- Each prefix has fifteen wires touched by quantum gates. The Aer adapter drops
  only other wires that remain product spectators under the specified Pauli
  faults, retaining all 33 quantum events. Thus these are exact active-wire
  prefix comparisons, not dense simulations of the entire 42-wire d5 protocol.
  The later influence of idle-wire faults remains in the full-history tests.
- Stim checks 158,034 signed flows over 2,928 ideal/physical-fault-basis histories:
  762 histories with 34 certificates for d3, and 2,166 with 61 for d5. The
  certificates include record constraints, all three joint Choi stabilizers,
  output code checks and spectator/resource preparation flows. Independent
  materialization retains reset feedback and all original noise locations.
- Tests verify all four matrix normalizations, the complete record-rank
  normalization argument, rejection after violating each constraint, binding
  with Stim disabled, end-to-end coefficient composition, and decline for a
  second magic gate. Five new tests bring the combined MSC suite to 42 tests.

The independent Aer evidence now includes the injection prefixes and the
previous nineteen-qubit terminal monomial checks. It still does not constitute
an external simulation of the full original d5 protocol. The authors' larger
MSC7/Reg5/f7 circuit-artifact gap also remains.

## Next bounded step

Replace the per-history offline gadget binder with fixed compiler-produced
fault maps for its two monomials, including their relative phase, root/reset
conditions, and ancilla actions. Validate those payloads against the existing
binder and independent gadget checks before composing record sampling. This
removes the remaining per-history topology work from the proposed path; it
must not be moved into a production hot executor.

After fixed binding, implement and validate full record sampling, then native
execution with preallocated coefficient/record storage. Fixed-history agreement
alone does not establish sampling throughput or an end-to-end speedup. The
reconstructed f7 remains the large-state performance target; original d3/d5
provide composition coverage of a different physical family.

Reproduction:

```
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_injection.py \
  --output tools/profile/research/msc_injection_data.json
CLIFFT_MSC_RECORD_PROBE=/tmp/msc-record-probe \
  MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p 'test_msc_*.py'
```
