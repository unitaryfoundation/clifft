# Smaller Active States in Clifft (v0.12.0, October 2026)

Clifft's sampling cost depends on the quantum state that remains after the
compiler has accounted for Clifford structure. A circuit can contain many
non-Clifford gates while preparing a state that needs only a small dense
array. Finding that smaller representation saves work on every shot.

Version 0.12.0 adds two default compiler passes that use this opportunity:
phase-polynomial reduction for commuting T rotations, and stabilizer-based
simplification for arbitrary-angle rotations. It also adds optional batch
calibration and extends leakage and loss modeling with in-record status checks
and effects on a gate's computational partner.

## Find the non-Clifford core

Clifft already combines nearby rotations and uses its default squeezing pass
to shorten their active lifetimes. An additional active-width scheduling
search remains opt-in. The new `PhasePolynomialPass` recognizes
relationships among a larger collection of commuting T rotations. It uses
signed stabilizer constraints established by the circuit's prepared input to
express those rotations in fewer independent phase variables.

The pass then synthesizes the remaining parity phases using a native
implementation of TOHPE, following
[Vandaele's T-count reduction algorithm](https://doi.org/10.22331/q-2025-09-16-1860).
It reconstructs the exact Clifford correction, retaining the signs and phase
relationships required by later gates and measurements. The pass rejects a
candidate that increases T count or peak active width; at unchanged peak
width, estimated dense work must not increase either.

Encoded transversal-T circuits are a useful application of this analysis.
Their physical rotation axes can look independent even when the encoded
state satisfies constraints that expose a much smaller non-Clifford core.
Triorthogonal codes connect this structure to
[magic-state distillation](https://arxiv.org/abs/1209.2426).

In the [development study](https://github.com/unitaryfoundation/clifft/pull/545),
phase reduction lowered final peak active width and estimated dense work on
all 1,037 circuits in an ideal, noiseless triorthogonal collection, both with
and without optional scheduling. The collection is still evolving and is not
part of the recurring release benchmark campaign. Those structural results
describe that collection; noisy protocol variants need separate evaluation.

The opportunity depends on the circuit. The same study found no accepted phase
rewrite on its ten general benchmark inputs and eight canary circuit inputs.
Compilation still has to look for an opportunity, so unchanged sampling plans
can come with additional compilation cost.

## Simplify rotations using the prepared state

`RotationSimplificationPass` extends state-dependent simplification to
arbitrary rotation angles. Two different Pauli products can act identically
on a constrained state. When the required constraints survive a commuting
region, the compiler can combine their angles and absorb any resulting
Clifford operation.

For example, qubit 1 remains in `|0>` in this circuit. Within that state,
`Z0*Z1` acts like `Z0`, so the two rotations cancel:

```python
import clifft

source = """
    H 0
    R_Z(0.137) 0
    R_PAULI(-0.137) Z0*Z1
"""
program = clifft.compile(source)
assert program.peak_active_width == 0
```

The compiler must establish that equivalence from the actual preparation.
Changing the preparation of qubit 1 can invalidate it.

A larger example is the fixed-input 16-bit Draper QFT adder from
[MQT Bench](../acknowledgments.md#mqt-bench), retained as a
[32-qubit fixture](https://github.com/unitaryfoundation/clifft/blob/037551a9/tools/bench/fixtures/draper_adder_16_basis.stim).
It prepares `a=37449` and `b=18724`, preserves `a`, and returns `b=56173`.
The following comparison uses the same circuit and source build, with optional
active-width scheduling disabled:

| HIR pipeline | Peak active width | Peak dense-state dimension |
| --- | ---: | ---: |
| Peephole fusion and squeezing | 15 | 32,768 |
| Add phase-polynomial reduction | 14 | 16,384 |
| Add rotation simplification, the new default | 0 | 1 |

These are structural compiler results, reproduced at
[`556c3e13`](https://github.com/unitaryfoundation/clifft/commit/556c3e13fc25f756d145ee2cde954aed853bff6a).
They measure the largest dense state required during execution, rather than
elapsed time or total process memory. The adder result applies to its prepared
basis inputs. Tests also compare smaller Draper and ripple-carry adders with
entangled inputs against Qiskit Aer, where the compiler must preserve the
remaining quantum state.

The fixture has also been added to
[clifft-bench](https://github.com/unitaryfoundation/clifft-bench/pull/67)
for the candidate's release comparison, including compilation and sampling
measurements.

## Preserve phases, correlations, and fault semantics

An optimizer must preserve more than state normalization or each measurement's
individual probability. Relative phases determine later interference, and
joint measurement records carry correlations consumed by feedback,
postselection, detectors, and observables.

The tests therefore combine exact phase-polynomial checks with independent
Aer amplitude and joint-record comparisons, and Stim references for applicable
stochastic Clifford cases. They cover noise before, within, and after rotation
regions; signed and unknown constraints; probability-one faults; conditioned
fault sampling; and the interaction of the default pipeline with optional
scheduling. Tests also require a rewrite to occur on designated examples, so
a pass that does nothing cannot satisfy those checks accidentally.

Both new passes analyze complete circuits from their known all-zero
initialization. They discard constraints invalidated by possible noise or
measurement outcomes, and never infer a constraint from future postselection.
They are excluded from leakage and loss continuations because their output
before an instrument can depend on the circuit after it. See the
[pass reference](../reference/passes.md) for the boundaries and controls.

The default pipeline is now:

```text
PeepholeFusionPass -> PhasePolynomialPass -> RotationSimplificationPass -> StatevectorSqueezePass
```

`ActiveWidthSchedulePass` remains opt-in. To retain the v0.11.0 default pass
selection, supply a manager containing `PeepholeFusionPass` and
`StatevectorSqueezePass`; `hir_passes=None` skips all HIR optimization. Equivalent
plans can produce different samples for the same seed while preserving the
sampling distribution. Explicitly selecting the old passes does not promise
bit-for-bit replay across package versions.

## Measure the batch size for longer jobs

Once a circuit is compiled, its fastest sampling configuration also depends on
the machine, output mode, and shot count. The default `batch_size="auto"`
uses conservative heuristics. The new `batch_size="tune"` option measures
eligible batch capacities before running the requested shots:

```python
import clifft

program = clifft.compile("H 0\nT 0\nH 0\nM 0")
result = clifft.sample(
    program,
    shots=100_000,
    batch_size="tune",
    tuning_budget_seconds=0.25,
    seed=42,
)
assert result.batch_tuning is not None
chosen_batch_size = result.batch_tuning.batch_size
```

Calibration shots are discarded and excluded from the production result.
The budget is a soft limit, and calibration adds latency, so it is most useful
for longer jobs. Small jobs or circuits already well served by `auto` can be
slower. Reuse the reported numeric batch size on later calls with the same
workload and settings to avoid recalibrating. See
[CPU Execution and Tuning](../guide/cpu-execution.md#budgeted-batch-calibration)
for the report fields and reproducibility considerations.

Packed execution also reduces memory traffic when copying expectation values
and sampling noise. These changes retain the existing execution model and
noise-draw order.

## Put leakage and loss information into the circuit

`HERALD_LEAKAGE_EVENT` and `HERALD_LOSS_EVENT` now append ordinary record bits
that report the site's current status without changing its state or
occupation. Detectors, observables, and classical feedback can consume those
bits in-circuit. The probes distinguish leakage from loss, are perfect, and
take no arguments. `READOUT_NOISE` models missed detections and false positives. See
[Nondestructive Status Checks](../guide/leakage-and-loss.md#nondestructive-status-checks).

Two-qubit gates involving a leaked or lost operand can also apply configured
effects to the computational partner. `PartnerEffect` describes Pauli noise
and, for leaked sources, leakage spreading. Model-wide defaults, directional
`InteractionRule` overrides, and explicit circuit annotations control where
the effects apply. Existing models retain their behavior unless these effects
are requested. The [Partner Interactions guide](../guide/partner-interactions.md)
explains probability conventions and ordering with other transitions.

## Measuring the candidate

The structural reductions above identify useful circuits for timing studies.
The v0.12.0 release comparison will measure the published candidate in
`clifft-bench`, retaining the existing workloads and adding the fixed-input
adder. The [Performance guide](../guide/performance.md) contains the completed
release campaigns and their measurement details.

A separate local study of the evolving triorthogonal collection will use a
frozen circuit snapshot and the published candidate. It will report
compilation separately from sampling, record the CPU and execution settings,
and distinguish pass-enabled comparisons from release-over-release results.
Timing results from one machine describe that configuration; broader hardware
claims require measurements on those other machines.
