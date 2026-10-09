# Richer Noise Models, Smaller Active States (v0.12.0, October 2026)

Clifft 0.12.0 extends leakage and loss modeling with configurable effects on
gate partners and status checks that circuits can act on. Two new default
compiler passes find smaller active states, and optional batch tuning helps
choose an efficient execution configuration for longer sampling jobs.

## Extend leakage and loss models

When a gate encounters a leaked or lost qubit, simply skipping that gate may
miss a disturbance to its partner. Clifft's new `PartnerEffect` lets a noise
model apply Pauli errors to the computational partner, or spread leakage from
an already-leaked source. The source keeps its status. These effects can
capture correlations that influence later gates and measurements.

Protocols also need to detect leakage or loss during a circuit. The new
`HERALD_LEAKAGE_EVENT` and `HERALD_LOSS_EVENT` probes append ordinary
measurement-record bits without changing the site's state or occupation.
Detectors, observables, and feedback can use those bits at the point of the
check.

This example combines both capabilities. Leakage on qubit 0 spreads to
qubit 1 at a two-qubit gate, and a detector reports the partner's leakage.
A separate lost qubit illustrates the distinction between leakage and loss:

```python
from clifft import noncomp

model = noncomp.Model(
    gate_partner_effects={
        "leaked": noncomp.PartnerEffect(spread_probability=1),
    }
)
result = noncomp.sample(
    """
    LEAKAGE(1) 0
    CX 0 1
    HERALD_LEAKAGE_EVENT 1
    DETECTOR rec[-1]
    LOSS(1) 2
    HERALD_LEAKAGE_EVENT 2
    HERALD_LOSS_EVENT 2
    """,
    model,
    shots=16,
    seed=1,
)
assert (result.measurements == [1, 0, 1]).all()
assert result.detectors.all()
```

The first record bit flags leakage on qubit 1. The next two show that qubit 2
is lost rather than leaked. Unit probabilities make the example deterministic;
a physical noise model supplies the appropriate rates. The probes themselves
are perfect and take no arguments; `READOUT_NOISE` can model missed detections
and false positives.

Partner effects are opt-in. A model can set defaults, use `InteractionRule`
to select particular gates and operand directions, or place explicit effects
in the circuit. Both leaked and lost sources can disturb a partner with Pauli
noise; only leaked sources can spread leakage. See
[Partner Interactions](../guide/partner-interactions.md) and
[Nondestructive Status Checks](../guide/leakage-and-loss.md#nondestructive-status-checks)
for the configuration options.

## Find smaller active states

Clifft's sampling cost depends on the quantum state that remains after the
compiler has accounted for Clifford structure. A circuit can contain many
non-Clifford gates while preparing a state that needs only a small dense
array. Finding that smaller representation saves work on every shot.

The compiler already combines nearby rotations and uses squeezing to shorten
active lifetimes. The two new passes also exploit what is known about the
state: Clifft starts with every qubit in `|0>` and follows the circuit to find
rotations that act identically on the state being simulated. If noise or
measurements invalidate a fact about that state, the passes stop using it.

### Reduce commuting T rotations

`PhasePolynomialPass` finds relationships across a group of commuting
T rotations and expresses their combined action using fewer independent phase
variables. It uses [Vandaele's TOHPE algorithm](https://doi.org/10.22331/q-2025-09-16-1860)
to reduce the T count and preserves the Clifford correction needed for the
same quantum evolution.

Encoded transversal-T circuits are a useful application. Preparing an encoded
state establishes relationships among physical qubits that the pass can use
to reduce the active state. Triorthogonal codes provide a family of these
circuits connected to [magic-state distillation](https://arxiv.org/abs/1209.2426).

### Simplify arbitrary-angle rotations

`RotationSimplificationPass` extends simplification to arbitrary angles. It
combines rotations around different Pauli products when they act identically
on the prepared state, then absorbs any resulting Clifford operations.
For example, qubit 1 remains in `|0>` here, so `Z0*Z1` acts like `Z0` and the
two rotations cancel:

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

Changing the preparation of qubit 1 can invalidate that equivalence.
A larger example is the fixed-input 16-bit Draper QFT adder from
[MQT Bench](../acknowledgments.md#mqt-bench), retained as a
[32-qubit fixture](https://github.com/unitaryfoundation/clifft/blob/037551a9/tools/bench/fixtures/draper_adder_16_basis.stim).
It adds `a=37449` to `b=18724`, preserving `a` and returning `b=56173`.

| HIR pipeline | Peak active width | Peak dense-state dimension |
| --- | ---: | ---: |
| Peephole fusion and squeezing | 15 | 32,768 |
| Add phase-polynomial reduction | 14 | 16,384 |
| Add rotation simplification, the new default | 0 | 1 |

The new defaults eliminate the dense-state growth for these prepared basis
inputs. The table compares pass selections on the same
[source build](https://github.com/unitaryfoundation/clifft/commit/556c3e13fc25f756d145ee2cde954aed853bff6a),
with optional active-width scheduling disabled. It measures state size. The
fixture is also available in
[clifft-bench](https://github.com/unitaryfoundation/clifft-bench/pull/67)
for compilation and sampling measurements.

Tests check relative phases and correlated measurement records against
independent Qiskit Aer and Stim references where applicable, including noisy
circuits and adders with entangled inputs. Both passes are enabled by default
for ordinary compilation and excluded from leakage/loss continuations.
`ActiveWidthSchedulePass` remains opt-in. See the
[pass reference](../reference/passes.md) for controls and applicability, and the
[release notes](https://github.com/unitaryfoundation/clifft/blob/main/CHANGELOG.md)
for the changed defaults and fixed-seed behavior.

## Tune batch sizes for longer jobs

A compiled circuit's fastest sampling configuration depends on the machine,
output mode, and shot count. The default `batch_size="auto"` uses heuristics.
The new `batch_size="tune"` option measures eligible batch capacities before
running the requested shots:

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

Calibration shots are excluded from the result. The budget is a soft limit,
and tuning adds latency, so it is most useful for longer jobs. Reuse the
reported numeric batch size on later calls with the same workload and settings
to avoid recalibrating. See
[CPU Execution and Tuning](../guide/cpu-execution.md#budgeted-batch-calibration)
for the report fields and reproducibility details.
