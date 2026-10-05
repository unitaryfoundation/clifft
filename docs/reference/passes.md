# Optimization Passes

Clifft's public optimization passes operate on the Heisenberg IR before
active-coordinate planning and executable-plan preparation.

## Default Pipeline

The default HIR pipeline:

{% for p in default_hir_passes %}
1. **{{ p['name'] }}** -- {{ p['summary'] }}
{% endfor %}

Use `clifft.default_hir_pass_manager()` to get these defaults, or build a
custom pipeline:

```python
import clifft

pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())
pm.add(clifft.PhasePolynomialPass())
pm.add(clifft.StatevectorSqueezePass())
pm.add(clifft.ActiveWidthSchedulePass())
```

Optimization passes end at the HIR boundary; `SamplingPlan` is not a second
public pass pipeline. See [Software Architecture](../theory/architecture.md)
for the private planning and executable-preparation stages that follow.

## Trajectory Safety Metadata

Some workflows require measurements to remain in their original order,
including hidden measurements introduced by the compiler. In particular,
`clifft.noncomp.sample` may force the result of a hidden trace-out measurement
when it resumes a trapped transition. Moving another measurement across that
collapse can change quantum correlations.

`clifft.noncomp.sample` therefore applies only passes that are enabled by
default, preserve measurement-record order, and preserve instrument prefixes.
Its HIR pipeline uses `PeepholeFusionPass` but omits
`PhasePolynomialPass`, `StatevectorSqueezePass` and `ActiveWidthSchedulePass`.

Record-order preservation is necessary but does not by itself make a
continuation compatible with an already-running executor. Trajectory passes
must also opt in to instrument-prefix stability: changing the circuit after an
instrument may not change optimized output through that instrument. The
runtime checks each recompiled prefix before resuming. For this reason,
`clifft.noncomp.sample` uses a fixed internal
pipeline and does not currently accept custom pass managers. See
[Leakage and Loss](../guide/leakage-and-loss.md#why-there-is-no-compile-step)
for how continuations are compiled and resumed.

---

## HIR Passes

{% for p in hir_passes %}
### {{ p['name'] }}

| | |
|---|---|
| **Kind** | HIR (pre-lowering) |
| **Default** | {{ '✅ Enabled' if p['default_enabled'] else '❌ Disabled' }} |
| **Preserves measurement-record order** | {{ 'Yes' if p['preserves_record_order'] else 'No' }} |
| **Preserves instrument prefix** | {{ 'Yes' if p['preserves_instrument_prefix'] else 'No' }} |
| **Python** | `clifft.{{ p['python_name'] }}()` |

{{ p['detail'] }}

{% if p['name'] == 'PhasePolynomialPass' %}
Pauli measurements, expectation-value probes and classically controlled Pauli
gates can move before a phase prefix only when their conjugated axes remain
Paulis and all required constraints survive. Pauli noise can cross only when
its nonzero-probability channels commute with that prefix and preserve those
constraints. These must hold on every trajectory and the noiseless reference,
so even probability-one noise discards affected constraints. Instruments clear
all constraints; postselection supplies none. Use this pass only on complete
circuits with an all-zero input; prior scheduling across noise causes it to skip
the circuit.

`max_variables=32` accepts integers from `0` to `64`; zero disables the pass.
This bounds independent axes modulo known constraints, not circuit qubits.
Exceeding the cap starts a new region. TOHPE searches at most 128 odd parity
terms per region; larger inputs retain their projected parity representation.
Increasing the variable cap can substantially increase compilation cost.

Rewrites cannot increase T count or current peak active width; at equal peak,
estimated dense work must not increase. This guard does not guarantee faster
sampling after subsequent passes. Arbitrary-angle rotations and incompatible
noise can prevent useful collection.

After a run, `input_t_count`, `output_t_count`, `blocks_reduced` and
`pauli_pullbacks` describe the accepted rewrite; `applied` reports acceptance.
Equivalent rewrites can change samples for a fixed random seed while preserving
their distribution. To opt out while retaining the other default passes:

```python
import clifft

pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())
pm.add(clifft.StatevectorSqueezePass())
program = clifft.compile("H 0\nT 0\nM 0", hir_passes=pm)
```

Pass `hir_passes=None` to `clifft.compile()` to disable all HIR optimization.
{% endif %}

{% if p['name'] == 'ActiveWidthSchedulePass' %}
See [Compiling Circuits](../guide/compilation.md#active-width-scheduling) for
setup and guidance on when to enable this pass.

| Option | Default | Meaning |
|---|---|---|
| `beam_width` | `8` | Positive number of partial schedules to retain. Larger beams explore more alternatives but do not guarantee better results. |
| `search_budget` | `16.0` | Search executions per HIR operation before switching to greedy continuation. `0` requests greedy continuation after the initial sweep; `None` disables budget-driven narrowing. |
| `noise_transparent` | `True` | Allow crossings of Pauli noise, with symbolic sign correction. |
| `sink_neutral_rotations` | `True` | Delay width-neutral rotations where legal to reduce dense work and expose fusion opportunities. |

The search narrows to one retained candidate at half the execution budget.
Ongoing sweeps and replays finish, so the thresholds can be exceeded. This
budget does not bound classification probes or wall time: probe counts can
grow quadratically on circuits with many independent rotations, and each
probe's cost also depends on the qubit count. Budgets must be finite and
nonnegative, or `None`.

After a run, `incumbent_peak` and `result_peak` report peak active width before
and after scheduling. `incumbent_dense_work` and `result_dense_work` estimate
work by summing $2^w$ over operations touching the active array at width $w$.
The pass accepts a lower peak even if that work estimate rises; neither metric
measures elapsed time or guarantees a globally optimal schedule.

`applied` reports whether the order changed. `swept_ops` counts search
executions, including discarded candidates and replays; `classification_probes`
counts checks for expanding operations. These counters exclude graph
construction and final analysis. For per-operation widths, use
[`clifft.active_width_trace`](python-api.md#clifft.active_width_trace).
{% endif %}

{% endfor %}
