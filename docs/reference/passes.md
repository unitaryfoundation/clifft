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
pm.add(clifft.RotationSimplificationPass())
pm.add(clifft.StatevectorSqueezePass())
pm.add(clifft.ActiveWidthSchedulePass())
```

Supply the manager as `hir_passes=pm` to `clifft.compile()` to replace the
defaults. Omit any pass you want to disable, or use `hir_passes=None` to skip
all HIR optimization.

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
`PhasePolynomialPass`, `RotationSimplificationPass`, `StatevectorSqueezePass`
and `ActiveWidthSchedulePass`.

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

`max_variables` limits how many independent phase variables the pass analyzes
together (default `32`, range `0` to `64`; zero disables it). Circuits can have
more qubits than this limit. Raising it may uncover more simplifications but
increases compilation cost. The pass rejects rewrites that increase T count
or peak active width; at unchanged width, estimated sampling work must not
increase either. Actual sampling speedups depend on the complete optimization
pipeline. Equivalent rewrites preserve sampling distributions but can change
samples for a fixed random seed.

After a run, `input_t_count`, `output_t_count`, `blocks_reduced` and
`pauli_pullbacks` describe the accepted rewrite; `applied` reports acceptance.
{% endif %}

{% if p['name'] == 'RotationSimplificationPass' %}
This pass simplifies commuting Pauli rotations using constraints known from the
complete circuit's all-zero input. Equivalent axes can combine even when their
angles are arbitrary. Newly exposed Clifford rotations update a compiler-side
frame immediately, allowing the same region to absorb more following rotations
and recover facts that would otherwise be lost. It makes one forward circuit
sweep and does not repeat earlier passes or add runtime stabilizer tracking.

Every nonrotation operation ends a region. Constraints must hold on every noise
trajectory and the noiseless reference; probability-one noise is no exception.
Measurements retain only outcome-independent facts, instruments clear all facts,
and postselection supplies none. The pass skips circuits whose noise has already
been rescheduled. Use it before squeezing and scheduling.

| Option | Default | Meaning |
|---|---|---|
| `max_region_ops` | `256` | Maximum live rotations in a region; range `0` to `4096`. Zero disables the pass. |
| `max_region_passes` | `8` | Maximum local retries without consuming fresh input; range `1` to `64`. |

The live-term cap bounds pairwise commutation checks within each collection step;
it does not bound the total number of input rotations a shrinking region can
consume. Consuming new input resets the retry budget. These bounds limit local
search, not total compilation time or source-provenance storage, and do not
guarantee a globally optimal result.

Analysis exits early if there are no rotations or no remaining known constraints.
The pass copies the HIR only after finding a rewrite. Nontrivial rewrites are
rejected if either peak active width or estimated dense work increases; deleting
only known scalar phases needs no additional width analysis. This guard does not
guarantee faster compilation or sampling. In particular, benefits observed on
fixed-input adders do not imply the same benefit for arbitrary superpositions.

After a run, `regions_examined` and `regions_capped` describe the attempted search.
`regions_reduced`, `rotations_removed` and `applied` describe only an accepted
rewrite. Merged rotations retain the union of their source locations. Equivalent
rewrites preserve output distributions but can change samples for a fixed seed.
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
