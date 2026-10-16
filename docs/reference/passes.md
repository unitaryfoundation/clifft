# Optimization Passes

Clifft's public optimization passes operate on the Heisenberg IR before
active-coordinate planning and executable-plan preparation.

## Default Pipeline

The default HIR pipeline:

{% for p in default_hir_passes %}
1. **{{ p['name'] }}** -- {{ p['summary'] }}
{% endfor %}

These passes are complementary; the default manager runs them in the order shown.

Use `clifft.default_hir_pass_manager()` to get these defaults. To opt into
128-variable phase collection while retaining the rest of the pipeline:

```python
import clifft

phase = clifft.PhasePolynomialPass(max_variables=128)
pm = clifft.default_hir_pass_manager(overrides=[phase])
program = clifft.compile("H 0\nT 0\nM 0", hir_passes=pm)
```

Each override replaces the corresponding pass in its original position,
regardless of the order of `overrides`. The manager retains the supplied
instances, so pass statistics such as `phase.output_t_count` remain accessible.
Duplicate pass types and passes outside the default pipeline raise `ValueError`.

For other pass choices or ordering, build a custom pipeline:

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

## Constraints from Measurement and Feedback

`PhasePolynomialPass` and `RotationSimplificationPass` recover fixed-sign
stabilizer constraints through measurement and feedback, including after earlier
non-Clifford work. Constraints must hold for every trajectory and the noiseless
reference; postselection supplies no additional constraints. See
[Optimization for Simulation](../theory/stabilizer-optimization.md) for the
analysis and reduction principles.

These are static rewrites shared by all shots. Unresolved affine signs do not
produce conditional rewrites or per-history specialization. To disable recovered
constraint rewrites, omit both passes from a custom `HirPassManager`, or use
`hir_passes=None` to disable all HIR optimization. Equivalent plans preserve
output distributions but do not guarantee historical fixed-seed samples.

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

`max_variables` is a hard ceiling on independent phase variables analyzed
together (default `64`, range `0` to `128`; zero disables the pass). Circuits
can have more physical qubits than this limit. Collection starts at
`min(32, max_variables)`, then tries a complete region up to 64 variables.
`max_variables=32` disables expansion. Fixed-constraint recovery through
measurement and feedback remains enabled with the default 64-variable ceiling.

Set `max_variables=128` explicitly to enable wider collection. If the 64-variable
attempt is capped, the pass can try a complete contiguous commuting T region
using two-word coordinates. This optional path keeps an
ordinary-basis candidate and tries one basis derived from the final Clifford
tableau. It crosses no measurement, feedback, noise, or instrument boundaries.
Synthesis still supports at most 64 core variables and searches at most 128 odd
parity terms. A signature-rank precheck rejects unsupported cores before cubic
polynomial expansion.

`max_wide_work` (default `50000000`) bounds deterministic work units shared by
all optional attempts and both basis candidates in one pass. The units include
polynomial updates, charged row operations, and an allowance for synthesis.
Work units are not a wall-clock limit; their cost depends on the polynomial.
Wide search can add substantial compilation time even when it finds no benefit,
so it is disabled by default.
`max_wide_terms` (default `200000`) bounds each polynomial's size. Setting either
to zero disables the optional path. Exhaustion retains a completed valid
candidate or falls back to the ordinary smaller-region policy. These limits
control compiler effort and do not change the execution representation.

Larger complete regions can uncover additional reductions, at extra compilation
cost. The pass rejects rewrites that increase T count
or peak active width; at unchanged width, estimated sampling work must not
increase either. Actual sampling speedups depend on the complete optimization
pipeline. Equivalent rewrites preserve sampling distributions but can change
samples for a fixed random seed.

After a run, `input_t_count`, `output_t_count`, `blocks_reduced` and
`pauli_pullbacks` describe the accepted rewrite; `applied` reports acceptance.
`wide_attempts`, `wide_rewrites`, `wide_work`, and `wide_budget_exhausted` report
the optional search, including rejected candidates; `wide_rewrites` counts only
accepted rewrites. All counters reset on each run.
{% endif %}

{% if p['name'] == 'RotationSimplificationPass' %}
Run this pass on complete circuits before squeezing or scheduling. It skips
circuits whose noise has already been rescheduled.

Stabilizer constraints must hold for every measurement outcome and noise
trajectory, including the noiseless reference. Every nonrotation operation ends
a region; instruments discard the known stabilizers and postselection adds none.

| Option | Default | Meaning |
|---|---|---|
| `max_region_ops` | `256` | Maximum rotations retained in a region; range `0` to `4096`. Zero disables the pass. |
| `max_region_passes` | `8` | Maximum local retries without adding new rotations; range `1` to `64`. |

Rewrites must not increase peak active width or estimated sampling work.
Equivalent rewrites preserve output distributions but can change samples for a
fixed seed.

After a run, `regions_examined` and `regions_capped` describe the attempted search.
`regions_reduced`, `rotations_removed` and `applied` describe the accepted rewrite.
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
