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
This pass uses signed Pauli constraints proved from the complete circuit's
all-zero input. It retains only fixed eigenvalues shared by every reachable
trajectory. Measurements and stochastic Pauli noise discard relations with
unknown signs; deterministic Pauli noise updates signs. Instruments discard
all entry knowledge. Future postselection never justifies a rewrite. Do not use
this pass on fragments with an unspecified input state.

For a commuting region, write its phase as a Boolean polynomial `f` modulo eight.
After quotienting by the known constraints, change coordinates to `(u, v)` so
that `f(u, v) = f(u, 0) + q(u, v)`, with `q` a diagonal Clifford polynomial.
Project the original parity terms onto the retained coordinates, reduce their
odd coefficients with TOHPE, and reconstruct the exact Clifford correction.
The Clifford part moves into a compiler-side frame; only the non-Clifford core
needs active-state rotations. The executor uses its existing operations.

TOHPE is the third-order homogeneous polynomial elimination algorithm from
[Vandaele, *Lower T-count with faster algorithms*, Algorithm 2](https://arxiv.org/abs/2407.08695).
Clifft implements it natively with deterministic elimination and tie-breaking.
A core of dimension `r` needs at least `r` odd parity terms, so the search skips
representations that already meet this bound. Search is limited to 128 odd
terms per region; larger tables retain the projected parity representation.
This bounds synthesis search, not total compilation time.

`max_variables=32` accepts integers from `0` to `64`; zero disables the pass.
This bounds independent axes modulo known constraints, not circuit qubits.
Exceeding the cap starts a new region. Increasing it can substantially increase
compilation cost because the polynomial is cubic.

Noncommuting rotations, arbitrary-angle rotations, instruments and observers
whose conjugates are not Paulis end a region. Crossed operations must preserve
every constraint the region uses. A Pauli-noise site can move before a phase
prefix only when its nonzero-probability channels commute with that prefix.
Prior scheduling across noise causes the pass to skip the circuit.

The pass does not increase T count. It accepts the complete candidate only if
peak active width decreases, or stays equal without increasing estimated dense
work. This is a guard at the current pipeline position; subsequent squeezing
or scheduling can change the outcome. Benchmark the complete pipeline for your
workload; arbitrary-angle and gate-depolarizing circuits may gain nothing.

After a run, `input_t_count`, `output_t_count`, `blocks_reduced` and
`pauli_pullbacks` describe the accepted rewrite; `applied` reports acceptance.
The pass preserves measurement-record order and is excluded from instrument
continuations.

The default pipeline runs this pass after peephole fusion and before squeezing.
An optional `ActiveWidthSchedulePass` can follow the defaults. Equivalent
rewrites can change samples for a fixed random seed while preserving their
distribution. To opt out of phase reduction while retaining the other defaults,
supply an explicit manager:

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
