# Preparation rendering and parsed/traced prefix reuse

Date: 2026-10-09. Branch: `codex/bt27-fault-specialization`.

## Finding

Directly emitting the already optimized preparation avoids about 0.7-0.8 ms
of repeated host work per factory/D/E shot. Reusing its parsed or traced native
representation adds a smaller benefit on D/E and little or none on the larger
factory. Further prefix-only tuning is not the next high-value target.

This experiment preserves the preceding reduction rules, all-zero circuit
entry assumption, conditional records, supported Pauli fault model, and
single-rotation carrier. All construction and composition occur before the
unchanged native optimizer/planner/executor. There are no production source,
public API, or execution lifecycle changes.

## What is shared

The preceding `ReusablePhase` optimized a fixed unmeasured state preparation
once. Its per-shot implementation nevertheless rendered the original larger
preparation, reparsed it to count surviving T gates, and then replaced it with
the optimized source. The preparation and T count are already known from
setup; only Clifford corrections and the continuation vary on this route.

The new research `PreparedPhase` emits the stored optimized preparation
directly. It reuses the physical wire map and constant step metadata while
evaluating the same existing correction algebra, original fault history, and
preparation measurement sampler. All outgoing records, offsets, annotations,
and tail operations are retained. The original preparation may contain live
measurements: the reusable unmeasured fragment reconstructs their conditional
state, and their sampled visible values remain in the following MPAD records.
This does not remove or fix those measurement outcomes across shots.

Eligible inputs take the same surviving-remainder route as before. Clifford
completion, the single-rotation carrier, and unsupported inputs retain the old
implementation. Direct emission is usable with the existing Python compiler;
it does not require the new native diagnostic worker.

The native worker compares three compilation routes for that identical source:

| Mode | Shared setup | Per-shot construction |
| --- | --- | --- |
| Fresh | Prefix text | Parse and trace the complete circuit |
| Parsed | Prefix AST | Parse the continuation, prepend copied prefix nodes, trace all |
| Traced | Prefix HIR and inverse final frame | Parse/trace the continuation independently, compose its HIR with the prefix |

The tracer has no resumable-prefix API. The traced mode uses existing HIR
builders and tableau operations in an opt-in research executable, like the
earlier boundary-composition studies. It neither patches the tracer nor moves
tableau evolution into native execution.

If the prefix has final Clifford frame F, each independently traced
continuation Pauli P becomes `F^-1 P F`. Prefix rotations precede those
transformed operations, and the final physical frame is the prefix frame
followed by the continuation frame. Signed axes matter. This transformation
applies equally to measurements, feedback Paulis, expectation probes, and
surviving rotations. The composition itself adds no new regional reduction or
ability to carry multiple rotations through the research analyzer.

The prefix must have no visible or hidden measurements, annotations with
outputs, noise, or instruments. Since all records belong to the continuation,
their visible and hidden indices stay unchanged. Both fragments must fit the
same physical width, padded by the prefix's explicit identity instruction.
Noise has already been drawn and materialized by the host. Unsupported
interfaces reject rather than being approximated. Source provenance is
rebased for the parsed assembly and cleared for independent HIR composition;
it is not part of the semantic trace comparison.

## Paired measurements

The retained [driver output](prefix-trace-study.json) uses 128 fresh physical
histories per eligible input, with paired preparation and native sampling
seeds. It alternates old/direct rendering order and rotates native mode order.
Each native mode receives exactly the same conditional source. Small
validation ran on another pinned CPU at the start of the timing run.

Reported costs sum fault drawing, the measured renderer, and its worker
request/response time. That boundary includes input framing, IPC, JSON output,
parsing/tracing, remaining optimization, width inspection, planning, executable
construction, and one-shot sampling. The old-rendering baseline and direct
rendering use the same fresh native backend measurement to isolate rendering
cost. Input generation, setup, parity checks, and separate exact-validation
requests are excluded. Every attempt is retained and width is checked before
dense execution; all reported attempts complete.

This worker adds an IPC boundary absent from the preceding Python benchmark,
and uses a new random seed. Compare columns within this table, rather than
treating differences from the preceding study's absolute timings as a
regression or gain. The native worker is a diagnostic, not an integrated
backend or an automatic cost-based selector.

All values are milliseconds per complete shot, excluding setup.

| Input | Prior rendering + fresh | Direct rendering + fresh | Direct + parsed | Direct + traced | Observed width, all modes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noisy factory | 22.626 | 21.780 | 21.689 | 21.938 | 7-9 |
| D | 4.876 | 4.202 | 4.066 | 3.946 | 8-9 |
| E | 3.230 | 2.551 | 2.441 | 2.391 | 8-9 |
| BT27 direct X | 3.774 | 3.693 | 3.676 | 3.655 | 3-7 |

| Rendering alone | Before ms | Direct ms |
| --- | ---: | ---: |
| Noisy factory | 4.378 | 3.532 |
| D | 1.351 | 0.677 |
| E | 1.267 | 0.587 |
| BT27 direct X | 1.105 | 1.023 |

Direct emission cuts complete cost by about 4%, 14%, and 21% on factory/D/E
within this harness. Beyond that improvement, traced reuse saves about 0.26 ms
on D and 0.16 ms on E. The factory's extra frame composition costs more than
its avoided prefix tracing. Tiny differences among the BT and factory native
modes should not be treated as robust throughput advantages from this short
local run.

| Native stages | Factory fresh ms | Factory traced ms | D fresh ms | D traced ms | E fresh ms | E traced ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parse | 1.131 | 0.971 | 0.451 | 0.254 | 0.280 | 0.133 |
| Trace | 3.349 | 3.313 | 0.559 | 0.515 | 0.297 | 0.258 |
| Compose | 0 | 0.431 | 0 | 0.057 | 0 | 0.041 |
| Remaining optimization + width inspection | 4.749 | 4.712 | 0.677 | 0.662 | 0.380 | 0.378 |
| Plan | 6.767 | 6.770 | 1.111 | 1.095 | 0.540 | 0.540 |
| Prepare + sample | 0.087 | 0.084 | 0.042 | 0.040 | 0.030 | 0.029 |

The standalone prefix contains 1,299 parsed nodes/33 HIR rotations for the
factory, 1,227/33 for D/E, and 347/23 for BT direct X. Removing those nodes
from tracing barely changes trace time: the entire physical-width continuation
and its frame still have to be traced. Prefix reuse also leaves subsequent
planning essentially unchanged. This locates the remaining work more precisely
than the total compilation cost did.

Total frontend setup is 0.751, 0.223, 0.191, and 0.762 seconds for factory, D,
E, and BT direct X. Worker startup adds 8.0, 2.5, 2.2, and 2.0 ms, including
native prefix setup of 6.1, 1.2, 0.7, and 0.7 ms. Setup builds both shared
representations for this comparison, even when timing the fresh mode.

The host peak is 167.0 MiB across all inputs. Per-input native process peaks,
read from Linux `/proc/self/status` rather than inherited child `getrusage`,
are 9.81, 6.92, 6.64, and 6.04 MiB. These include validation requests and all
three modes within each worker; they do not isolate the cost of one mode or
coefficient storage. The separate process peaks are not simultaneous unique
memory and should not simply be added.

## Correctness evidence

For each native validation request, the worker builds a fresh full-circuit
trace and compares every operation's type, payload, signed Pauli mask, and
flags, plus record counts, detector/observable target lists, expectation
metadata, and the complete final Clifford frame. Arena handles and source
line provenance are deliberately excluded. Equality establishes the same
conditional HIR semantics, including its physical-state interface, rather
than merely agreement on a finite output sample.

The large run passes:

- **557 identical-source comparisons:** 512 freshly sampled trajectories and
  45 selected stress histories produce byte-identical source and identical
  frontend metadata under old versus direct rendering.
- **1,069 exact native trace comparisons:** both alternative constructions on
  all 512 trajectories, plus traced composition on the 45 stress histories.
  Stress inputs include identity, selected individual outcomes, and selected
  multiple-fault histories without conditioning the ordinary draws.
- **1,536 completed native samples:** three modes on the 512 trajectories.
  With matched seeds, returned records, detectors, observables, and widths
  are identical. Declared parities are independently recomputed from records.
- **34 unchanged fallback checks:** scored BT27, cultivation d3/d5, and an
  unsupported initial rotation retain the prior route on selected histories.

The [small validator](prefix-trace-validation.json) additionally includes:

- 48 randomized three-qubit prefix/continuation pairs, including noncommuting
  gates, signed T rotations, and arbitrary-angle tail rotations. Composed
  native physical states match Aer density matrices to below `1.2e-15`.
- An empty continuation and an independently evaluated Pauli expectation.
- Four measured continuations checked in both parsed and traced modes against
  complete Aer record laws, summing all hidden reset branches. Maximum error
  is below `2.8e-16`. These include MPAD records, inverted observables, feedback,
  reset, later magic, detectors, and logical observables.
- Ten original/renamed frontend circuits, 20 fault histories, and 34 enumerated
  branches. Complete record-conditioned density matrices match the original
  input, with maximum error below `1.3e-16`; native record-law error is below
  `4.5e-16`. There are 28 additional exact composed-trace comparisons and
  complete conditional record-law comparisons through the native worker.
- Rejection controls for a measured prefix, noisy prefix, live tail noise,
  and a continuation wider than the declared interface.

These new checks target composition and serialization reuse. The original
phase reduction and large noisy-law evidence remain those of the preceding
[factory assessment](FACTORY_FRONTEND_ASSESSMENT.md) and
[compiled-prefix study](COMPILED_PREFIX_REUSE.md). Exact agreement with the
already reduced conditional source does not independently certify every
large-circuit reduction or a rare accepted-error rate.

## Decision and next bounded question

Keep direct preparation emission as a useful host-side improvement. Preserve
the native modes as reproducible experiments; this evidence does not support
making traced-prefix composition the default for every input. Do not expand
the non-Clifford carrier in response to these costs.

The next reuse question is the **fixed continuation and its noise responses**:
can it be traced once, then attached to the changing Clifford correction while
preserving all records and outgoing state? A bounded follow-up should derive
the Pauli/readout-fault responses of a Clifford continuation from its raw
operations and test construction cost against fresh continuation tracing.
It should use the same factory/D/E and smaller controls, including feedback
and reset, rather than introducing protocol-specific matrices or boundaries.

Such a study should initially retain ordinary per-trajectory planning. On the
factory that still costs about 6.8 ms; prefix reuse alone has not reduced it.
Earlier fault-dependent measurement-law ranks rule out sharing the ideal plan
with only output-sign changes. Any proposal to share more of the planner or
change the execution lifecycle must account for those dependencies explicitly
and receive the separate architectural review required by the repository.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#parsed-and-traced-prefix-reuse).
