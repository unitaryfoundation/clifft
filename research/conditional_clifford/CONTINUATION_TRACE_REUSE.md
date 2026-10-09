# Fixed Clifford-continuation response reuse

This study reuses the Clifford continuation after the automatically reduced
phase region. It extends the preceding
[prefix-rendering and tracing study](PREFIX_TRACE_REUSE.md), keeping the
single-rotation carrier and production compiler/executor unchanged.

## Construction

`ContinuationPhase` accepts the same raw complete circuit as `PreparedPhase`.
It uses the already discovered phase region, optimized deterministic
preparation, encoder, noise-site map, and sampled-prefix interface. No circuit
name, hand-specified boundary, or protocol-specific matrix enters the new
construction. This path applies when the preceding preparation-reuse route is
eligible and its remaining continuation is Clifford. Later non-Clifford gates,
the existing Clifford/one-rotation routes, and unsupported initial regions
retain their preceding behavior.

Setup builds one continuation containing the fixed encoder, placeholder
records for the sampled prefix, and the complete ideal tail. At every possible
Pauli-fault location it inserts synthetic single-generator probes. A categorical
Y outcome uses both X and Z generators; correlated multi-qubit outcomes select
the corresponding product. Original categorical probabilities and exclusive
outcome selection are unchanged. The synthetic `E(.5)` operations only expose
native traced Pauli axes: the worker removes all these probe operations before
optimization or execution. Their probability has no role in sampling noise.

In the continuation's own input coordinates, let `Q_j` be the rewound Pauli
for fault generator j and `P_i` the signed axis of a later HIR operation. Setup
stores a packed response bit for every anticommuting pair after that fault's
location. A selected fault flips that operation's sign precisely when
`Q_j` and `P_i` anticommute. Multiple selected faults XOR their response bits;
the product of their Pauli axes updates the complete final Clifford frame.
No entire-history cache, repeat-history assumption, or fault-weight cutoff is
needed. Storage scales with generators times operations, rather than the
number of possible histories.

Measurement and reset operations remain explicit in the native trace. Its
Clifford frame does not discard faults at a reset: the signed measurement and
conditional correction together implement the physical erasure. Retaining the
accumulated frame is necessary for equality to fresh tracing. Readout faults
and sampled-prefix MPAD values flip the corresponding measurement sign, so
subsequent feedback consumes the correct visible record. Restoring bits only
after execution would give the wrong conditional physical state.

For each shot, the host draws the original fault history and ideal prefix
outcomes, evaluates the existing boundary controls, and sends only:

- The diagonal Clifford boundary correction, using S, S_DAG, Z, and CZ.
- Selected continuation/boundary Pauli-generator indices.
- Prefix-record and readout-flip indices.

The worker copies the pretraced preparation's forward/inverse frames, applies
the boundary correction, patches the stored continuation, and composes the two
fragments. It then runs the same remaining optimization passes, width check,
ordinary planner, preparation, and sampler as the fresh reference. All of this
construction remains in the research host before native execution. It does
not introduce runtime topology discovery inside a native dispatch loop.

## Measured cost

The retained run uses 128 ordinary trajectories per input, pinned to CPU 2,
with a width budget of 12. Complete per-attempt means include fault drawing,
host control construction, IPC, native construction, optimization, width
inspection, planning, and one sample. Setup and separate correctness requests
are excluded. The fresh reference uses direct optimized-prefix rendering from
the preceding study. Both modes use the same persistent worker and already
constructed templates; this is a paired warm-process comparison, not a
cold-start comparison with Python-only Clifft or Merlin.

| Complete input | Fresh trace | Reused continuation | Throughput ratio | Observed width |
| --- | ---: | ---: | ---: | ---: |
| Noisy factory | 21.699 ms | 17.811 ms | 1.22x | 7-9 |
| Steane D | 4.081 ms | 3.633 ms | 1.12x | 8-9 |
| Steane E | 2.538 ms | 2.358 ms | 1.08x | 8-9 |
| BT27 direct-X | 3.598 ms | 4.412 ms | 0.82x | 3-7 |

The factory takes about 18% less time per shot, D 11% less, and E 7% less.
BT direct-X takes 23% more time. All 512 ordinary histories complete within
the width budget under both constructions; there is no postselection or
discarded noisy tail.

Selected stage means in milliseconds per shot:

| Stage | Factory fresh / reuse | D fresh / reuse | E fresh / reuse | BT fresh / reuse |
| --- | ---: | ---: | ---: | ---: |
| Host rendering / controls | 3.538 / 2.076 | 0.669 / 0.359 | 0.580 / 0.309 | 1.019 / 0.671 |
| Native parse + trace + composition | 4.594 / 3.636 | 1.051 / 1.246 | 0.629 / 0.916 | 0.434 / 1.677 |
| Optimization + width inspection | 4.753 / 4.702 | 0.675 / 0.668 | 0.376 / 0.372 | 0.447 / 0.435 |
| Ordinary planning | 6.664 / 6.637 | 1.096 / 1.095 | 0.544 / 0.540 | 0.749 / 0.746 |
| Prepare + sample | 0.082 / 0.081 | 0.038 / 0.039 | 0.028 / 0.028 | 0.040 / 0.040 |

Native composition alone now costs 3.62, 1.23, 0.90, 1.66 ms
for factory, D, E, and BT respectively. The response patch avoids tracing,
but the current general frame/axis composition can exceed fresh trace cost.
Much of the total benefit comes from smaller host construction and requests.
Mean request sizes fall from 75032, 15416, 7871, 2991
bytes to 629, 304, 324, 627 bytes, respectively.

Frontend setup takes 0.783, 0.229, 0.214, 0.758 seconds;
worker startup adds 25.8, 6.2, 4.2, 2.8 ms, including native
template setup of 23.9, 4.3, 2.5, 1.5 ms. Setup is shared by the
comparison modes and is not included in the per-shot means. The full-panel
host peak is 167.5 MiB; native worker peaks are
9.30, 6.79, 6.58, 6.15 MiB.

The factory's scheduled continuation has no remaining Pauli-channel sites:
its selected earlier faults enter through the boundary response. D and E also
exercise noise genuinely inside the continuation, with 243 and 189 categorical
tail sites respectively. Readout sites remain in all four inputs. Dense stress
histories select an outcome at every original site, including 2,349 sites for
the factory, 459 for D, 351 for E, and 4,170 for BT direct-X. These are algebra
checks, not draws from a conditioned noise distribution.

The response-bit payload is 93.0 KiB for the factory, 19.0 KiB for D, 10.1 KiB
for E, and 4.2 KiB for BT. These figures exclude generator axes, frames,
container overhead, and the existing phase frontend's storage. Each native
process peak includes both modes and validation requests; host and child
peaks are neither isolated per-mode costs nor simultaneous unique memory.

## Correctness evidence

The [large-run artifact](continuation-trace-study.json) contains 128 freshly
drawn histories per eligible input, alternating construction order with
matched fault/prefix/native-sample seeds. Correctness requests run outside
timed requests. It records:

- **561 exact native HIR comparisons:** 512 ordinary draws and 49 selected
  stress histories. Comparisons include operation payloads, signed axes,
  flags, visible/hidden record metadata, detector/observable maps, expectation
  metadata, and the complete final Clifford frame. Arena handles and source
  provenance are excluded.
- **1,024 timed native samples:** both constructions return identical records,
  declared detector/observable parities, widths, and expectations for matched
  seeds. Parities are independently recomputed from the visible records.
- **34 unchanged fallback checks:** scored BT27, cultivation d3/d5, and an
  unsupported initial rotation preserve the preceding frontend route.

The [independent small validator](continuation-trace-validation.json) includes:

- 32 random three-qubit prefix/continuation pairs, each with all eight
  subsets of three Pauli-product insertions and varying boundary corrections.
  All **256 full density matrices** match Aer within `3.4e-16`.
- Two measured two-qubit circuits, one Clifford and one non-Clifford, with
  all 16 combinations of three faults and one readout flip. For each combination,
  all nine local Pauli measurement bases check the outgoing state conditioned
  on the earlier records. These **288 complete joint record laws** match Aer
  within `1.7e-16`, including hidden reset branches. This informationally
  complete tomography checks more than the original circuit's output bits.
- **144 exact Stim record-law comparisons** for the Clifford half of that
  panel, with maximum error below `5.6e-17`.
- Ten original/renamed frontend inputs, 66 fault histories, and 80 enumerated
  prefix/decision branches. Complete record-conditioned density matrices of
  the preceding conditional source match the original input within `7.0e-17`.
  The new construction additionally matches **72 exact conditional traces**
  and their complete native record laws, within `3.9e-16` of Aer. These cover
  correlated categorical channels, sampled prefix records, inversions,
  feedback, reset, annotations, and the later-magic/carrier fallbacks.
- Seven rejection controls for live non-probe noise, live readout noise,
  non-Clifford or wider continuations, invalid generator/record indices, and
  unsupported boundary corrections. A readout-postprocessing-only negative
  control changes the conditional state by 1.0 in a density-matrix entry.

The preceding prefix-fragment validator also passes against the extended
worker; its [retained regression result](continuation-prefix-regression.json)
includes arbitrary-angle tails on the old path, full-state composition,
complete instrument record laws, and earlier rejection controls.

Large-circuit exact HIR equality establishes equivalence to the already reduced
conditional source, not an independent certification of every original large
reduction or a rare accepted-error rate. Earlier evidence for those reductions
remains in the [factory assessment](FACTORY_FRONTEND_ASSESSMENT.md). The 128-shot
timing sample is not used to claim statistical validation of rare events.

## Assessment and next bounded question

Fixed-continuation reuse is algebraically sound within this interface and
removes repeated tail parsing/tracing. It gives a useful factory improvement,
smaller D/E gains, and a BT slowdown. Keep it as an explicit research option;
these results do not justify selecting it for every eligible input.

The cost has partly moved into composing the changing boundary with the
stored continuation. The next bounded compilation study should distinguish
frame multiplication, axis transformation, copying, and control application
inside that composition step, then test whether fixed frame products or
boundary-action responses can be precomputed. A targeted native profile or
substage timings would now give useful guidance. Retain ordinary per-shot
planning until that experiment demonstrates a benefit without changing the
complete state/record interface.

The factory's ordinary planning still takes about 6.6 ms per shot. A separate
planner-reuse proposal must account for the earlier observed fault-dependent
measurement-law ranks; an ideal plan with output-bit flips alone is insufficient.
Changing the execution lifecycle requires the repository's separate architectural
review. Expanding beyond the single-rotation carrier is still deferred.

This checkpoint concerns construction cost on the existing circuit panel.
It does not expand the structural applicability of phase reduction or solve
arbitrary non-Clifford continuation entry. Future capability work needs its own
candidate circuits and feedback loop. Independent noise-free optimization
remains worthwhile, as previously agreed, but is outside this noisy study.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#fixed-clifford-continuation-reuse).
