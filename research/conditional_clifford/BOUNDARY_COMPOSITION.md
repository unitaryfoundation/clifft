# Algebraic diagonal-boundary composition

The [continuation-reuse study](CONTINUATION_TRACE_REUSE.md) removed repeated
tracing, but attaching the changing Clifford correction still cost enough to
slow down BT direct-X. This follow-up locates and removes that bottleneck while
retaining the same phase frontend, fault laws, native HIR, per-shot planning,
and execution lifecycle.

## Finding and change

New native substage timers separate boundary-frame construction, fault/record
patching, axis transformation and HIR assembly, and final frame multiplication.
They show that boundary-frame construction dominates the old composition.
Inspecting the current native tableau implementation explains why: appending
each small gate to a full forward tableau visits every row and builds temporary
local Pauli objects. Updating its inverse by prepending the inverse gate is
already much cheaper. Whole-frame multiplication and fault-response patching
are not the principal bottlenecks here.

The correction at this interface is a diagonal Clifford. Its full action can
be represented by single-wire S powers `k_a` modulo four and CZ edges `(a,b)`.
For a tableau row written in the native convention `i^p X^x Z^z`, conjugation
by the complete correction preserves `x` and applies:

```text
S_a^k: p += k*x_a                    (mod 4)
       z_a ^= (k mod 2)*x_a

CZ_ab: p += 2*x_a*x_b                (mod 4)
       z_a ^= x_b
       z_b ^= x_a
```

All terms use unchanged X bits, so they can be accumulated in one traversal of
each row. Preserving the raw phase, rather than only the unsigned X/Z masks,
is essential for Y-containing rows and inverted observables. Repeated S powers
combine modulo four; repeated CZ edges cancel by repeated XOR/phase updates.
The final complete frame remains exactly equal to generic gate-by-gate
construction, up to the physically irrelevant global circuit phase already
absent from a Clifford tableau.

The new `diagonal` worker mode uses this formula for the forward frame and
keeps the existing cheap inverse updates. It retains both `fresh` and
`continuation` as explicit comparison modes. There is no cache of corrections
or complete fault histories, and no added persistent response table. This is
an algebraic batching improvement to the research host, rather than additional
compile-once preparation reuse. The previous continuation template remains
shared exactly as before.

The formula applies to any diagonal Clifford acting on any Clifford frame;
it does not inspect circuit names or preparation patterns. The enclosing
frontend still has all of its previous structural limits. There are no changes
to `src/clifft`, Stim, native hot dispatch, or the single-rotation carrier.

## Paired results

Complete per-attempt means:

| Complete input | Fresh trace | Previous reuse | Diagonal update | Less time vs previous reuse |
| --- | ---: | ---: | ---: | ---: |
| Noisy factory | 21.523 ms | 17.553 ms | 14.536 ms | 17% |
| Steane D | 4.032 ms | 3.672 ms | 2.435 ms | 34% |
| Steane E | 2.652 ms | 2.409 ms | 1.574 ms | 35% |
| BT27 direct-X | 3.716 ms | 4.374 ms | 2.940 ms | 33% |

Composition substage means in milliseconds, comparing previous reuse with
the diagonal update:

| Substage | Factory old / new | D old / new | E old / new | BT old / new |
| --- | ---: | ---: | ---: | ---: |
| Boundary frames and input copies | 3.103 / 0.116 | 1.239 / 0.045 | 0.854 / 0.035 | 1.480 / 0.052 |
| Fault/record patches and tail copy | 0.049 / 0.045 | 0.007 / 0.006 | 0.005 / 0.005 | 0.009 / 0.008 |
| Axis pullback and HIR assembly | 0.309 / 0.307 | 0.029 / 0.030 | 0.020 / 0.020 | 0.019 / 0.019 |
| Final frame product | 0.072 / 0.072 | 0.024 / 0.024 | 0.016 / 0.016 | 0.018 / 0.018 |

Boundary-frame construction accounts for 88%, 95%, 95%, 97%
of the previous composition cost, respectively. The new update removes most
of that work. Total new composition is only 0.544, 0.107, 0.078, 0.099
ms per shot. The few substage timers do not isolate allocator cost, and their
sum excludes some function-return/destruction overhead; whole-shot means
include that overhead.

| Remaining stage with diagonal update | Factory | D | E | BT |
| --- | ---: | ---: | ---: | ---: |
| Optimization plus width inspection | 4.550 ms | 0.657 ms | 0.379 ms | 0.445 ms |
| Ordinary planning | 6.608 ms | 1.088 ms | 0.540 ms | 0.752 ms |
| Prepare and sample | 0.081 ms | 0.037 ms | 0.030 ms | 0.041 ms |

Factory optimization/width inspection and planning now account for about
77% of complete per-shot time. Their costs remain essentially unchanged.
Observed active widths are 7-9, 8-9, 8-9, 3-7
for factory, D, E, and BT respectively, matching both reference constructions.

Frontend setup is 0.741, 0.222, 0.213, 0.750 seconds;
native worker startup adds 25.3, 5.8, 3.8, 2.7 ms, including template
setup. Host peak across the panel is 166.8 MiB; native process peaks
are 9.39, 6.94, 6.60, 6.12 MiB.

Setup still builds both shared preparation and continuation representations.
No additional persistent data structure is required by diagonal batching;
its per-shot powers and edge list live in the research host. Process peaks
include all modes and validation and do not isolate one construction's memory.
The same width guard executes before dense coefficient allocation. All sampled
histories in this panel finish under the width budget, and the timing draw law
is unconditioned.

These are local warm-process means, with 128 ordinary trajectories per input,
matched fault/prefix/native-sample seeds, and cyclic mode order on CPU 2. They
include fault drawing, host construction, IPC, native construction, remaining
optimization, width analysis, ordinary planning, and sampling. Setup and
correctness requests are outside the timed requests. Compare the three modes
within this run; this is not a direct performance comparison with ordinary
compile-once Clifft or Merlin.

## Correctness

The [timing and exact-comparison artifact](boundary-composition-study.json)
records **1,122 exact HIR comparisons**: both reuse modes against full fresh
tracing for 512 ordinary histories and 49 selected stress histories. The latter
include an outcome at every original fault site. Signed operation payloads,
visible and hidden record metadata, detector/observable maps, and the complete
final Clifford frame agree. Its **1,536 timed samples** also agree exactly for
matched seeds, including detector/observable parity audits. Existing scored
BT27, cultivation d3/d5, and unsupported-input routes pass 34 fallback checks.

The [boundary algebra validator](boundary-algebra-validation.json) adds:

- Every element of the three-qubit diagonal Clifford group modulo global phase:
  `4^3 * 2^3 = 512` combinations of S powers and CZ edges. The tests also insert
  canceling/repeated gates and alternate a multi-qubit Pauli fault. Every
  composed HIR equals fresh tracing, and its full final density matrix agrees
  with Aer. This exhausts corrections for the chosen preparation/continuation,
  not all possible quantum circuits.
- 96 constructions at physical widths 65, 129, and 193, with gates on both
  sides of mask-word boundaries, sampled Pauli controls, readout inversions,
  feedback, reset, and detector/observable records. Both construction modes
  pass **192 exact full-HIR comparisons** to fresh tracing and return identical
  seeded samples. These large physical-width checks use a small active state;
  their complete-frame comparisons cover all physical wires.

The existing independent continuation validator also passes with
`--mode diagonal`; its [new result](boundary-continuation-validation.json)
retains 256 randomized full-state comparisons with Aer, 288 complete joint
record laws with informationally complete final-state tomography, 144 exact
Clifford-law comparisons with Stim, 72 frontend conditional-trace comparisons,
original-source instrument checks, and all seven rejection controls.
The [prefix regression result](boundary-prefix-regression.json) retains checks
of the earlier worker modes, including non-Clifford continuations on those
paths.

Exact large-circuit HIR equality establishes equivalence to the already
reduced conditional source. Independent small-state checks support the new
algebra and existing interface; this does not independently establish every
large original reduction or certify rare accepted-error rates. The 128-shot
runs serve timing and matched-trajectory checks, not rare-event statistics.

## Assessment checkpoint

The focused change removes the dominant boundary-construction overhead and
improves all four measured eligible inputs, recovering the earlier BT slowdown.
Keep diagonal batching as the preferred experimental continuation construction
when using this research path. It does not imply that the conditional frontend
beats ordinary Clifft on every circuit, or that it should replace that backend.

Boundary composition is now a small part of the complete shot cost. Further
micro-optimization there has limited upside. The useful next question is which
parts of the remaining optimization/width analysis and planner depend on the
sampled boundary, and whether any invariant portion is worth reusing. Start
with a cost and dependency assessment on this same panel, preserving complete
state/record equivalence and using fresh planning as the reference. Matching
active widths alone is not evidence that plans can be shared.

Earlier fault-dependent measurement-law ranks still rule out an ideal plan
with only output-bit corrections. No planner reuse or executor lifecycle change
has been implemented here; an architecture change would need its own proposal
and approval. Expanding the non-Clifford carrier remains deferred. This is a
checkpoint in compilation cost, not an expansion of the circuit families the
phase reduction can recognize.

Commands are in the
[profiling guide](../../tools/profile/README.md#diagonal-boundary-composition).
