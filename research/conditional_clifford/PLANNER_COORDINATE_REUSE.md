# Coordinate reuse during fresh planning

This follows [guarded squeeze scheduling](SQUEEZE_SCHEDULE_REUSE.md). The
experiment measures how often a planner frame survives multiple Pauli-coordinate
queries and compares three alternatives with the existing planner. The strongest
general candidate is an identity-Pauli shortcut: factory shot time falls from
12.41 to 9.55 ms, and BT27 direct-X from 2.75 to 2.52 ms. More ambitious inverse
reuse has mixed results and is not selected automatically.

Every shot still receives a newly computed plan. All changes live in profiling
tools and their opt-in CMake target; no production source, executor contract,
circuit eligibility rule, or non-Clifford carrier changes.

## What the coordinate audit found

The current `CoordinateFrame` converts an initial Pauli into the planner's
selected stabilizer coordinates by commuting with the frame's generator rows,
then evaluating a forward round trip to recover its sign. It builds a full
inverse after `2 * num_qubits` direct lookups in one unchanged frame. Promotion
and measurement changes invalidate that inverse and reset the counter.

The [retained study](coordinate-reuse-study.json) audits 32 ordinary noisy
trajectories per input, separately from timing. A frame interval ends at a
coordinate-basis change or the end of planning; empty terminal intervals are
excluded from the one-query percentage below.

| Input | Physical qubits | Queries per shot | Identity queries per shot | Nonempty intervals with one query | Largest interval | Native inverse uses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Factory | 432 | 402 | 189 | 91.6% | 194 | 0 |
| D | 162 | 186 | 0 | 65.4% | 73 | 0 |
| E | 108 | 132 | 0 | 60.2% | 73 | 0 |
| BT27 direct-X | 135 | 158 | 54 | 38.3% | 127 | 0 |

None of these inputs reaches the native inverse threshold in the audited
trajectories. In the factory and BT, every repeated complete query within a
frame is the identity Pauli. D/E have no repeated complete queries in a frame.
A cache of complete Pauli queries would therefore add little beyond recognizing
identity here. This says nothing about repeated complete fault histories:
all the candidate storage is local to one frame during one shot's planning.

The identity queries arise from representing already sampled records with
`MPAD` operations. The frontend traces these as signed identity measurements,
so they still pass through coordinate resolution. Any Clifford frame sends
identity to identity. Returning that Pauli directly preserves its phase and
avoids scanning every physical generator; records and their symbolic signs
continue through the ordinary planner. This is a Pauli-algebra rule, with no
circuit-name or gate-sequence lookup.

Some intervals contain dozens of distinct queries, so the audit also counts
which individual X/Z generators those queries touch. The `input_weight` and
`unique_generators` fields count X and Z factors separately; a Y contributes
both. They measure the demand for a partial inverse rather than a whole-query
cache. Many frame changes still occur after just one query, limiting reuse.

## Compared policies

All modes retain diagonal boundary composition, guarded squeeze scheduling,
fresh planning, and ordinary executable lowering/sampling.

- `squeeze`: preceding worker path and ordinary linked planner.
- `coordinate-native`: the unchanged coordinate algorithm inside the research
  wrapper. This controls for compiling/instrumenting the planner differently.
- `coordinate-identity`: return a Pauli with zero X/Z support directly;
  otherwise use ordinary coordinate conversion.
- `coordinate-columns`: include the identity shortcut and lazily construct
  inverse images of only the physical X/Z generators requested in the current
  frame. Reuse each image until that frame changes. Multiply images in native
  `i^p X^x Z^z` order, retaining the raw input phase and the signed images.
- `coordinate-inverse32`: include the identity shortcut and build the full
  inverse after 32 nonidentity queries in one frame. This is an experimental
  comparison point, not a newly chosen production threshold.

For lazy inverse images, the symplectic transpose determines each generator's
body. A forward evaluation through the current frame determines its sign.
This uses the same algebra as full tableau inversion, but computes only
requested rows. Every basis change clears all images, so no stale coordinates
can cross a promotion or measurement. Storage is bounded by `2n` Pauli images
for one frame and is discarded before native execution.

The research translation unit includes the existing `planner.cc` with alternate
entry-point and coordinate-wrapper type names, after including its original
headers. It inherits the production frame's basis updates and uses the unchanged
planner body. It neither maintains a copied planner implementation nor inserts
hooks into the production build. The alternate planner is compiled only into
`profile_prefix_trace_reuse`. The wrapper control is within 1% of ordinary
complete-shot time on all four inputs in this run.

## Paired cost results

There are 128 ordinary trajectories per eligible input, with mode order rotated
and identical fault histories, prefix seeds, and sample seeds. Complete-shot
time includes drawing faults, constructing the payload, IPC, optimization,
planning, lowering, and sampling. Setup, interval diagnostics, and reference
verification requests are excluded and run separately. CPU affinity is two,
with one native thread. Validation processes used distinct pinned cores, so
shared-machine effects can still influence small differences.

| Input | Previous squeeze | Identity shortcut | Lazy inverse images | Earlier full inverse |
| --- | ---: | ---: | ---: | ---: |
| Factory | 12.411 ms | 9.550 ms | 10.264 ms | 9.583 ms |
| D | 2.294 ms | 2.317 ms | 2.216 ms | 2.581 ms |
| E | 1.408 ms | 1.408 ms | 1.452 ms | 1.499 ms |
| BT27 direct-X | 2.750 ms | 2.520 ms | 2.313 ms | 2.596 ms |

The identity shortcut saves 23.1% on the factory and 8.3% on BT. It leaves D/E
essentially unchanged; D measures about 1% slower. All four 32-shot blocks show
positive factory and BT savings. Factory planning itself drops from 6.70 to
3.83 ms, accounting for almost all its 2.86 ms whole-shot reduction.

Lazy inverse images improve BT by 15.9% relative to the preceding worker, or
about 8.2% beyond the identity shortcut. They improve D by 3.4%, but cost more
than the identity-only route on the factory and E. Every 32-shot block has
the same direction for those comparisons. The method changes both the way
sparse queries are evaluated and the amount of reused work; these timings do
not attribute its entire gain to cache hits alone. In particular BT's queries
are sparse even when an image is used only once.

Earlier full inversion is slower than the identity-only route on D/E/BT. It
builds one inverse per ordinary D/E shot and 55 across 128 BT shots. On the
factory it builds none after identity queries are skipped, so it behaves like
the identity shortcut with a different counter. There is no evidence here for
lowering the native threshold uniformly.

The artifact retains stage means, paired savings in successive 32-shot blocks,
frame-length histograms, generator-use counts, and per-policy counters. Values
under `coordinate_totals` are summed over ordinary shots, including the sum
of per-shot `maximum_cached_columns`. These are local cost measurements, not
rare-event accuracy studies or precise throughput guarantees.

## Correctness checks

The large panel includes 512 ordinary noisy trajectories plus 241 no-fault,
fixed-prefix-seed, and selected/dense fault-stress trajectories. Across five
modes this gives 3,765 exact raw-and-optimized HIR comparisons, including the
final Clifford frame. Across the four research modes there are 3,012 comparisons
of complete untruncated plan inspection and exact final tableaus, plus 128
separate audit-plan checks. Every ordinary sample's records, detectors,
observables, expectations, width, and T count matches the preceding worker.

For the three changed coordinate policies, reference requests additionally
compare every returned Pauli body and phase exactly with native coordinate
conversion on the same current frame: **494,940 coordinate checks** on the
large panel. No digest or matching width substitutes for this comparison.
These primitive checks matter because textual plan inspection alone is not a
bitwise comparison of every floating-point payload. The actual planner body,
HIR angles, and symbolic dependency logic remain the same.

Independent validation includes:

- The complete continuation validator in
  [identity](coordinate-identity-validation.json),
  [lazy-image](coordinate-columns-validation.json), and
  [earlier-inverse](coordinate-inverse-validation.json) modes. Each passes 256
  full-state Aer comparisons, 288 complete record laws with output tomography,
  144 exact Stim Clifford-law checks, 72 frontend conditional-trace checks,
  original-source instrument checks, and seven rejection controls. Each also
  checks 2,344 coordinate results exactly. Maximum recorded state/record error
  is below `3.4e-16`.
- The [boundary validator](coordinate-boundary-validation.json) in lazy-image
  mode passes all 512 three-qubit diagonal Clifford corrections against Aer
  and 96 wide cases at physical widths 65, 129, and 193, with measurement,
  reset, feedback, faults, and signed readout. It checks 2,272 coordinate results
  exactly and reports maximum small-state error below `3.9e-16`.
- [Targeted controls](coordinate-reuse-controls.json) make 60 candidate requests
  across identity records, long stable frames, dense Pauli axes, repeated
  nonidentity queries, and changing measured/reset frames. They require actual
  inverse builds/cache use where intended, match Aer states or complete record
  laws, and check 1,752 coordinate results. Long stable controls also exercise
  the original native inverse cache, unlike the large benchmark panel.
- The [prefix regression](coordinate-prefix-regression.json) passes. Scored
  BT27, cultivation d3/d5, and the unsupported-rotation frontend route retain
  their previous behavior in 34 fallback checks.

All timing and validation artifacts record source/binary hashes. The large
checks validate this planner experiment relative to the preceding conditional
source; they do not independently certify every possible noisy instrument of
the original large circuit. Factory/D/E remain related diagnostic constructions.

## Decision and next assessment

Keep the identity shortcut as the preferred candidate for this research path.
It has a direct algebraic justification and removes a measured cost without
requiring reuse across shots. Keep lazy images and the earlier full inverse
as experimental comparisons; do not add circuit-name dispatch or a universal
cache policy from these four inputs. No public/default route is changed here.

This completes the planned coordinate-reuse experiment and is a useful pause
in performance tuning. The next study should assess the combined prototype
on the broader complete-circuit panel: which inputs reduce, chain, retain
non-Clifford work, or fall back, and how compilation cost compares with the
width reduction achieved. Include the existing scored BT, distillation, and
cultivation controls, and prioritize a structurally different new complete
family when one is available. That offers more signal on automatic reduction
generality than another threshold adjustment on these factory variants.

The all-zero entry assumption, current non-Clifford carrier bound, and separate
interest in noise-free optimization remain unchanged. Reproduction commands
are in the [profiling guide](../../tools/profile/README.md#planner-coordinate-reuse).
