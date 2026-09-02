# Active-width certificate search

A standalone research tool that certifies the minimum peak active width a
fixed, already-compiled HIR can reach over a stated rewrite class. It is the
exact counterpart to the library's `ActiveWidthSchedulePass`: where that pass
runs a beam search to find a good schedule quickly, this tool runs an
exhaustive, budgeted search to either prove a schedule is the best possible
or report how far the search got before giving up.

## What it is

`width_certificate` loads one `.stim` circuit, traces and optimizes it with a
chosen pass pipeline, builds the `ScheduleDependence` relation over the
resulting HIR's operations (see
`src/clifft/optimizer/schedule_dependence.h`), and runs a closure-based
threshold search (`active_width_search.h`) that either certifies the peak
active width is minimal over every legal reordering of that exact HIR, or
reports the best bounds it proved before a node budget ran out.

Two numbers bound the answer:

- **upper_bound**: the peak active width of the best schedule the search
  found (never worse than the input's own incumbent peak).
- **lower_bound**: the tightest peak the search proved no schedule can beat,
  starting from the order-invariant final active width and rising as
  successive thresholds are proven infeasible.

The result is a certificate exactly when the two meet (`optimal()` is true).
A budget-exhausted run still reports both bounds; they just may not have
converged.

### Scope and non-claims

The certificate is about **one fixed HIR**, over **one fixed dependence
relation** (plain `can_swap`, or the noise-transparent relaxation), scored by
the sampling planner's own structural width model
(`analyze_active_width` / `DormantSubspace`). It certifies a minimum over
every linear extension of that relation -- not:

- the minimum width among all semantically equivalent circuits, including
  ones reachable only through a rewrite that exposes new gate fusion;
- anything about amplitude-level stabilizers the structural planner does not
  track;
- a claim about any other circuit that merely samples the same distribution
  as this one;
- a bound on the noncomputational pipeline, which this tool never schedules.

A "certified" number in this document means `search_width_schedule` returned
`optimal() == true` for that circuit, pipeline, and relation. A number
without a certificate (budget exhausted) is only an upper bound: a longer
search, or a smarter one, might still find something lower.

## When to use it

- **After changing the scheduling pass's ranking or beam.** Run this tool on
  the corpus to measure the gap between what the beam search finds and what
  is actually achievable, rather than guessing from the beam's own output
  whether it left width on the table.
- **Before redesigning a gadget for lower width**, to learn whether its
  current width is intrinsic to the structural model (the search cannot beat
  the incumbent under either relation) before spending effort on a rewrite
  that cannot help.
- **To produce certified numbers for a paper or a design document**, where
  "the scheduler found width W" and "W is provably minimal for this HIR
  under this relation" are different claims and only the search can make the
  second one.

## Why it is not in the library

This is a diagnostic instrument for people changing the scheduler, not a
capability any clifft user needs at compile time: nothing in the production
pipeline calls an exact, potentially-exponential search, and there is no
Python binding because there is no Python-facing use case to bind. It links
against `clifft_core` and reuses the library's own closure machinery
(`clifft/optimizer/active_width_closure.h`) so its notion of "ready" and
"expanding" cannot drift from the scheduling pass's, but the search itself,
its CLI, and its tests live here, out of the way of the code every build and
every wheel ships.

## Building and running

```bash
cmake -B build -S research/active_width_certificates -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/width_certificate tests/fixtures/coherent_d3_r3.stim --pipeline production
```

`CLIFFT_SOURCE_DIR` defaults to the repository root two levels up from this
directory; pass `-DCLIFFT_SOURCE_DIR=/path/to/clifft` to point the build at a
different checkout.

Driver usage:

```text
width_certificate <circuit.stim> [--pipeline none|peephole|production]
                   [--no-noise-transparency] [--budget N] [--print-order]
```

- `--pipeline`: `none` runs the search directly on the traced HIR;
  `peephole` runs `PeepholeFusionPass` first; `production` (the default)
  runs `PeepholeFusionPass` then `StatevectorSqueezePass`, matching the
  library's own default-enabled pass set.
- `--no-noise-transparency` builds the dependence relation from plain
  `can_swap` only, instead of also allowing a movable operation to cross a
  `NOISE` op (the default; see `schedule_dependence.h` for why that
  relaxation is sound).
- `--budget N` sets the shared node budget across every threshold the outer
  loop tries (default 200000).
- `--print-order` additionally prints the witness schedule as a space
  separated list of original op indices.

Run the tests from the same build:

```bash
ctest --test-dir build --output-on-failure
```

## Corpus certificate table

Regenerated with this tool against the tip of this branch. `coherent_d3_r3`,
`coherent_d5_r5`, and `cultivation_d5` are the fixtures under
`tests/fixtures/` in this repository; `distillation` and `cultivation_d3` are
from the `clifft-paper` QEC benchmark corpus (not included in this
repository). All rows use `--pipeline production`, the default node budget
(200000), and both relation options.

| Circuit | Relation | Peak | Certificate | Explored nodes |
|---|---|---:|---|---:|
| coherent_d3_r3 | can_swap | 5 | optimal | 71 |
| coherent_d3_r3 | + noise-transparent | 4 | optimal | 64 |
| coherent_d5_r5 | can_swap | <=13 | budget exhausted | 200000 |
| coherent_d5_r5 | + noise-transparent | <=13 | budget exhausted | 200000 |
| distillation | can_swap | 5 | optimal | 31 |
| distillation | + noise-transparent | 3 | optimal | 21 |
| cultivation_d3 | can_swap | 4 | optimal | 16 |
| cultivation_d3 | + noise-transparent | 4 | optimal | 51 |
| cultivation_d5 | can_swap | 10 | optimal | 25862 |
| cultivation_d5 | + noise-transparent | 10 | optimal | 25897 |

`cultivation_d3` and `cultivation_d5` certifying the same peak under both
relations means both are intrinsic: no legal reordering, noise-transparent or
not, beats the production incumbent.

Every row above was regenerated with this tool rather than copied from prior
prototype or design-doc numbers. One regenerated number disagrees with an
earlier expectation: `cultivation_d5` was previously reported as
budget-exhausted with no certificate at a 100000-node budget from an
uncommitted prototype tool; the in-tree implementation certifies it optimal
at peak 10 (intrinsic, matching both relations) in under 26000 nodes at the
default 200000-node budget. That discrepancy is expected -- the prototype was
explicitly a scratch tool superseded by this in-tree implementation -- and is
recorded here rather than silently overwritten. `coherent_d5_r5` matches the
earlier expectation: both relations run the full 200000-node budget without
resolving the threshold below the incumbent peak, so 13 remains an upper
bound only, not a certificate.
