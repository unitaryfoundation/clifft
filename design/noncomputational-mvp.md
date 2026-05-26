<!--pytest-codeblocks:skipfile-->
# Noncomputational Structural-History MVP — Design Note

Working contract for the long-running feature branch
`codex/noncomputational-structural-mvp`. This document is local to the
branch; it is not intended to ship to `main` as a long-lived
architecture doc. It exists so that API and policy decisions are
visible and pushback can happen before code accumulates.

Authoritative inputs:

- GitHub issue
  [#104](https://github.com/unitaryfoundation/clifft/issues/104) (scope
  and constraints).
- Theory note `~/NONCOMPUTATIONAL_TRAJECTORY_MODEL.tex` (semantics).
- Prior sqale-sim draft
  [Infleqtion/client-superstaq#1353](https://github.com/Infleqtion/client-superstaq/pull/1353)
  (compatibility target for the exact subset).
- Reviewer answers in the implementation-kickoff thread (this file
  captures the resolved decisions).

## 1. Scope summary

**In:** state-independent structural loss; known-source classical
transitions; classical status-to-status transitions; reset/reload
restoring a site when the policy allows it; per-trajectory concrete
statuses; per-trajectory initial-state sampling; semantic validation
that the configured instrument set is pre-sampleable.

**Out:** unknown coherent state-dependent transitions; exact diagonal
no-jump filters; segmented JIT replan/resume; dynamic control-flow IR
above HIR; per-site classical distributions; full qudit simulation;
new circuit-level instructions (no `LOSS(p)`, no
`TRANSITION_INSTRUMENT`, no parser changes); compile cache.

**Status of `LOSS(p)` syntax:** deferred. A later PR may add it as
syntactic sugar over this model. The prior LOSS design notes (`R` vs
`RL` split, random lost-measurement default, deltakit-style
`HERALD_LEAKAGE_EVENT` shape) feed the rewrite-policy table in §5; they
do not feed a new instruction surface in this MVP.

## 2. Status representation

### 2.1 Categories (fixed C++ enum)

```cpp
enum class LevelCategory : uint8_t {
    Computational      = 0,  // coherent, in superposition
    KnownComputational = 1,  // coherent level whose physical bit
                             // has been sampled (post-Z-collapse)
    Leaked             = 2,  // present, outside computational subspace
    Lost               = 3,  // absent / vacuum
};
```

Rewrite policies key off `LevelCategory`. The set is fixed; users may
not extend it.

### 2.2 Levels (model-defined)

A `Level` is a model-defined tag with a stable integer id, a label, and
an *intrinsic* category. Intrinsic category is the default category for
a freshly-sampled site holding that level. A `Computational` level can
be promoted to `KnownComputational` at runtime (via a sampled
Z-collapse); the reverse promotion (re-coherence) is not in MVP scope.

A `Computational`-intrinsic level must also declare a `basis_bit`
(0 or 1) identifying which computational basis state it represents.
The rewriter uses this to prepend a preparation gate when an initial
sample places the site in a `Computational` level whose `basis_bit`
is not 0 (the SVM default initialization). `Leaked` and `Lost` levels
have no `basis_bit`.

Default level set for the MVP (sqale-aligned):

| id | label    | intrinsic category | basis_bit | notes                          |
|----|----------|--------------------|-----------|--------------------------------|
| 0  | `g`      | Computational      | 0         | logical 0                      |
| 1  | `e`      | Computational      | 1         | logical 1                      |
| 2  | `leak_g` | Leaked             | —         | metastable / Rydberg "leak g"  |
| 3  | `leak_e` | Leaked             | —         | metastable / Rydberg "leak e"  |
| 4  | `lost`   | Lost               | —         | empty trap / vacuum            |

Users may construct a `NonComputationalModel` with a different level
set, but the default ships unchanged.

### 2.3 Per-trajectory site state

```cpp
// One byte per site: bit 7 carries `known`, bits 0-6 carry `level_id`.
// (A `: 1` / `: 7` bitfield struct is not guaranteed by the standard
// to pack into one byte; explicit masking guarantees the layout.)
using SiteStatus = uint8_t;

constexpr uint8_t kSiteStatusKnownBit  = 0x80;
constexpr uint8_t kSiteStatusLevelMask = 0x7F;  // supports up to 128 levels

inline uint8_t site_level(SiteStatus s)        { return s & kSiteStatusLevelMask; }
inline bool    site_known(SiteStatus s)        { return (s & kSiteStatusKnownBit) != 0; }
inline SiteStatus make_site_status(uint8_t level_id, bool known) {
    return static_cast<SiteStatus>(level_id | (known ? kSiteStatusKnownBit : 0));
}
```

Trajectory state during history sampling is `std::vector<SiteStatus>
statuses;` — one byte per site, allocated once per history. The
runtime-effective category at site `i` is:

- `model.levels[site_level(statuses[i])].intrinsic_category`, *unless*
- the intrinsic category is `Computational` and `site_known(statuses[i])`
  is true, in which case the effective category is
  `KnownComputational`.

"Known" here means known *in the energy basis* (`g`/`e`) — the basis
in which transition matrices are indexed. X- and Y-basis measurements
or resets do not produce `known=1`; see §5.2.1.

## 3. `NonComputationalModel` API

### 3.1 Python ergonomic shape

```python
import clifft

model = clifft.NonComputationalModel(
    # Optional; defaults to the sqale-aligned 5-level set above.
    levels=[
        clifft.Level("g",      category="computational", basis_bit=0),
        clifft.Level("e",      category="computational", basis_bit=1),
        clifft.Level("leak_g", category="leaked"),
        clifft.Level("leak_e", category="leaked"),
        clifft.Level("lost",   category="lost"),
    ],

    # Independent per-site initial level distribution.
    # Indices into levels[]; sums to 1 per site (validated in C++).
    initial_state=[0.0065, 0.96, 0.0065, 0.027, 0.0],

    # Per-gate transition instruments. Matrix shorthand uses T[to, from]
    # convention. Keys are gate names from the existing clifft circuit
    # vocabulary; unknown keys reject at validation time.
    transitions={
        "CZ": clifft.TransitionInstrument(matrix=[...]),   # 5x5
        "RZ": clifft.TransitionInstrument(matrix=[...]),
    },

    # Level -> visible-symbol classifier applied at measurement.
    # Row-stochastic `symbols x levels` matrix using
    # matrix[symbol_idx, level_id] = P(symbol | level).
    # Optional; default is identity-on-computational + reject on
    # leaked/lost (i.e. policy must say what M of a noncomputational
    # site returns).
    classifier=clifft.MeasurementClassifier(
        symbols=["0", "1"],     # 2-symbol binary record by default
        matrix=[[...], [...]],  # shape (len(symbols), len(levels))
    ),

    # Policy hooks for downstream operations on noncomputational
    # sites. See §5 for the default table.
    policy=clifft.NonComputationalPolicy(
        lost_measurement="random",  # | "zero" | "one" | "reject"
        # ... other knobs, default-reject for ambiguous cases.
    ),
)
```

`clifft.Level`, `clifft.TransitionInstrument`,
`clifft.MeasurementClassifier`, `clifft.NonComputationalPolicy`, and
`clifft.NonComputationalModel` are all nanobind-bound C++ classes.
Python provides keyword constructors and `__repr__`. No pydantic;
light shape checks (e.g., `isinstance(matrix, list)`) on the Python
side, semantic validation (column sums, unknown-gate rejection,
intrinsic-category consistency, classifier shape) on the C++ side at
model construction. Source-independence is *cached* on each
`TransitionInstrument` at construction but is not a construction-time
invariant; the runtime sampler uses it as the §4.2 gate.

### 3.2 C++ side

```cpp
namespace clifft {

struct Level {
    std::string label;
    LevelCategory intrinsic_category;
    std::optional<uint8_t> basis_bit;  // required if intrinsic_category == Computational
};

class TransitionInstrument {
public:
    // Construct from a square matrix using T[to, from] convention.
    // Validates matrix shape and that every column sum lies in [0, 1].
    // Computes the cached flag `is_source_independent_on_computational`
    // but does NOT reject source-dependent matrices here; whether a
    // source-dependent matrix is applicable is checked at sample time
    // (see section 4.2).
    static TransitionInstrument from_matrix(
        std::vector<std::vector<double>> matrix);

    bool is_source_independent_on_computational() const;
    // ... accessors for per-source branches, no-jump weight per source, etc.
};

// Distinct from TransitionInstrument: rectangular row-stochastic map
// from levels to reported user-facing symbols. No no-jump branch, no
// concept of source-independence on Computational levels.
class MeasurementClassifier {
public:
    static MeasurementClassifier from_matrix(
        std::vector<std::string> symbols,
        std::vector<std::vector<double>> matrix);  // shape (symbols, levels)

    // ... accessors
};

class NonComputationalModel {
public:
    NonComputationalModel(
        std::vector<Level> levels,
        std::vector<double> initial_state,
        std::map<std::string, TransitionInstrument> transitions,
        std::optional<MeasurementClassifier> classifier,
        NonComputationalPolicy policy);

    // Throws at construction on shape/probability/key validation
    // failures (section 4.1). Source-context validity is enforced at
    // sample time (section 4.2).
    // ... accessors
};

}  // namespace clifft
```

## 4. Validation: construction-time vs. sample-time

Validation is split: model construction checks shape and self-consistency;
the runtime sampler checks whether each transition's source-context is
representable in the MVP.

### 4.1 Construction-time checks (always run, throw on failure)

1. **Level set well-formed.** Every level id in `[0, len(levels))`,
   intrinsic categories all in the enum. Every `Computational`-intrinsic
   level declares a `basis_bit` in `{0, 1}`; non-`Computational` levels
   omit it.
2. **Initial state is a probability vector** over `levels`. Sums to 1
   within tolerance.
3. **Transition matrices are square**, of size `len(levels)`, entries
   in `[0, 1]`.
4. **Column sums in [0, 1].** Implied no-jump weight per source is
   `1 - sum(col)`.
5. **Classifier matrix shape and rows.** If a `MeasurementClassifier`
   is provided, its matrix has shape `(len(symbols), len(levels))`,
   every entry is in `[0, 1]`, and every column (per-level row over
   symbols) sums to 1 within tolerance.
6. **Policy values are well-formed** (enum values, not free strings on
   the C++ side; Python sugar translates).
7. **Transition keys reference known gate names** in clifft's circuit
   vocabulary; unknown keys reject.

Each `TransitionInstrument` also computes and caches a derived flag
at construction:

- `is_source_independent_on_computational`: true iff every column whose
  source level is intrinsic `Computational` is bit-identical (within
  tolerance) to the others.

This flag is *not* a validation gate — source-dependent matrices are
fine to declare. Whether they are applicable depends on the source
context at sample time. The sqale-aligned `cz_transition_matrix` and
`rz_transition_matrix` in §1's prior art have distinct `g` and `e`
columns; they are valid `TransitionInstrument`s and will be accepted
here.

### 4.2 Sample-time check (per (transition, target-site) firing)

When a transition fires on a target site at category `c`:

- `c == KnownComputational`: use the column matching the known level
  directly. Always allowed.
- `c == Leaked` or `c == Lost`: use the column matching the level
  directly. Classical Markov update. Always allowed.
- `c == Computational` (coherent, unknown): require
  `is_source_independent_on_computational` on this instrument. If
  false, reject with an error naming the op index, the site, and the
  instrument — pointing the user at the cut between MVP and the later
  diagonal-filter extension. If true, the no-jump branch is scalar on
  `H_C` and the jump branches are source-independent; sample without
  consulting amplitudes.

This is the "pre-sampleable" boundary, enforced where it actually
matters (at the unknown-coherent-source point) rather than at model
construction. It also means a model with a source-dependent instrument
that only ever fires on known-or-classical sources runs fine in MVP.

## 5. Sampling, rewrite, and policy table

### 5.1 Pipeline

For each shot:

1. **Sample initial statuses.** One draw from `initial_state` per
   site. The sampled level is a classical fact, so the initial
   `known` bit is 1 for every site (it is moot on `Leaked`/`Lost`).
2. **Translate known computational initial levels into prep gates.**
   For every site sampled to a `Computational` level whose
   `basis_bit == 1`, the rewriter prepends an `X` on that site so the
   SVM's `|0...0>` initial state matches the sampled known level.
   `basis_bit == 0` requires no prep. `Leaked`/`Lost` initial sites
   need no quantum prep (their computational amplitude is irrelevant);
   the lost/leaked policy gates downstream ops on them from op 0.
3. **Walk the circuit ops in order, sampling transitions and updating
   statuses.** For each op, consult the model's transitions and the
   site statuses. Apply the §4.2 sample-time check on the source
   context. Sample any per-target branches. Record the resulting
   `NonComputationalHistory` (sequence of (op-index, site, sampled
   branch, status-update, optional herald)). Update the `known` bit
   per §5.2.1 below.
4. **Rewrite the original circuit using the history.** Produce a new
   ordinary clifft circuit (no new instructions). Drop, keep, or
   replace ops per the policy table. Insert an `R` op at structural
   loss points where the lost site was previously coherent (see §5.3).
5. **Compile the rewritten circuit** through the ordinary
   trace/lower/bytecode pipeline.
6. **Sample one shot** on the SVM.
7. **Return** the user-facing `(measurements, detectors, observables)`
   unchanged from the existing API, plus the noncomputational metadata
   sidecar: `(history, per-site final status, classifier output, herald
   bits)`.

The C++ orchestrator owns steps 1-7. Caching of step 5 across shots
is deferred (see §1, "Out").

### 5.2 Default rewrite-policy table

Rows are (operation kind, site category at op time). "Reject" means
the C++ rewrite raises with a clear message naming the op index and
site; no silent approximation. "Apply, clear known" means the op runs
unchanged but the site's `known` bit is set to 0 afterward (per §5.2.1).

| Operation                | Computational         | KnownComputational                  | Leaked                                  | Lost                                              |
|--------------------------|-----------------------|--------------------------------------|------------------------------------------|---------------------------------------------------|
| Single-qubit gate        | apply                 | apply, clear known                  | reject (policy override allowed)        | drop                                              |
| Single-qubit Pauli noise | apply                 | apply, clear known                  | drop                                    | drop                                              |
| Two-qubit gate           | apply                 | apply, clear known on both           | reject                                  | reject (policy override: drop)                    |
| MPP / multi-target meas  | apply                 | apply, clear known on all            | reject                                  | reject                                            |
| Visible Z-basis meas (`M`/`MR`) | apply, set known after | apply, outcome = sampled known value | classifier → visible bit; default reject | policy: random / zero / one / reject |
| Visible X/Y-basis meas (`MX`/`MY`/`MRX`/`MRY`) | apply, **clear known** | apply, **clear known** | classifier → visible bit; default reject | policy: random / zero / one / reject |
| Z-basis reset `R`        | apply, set known after | apply, set known after | restore to known computational          | reject unless `policy.reset_restores_lost` is set |
| X/Y-basis reset `RX`/`RY` | apply, **clear known** | apply, **clear known** | restore to coherent computational (known = 0) | reject unless `policy.reset_restores_lost` is set |
| Detector / Observable    | unchanged             | unchanged                            | unchanged                               | unchanged                                         |

Notes:

- "Policy override allowed" means the cell rejects by default but the
  user may set a `NonComputationalPolicy` field to flip it to a
  specific behavior. The MVP exposes overrides only where listed.
- There is no separate `RL` op in MVP. Lost-site reset rejects by
  default; the `policy.reset_restores_lost` flag turns it into a
  reload that restores the site to a known computational state. See
  §8 for the rationale.
- "Apply, clear known" is the conservative default: we drop knownness
  on any quantum gate touching a `KnownComputational` site. This loses
  some optimization opportunity (X on a known site could keep
  knownness with a flipped value; Z/S/T preserve it as a phase), but
  the conservative rule sidesteps gate-by-gate classification in MVP.
  Refinement is a later pass once the trajectory infrastructure is in
  place. The reviewer's framing applies: knownness is produced by
  initial sampling, measurement/collapse, and reset/reload; ordinary
  quantum gates clear it unless explicitly handled.

### 5.2.1 Knownness transitions

A site's `known` bit transitions as follows during history sampling:

| Cause                                            | Effect                                                  |
|--------------------------------------------------|---------------------------------------------------------|
| Initial-state sample (any level)                 | `known = 1`                                             |
| Any quantum gate touching site (single or multi) | `known = 0` on every touched site                       |
| Z-basis measurement (`M`/`MR`)                   | `known = 1`, level updated to `g` or `e` per outcome    |
| X- or Y-basis measurement (`MX`/`MY`/`MRX`/`MRY`)| `known = 0` (post-state is in X- or Y-basis, not energy basis) |
| Z-basis reset `R`                                | `known = 1` (reset is to a known energy-basis state)    |
| X- or Y-basis reset `RX`/`RY`                    | `known = 0` (reset is to a known X/Y-basis state, but `known` is energy-basis) |
| Reload via `policy.reset_restores_lost` (Z-basis) | `known = 1`                                            |
| Transition jump to a new computational-intrinsic level | `known = 1` (destination level is the known value)      |
| Transition jump to a noncomputational level      | `known` bit moot; site category becomes Leaked or Lost   |

The "known" bit means *known in the energy basis* (`g`/`e`) — the
basis transition matrices are indexed in. A site that is in a known
X-basis or Y-basis state after `RX`/`MY`/etc. is still coherent in
the energy basis (a superposition of `g` and `e`), so its category
remains `Computational`. Source-dependent transitions on it will hit
the §4.2 sample-time rejection unless the instrument is
source-independent on Computational sources.

### 5.3 Hidden trace-out at structural loss

When a site that is currently `Computational` (coherent) transitions
to `Lost`, theory requires a hidden Z-basis measurement to unravel the
partial trace. clifft already has the infrastructure for this: the
`R`/`RX`/`RY` reset ops are lowered by the frontend
(`src/clifft/frontend/frontend.cc:618-694`) to a *hidden* measurement
slot — tracked on a separate `hidden_meas_idx` counter, marked with
`meas_op.set_hidden(true)`, never exposed to the user-facing record,
followed by a conditional Pauli correction so the residual state on
survivors is the correct conditional state. This is exactly the
trace-out unraveling the loss event needs.

The MVP rewriter takes advantage of this directly. At a structural
loss event on a previously-coherent site, the rewriter inserts a
single `R` op on that site in the rewritten circuit. Concretely:

1. **No new HIR op or circuit instruction is introduced.** The
   rewriter emits an existing `R` op; the existing frontend lowering
   handles the hidden measurement, the corrective Pauli, and the
   hidden-slot accounting.
2. **No `rec[-k]` renumbering is needed.** Visible measurement counts
   only advance on visible `M` slots; hidden-measurement slots live on
   their own counter and never shift the user-facing record. Detectors,
   observables, and feedback continue to reference the original
   visible-record indices unchanged.
3. **No output stripping pass is needed.** The SVM does not emit
   hidden-measurement slots into the user-facing measurement array;
   `compile()` / `sample()` already enforce that.
4. **The trace-out outcome is consumed internally and not exposed.**
   The frontend's hidden-measurement slot drives the corrective Pauli
   on the residual state, but `sample()` sizes `result.measurements`
   to `program.num_measurements` (visible only) and never propagates
   hidden slots out (see `src/clifft/svm/svm.cc:199-205`). Surfacing
   the outcome would require extending the SVM result struct; that is
   out of scope for MVP. The noncomputational sidecar therefore does
   not include trace-out outcomes in v1.

After the inserted `R`, the rewriter marks the site `Lost` in the
trajectory and drops every subsequent op on that site per the policy
table. Note that `R` semantically leaves the site in `|0>` known
computational — that residual state is immediately reinterpreted as
`Lost` by the trajectory; the SVM's own post-`R` state on that qubit
is irrelevant because no later op reads from it.

If a site transitions to `Lost` while it is already `KnownComputational`
or already noncomputational (`Leaked` or starting `Lost`), no quantum
trace-out is needed — the destination is purely a status update with
no `R` insertion required.

## 6. New C++ headers and dependency order

Proposed layout under `src/clifft/noncomp/` (new directory):

1. `level.h` — `LevelCategory` enum, `Level` struct. Zero deps.
2. `transition_instrument.h/.cc` — `TransitionInstrument`,
   matrix-to-branch expansion, derived
   `is_source_independent_on_computational` flag. Depends on `level.h`.
3. `classifier.h/.cc` — `MeasurementClassifier`, row-stochastic
   `(symbols, levels)` map. Depends on `level.h`.
4. `policy.h` — `NonComputationalPolicy` struct with enums for the
   ambiguous-case overrides. Zero deps.
5. `model.h/.cc` — `NonComputationalModel`, all construction-time
   validation per §4.1. Depends on (1)-(4).
6. `history.h` — `SiteStatus` (the packed byte from §2.3),
   `NonComputationalHistory` (sequence of (op-index, site, branch,
   status-update, herald)). Depends on (1).
7. `sampler.h/.cc` — history sampler that walks the parsed circuit,
   updates `SiteStatus` per shot, and enforces the §4.2 sample-time
   source-context check. Depends on (1)-(6), `clifft::Circuit`,
   `clifft::AstNode`.
8. `rewriter.h/.cc` — produces a new `clifft::Circuit` from
   `(original, history, model)`, inserting `R` at coherent-site loss
   points and `X` prep for `basis_bit==1` initial samples. Depends on
   (1)-(7) and `clifft::Circuit`.
9. `orchestrator.h/.cc` — top-level `sample_noncomputational` entry
   point that runs the full pipeline. Depends on (1)-(8), the existing
   compile pipeline, and the SVM.

Python bindings in `src/python/bindings.cc` expose:
`Level`, `TransitionInstrument`, `MeasurementClassifier`,
`NonComputationalPolicy`, `NonComputationalModel`,
`sample_noncomputational(circuit_text, model, shots, seed=None)`.

## 7. Test plan (dependency order)

Each test category lands once the headers it needs are in place. C++
tests under `tests/test_noncomp_*.cc`; Python tests under
`tests/python/test_noncomputational.py`.

1. **`level`, `transition_instrument`, `classifier` unit tests.**
   - Default level set roundtrips; `basis_bit` required iff
     intrinsic category is Computational.
   - Matrix orientation: `T[to, from]` matches a known per-column
     no-jump weight.
   - `is_source_independent_on_computational` is true for identical
     `g`/`e` columns, false when they differ. Construction does *not*
     throw on differing-column instruments — both must build cleanly.
   - `MeasurementClassifier` matrix shape and per-level column sums
     enforced; mismatched (symbols, levels) dimensions rejected.
2. **`model` validation tests.**
   - Initial-state sum, unknown-gate keys, classifier shape, policy
     enum values all reject at construction with a named field.
   - Source-dependent transitions in `model.transitions` build fine;
     no rejection at construction.
3. **`history` sampler tests.**
   - Deterministic seed → fixed history.
   - Initial-state-only sampling produces marginal frequencies within
     a binomial bound at large N.
   - Sample-time §4.2 rejection: applying a source-dependent
     transition to an unknown-coherent Computational site raises an
     error naming the op, site, and instrument. Applying the same
     transition to a `KnownComputational` site (post Z-basis `M`)
     succeeds.
4. **`rewriter` unit tests.**
   - Initial `e` sample on a site emits a leading `X` prep; initial
     `g` does not.
   - Lost-site single-q gate dropped; two-q gate rejected by default.
   - Reset on lost rejected by default; `policy.reset_restores_lost`
     enables restoration.
   - **Trace-out via inserted `R`**: at a structural-loss event on a
     previously-coherent site, an `R` is emitted on that site. The
     compiled rewritten circuit's
     `hir.num_hidden_measurements` increments by one, and the
     visible-record layout (and `rec[-k]` references in detectors and
     observables) is unchanged. Verified on a circuit that has
     detectors before *and* after the loss point.
   - Z-basis vs X/Y-basis measurement/reset: a `KnownComputational`
     site produced by `M` retains `known=1`; the same site after `MX`
     has `known=0`. (Single-site assertions on the sampler output.)
   - Lost visible-measurement policy `"random"` and `"zero"` each
     produce the configured output.
5. **End-to-end `sample_noncomputational` Python tests.**
   - Visible measurement record layout unchanged vs. a lossless run
     where the model is configured with zero loss probability.
   - Survivor statistics on a small (3-qubit) entangled circuit with
     forced loss match a Python brute-force enumerator within shot
     noise.
   - "Everything lost initially" (initial_state concentrates on
     `lost`) returns a trivial output and a full noncomp sidecar.
6. **Regression / smoke.**
   - Existing `compile()` / `sample()` paths unchanged. Run the
     current C++ and Python suites; both stay green.

Validation oracle for (5): a small Python brute-force enumerator that
walks the circuit with explicit density matrices over computational
qubits + classical statuses. Bounded to ~4 qubits and ~10 events;
fast enough for parameter sweeps. No sqale-sim dependency in the
MVP.

## 8. Open questions to settle before §6 step 7 (`rewriter`)

These do not block the design note but should be settled before the
rewriter PR:

- **Surfacing trace-out outcomes in the sidecar.** Hidden-measurement
  outcomes are available from the SVM's hidden-record buffer; whether
  to expose them as part of the noncomputational metadata sidecar is
  not load-bearing for correctness. Provisional: do not expose them in
  MVP. Re-enable later if a decoder or debugging workflow needs them.
- **Herald metadata transport.** The orchestrator sidecar shape is
  written above as "per-site final status + classifier output + herald
  bits". We may want a more structured sidecar (e.g., numpy structured
  array) for performance / decoder integration. Provisional: keep the
  sidecar a typed dict in Python for the MVP; convert to a structured
  array if/when a decoder consumer demands it.

### Resolved during note revision

- **`RL` (loss reload).** Closed: no new circuit op in MVP. Lost-site
  reset rejects unless `policy.reset_restores_lost` is set, in which
  case existing `R`/`RX`/`RY` ops restore the site to a known
  computational state. The table in §5.2 reflects this.
- **Initial-state correlation across sites.** Closed: out of scope.
  `initial_state` is an independent per-site distribution; a future
  feature can add correlated initial states without breaking this
  schema.
- **Source-independence enforcement point.** Closed: not a
  construction-time invariant. Source-dependent matrices construct
  fine; the runtime sampler rejects only when a transition fires on a
  site whose source category is unknown-coherent `Computational` and
  the instrument is not source-independent on `Computational` levels.
  See §4.

## 9. Out of scope but planned-for

- `LOSS(p) targets...` Stim instruction as syntactic sugar.
- Diagonal `aI + bZ` filter for state-dependent no-jump (the natural
  next exact-mode extension).
- Compile cache by rewritten-circuit hash, with observable compile
  count.
- Segmented JIT / replan for true topology changes mid-shot.
- Decoder-side integration of herald bits.

These are explicitly outside the MVP and should not constrain the
schemas above beyond what is already noted.
