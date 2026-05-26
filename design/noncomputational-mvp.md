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

Default level set for the MVP (sqale-aligned):

| id | label    | intrinsic category | notes                          |
|----|----------|--------------------|--------------------------------|
| 0  | `g`      | Computational      | logical 0                      |
| 1  | `e`      | Computational      | logical 1                      |
| 2  | `leak_g` | Leaked             | metastable / Rydberg "leak g"  |
| 3  | `leak_e` | Leaked             | metastable / Rydberg "leak e"  |
| 4  | `lost`   | Lost               | empty trap / vacuum            |

Users may construct a `NonComputationalModel` with a different level
set, but the default ships unchanged.

### 2.3 Per-trajectory site state

```cpp
struct SiteStatus {
    uint8_t level_id;   // index into model.levels
    uint8_t known : 1;  // only meaningful when category == Computational
    uint8_t reserved : 7;
};
```

Trajectory state during history sampling is `std::vector<SiteStatus>
statuses;` — one byte per site, allocated once per history. The
runtime-effective category at site `i` is:

- `model.levels[statuses[i].level_id].intrinsic_category`, *unless*
- the intrinsic category is `Computational` and `statuses[i].known == 1`,
  in which case the effective category is `KnownComputational`.

## 3. `NonComputationalModel` API

### 3.1 Python ergonomic shape

```python
import clifft

model = clifft.NonComputationalModel(
    # Optional; defaults to the sqale-aligned 5-level set above.
    levels=[
        clifft.Level("g",      category="computational"),
        clifft.Level("e",      category="computational"),
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
    # Optional; default is identity-on-computational + reject on
    # leaked/lost (i.e. policy must say what M of a noncomputational
    # site returns).
    classifier=clifft.TransitionInstrument(matrix=[...]),

    # Policy hooks for downstream operations on noncomputational
    # sites. See §5 for the default table.
    policy=clifft.NonComputationalPolicy(
        lost_measurement="random",  # | "zero" | "one" | "reject"
        # ... other knobs, default-reject for ambiguous cases.
    ),
)
```

`clifft.Level`, `clifft.TransitionInstrument`,
`clifft.NonComputationalPolicy`, and `clifft.NonComputationalModel` are
all nanobind-bound C++ classes. Python provides keyword constructors
and `__repr__`. No pydantic; light shape checks (e.g.,
`isinstance(matrix, list)`) on the Python side, semantic validation
(column sums, source-independence, unknown-gate rejection,
intrinsic-category consistency) on the C++ side at model construction.

### 3.2 C++ side

```cpp
namespace clifft {

struct Level {
    std::string label;
    LevelCategory intrinsic_category;
};

class TransitionInstrument {
public:
    // Construct from a square matrix using T[to, from] convention.
    // Validates column sums in [0, 1] and source-independence
    // (every column on Computational sources must have the same
    // probability vector); throws on failure.
    static TransitionInstrument from_matrix(
        std::vector<std::vector<double>> matrix);

    // ... accessors for branches, no-jump weight per source, etc.
};

class NonComputationalModel {
public:
    NonComputationalModel(
        std::vector<Level> levels,
        std::vector<double> initial_state,
        std::map<std::string, TransitionInstrument> transitions,
        std::optional<TransitionInstrument> classifier,
        NonComputationalPolicy policy);

    // Throws at construction if any invariant violates pre-sampleability.
    // ... accessors
};

}  // namespace clifft
```

## 4. Pre-sampleability validation

Validation runs once at `NonComputationalModel` construction. Failures
throw with a message naming the offending field. Checks:

1. **Level set well-formed.** Every level id in `[0, len(levels))`,
   intrinsic categories all in the enum.
2. **Initial state is a probability vector** over `levels`. Sums to 1
   within tolerance.
3. **Transition matrices are square**, of size `len(levels)`, entries
   in `[0, 1]`.
4. **Column sums in [0, 1].** Implied no-jump weight per source is
   `1 - sum(col)`.
5. **Source-independence for Computational sources.** For every
   transition, every column whose source is a `Computational` level
   must be identical. This is the "pre-sampleable" condition: no two
   Computational source levels have distinguishable jump rates. If
   violated, raise with a message naming the operation and the
   differing columns. (This is exactly the cut between the MVP and the
   later diagonal-filter work.)
6. **Classifier sources cover the full level set** if provided.
7. **Policy values are well-formed** (enum values, not free strings on
   the C++ side; Python sugar translates).
8. **Transition keys reference known gate names** in clifft's circuit
   vocabulary; unknown keys reject.

Sources that are intrinsic `Leaked` or `Lost` are not subject to (5) —
their transitions are classical Markov updates by definition.

## 5. Sampling, rewrite, and policy table

### 5.1 Pipeline

For each shot:

1. **Sample initial statuses.** One draw from `initial_state` per
   site. Initial `known` bit is 1 (the sampled level is, by
   construction, a classical fact).
2. **Walk the circuit ops in order, sampling transitions and updating
   statuses.** For each op, consult the model's transitions and the
   site statuses. Sample any per-target branches. Record the resulting
   `NonComputationalHistory` (sequence of (op-index, site, sampled
   branch, status-update, optional herald)).
3. **Rewrite the original circuit using the history.** Produce a new
   ordinary clifft circuit (no new instructions). Drop, keep, or
   replace ops per the policy table. Insert hidden trace-out
   measurements at structural loss points where the lost site was
   coherent (see §5.3).
4. **Compile the rewritten circuit** through the ordinary
   trace/lower/bytecode pipeline.
5. **Sample one shot** on the SVM.
6. **Strip hidden record positions** from the measurement array (see
   §5.3) and return the user-facing `(measurements, detectors,
   observables)` plus the noncomputational metadata sidecar:
   `(history, per-site final status, classifier output, herald bits)`.

The C++ orchestrator owns steps 1-6. Caching of step 4 across shots is
deferred (see §1, "Out").

### 5.2 Default rewrite-policy table

Rows are (operation kind, site category at op time). "Reject" means
the C++ rewrite raises with a clear message naming the op index and
site; no silent approximation.

| Operation                | Computational     | KnownComputational            | Leaked                          | Lost                            |
|--------------------------|-------------------|-------------------------------|---------------------------------|---------------------------------|
| Single-qubit gate        | apply             | apply (rehydrates known state) | reject (policy override allowed) | drop                            |
| Single-qubit Pauli noise | apply             | apply                          | drop                            | drop                            |
| Two-qubit gate           | apply             | apply                          | reject                          | reject (policy override: drop)  |
| MPP / multi-target meas  | apply             | apply                          | reject                          | reject                          |
| Visible single-q meas    | apply             | apply (sampled-known outcome)  | classifier → visible bit; default reject | policy: random / zero / one / reject |
| Reset `R`                | apply             | apply                          | restore to known computational  | reject (use `RL` for reload)    |
| Reload `RL`              | apply             | apply                          | restore                         | restore                         |
| Detector / Observable    | unchanged         | unchanged                      | unchanged                       | unchanged                       |

Notes:

- "Policy override allowed" means the cell rejects by default but the
  user may set a `NonComputationalPolicy` field to flip it to a
  specific behavior. The MVP exposes overrides only where listed.
- `RL` (reload) is the loss-restoration operation analog from the
  prior LOSS design notes; it differs from `R` (computational reset)
  in that it can promote a `Lost` site back to a known computational
  state. If we do not want to add `RL` as a new HIR-level op in this
  MVP, `RL` collapses to a model-config flag on existing `R` ops; that
  decision is open (see §8).

### 5.3 Hidden trace-out at structural loss

When a site that is currently `Computational` (coherent) transitions
to `Lost`, theory requires a hidden Z-basis measurement to unravel the
partial trace. The rewriter inserts an ordinary `M` op at that point.
Two consequences:

1. **The hidden M consumes a measurement slot** in the rewritten
   circuit. The rewriter tracks the slot offset and **renumbers every
   subsequent `rec[-k]` reference** in detectors, observables, and
   classical feedback so visible record layout is unchanged. This is a
   straightforward pre-compile pass over the rewritten ops.
2. **The hidden slot is stripped from user output.** The orchestrator
   carries a `std::vector<uint64_t> hidden_slot_indices` per
   rewritten-circuit instance; after the SVM returns the full
   measurement array it removes those indices before exposing the
   `measurements` array. The hidden outcome is *not* discarded
   internally — it becomes part of the noncomputational metadata
   sidecar (under "trace-out outcomes") for debugging and validation.

This keeps the rewriter's output in ordinary clifft syntax (no new HIR
op required) and isolates the hidden-record handling to two small
helpers on the orchestrator path. It does mean every history pays one
extra M per coherent-site loss in compile work; for the MVP that is
acceptable.

## 6. New C++ headers and dependency order

Proposed layout under `src/clifft/noncomp/` (new directory):

1. `level.h` — `LevelCategory` enum, `Level` struct. Zero deps.
2. `transition_instrument.h/.cc` — `TransitionInstrument`,
   matrix-to-branch expansion, source-independence check. Depends on
   `level.h`.
3. `policy.h` — `NonComputationalPolicy` struct with enums for the
   ambiguous-case overrides. Zero deps.
4. `model.h/.cc` — `NonComputationalModel`, all semantic validation.
   Depends on (1)-(3).
5. `history.h` — `SiteStatus`, `NonComputationalHistory` (sequence of
   (op-index, site, branch, status-update, herald)). Depends on (1).
6. `sampler.h/.cc` — history sampler that walks the parsed circuit and
   updates `SiteStatus` per shot. Depends on (1)-(5), `clifft::Circuit`,
   `clifft::AstNode`.
7. `rewriter.h/.cc` — produces a new `clifft::Circuit` from
   `(original, history, model)` and computes the
   `hidden_slot_indices` sidecar. Depends on (1)-(6) and
   `clifft::Circuit`.
8. `orchestrator.h/.cc` — top-level `sample_noncomputational` entry
   point that runs the full pipeline. Depends on (1)-(7), the existing
   compile pipeline, and the SVM.

Python bindings in `src/python/bindings.cc` expose:
`Level`, `TransitionInstrument`, `NonComputationalPolicy`,
`NonComputationalModel`, `sample_noncomputational(circuit_text, model,
shots, seed=None)`.

## 7. Test plan (dependency order)

Each test category lands once the headers it needs are in place. C++
tests under `tests/test_noncomp_*.cc`; Python tests under
`tests/python/test_noncomputational.py`.

1. **`level` and `transition_instrument` unit tests.**
   - Default level set roundtrips.
   - Matrix orientation: `T[to, from]` matches a known per-column
     no-jump weight.
   - Source-independence check rejects a hand-crafted Computational
     pair of differing columns; accepts identical columns.
2. **`model` validation tests.**
   - Initial-state sum, unknown-gate keys, classifier shape, policy
     enum values.
3. **`history` sampler tests.**
   - Deterministic seed → fixed history.
   - Initial-state-only sampling produces marginal frequencies within
     a binomial bound at large N.
4. **`rewriter` unit tests.**
   - Lost-site single-q gate dropped; two-q gate rejected by default.
   - Reset on lost rejected; `RL` (or its config-flag analog) restores.
   - Hidden M insertion at coherent-site loss, with `rec[-k]`
     renumbering verified on a circuit that has detectors before *and*
     after the loss point.
   - Lost visible-measurement policy `"random"` and `"zero"` each
     produce the configured output.
5. **End-to-end `sample_noncomputational` Python tests.**
   - Visible measurement record layout unchanged vs. lossless run with
     `LOSS=0`.
   - Survivor statistics on a small (3-qubit) entangled circuit with
     forced loss match a Python brute-force enumerator within shot
     noise.
   - `LOSS=1` (every site lost initially) returns a trivial output and
     a full noncomp sidecar.
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

- **`RL` as a new HIR op vs. a config flag on existing `R`.** A new op
  adds a parser/lower change (which we said we would not do in MVP),
  but folding it into `R` via a policy flag makes the semantics
  context-sensitive (same circuit text means different things
  depending on the model). Provisional: config flag on the model
  (`policy.reset_restores_lost = bool`), no new circuit op. If we
  later add `LOSS(p)` syntax, `RL` rides with it.
- **Herald metadata transport.** The orchestrator sidecar shape is
  written above as "per-site final status + classifier output + herald
  bits". We may want a more structured sidecar (e.g., numpy structured
  array) for performance / decoder integration. Provisional: keep the
  sidecar a typed dict in Python for the MVP; convert to a structured
  array if/when a decoder consumer demands it.
- **Initial-state correlation across sites.** §3 makes
  `initial_state` an independent per-site distribution. sqale-sim
  matches this. If a hardware model needs correlated initial states,
  that is a separate feature; reject correlated initial-state config
  in this MVP.

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
