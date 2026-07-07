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
restoring a qubit when the policy allows it; per-trajectory concrete
statuses; per-trajectory initial-state sampling; semantic validation
that the configured instrument set is pre-sampleable.

**Out:** dynamic control-flow IR above HIR; per-qubit classical
distributions; full qudit simulation. (Several items originally out of
the MVP have since shipped: unknown coherent state-dependent
transitions, the exact no-jump filter, trap-and-resume execution, and
the continuation cache landed via
[state-dependent-jumps.md](state-dependent-jumps.md), and the
`LOSS(p)` / `LEVEL_TRANSITION[name]` circuit annotations landed with
the annotation layer, §5.0.)

**Status of `LOSS(p)` syntax:** deferred. A later PR may add it as
syntactic sugar over this model. The prior LOSS design notes (`R` vs
`RL` split, random lost-measurement default, deltakit-style
`HERALD_LEAKAGE_EVENT` shape) feed the rewrite-policy table in §5; they
do not feed a new instruction surface in this MVP.

## 2. Status representation

### 2.0 Terminology

This MVP tracks noncomputational level by clifft circuit qubit index.
The doc uses "qubit" consistently with the rest of clifft's API and
internals (`Target::qubit`, `num_qubits`). It does not yet model atom
identity or trap site separately; a later movement-aware extension can
distinguish atom, trap site, and circuit qubit.

### 2.1 Level categories vs. qubit status kinds

The model has two related-but-distinct enums.

`LevelCategory` is a property of a *level* in the model:

```cpp
enum class LevelCategory : uint8_t {
    Computational = 0,  // coherent, in superposition by default
    Leaked        = 1,  // present, outside computational subspace
    Lost          = 2,  // absent / vacuum
};
```

`QubitStatusKind` is a per-qubit *runtime* tag carried in the
trajectory. It splits the computational case by whether the
energy-basis value has been resolved:

```cpp
enum class QubitStatusKind : uint8_t {
    ComputationalUnknown = 0,  // in H_C, energy-basis value unresolved
    ComputationalKnown   = 1,  // in H_C with a known g/e value
    Leaked               = 2,
    Lost                 = 3,
};
```

Splitting the runtime tag from the level-property enum lets us
encode "computational but unknown" as a kind that *cannot* carry a
specific level id, enforcing the source-resolved invariant
structurally rather than by convention. See §2.3.

`ComputationalKnown` earns its place as a kind distinct from an opaque
`Computational`: three code paths branch on it. The sampler takes the
*exact* known-source column for a state-dependent transition on a
`ComputationalKnown` qubit and only rejects (or approximates) on
`ComputationalUnknown` (§4.2); the rewriter prepends an `X` prep for a
qubit whose sampled initial level is the known `|1>` state; and a
classically-controlled `X` demotes `Known → Unknown` while leaving an
already-`Unknown` qubit untouched (§5.2.1).

### 2.2 Levels (model-defined)

A `Level` is a model-defined tag with a stable integer id, a label, and
a `LevelCategory`. Qubits transition between status kinds during
simulation:

- `ComputationalUnknown ↔ ComputationalKnown`: a Z-basis measurement
  or Z-basis reset promotes `Unknown → Known` with the sampled level
  id (`g` or `e`). Any subsequent quantum gate that does not preserve
  the energy basis demotes `Known → Unknown`, discarding the level id
  (Hadamard on `|0>` is the canonical example).
- `(anything) → Leaked` or `Lost`: via a transition jump.
- `Leaked / Lost → Computational`: only via reset/reload, gated by
  policy. Z-basis reset (`R`) lands in `ComputationalKnown(g)`;
  X/Y-basis reset (`RX`/`RY`) lands in `ComputationalUnknown` (the
  post-reset state is in a non-energy basis). `Leaked → Computational`
  via any of these is allowed; `Lost → Computational` only when
  `policy.reset_restores_lost` is set. There is no spontaneous
  coherent return from a leaked level back into a superposition with
  `g`/`e`.

A level table must contain exactly two `Computational` levels; in table
order the first is the `|0>` state and the second is `|1>`. The rewriter
uses `computational_one_id()` to prepend a preparation gate when an
initial sample places the qubit in `ComputationalKnown` with the `|1>`
level, or when a transition materializes the carrier at the `|1>` level
(the SVM default initialization is `|0...0>`). `Leaked` and `Lost` levels
carry no basis information — "lost from `|1>`" provenance, if ever needed,
belongs in an event record or in distinct levels, not in the level tag.

`LevelSet` validation rejects any table without exactly two `Computational`
levels; downstream paths (visible Z-basis measurement, Z-basis reset,
initial prep, classifier defaults) need unambiguous `g`/`e` ids.

Default level set for the MVP:

| id | label    | category      | notes                          |
|----|----------|---------------|--------------------------------|
| 0  | `g`      | Computational | logical 0                      |
| 1  | `e`      | Computational | logical 1                      |
| 2  | `leak_g` | Leaked        | metastable / Rydberg "leak g"  |
| 3  | `leak_e` | Leaked        | metastable / Rydberg "leak e"  |
| 4  | `lost`   | Lost          | empty trap / vacuum            |

Users may construct a `NonComputationalModel` with a different level
set, but the default ships unchanged.

### 2.3 Per-trajectory qubit state

```cpp
constexpr uint8_t kInvalidLevel = 0xFF;

class QubitStatus {
public:
    static QubitStatus computational_unknown();

    // Build a known-source status without validating level_id against
    // a table. Reserved for tests and interior code; user code should
    // go through LevelSet's validated factories.
    static QubitStatus computational_known_unchecked(uint8_t level_id);
    static QubitStatus leaked_unchecked(uint8_t level_id);
    static QubitStatus lost_unchecked(uint8_t level_id);

    QubitStatusKind kind() const;
    uint8_t level_id() const;
    bool is_unknown_computational() const;
    std::optional<uint8_t> known_source_level() const;
    uint8_t require_classical_source_level() const;  // throws on Unknown

private:
    QubitStatus(QubitStatusKind kind, uint8_t level_id);
    QubitStatusKind kind_;
    uint8_t level_id_;
};
```

The two fields together carry the runtime status of one qubit during
history sampling. Trajectory state is `std::vector<QubitStatus>
statuses;`, two bytes per qubit.

**Invariants:**

| `kind`                  | `level_id` constraint                          |
|-------------------------|------------------------------------------------|
| `ComputationalUnknown`  | `kInvalidLevel` (no source level is resolved)  |
| `ComputationalKnown`    | must be a `Computational`-category level id    |
| `Leaked`                | must be a `Leaked`-category level id           |
| `Lost`                  | must be a `Lost`-category level id             |

The canonical construction path is through `LevelSet` (see §3.2),
which validates the (kind, level_id) pair against its table:

```cpp
QubitStatus s = level_set.computational_known(g_id);
QubitStatus t = level_set.leaked(leak_g_id);
QubitStatus u = level_set.lost(lost_id);
QubitStatus v = QubitStatus::computational_unknown();  // no level id needed
```

`QubitStatus` is non-aggregate; its constructor is private. The
`_unchecked` static factories exist for tests and for interior code
where the invariant is already established (the name is the
warning); user code goes through `LevelSet`.

Source-dependent transitions must call `known_source_level()` or
`require_classical_source_level()` so the "unknown coherent"
rejection path stays consistent: most rewrite-policy entries only
need `s.kind`; only the source-dependent column lookup needs the
level id, and the accessors are the only safe way to read it.

"Known" here means known in the computational basis (`|0>` or `|1>`)
— the basis in which transition matrices are indexed. X- and Y-basis
measurements or resets do not produce `ComputationalKnown`; they
leave the qubit in `ComputationalUnknown` (see §5.2.1).

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

    # Independent per-qubit initial level distribution.
    # Indices into levels[]; sums to 1 per qubit (validated in C++).
    initial_state=[0.0065, 0.96, 0.0065, 0.027, 0.0],

    # Per-gate transition instruments. Matrix shorthand uses T[to, from]
    # convention. Keys are gate names from the existing clifft circuit
    # vocabulary; unknown keys reject at validation time.
    #
    # MVP semantics for every entry:
    #   - The instrument fires AFTER the gate.
    #   - "target" means a qubit operand of the gate, not a record
    #     target.
    #   - For multi-qubit gates, the instrument applies INDEPENDENTLY
    #     to each qubit operand (per-qubit marginal). Joint/correlated
    #     two-qubit instruments are out of scope; see section 9.
    transitions={
        "CZ": clifft.TransitionInstrument(matrix=[...]),   # 5x5
        "RZ": clifft.TransitionInstrument(matrix=[...]),
    },

    # Level -> visible-symbol classifier applied at measurement.
    # Shape: (len(symbols), len(levels)). Entry matrix[s, l] is
    # P(symbol=s | level=l).
    #
    # The matrix is COLUMN-SUBSTOCHASTIC: each column sum lies in
    # [0, 1] and the deficit `1 - sum(column)` is implicit
    # P(reject | level). A classifier call that samples "reject"
    # raises with the offending qubit and level. Default identity +
    # reject is just:
    #     [[1, 0, 0, 0, 0],   # P("0" | level) over [g, e, leak_g, leak_e, lost]
    #      [0, 1, 0, 0, 0]]   # P("1" | level)
    # which has column sums [1, 1, 0, 0, 0] — leaked/lost reject
    # with probability 1.
    classifier=clifft.MeasurementClassifier(
        symbols=["0", "1"],
        matrix=[[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0]],
    ),

    # Policy for downstream operations on noncomputational qubits.
    # See section 5 for the per-op table (ops with no representable
    # effect on a vacated site are dropped).
    policy=clifft.NonComputationalPolicy(
        reset_restores_lost=False,
    ),
)
```

There is intentionally *no* separate `lost_measurement` policy knob.
Lost-qubit measurement behavior is fully specified by the classifier
matrix's `lost`-level column (with deficit = reject). Users who want
random-bit-on-lost configure
`classifier.matrix[:, lost_id] = [0.5, 0.5]`; users who want
deterministic-0-on-lost set `[1.0, 0.0]`; users who want reject leave
the column at zero. One way to specify, not two.

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
    LevelCategory category;
};

// Validated level table. Construction runs the section 4.1 level-set
// checks and throws std::invalid_argument on failure. Owns the
// QubitStatus factories so that level ids and tables stay paired.
class LevelSet {
public:
    explicit LevelSet(std::vector<Level> levels);
    static LevelSet default_set();

    std::span<const Level> levels() const;
    const Level& at(uint8_t level_id) const;
    size_t size() const;

    QubitStatus computational_known(uint8_t level_id) const;
    QubitStatus leaked(uint8_t level_id) const;
    QubitStatus lost(uint8_t level_id) const;
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

// Distinct from TransitionInstrument: rectangular column-substochastic
// map from levels to reported user-facing symbols. Column sums lie in
// [0, 1]; the deficit per level is implicit P(reject | level). No
// no-jump branch, no concept of source-independence on Computational
// levels.
class MeasurementClassifier {
public:
    static MeasurementClassifier from_matrix(
        std::vector<std::string> symbols,
        std::vector<std::vector<double>> matrix);  // shape (symbols, levels)

    // P(reject | level) = 1 - sum_s matrix[s, level], computed at
    // construction and used by the sampler.
    double reject_probability(uint8_t level_id) const;
    // ... accessors
};

class NonComputationalModel {
public:
    NonComputationalModel(
        LevelSet levels,
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

### 3.3 Construction paths (compositional now, spec-based later)

The C++ constructor above is *compositional*: the caller builds each
`TransitionInstrument` / `MeasurementClassifier` against a `LevelSet`,
then hands the pre-built objects to the model. This keeps instruments
independently constructible and testable, but it admits a misuse a
level-count check cannot catch — a component built against a
different but same-sized `LevelSet` would bind its columns to the
wrong level ids. To close that, each instrument and classifier records
a deterministic `LevelSet::fingerprint()` (over each level's label
and category, in order) at construction, and the model
rejects any component whose fingerprint does not match its own table.

The fingerprint exists *only* to guard the compositional path. The
intended **primary, Python-facing** construction is spec-based: the
model receives raw matrices and symbols and constructs every bound
component against its single `LevelSet` internally, so there is only
one level table in scope and no fingerprint is needed or surfaced.

Plan: add a spec-based `NonComputationalModel` builder (raw
`transition` matrices + classifier spec, built against the model's
`LevelSet`) when bindings land (§6 step 9/10), and bind *that* shape
in Python. At that point decide whether the compositional constructor
stays public or becomes an internal/test-only entry. Fingerprints
must not leak into the Python API or user docs. Transition keys are
arbitrary names, stored verbatim and resolved by exact key from
`LEVEL_TRANSITION[key]` annotations; a key that names a hookable
physical gate additionally registers a gate hook (§5.0).

## 4. Validation: construction-time vs. sample-time

Validation is split: model construction checks shape and self-consistency;
the runtime sampler checks whether each transition's source-context is
representable in the MVP.

### 4.1 Construction-time checks (always run, throw on failure)

1. **Level set well-formed.** Every level id in `[0, len(levels))`,
   categories all in `LevelCategory` (unrecognized enum values reject).
   The table must contain exactly two `Computational` levels; in table
   order the first is the `|0>` (g) state and the second is `|1>` (e).
   Downstream code needs unambiguous `g`/`e` ids.
2. **Initial state is a probability vector** over `levels`. Sums to 1
   within tolerance.
3. **Transition matrices are square**, of size `len(levels)`, entries
   in `[0, 1]`. (Square because each instrument fires per qubit
   operand; per-qubit shape is `len(levels) x len(levels)`. Joint
   correlated two-qubit instruments would be a different type, out
   of scope.)
4. **Transition matrix column sums in [0, 1].** Implied no-jump weight
   per source is `1 - sum(col)`.
5. **Classifier matrix is column-substochastic.** If a
   `MeasurementClassifier` is provided, its matrix has shape
   `(len(symbols), len(levels))`, every entry is in `[0, 1]`, and
   every column sum lies in `[0, 1]`. The deficit per column is
   `P(reject | level)`.
6. **Policy values are well-formed** (enum values, not free strings on
   the C++ side; Python sugar translates).
7. **Transition keys are tag-safe names**: nonempty, with no `]` or
   newline, so a `LEVEL_TRANSITION[key]` annotation can reference them.
   A key that names a hookable gate registers a gate hook, and two keys
   resolving to the same gate reject; any other key is a named-only
   reference, not an error.

Each `TransitionInstrument` also computes and caches a derived flag
at construction:

- `is_source_independent_on_computational`: true iff every column whose
  source level has category `Computational` is bit-identical (within
  tolerance) to the others.

This flag is *not* a validation gate — source-dependent matrices are
fine to declare. Whether they are applicable depends on the source
context at sample time. The sqale-aligned `cz_transition_matrix` and
`rz_transition_matrix` in §1's prior art have distinct `g` and `e`
columns; they are valid `TransitionInstrument`s and will be accepted
here.

### 4.2 Sample-time check (per (transition, target-qubit) firing)

When a transition fires on a target qubit with `QubitStatus s`:

- `s.kind == ComputationalKnown`: use the column for `s.level_id`
  directly. Always allowed.
- `s.kind == Leaked` or `s.kind == Lost`: use the column for
  `s.level_id` directly. Classical Markov update. Always allowed.
- `s.kind == ComputationalUnknown`: if
  `is_source_independent_on_computational` is true, the no-jump branch
  is scalar on `H_C` and the jump branches are source-independent;
  sample without consulting amplitudes. Otherwise the fire is resolved
  at runtime by the exact-mode driver
  (design/state-dependent-jumps.md): the circuit compiles once with
  every annotation kept as a runtime instrument site, fire
  probabilities are evaluated on the live state, and a fire that
  cannot resolve in-line traps to the driver, which recompiles the
  remaining circuit under the now-known status and resumes. Exact for
  every source context, at a cost exponential in the number of
  damping-expanded sites (see the `damping` policy there). This is the
  only sampling path; a "reject if my model needs runtime resolution"
  contract check is a future validator's job
  (`noncomp.validate_static`), not a sampling policy.

This is the "pre-sampleable" boundary, enforced where it actually
matters (at the unknown-coherent-source point) rather than at model
construction. It also means a model with a source-dependent instrument
that only ever fires on known-or-classical sources runs fine in MVP.

## 5. Sampling, rewrite, and policy table

### 5.0 Transitions are circuit annotations

Transition consult points are first-class circuit instructions:

- `LEVEL_TRANSITION[name] q...` applies the model transition matrix stored
  under `name` to each target independently. Any key in the model's
  `transitions` map is referenceable; a key that names a gate (e.g.
  `"CZ"`) additionally acts as a *hook*, expanded by the annotation
  layer into a `LEVEL_TRANSITION[key]` annotation after every occurrence of
  that gate (one single-target annotation per Physical operand, in
  operand order). Feedback operands are virtual and get none.
- `LOSS(p) q...` applies a uniform loss with probability `p` inline; it
  requires a level table with exactly one Lost-category level.

**A transition fires at its circuit position, and the source column is
the qubit's status there.** This is the standard noise-follows-the-
ideal-operation convention; the annotation carries no reference to any
gate, and placement is meaning: putting an annotation before or after
an operation selects which state it consults. Consequences relative to
attaching transitions to gates with entry-status sources:

- A hook on a Z-diagonal gate (`CZ`, `S`, ...) consults the same
  definite level the gate entered with (the status stepper tracks
  knownness through diagonal and X-type gates), so the exact
  known-source path survives where it physically should.
- A hook on a basis-mixing gate (`H`, ...) consults a genuine
  superposition: a source-dependent matrix there is an unknown-source
  consult (rejected, or runtime-resolved under the exact policy). The
  old entry-status column
  choice was an artifact of attachment, not physics.
- A hook on a measure-and-reset consults the post-reset state. A
  transition acting on the pre-reset level -- readout-induced loss --
  is written explicitly as `M q`, `LEVEL_TRANSITION[name] q`, `R q`.

Annotations whose qubit has a definite (leaked/lost) level are consumed
by the rewriter against pre-drawn outcomes, leaving only their carrier
edits; annotations on computational qubits stay runtime instrument
sites, consulted by the VM. `clifft.compile()` rejects circuits
containing them.

### 5.1 Pipeline

The sampling pipeline is the exact-mode driver
([state-dependent-jumps.md](state-dependent-jumps.md)): the annotated
circuit compiles once as a shared main line with every annotation kept as
a runtime instrument site; each shot draws its initial levels (known |1>
levels become per-shot Pauli-frame preloads, leaked/lost levels compile
their own from-the-top continuation), executes, and resolves transition
fires against the live state -- in-line where possible, by trap,
continuation recompile, and resume where not. The driver returns the
user-facing `(measurements, detectors, observables)` unchanged from the
existing API plus the noncomputational sidecar (per-qubit final status
and herald bits).

An earlier revision of this note specified a per-shot ahead-of-time
pipeline here (sample a trajectory, rewrite, compile, run -- one compile
per shot); it shipped as the MVP and was retired once exact runtime
resolution was validated, together with its `equalize_rates` and
`reject` policies (state-dependent-jumps.md §6, post-plan note). The
rewrite-policy table below survives it unchanged: the per-node
semantics live in the shared rewriter walk the continuation rewrite
runs on.

### 5.2 Default rewrite-policy table

Rows are (operation kind, `QubitStatusKind` at op time). "Reject"
means the C++ rewrite raises with a clear message naming the op index
and qubit; no silent approximation. "Apply, demote to Unknown" means
the op runs unchanged but the qubit's status transitions
`ComputationalKnown → ComputationalUnknown` afterward (per §5.2.1).

| Operation                | ComputationalUnknown | ComputationalKnown                  | Leaked                                  | Lost                                              |
|--------------------------|-----------------------|--------------------------------------|------------------------------------------|---------------------------------------------------|
| Single-qubit gate        | apply                 | apply, demote to Unknown            | drop                                    | drop                                              |
| Single-qubit Pauli noise | apply                 | apply, demote to Unknown            | drop                                    | drop                                              |
| Two-qubit gate           | apply                 | apply, demote to Unknown on both    | drop                                    | drop                                              |
| MPP / multi-target meas  | apply                 | apply, demote to Unknown on all     | reject                                  | reject                                            |
| Visible Z-basis meas `M`              | apply, promote to `ComputationalKnown(outcome)` | apply, visible outcome = current `level_id` (status unchanged) | classifier; reject probability per `leak_*` column | classifier; reject probability per `lost` column |
| Visible Z-basis meas-and-reset `MR`   | apply, **post-op `ComputationalKnown(g)`**; visible outcome reflects sampled value | apply, visible outcome = current `level_id`; **post-op `ComputationalKnown(g)`** | classifier; post-op `ComputationalKnown(g)` if `reset_restores_lost` applies, else status preserved | classifier; post-op `ComputationalKnown(g)` if `reset_restores_lost`, else the site stays lost |
| Visible X/Y-basis meas (`MX`/`MY`)    | apply, stays `ComputationalUnknown`  | apply, demote to Unknown | reject | reject |
| Visible X/Y-basis meas-and-reset (`MRX`/`MRY`) | apply, stays `ComputationalUnknown`; **post-op also `ComputationalUnknown`** | apply, demote to Unknown | classifier; post-op `ComputationalUnknown` if `reset_restores_lost` applies, else preserved | classifier; post-op `ComputationalUnknown` if `reset_restores_lost`, else the site stays lost |
| Z-basis reset `R`        | apply, promote to Known (`g`) | apply, set `level_id = g` | restore to ComputationalKnown (`g`) | drop unless `policy.reset_restores_lost` is set; then restore to ComputationalKnown (`g`) |
| X/Y-basis reset `RX`/`RY` | apply, stays Unknown | apply, demote to Unknown | restore to ComputationalUnknown      | drop unless `policy.reset_restores_lost` is set; then restore to ComputationalUnknown |
| Detector / Observable    | unchanged             | unchanged                            | unchanged                               | unchanged                                         |

Notes:

- An operation with no representable effect on a leaked or lost operand
  is dropped whole (identity on the surviving operands). This is the
  only behavior, not a policy the caller selects. The exception is a
  non-reset X/Y-basis measurement (`MX`/`MY`) or a multi-target/parity
  measurement (`MPP`): it rejects, because the classifier defines a
  single Z-basis-like record bit that is not a faithful X/Y or parity
  outcome. A measure-and-reset (`MR`/`MRX`/`MRY`) is kept instead — its
  record is the classifier bit and its reset re-prepares the site, so
  the readout basis is incidental on a vacated carrier.
- There is no separate `RL` op in MVP. Lost-qubit reset drops by default
  (the vacated site is unaffected); the `policy.reset_restores_lost`
  flag turns it into a reload that restores the qubit to a
  `ComputationalKnown` status. See §8 for the rationale.
- **Lost-qubit measurement is fully specified by the classifier
  matrix's `lost` column.** There is no separate `lost_measurement`
  policy knob: random-bit is `[0.5, 0.5]`, deterministic-0 is
  `[1, 0]`, reject is `[0, 0]`. A classifier may carry a third
  symbol that *heralds* the slot — `[0, 0, 1]` is a deterministic
  loss herald. The herald is reported per measurement in the result
  sidecar; the visible record keeps a uniformly drawn bit (a heralded
  outcome carries no preferred computational value, and a pinned bit
  would silently bias detectors), so the record layout and every
  `rec[-k]` reference are unchanged. Alphabets beyond three symbols
  have no defined mapping onto (bit, herald) and reject at rewrite.
  Mechanically, the rewrite replaces the measurement with an `MPAD`
  padding its visible slot; a stochastic column adds a
  `READOUT_NOISE` on that slot so the bit is drawn at sample time
  inside the VM, while a deterministic column pads the literal bit
  with no draw.
- **Computational-qubit Z measurements (`M`, `MR`) receive the
  classifier's computational columns as readout confusion**: the true
  outcome is misreported at the column's off-diagonal rate via an
  asymmetric in-record flip (`READOUT_NOISE(p01, p10)` on the slot).
  The qubit still collapses to its true state -- real readout error is
  a misreport. X/Y-basis measurements are not level readouts and carry
  no confusion. Identity computational columns add nothing, and a
  computational column must place all its probability on the two
  record symbols.
- "Apply, demote to Unknown" is the conservative default for
  basis-mixing gates. Z-diagonal gates preserve a known level and
  X-type gates flip it (the gate classification lives in the status
  stepper and defaults to demotion for anything unclassified), so the
  exact known-source path survives diagonal circuits.

### 5.2.1 QubitStatusKind transitions

A qubit's `QubitStatusKind` transitions as follows during history
sampling:

| Cause                                            | Effect                                                                  |
|--------------------------------------------------|-------------------------------------------------------------------------|
| Initial-state sample (Computational level)       | `ComputationalKnown(level_id)`                                          |
| Initial-state sample (Leaked level)              | `Leaked(level_id)`                                                      |
| Initial-state sample (Lost level)                | `Lost(level_id)`                                                        |
| Z-diagonal gate (`Z`/`S`/`T`/`CZ`/`R_Z`...), kind == Known | level preserved: `ComputationalKnown(level)`                  |
| X-type gate (`X`/`Y`), kind == Known             | level flipped: `ComputationalKnown(other level)`                        |
| Any other quantum gate touching qubit, kind == Known | demote to `ComputationalUnknown`                                    |
| Z-basis measurement `M`                          | `ComputationalKnown(g)` or `ComputationalKnown(e)` per outcome          |
| Z-basis measurement-and-reset `MR`               | `ComputationalKnown(g)` (post-op state is reset, regardless of outcome) |
| X- or Y-basis measurement (`MX`/`MY`/`MRX`/`MRY`)| post-state is in X/Y basis, not energy basis: `ComputationalUnknown`    |
| Z-basis reset `R`                                | `ComputationalKnown(g)`                                                 |
| X- or Y-basis reset `RX`/`RY`                    | `ComputationalUnknown`                                                  |
| Reload via `policy.reset_restores_lost` (Z-basis) | `ComputationalKnown(g)`                                                |
| Transition jump to a Computational-category level | `ComputationalKnown(destination_level_id)`                             |
| Transition jump to a Leaked-category level        | `Leaked(destination_level_id)`                                         |
| Transition jump to a Lost-category level          | `Lost(destination_level_id)`                                           |

"Known" here means known *in the energy basis* (`g`/`e`) — the basis
transition matrices are indexed in. A qubit that is in a known X- or
Y-basis state after `RX`/`MY`/etc. is still coherent in the energy
basis (a superposition of `g` and `e`), so its kind is
`ComputationalUnknown`. Source-dependent transitions on it will hit
the §4.2 sample-time rejection unless the instrument is
source-independent on Computational sources.

### 5.2.2 Ledger status is *pre-SVM-known*, not trajectory-physical

The status the classical walk tracks is what is **classically known
before SVM execution**, which is narrower than the physically-collapsed
trajectory state: the walk compiles circuits and pre-draws classical
consults, so it cannot consult a measurement outcome that only exists
inside the SVM.

The consequences, which override the naive reading of the §5.2.1 table:

- **`M` on `ComputationalKnown(g/e)`**: status stays known. The value
  was already classically known, so a later source-dependent
  transition may use it.
- **`M` on `ComputationalUnknown`**: the ledger status stays
  `ComputationalUnknown`. It does **not** promote to
  `ComputationalKnown`, because the outcome is produced by the SVM, not
  the classical walk. (The SVM still performs the real quantum
  measurement and collapse; that is independent of the ledger.) A
  source-dependent transition fired on this qubit afterward is resolved
  at runtime per §4.2.
- **`R` / `MR` (Z-basis)**: produce `ComputationalKnown(g)` — the value
  comes from the instruction, not an SVM outcome, so it is pre-SVM
  known.
- **`RX` / `RY` / `MRX` / `MRY`**: produce `ComputationalUnknown` (not a
  `g`/`e` energy eigenstate), as in §5.2.1.

The §5.2.1 row "`M` → `ComputationalKnown(g)` or `(e)` per outcome"
describes the *physical* trajectory inside a single shot; it is **not**
the precompile history status unless the value was already known before
SVM execution. "Measure, then use the measured value to drive a later
noncomputational transition" requires a segmented runtime / replan and
is out of scope for the pre-sampled MVP (§9). The history layer must
not pre-sample measurement outcomes: doing so would either skip the
measurement back-action on the computational state or force a
branch-and-continue boundary, which is exactly the dynamic/JIT path
deferred in §1.

### 5.2.3 Operations with no representable effect on leaked/lost operands

An operation whose physical effect on a leaked or lost operand is not
representable is excised whole — identity on the surviving operands —
modeling the reading that an interaction with a vacated or leaked site
does not happen. This is the only behavior; there is no policy knob to
turn it off. Concretely:

- single-qubit gate on a Leaked operand: drop;
- two-qubit gate, multi-qubit noise channel, or classical feedback with
  a Leaked or Lost operand: drop whole;
- non-restoring lost-qubit reset: drop;
- non-restoring lost-qubit measure-and-reset: kept — the visible record
  slot must survive, the classifier supplies the bit, and the site
  simply stays lost;
- a non-reset X/Y-basis or multi-target measurement (`MX`/`MY`/`MPP`)
  rejects: dropping a measurement would shift the record, and no
  single-bit substitution is a faithful X/Y or parity outcome.

A "does this model ever drive a dead qubit" contract check — distinct
from these per-op representability limits — is a static validator's job
(a future `noncomp.validate_static`), not a sampling policy: as a
sampling-time reject it would fire seed-dependently, since whether a
qubit is vacated before a given op is a per-shot draw.

A dropped operation has no physical effect, so a surviving operand's
status keeps its entry value (it is not demoted). Transition
annotations are separate instructions and are never policy-gated: the
noise process fires whether or not the intended gate could act. The
sampler and the rewriter advance statuses through this same rule.

### 5.3 Hidden carrier edits at transition jumps

Whenever a qubit whose status is `ComputationalUnknown` transitions
to a noncomputational kind (either `Leaked` or `Lost`), theory
requires a hidden Z-basis measurement to unravel the partial trace.
The qubit's computational amplitude is entangled with the rest of the
state, and "removing" it from the coherent subspace — whether by
hardware loss or by escape to a noncomputational level — must trace
that amplitude out. The post-tag (`Leaked` vs `Lost`) only changes
the metadata; the trace-out unraveling is identical.

clifft already has the infrastructure for this: the `R`/`RX`/`RY`
reset ops are lowered by the frontend
(`src/clifft/frontend/frontend.cc:618-694`) to a *hidden* measurement
slot — tracked on a separate `hidden_meas_idx` counter, marked with
`meas_op.set_hidden(true)`, never exposed to the user-facing record,
followed by a conditional Pauli correction so the residual state on
survivors is the correct conditional state. This is exactly the
trace-out unraveling the noncomputational transition needs.

The MVP rewriter takes advantage of this directly. At a transition
`ComputationalUnknown → (Leaked or Lost)`, the rewriter inserts a
single `R` op on that qubit in the rewritten circuit. Concretely:

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

After the inserted `R`, the rewriter sets the qubit's status to the
destination noncomputational kind (`Leaked(level)` or `Lost(level)`)
in the trajectory and dispatches every subsequent op per the policy
table. Note that `R` semantically leaves the qubit in `|0>`
`ComputationalKnown` — that residual state is immediately
reinterpreted as the noncomputational kind by the trajectory; the
SVM's own post-`R` state on that qubit is irrelevant because no
later op reads computational amplitude from it.

The decision to insert `R` is made on the **computational carrier
state immediately before the jump destination is installed**, not on
the qubit's status at op entry. The base operation can make a qubit
coherent before the after-gate transition fires: a qubit can enter an
op as `ComputationalKnown(g/e)` and be demoted to `ComputationalUnknown`
by the gate, after which a jump to `Leaked`/`Lost` still has entangled
computational amplitude to trace out. Conversely, a qubit that is still
a definite `ComputationalKnown` atom at jump time (the base op did not
make it coherent) carries no entanglement to unravel, so no `R` is
needed. The relevant carrier state is therefore
`normal_post_op_status(entry, op)` — the status the base operation
would leave if no jump fired — evaluated with the *entry* status still
used for transition source-column selection.

#### 5.3.1 Carrier materialization at computational destinations

A jump can also land *inside* the computational subspace (§5.2.1 row
"jump to a Computational-category level →
`ComputationalKnown(destination_level_id)`"): a relaxation entry such
as `T[g][e]`, or a recapture entry such as `T[g][lost]`. Updating the
trajectory status alone is not enough there — the SVM carrier must be
*materialized* at the destination level, or the bookkeeping silently
diverges from the quantum state (a `Known(e)` qubit whose matrix says
it relaxed to `g` would keep measuring 1).

Conditional on the jump branch, the destination state is the definite
level `|d>` regardless of what the base op left behind, i.e. the
channel `rho -> |d><d| (x) Tr_q(rho)`. The reset lowering implements
exactly that: a hidden Z-basis measurement plus a corrective Pauli
collapses and rezeros the carrier to `|0>`. So the rewriter inserts,
immediately after the base op:

- an `R` on the qubit, always; and
- an `X` after it when the destination is the `|1>` computational
  level.

The edit is deliberately uniform over the carrier state the base op
would leave:

- **Coherent / entangled** (`ComputationalUnknown`): the hidden
  measurement is the correct unraveling of the collapse, exactly as in
  the trace-out case above.
- **Definite** (`ComputationalKnown(k)`): the hidden measurement is
  deterministic (no branch, no rank growth); the `R`(+`X`) prepares
  `|d>` whether or not `k == d`.
- **Stale noncomputational residual** (`Leaked`/`Lost` source whose
  carrier was never traced out because it was a definite atom at
  departure time): the `R` rezeros the leftover amplitude before the
  destination is prepared, so a recapture never resurrects a stale
  bit.

Both inserted ops are invisible to the record: `R` lowers to a hidden
measurement slot and `X` is a unitary, so points 1-4 above apply
unchanged and no `rec[-k]` index moves.

Timing convention: the transition's source column is selected from the
**op-entry** status, but the jump branch is applied **after** the base
operation. A measurement whose instrument relaxes `e -> g` therefore
records the pre-relaxation bit, and the qubit is prepared at `g` after
the readout.

Summary rule, per qubit operand of an op:

    pre             = status entering the op
    post_if_no_jump = normal_post_op_status(pre, op)
    outcome         = sampled/replayed transition branch (source column from pre)
    if outcome jumps to a Computational level d:
        insert R, then X if d is the One level   (materialize the carrier)
    else if outcome jumps to a Leaked/Lost level
                 AND post_if_no_jump is ComputationalUnknown:
        insert R                                  (hidden trace-out)
    final status    = outcome destination (jump wins)

## 6. New C++ headers and dependency order

Proposed layout under `src/clifft/noncomp/` (new directory):

1. `qubit_status.h` — `QubitStatusKind` enum, `QubitStatus` aggregate
   with `computational_unknown()` factory and accessors
   (`is_unknown_computational`, `known_source_level`,
   `require_classical_source_level`). Zero clifft-internal deps.
2. `level.h` — `LevelCategory` enum, `Level` struct,
   `LevelSet` (validated table that owns the `QubitStatus` factories
   `computational_known` / `leaked` / `lost`). Depends on (1).
3. `transition_instrument.h/.cc` — `TransitionInstrument`,
   matrix-to-branch expansion, derived
   `is_source_independent_on_computational` flag. Depends on (2).
4. `classifier.h/.cc` — `MeasurementClassifier`, column-substochastic
   `(symbols, levels)` map with derived `reject_probability(level_id)`.
   Depends on (2).
5. `policy.h` — `NonComputationalPolicy` struct. Zero deps.
6. `model.h/.cc` — `NonComputationalModel`, all construction-time
   validation per §4.1. Depends on (1)-(5).
7. `history.h`, `sampler.h/.cc` — retired with the ahead-of-time
   pipeline (the trajectory record and its per-shot sampler); the
   initial-level draw lives in the driver.
8. `rewriter.h/.cc` — produces a new `clifft::Circuit` from
   `(annotated, events, model)` via `rewrite_continuation`: the shared
   per-node walk (policy table, classifier record writes, confusion),
   carrier edits for recorded jumps, and live instrument sites for
   coherent qubits. Depends on (1)-(6) and `clifft::Circuit`.
9. `exact_driver.h/.cc`, `seed.h` — the runtime driver: shared main
   line, trap resolution, continuation cache, forced trace-outs, and
   per-shot seed derivation. Depends on (1)-(8), the compile pipeline,
   and the SVM.
10. `orchestrator.h/.cc` — top-level `sample_noncomputational` entry
    point: validation and seed resolution, then the driver. Depends on
    (9).

Python bindings in `src/python/bindings.cc` expose: `Level`,
`LevelSet`, `TransitionInstrument`,
`MeasurementClassifier`, `NonComputationalPolicy`,
`NonComputationalModel`,
`sample_noncomputational(circuit_text, model, shots, seed=None)`.

When step 10 lands, add the spec-based `NonComputationalModel`
construction path described in §3.3 (raw transition matrices +
classifier spec built against the model's own `LevelSet`) and bind
that shape in Python, rather than exposing the compositional
constructor and its `LevelSet`-fingerprint mechanics to users.

## 7. Test plan (dependency order)

Each test category lands once the headers it needs are in place. C++
tests under `tests/test_noncomp_*.cc`; Python tests under
`tests/python/test_noncomputational.py`.

1. **`qubit_status`, `level`, `transition_instrument`, `classifier`
   unit tests.**
   - `LevelSet::default_set()` validates; a table without exactly two
     Computational levels rejects; unrecognized `LevelCategory` enum
     value rejects; oversized level set rejects.
   - Matrix orientation: `T[to, from]` matches a known per-column
     no-jump weight.
   - `is_source_independent_on_computational` is true for identical
     `g`/`e` columns, false when they differ. Construction does *not*
     throw on differing-column instruments — both must build cleanly.
   - `MeasurementClassifier` matrix shape and per-column substochasticity
     enforced; column-sum > 1 rejected; mismatched (symbols, levels)
     dimensions rejected. `reject_probability(level_id)` matches
     `1 - sum(column)` for several hand-built matrices.
   - `LevelSet` status factories: `computational_known(g_id)` builds
     fine; `computational_known(leak_g_id)` rejects;
     `require_classical_source_level()` on a `ComputationalUnknown`
     status throws; `known_source_level()` returns `nullopt` on
     `ComputationalUnknown` and the carried level id otherwise.
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
     transition to a `ComputationalUnknown` qubit raises an error
     naming the op, qubit, and instrument. Applying the same
     transition to a `ComputationalKnown` qubit (post Z-basis `M`)
     succeeds.
4. **`rewriter` unit tests.**
   - Initial `e` sample on a qubit emits a leading `X` prep; initial
     `g` does not.
   - Lost-qubit single-q gate dropped; two-q gate rejected by default.
   - Reset on lost rejected by default; `policy.reset_restores_lost`
     enables restoration to `ComputationalKnown(g)`.
   - **Trace-out via inserted `R`**: at a structural-loss event on a
     previously-`ComputationalUnknown` qubit, an `R` is emitted on
     that qubit. The compiled rewritten circuit's
     `hir.num_hidden_measurements` increments by one, and the
     visible-record layout (and `rec[-k]` references in detectors and
     observables) is unchanged. Verified on a circuit that has
     detectors before *and* after the loss point.
   - Z-basis vs X/Y-basis measurement/reset: a qubit kind is
     `ComputationalKnown` after `M`; the same qubit's kind after `MX`
     is `ComputationalUnknown`. (Single-qubit assertions on sampler
     output.)
   - Lost-qubit visible measurement: classifier columns `[0.5, 0.5]`,
     `[1, 0]`, `[0, 0]` produce random / deterministic-0 / reject
     respectively.
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

## 8. Open question to settle before §6 step 7 (`rewriter`)

One remaining sidecar-shape question, not load-bearing for the
contract:

- **Herald metadata transport.** The orchestrator sidecar shape is
  written above as "per-qubit final status + classifier output +
  herald bits". We may want a more structured sidecar (e.g., numpy
  structured array) for performance / decoder integration.
  Provisional: keep the sidecar a typed dict in Python for the MVP;
  convert to a structured array if/when a decoder consumer demands it.

## 9. Out of scope but planned-for

Exact state-dependent jumps — the diagonal no-jump filter and the
trap-and-resume machinery — **have since shipped**, designed and
implemented in [state-dependent-jumps.md](state-dependent-jumps.md);
the `LOSS(p)` annotation shipped with them. Still open:

- Joint / correlated multi-qubit `TransitionInstrument` (instead of
  per-qubit marginal): a different type with shape
  `len(levels)^k x len(levels)^k` for `k` operands. Out of scope for
  MVP; CZ and similar gates use independent marginals.
- Compile cache by rewritten-circuit hash, with observable compile
  count.
- Segmented JIT / replan for true topology changes mid-shot.
- Decoder-side integration of herald bits.

These are explicitly outside the MVP and should not constrain the
schemas above beyond what is already noted.
