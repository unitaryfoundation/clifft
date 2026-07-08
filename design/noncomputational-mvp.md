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

### 2.1 Levels vs. qubit statuses

The model has two related-but-distinct enums, both fixed at compile
time.

`Level` names the five levels, in matrix index order, and
`LevelCategory` groups them:

```cpp
enum class Level : uint8_t { G, E, LeakG, LeakE, Lost };

enum class LevelCategory : uint8_t {
    Computational = 0,  // coherent, in superposition by default
    Leaked        = 1,  // present, outside computational subspace
    Lost          = 2,  // absent / vacuum
};
```

`QubitStatus` is the per-qubit *runtime* ledger entry: computational,
or the definite noncomputational level the qubit holds. Which basis
state a computational qubit holds -- if either definitely -- is SVM
runtime information the classical ledger never tracks, so `G` and `E`
collapse to one `Computational` status and a status carries no
auxiliary level field at all -- an invalid (status, level) pairing is
unrepresentable rather than validated.

An earlier revision split the computational kind into
`ComputationalKnown`/`ComputationalUnknown` so the ahead-of-time
sampler could resolve state-dependent transitions classically on
definite carriers. That consumer retired with the AOT pipeline: fires
now resolve inside the VM against the live state, so a ledger-side
level claim for a computational qubit is not per-shot truth, and the
split was removed with the pipeline.

### 2.2 Level structure

Qubits transition between statuses during simulation:

- `Computational → Leaked` or `Lost`: via a transition jump.
- `Leaked / Lost → Computational`: only via reset/reload, gated by
  policy. Any reset flavor restores the category. `Leaked →
  Computational` is always allowed; `Lost → Computational` only when
  `policy.reset_restores_lost` is set. There is no spontaneous
  coherent return from a leaked level back into a superposition with
  `g`/`e`.

A level table must contain exactly two `Computational` levels; in table
order the first is the `|0>` state and the second is `|1>`. The driver
uses `computational_one_id()` to preload the Pauli frame when an
initial draw places a qubit at `|1>`, and the rewriter uses it to
append the `X` when a recorded jump materializes the carrier at the
`|1>` level (the SVM default initialization is `|0...0>`). `Leaked` and `Lost` levels
carry no basis information — "lost from `|1>`" provenance, if ever needed,
belongs in an event record or in distinct levels, not in the level tag.

The level table is fixed at compile time:

| id | label    | category      | notes                          |
|----|----------|---------------|--------------------------------|
| 0  | `g`      | Computational | logical 0                      |
| 1  | `e`      | Computational | logical 1                      |
| 2  | `leak_g` | Leaked        | metastable / Rydberg "leak g"  |
| 3  | `leak_e` | Leaked        | metastable / Rydberg "leak e"  |
| 4  | `lost`   | Lost          | empty trap / vacuum            |

The structure is deliberately not configurable: the known use cases
fit it, the Python surface has only ever exposed this table, and a
future level would be added explicitly -- one enum value and one
matrix row/column -- rather than through a runtime table. (An earlier
revision carried a runtime `LevelSet` with per-component fingerprint
validation to guard user-defined tables; it was removed once it was
clear no public path could reach a non-default table.)

### 2.3 Per-trajectory qubit state

```cpp
// Per-qubit ledger status: computational, or the definite
// noncomputational level the qubit holds.
enum class QubitStatus : uint8_t {
    Computational = 0,
    LeakG = 1,
    LeakE = 2,
    Lost = 3,
};

constexpr QubitStatus status_for(Level level);
Level noncomp_level(QubitStatus status);  // logic error on Computational
```

Trajectory state is `std::vector<QubitStatus> statuses;`, one byte per
qubit.

`QubitStatus` and `Level` are plain enums: a status either is the one
computational value or *is* its definite noncomputational level, so no
(kind, level_id) pair exists to validate. `status_for(Level)` collapses
the computational levels; `noncomp_level(QubitStatus)` recovers the
definite level of a noncomputational status and is a logic error on the
computational one -- callers branch on the category first.

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
    # Optional; defaults to [1.0, 0.0, 0.0, 0.0, 0.0] (all qubits in g).
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
    # Two rows (record symbols 0 and 1), or three when the third row
    # is the herald symbol; columns run over the levels. Entry
    # matrix[s, l] is P(symbol=s | level=l).
    #
    # The matrix is COLUMN-STOCHASTIC: every column sums to 1 within
    # tolerance. Substochastic "reject" columns are not supported and
    # fail at construction, and a computational column places no mass
    # on the herald row. The faithful identity readout is:
    #     [[1, 0, 1, 0, 1],   # P("0" | level) over [g, e, leak_g, leak_e, lost]
    #      [0, 1, 0, 1, 0]]   # P("1" | level)
    classifier=clifft.MeasurementClassifier(
        matrix=[[1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]],
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

// The five levels, fixed at compile time (matrix index order).
enum class Level : uint8_t { G, E, LeakG, LeakE, Lost };
inline constexpr uint8_t kNumLevels = 5;

constexpr LevelCategory category(Level level);
constexpr const char* level_name(Level level);
constexpr QubitStatus status_for(Level level);

class TransitionInstrument {
public:
    // Construct from a square matrix using T[to, from] convention.
    // Validates matrix shape and that every column sum lies in [0, 1].
    // Source-dependent matrices are fully supported; the source level
    // is resolved at sample time (see section 4.2).
    static TransitionInstrument from_matrix(
        std::vector<std::vector<double>> matrix);

    // ... accessors for per-source branches, no-jump weight per source, etc.
};

// Distinct from TransitionInstrument: a rectangular column-stochastic
// map from levels to record symbols -- two rows, or three when the
// third row is the herald symbol. Every column sums to 1 within
// tolerance; substochastic (reject) columns fail at construction, and
// a computational column places no mass on the herald row. No no-jump
// branch, no concept of source-independence on Computational levels.
class MeasurementClassifier {
public:
    static MeasurementClassifier from_matrix(
        size_t num_symbols,
        std::vector<std::vector<double>> matrix);  // shape (num_symbols, levels)

    // ... accessors
};

class NonComputationalModel {
public:
    // The one construction path: raw matrices in, validated model out.
    // Throws at construction on shape/probability/key validation
    // failures (section 4.1).
    static NonComputationalModel from_spec(
        std::vector<double> initial_state,
        const std::map<std::string,
                       std::vector<std::vector<double>>>& transition_matrices,
        std::optional<ClassifierSpec> classifier_spec,
        NonComputationalPolicy policy);

    // ... accessors
};

}  // namespace clifft
```

### 3.3 Construction path

Construction is spec-based and there is one public path: `from_spec`
receives raw matrices and symbols and builds every component against
the fixed level structure, validating each and naming the offending
component on failure. (An earlier revision also had a *compositional*
constructor taking pre-built `TransitionInstrument` /
`MeasurementClassifier` objects, guarded by per-component `LevelSet`
fingerprints against mixing tables; both the second path and the
fingerprint machinery retired with the runtime level table.)
Transition keys are
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
5. **Classifier matrix is column-stochastic.** If a
   `MeasurementClassifier` is provided, its matrix has two or three
   rows (the third is the herald symbol), every entry is in `[0, 1]`,
   and every column sums to 1 within tolerance; a computational column
   places no mass on the herald row. Substochastic (reject) columns
   fail at construction.
6. **Policy values are well-formed** (enum values, not free strings on
   the C++ side; Python sugar translates).
7. **Transition keys are tag-safe names**: nonempty, with no `]` or
   newline, so a `LEVEL_TRANSITION[key]` annotation can reference them.
   A key that names a hookable gate registers a gate hook, and two keys
   resolving to the same gate reject; any other key is a named-only
   reference, not an error.

Source-dependent matrices are fine to declare; the source level is
resolved at sample time. The sqale-aligned `cz_transition_matrix` and
`rz_transition_matrix` in §1's prior art have distinct `g` and `e`
columns; they are valid `TransitionInstrument`s and are accepted
here.

### 4.2 Sample-time check (per (transition, target-qubit) firing)

When a transition fires on a target qubit with `QubitStatus s`:

- `s.kind == Leaked` or `s.kind == Lost`: use the column for
  `s.level_id` directly. Classical Markov update, consumed by the
  status walk without touching the SVM.
- `s.kind == Computational`: the annotation stays a runtime instrument
  site, resolved by the exact-mode driver
  (design/state-dependent-jumps.md): the circuit compiles once with
  every such site kept live, fire probabilities are evaluated on the
  live state, and a fire that cannot resolve in-line traps to the
  driver, which recompiles the remaining circuit under the now-known
  status and resumes. Exact for every source context, at a cost
  exponential in the number of damping-expanded sites (see the
  `damping` policy there). This is the only sampling path; a "reject
  if my model needs runtime resolution" contract check is a future
  validator's job (`noncomp.validate_static`), not a sampling policy.

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

- A hook on a Z-diagonal gate (`CZ`, `S`, ...) consults a state with
  the same level populations the gate entered with (diagonal gates do
  not mix the basis), and a leaked/lost level is untouched by the
  dropped gate -- post-op placement changes nothing there.
- A hook on a basis-mixing gate (`H`, ...) consults a genuine
  superposition: the site is a runtime instrument and the fire is
  evaluated on the live post-gate state. The old entry-status column
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
and herald bits). The final status reports leaked/lost per-shot truth
(the driver drew those jumps itself); computational qubits report as a
bare category on both the C++ and Python surfaces, because fires with
computational destinations resolve inside the VM and no final level is
knowable outside it.

An earlier revision of this note specified a per-shot ahead-of-time
pipeline here (sample a trajectory, rewrite, compile, run -- one compile
per shot); it shipped as the MVP and was retired once exact runtime
resolution was validated, together with its `equalize_rates` and
`reject` policies (state-dependent-jumps.md §6, post-plan note). The
rewrite-policy table below survives it unchanged: the per-node
semantics live in the shared rewriter walk the continuation rewrite
runs on.

### 5.2 Default rewrite-policy table

Rows are (operation kind, status category at op time). "Reject"
means the C++ rewrite raises with a clear message naming the op index
and qubit; no silent approximation.

| Operation                | Computational | Leaked                                  | Lost                                              |
|--------------------------|---------------|------------------------------------------|---------------------------------------------------|
| Single-qubit gate        | apply         | drop                                    | drop                                              |
| Single-qubit Pauli noise | apply         | drop                                    | drop                                              |
| Correlated-error chain (`E`/`ELSE_CORRELATED_ERROR`) | apply | apply | apply |
| Two-qubit gate           | apply         | drop                                    | drop                                              |
| MPP / multi-target meas  | apply         | reject (up-front gate A)                | reject (up-front gate A)                         |
| Visible Z-basis meas `M`              | apply; visible outcome from the SVM | classifier; reject probability per `leak_*` column | classifier; reject probability per `lost` column |
| Visible Z-basis meas-and-reset `MR`   | apply; visible outcome from the SVM, reset re-prepares the site | classifier; restores if `reset_restores_lost` would allow it (leaked always restores) | classifier; restores if `reset_restores_lost`, else the site stays lost |
| Visible X/Y-basis meas (`MX`/`MY`)    | apply | classifier; same cell as `M` — readout basis is incidental on a vacated carrier | classifier; same cell as `M` |
| Visible X/Y-basis meas-and-reset (`MRX`/`MRY`) | apply | classifier; restores (leaked always restores) | classifier; restores if `reset_restores_lost`, else the site stays lost |
| Reset `R`/`RX`/`RY`      | apply         | restore to `Computational` | drop unless `policy.reset_restores_lost` is set; then restore to `Computational` |
| Detector / Observable    | unchanged     | unchanged                               | unchanged                                         |

Notes:

- An operation with no representable effect on a leaked or lost operand
  is dropped whole (identity on the surviving operands). This is the
  only behavior, not a policy the caller selects. Single-qubit
  measurements (`M`, `MX`, `MY`) classify: on a vacated carrier the
  readout basis is incidental, so the classifier's single record bit is
  equally valid for Z-, X-, or Y-basis forms. A measure-and-reset
  (`MR`/`MRX`/`MRY`) keeps its record the same way; its reset half
  re-prepares the site only when the reset restores it (a leaked qubit
  always; a lost qubit only by policy). The exception is a multi-qubit parity
  measurement (`MPP`): it spans more than one qubit and has no faithful
  single-bit substitution, so it is rejected before sampling begins
  when the model is capable (gate A). The supported workaround is an
  explicit ancilla circuit — the ancilla ladder gates drop on the
  vacated operand, yielding the survivors' parity. A model that can
  leak or lose qubits also requires a classifier when the circuit
  measures (gate B); both checks are capability boundaries, not
  per-qubit reachability analyses.
- A correlated-error chain is never dropped, whatever its operands'
  statuses: each member must keep its slot in the else-conditioning
  (dropping the head would orphan the later members; dropping a middle
  member would hand its firing probability to them), and a mixed-operand
  member must keep its surviving qubits' Pauli components. Separable
  noise needs no such care because the parser splits it one node per
  target. Applying a chain member to a vacated carrier is sound because
  **vacated carriers are unobservable** — the one contract this whole
  table implements: records come from the classifier, gates on
  noncomputational operands drop, every restoration begins with a
  reset, and expectation probes are rejected. A Pauli frame flip parked
  on such a carrier is therefore never read.
- There is no separate `RL` op in MVP. Lost-qubit reset drops by default
  (the vacated site is unaffected); the `policy.reset_restores_lost`
  flag turns it into a reload that restores the qubit to a
  computational status. See §8 for the rationale.
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

### 5.2.1 QubitStatus transitions

A qubit's `QubitStatus` transitions as follows during the status
walk:

| Cause                                              | Effect                            |
|----------------------------------------------------|-----------------------------------|
| Initial-state draw                                 | the drawn level's category        |
| Transition jump                                    | the destination level's category  |
| Reset (any flavor) on `Leaked`                     | `Computational`                   |
| Reset on `Lost` with `policy.reset_restores_lost`  | `Computational`                   |
| Every other operation                              | unchanged                         |

No gate, measurement, or reset moves a computational qubit out of its
category, and which basis state it holds is SVM runtime information
the ledger never tracks.

### 5.2.2 Ledger status is *pre-SVM-known*, not trajectory-physical

The status the classical walk tracks is what is **classically known
before SVM execution**, which is narrower than the physically-collapsed
trajectory state: the walk compiles circuits and pre-draws classical
consults, so it cannot consult a measurement outcome that only exists
inside the SVM.

That is why a computational status carries no basis information: an
outcome-dependent refinement would mean reading the SVM's records
mid-walk. The SVM still performs every real quantum measurement and
collapse; that is independent of the ledger. "Measure, then let the
measured state drive a later noncomputational transition" is exactly
what the runtime instrument sites provide: the fire probability is
evaluated on the live (post-measurement) state, and an unresolvable
fire traps to the driver (§4.2). The classical walk itself never
pre-samples measurement outcomes.

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
- single-qubit measurement (`M`, `MX`, `MY`) on a Leaked or Lost
  operand: classifies — the readout basis is incidental on a vacated
  carrier; all three forms map to the same classifier record bit;
- multi-qubit parity measurement (`MPP`) with any Leaked or Lost
  operand: rejected up front (gate A) before sampling begins, because
  no single-bit substitution faithfully represents a parity outcome;
  the supported workaround is an explicit ancilla expansion.

Capability contract checks (gate A — parity under capable model; gate
B — no classifier when capable and measuring) run up front, before the
first shot, as a capability boundary rather than per-qubit reachability.

A dropped operation has no physical effect, so a surviving operand's
status keeps its entry value (it is not demoted). Transition
annotations are separate instructions and are never policy-gated: the
noise process fires whether or not the intended gate could act. The
sampler and the rewriter advance statuses through this same rule.

### 5.3 Hidden carrier edits at transition jumps

Whenever a computational qubit transitions to a noncomputational kind
(either `Leaked` or `Lost`), theory requires a hidden Z-basis
measurement to unravel the partial trace.
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

The rewriter takes advantage of this directly. At a recorded jump
`Computational → (Leaked or Lost)`, it inserts a single `R` op on that
qubit in the rewritten circuit. Concretely:

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
table. Note that `R` semantically leaves the qubit in `|0>` — that
residual state is immediately
reinterpreted as the noncomputational kind by the trajectory; the
SVM's own post-`R` state on that qubit is irrelevant because no
later op reads computational amplitude from it.

The `R` is inserted for **every** recorded jump, deliberately uniform
over the carrier state: on an entangled carrier the hidden measurement
is the trace-out unraveling itself, and on a definite carrier — one
the site collapsed before trapping, or one prepared by a deterministic
chain — the same hidden measurement is deterministic (no branch, no
rank growth) and the reset merely rezeros the vacated slot. Eliding
the reset for provably-definite carriers would require the ledger to
track basis information the SVM owns; the uniform edit costs one
deterministic hidden measurement instead.

#### 5.3.1 Carrier materialization at computational destinations

A jump can also land *inside* the computational subspace: a relaxation
entry such as `T[g][e]`, or a recapture entry such as `T[g][lost]`.
Updating the trajectory status alone is not enough there — the SVM
carrier must be *materialized* at the destination level, or the
bookkeeping silently diverges from the quantum state (a qubit sitting
in `|e>` whose matrix says it relaxed to `g` would keep measuring 1).

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

- **Coherent / entangled carrier**: the hidden measurement is the
  correct unraveling of the collapse, exactly as in the trace-out case
  above.
- **Definite carrier** (in some `|k>`): the hidden measurement is
  deterministic (no branch, no rank growth); the `R`(+`X`) prepares
  `|d>` whether or not `k == d`.
- **Noncomputational residual** (a `Leaked`/`Lost` source being
  recaptured): the `R` rezeros whatever the carrier holds before the
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
    outcome         = recorded/replayed transition branch (source column from pre)
    if outcome jumps to level d:
        insert R, then X if d is the One level
        (trace-out for a noncomputational d, carrier materialization
         for a computational one)
    final status    = outcome destination (jump wins)

## 6. New C++ headers and dependency order

Proposed layout under `src/clifft/noncomp/` (new directory):

1. `level.h` — the fixed level structure: the `Level` and `QubitStatus`
   enums, `LevelCategory` with the constexpr `category` / `status_for` /
   `noncomp_level` dispatchers, and the diagnostic name tables. Zero
   clifft-internal deps. (`qubit_status.h` merged into it when the
   status became an enum.)
3. `transition_instrument.h/.cc` — `TransitionInstrument`: validated
   `T[to, from]` matrix with cached column sums. Depends on (1).
4. `classifier.h/.cc` — `MeasurementClassifier`: stochastic
   `(symbols, levels)` map, shape-validated at construction.
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

Python bindings in `src/python/bindings.cc` expose the spec-based
model builder (raw matrices in) and
`sample_noncomputational(circuit_text, model, shots, seed=None)`; the
C++ value types never cross the boundary.

## 7. Test plan (dependency order)

This is the original MVP test plan, kept as a record of what each
layer had to demonstrate. The suites evolved with the pipeline -- the
AOT-era items (the history sampler, the known/unknown status split and
its accessors, the source-independence flag) retired with their code,
and the current shape is what `tests/test_noncomp_*.cc` and
`tests/python/` contain.

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

Note on oracle independence: the enumerator (`utils_noncomp_enumerator`)
imports the oracle's channel primitives (`utils_noncomp_oracle`) for its
quantum-core steps, so the Python validation campaign constitutes one
independent reference, not two.  A shared misconception of the physical
channel would pass both; the oracle and enumerator self-checks, together
with several hand-derived closed-form checks (Bell joint, |+> marginal,
initial-leak recapture), anchor the campaign at concrete analytic points
that do not rely on the oracle's own machinery.

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
